"""
Experiment: VAE encode→decode projection comparison.

Encoders:
  - SD-VAE           (AutoencoderKL,          runwayml/stable-diffusion-v1-5)
  - SD-Turbo         (AutoencoderKL,          stabilityai/sd-turbo)
  - Asymmetric VQ-GAN (AsymmetricAutoencoderKL, cross-attention/asymmetric-autoencoder-kl-x-1-5)
  - Consistency Decoder (ConsistencyDecoderVAE, openai/consistency-decoder)
  - Tiny AutoEncoder (AutoencoderTiny,         madebyollin/taesd)

Images: N_IMAGES randomly sampled from different classes in data/imgnet512.
Per-image section: table of encoder × input size (64, 128, 256, 384, 512 bilinear-up, 512 original).
All images displayed at DISPLAY_SIZE for consistent grid layout.
Output: diagnostics/vae_projection.html
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from diagnostics import DiagnosticsReport
from ppg.ppg import HFLatentMap

# ── Config ─────────────────────────────────────────────────────────────────────

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_IMAGES     = 6
DISPLAY_SIZE = 512   # all grid images are resized to this for consistent layout
BENCHMARK_BATCH_SIZE = 1
BENCHMARK_N_RUNS     = 20
BENCHMARK_N_WARMUP   = 5

ENCODERS = [
    ("SD-VAE (KL)",         "kl",          "runwayml/stable-diffusion-v1-5"),
    ("SD-Turbo (KL)",       "kl",          "stabilityai/sd-turbo"),
    ("Asymmetric VQ-GAN",   "asymmetric",  "cross-attention/asymmetric-autoencoder-kl-x-1-5"),
    ("Consistency Decoder", "consistency", "openai/consistency-decoder"),
    ("Tiny AutoEncoder",    "tiny",        "madebyollin/taesd"),
]

UPSAMPLE_SIZES = [128, 256, 384, 512]   # bilinear-upscaled from 64px before encoding

# ── Data helpers ───────────────────────────────────────────────────────────────

def collect_diverse_paths(root: Path, ext: str, n: int, seed: int = 0) -> list[Path]:
    """One image from each randomly sampled class directory, up to n images."""
    import random
    rng = random.Random(seed)
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    rng.shuffle(class_dirs)
    paths = []
    for class_dir in class_dirs:
        files = sorted(class_dir.glob(f"*{ext}"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        if files:
            paths.append(rng.choice(files))
            if len(paths) == n:
                break
    return paths


_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [0,1] -> [-1,1]
])


def pil_load_at_size(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = transforms.functional.resize(img, size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    img = transforms.functional.center_crop(img, size)
    return img


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return _to_tensor(img)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(3, H, W) in [-1, 1] -> PIL RGB image."""
    arr = ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def pil_resize(img: Image.Image, size: int, resample=Image.BICUBIC) -> Image.Image:
    return img.resize((size, size), resample)


# ── Projection & benchmarking ──────────────────────────────────────────────────

@torch.no_grad()
def project(latent_map: HFLatentMap, batch: torch.Tensor) -> list[Image.Image]:
    """Encode → decode each image in batch (one at a time). Returns PIL images at DISPLAY_SIZE."""
    results = []
    for img in batch:
        x = img.unsqueeze(0).to(DEVICE)
        z = latent_map(x)
        out = tensor_to_pil(latent_map.inv(z).cpu().squeeze(0))
        results.append(pil_resize(out, DISPLAY_SIZE))
    return results


@torch.no_grad()
def benchmark(latent_map: HFLatentMap, batch: torch.Tensor) -> tuple[float, float]:
    """Returns (mean, std) per-image latency in ms over a fixed batch."""
    x = batch.to(DEVICE)
    for _ in range(BENCHMARK_N_WARMUP):
        latent_map.inv(latent_map(x))
    torch.cuda.synchronize()

    times = []
    for _ in range(BENCHMARK_N_RUNS):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        latent_map.inv(latent_map(x))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / len(batch))

    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    report = DiagnosticsReport("VAE Projection: encoder × input size")
    report.add_note(
        "Each section is one image. Rows: input + one row per encoder. "
        "Columns: 64×64 direct, bilinear-upscaled 128/256/384/512, original 512×512. "
        f"All images displayed at {DISPLAY_SIZE}×{DISPLAY_SIZE}."
    )

    root   = Path(__file__).parent.parent
    imgnet = root / "data/imgnet512"
    paths  = collect_diverse_paths(imgnet, ".JPEG", N_IMAGES, seed=1)
    print(f"Sampled {len(paths)} images from {len(paths)} different classes.")

    # ── Precompute inputs ──────────────────────────────────────────────────────
    pils_512 = [pil_load_at_size(p, 512) for p in paths]
    pils_64  = [pil_resize(img, 64) for img in pils_512]
    pils_up  = {size: [pil_resize(img, size, Image.BILINEAR) for img in pils_64]
                for size in UPSAMPLE_SIZES}

    batch_512 = torch.stack([pil_to_tensor(img) for img in pils_512])
    batch_64  = torch.stack([pil_to_tensor(img) for img in pils_64])
    batch_up  = {size: torch.stack([pil_to_tensor(img) for img in imgs])
                 for size, imgs in pils_up.items()}

    # benchmark uses a fixed-size batch at 512px (representative of generation)
    bench_input = torch.stack([pil_to_tensor(pils_512[0])] * BENCHMARK_BATCH_SIZE)

    # col_keys orders all input conditions; "64" and "orig" are special
    col_keys   = ["64"] + UPSAMPLE_SIZES + ["orig"]
    col_labels = ["64×64"] + [f"{s}×{s}↑" for s in UPSAMPLE_SIZES] + ["512×512"]

    # display-size inputs (resize for consistent grid)
    disp_inputs = {
        "64":   [pil_resize(img, DISPLAY_SIZE) for img in pils_64],
        "orig": [pil_resize(img, DISPLAY_SIZE) for img in pils_512],
        **{size: [pil_resize(img, DISPLAY_SIZE) for img in pils_up[size]]
           for size in UPSAMPLE_SIZES},
    }

    # ── Run all encoders ───────────────────────────────────────────────────────
    # recons[enc_label][key] = list[PIL at DISPLAY_SIZE], one per image
    recons:     dict[str, dict] = {}
    bench_stats: dict[str, tuple[float, float]] = {}

    for enc_label, ae_type, hf_name in ENCODERS:
        print(f"\nLoading {enc_label} ({hf_name})...")
        latent_map = HFLatentMap(ae_type, hf_name).to(DEVICE).eval()
        recons[enc_label] = {}

        print("  → 64px")
        recons[enc_label]["64"]   = project(latent_map, batch_64)
        for size in UPSAMPLE_SIZES:
            print(f"  → 64→{size}px")
            recons[enc_label][size] = project(latent_map, batch_up[size])
        print("  → 512px (original)")
        recons[enc_label]["orig"] = project(latent_map, batch_512)

        print("  → benchmarking...")
        bench_stats[enc_label] = benchmark(latent_map, bench_input)

        del latent_map
        torch.cuda.empty_cache()

    enc_labels = [l for l, *_ in ENCODERS]

    # ── Benchmark summary ──────────────────────────────────────────────────────
    report.add_header("Benchmark — encode+decode latency", level=2)
    report.add_note(
        f"Batch size: {BENCHMARK_BATCH_SIZE} images at 512×512. "
        f"{BENCHMARK_N_RUNS} runs after {BENCHMARK_N_WARMUP} warmup. "
        "Measured with CUDA events."
    )
    bench_lines = [f"{l}: {bench_stats[l][0]:.1f} ± {bench_stats[l][1]:.1f} ms/image"
                   for l in enc_labels]
    report.add_note(" | ".join(bench_lines))
    report.add_separator()

    # ── One section per image ──────────────────────────────────────────────────
    for i, path in enumerate(paths):
        report.add_header(f"Class {path.parent.name}", level=2)
        grid = (
            [[disp_inputs[k][i] for k in col_keys]]
            + [[recons[l][k][i]  for k in col_keys] for l in enc_labels]
        )
        report.add_image_grid(
            grid,
            row_labels=["Input"] + enc_labels,
            col_labels=col_labels,
        )
        report.add_separator()

    out = Path(__file__).parent / "vae_projection.html"
    report.save(str(out))


if __name__ == "__main__":
    main()
