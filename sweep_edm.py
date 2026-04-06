"""
Sweep script (EDM): run parameter grids over PPG settings and build an HTML viewer.

Usage:
    python sweep_edm.py sweeper/configs/example_edm.yaml                   # single GPU
    torchrun --nproc_per_node=N sweep_edm.py sweeper/configs/example_edm.yaml  # multi-GPU
"""

import gc
import json
import os
import random
import sys
import torch
import torch.distributed

from dataclasses import dataclass
from typing import Optional

from edm.dnnlib.util import import_net_from_url
from util import Logger
from ppg.ppg import create_sgdm, create_ppg_linear, make_projection_matrix
from generate import (EDMDynamics, EDMSolver, generate_images_local,
                       InputsIterable, NoiseIterable, LabelsIterable,
                       ExampleImagesIterable, CombinedInputs)
from sweeper import Gallery

SEED_BY_DATASET_INDEX = True
LATENT_DIM = 3 * 64 * 64  # EDM 64x64 RGB


# ---------------------------------------------------------------------------
# RunResult

@dataclass
class RunResult:
    """Auxiliary outputs from one sweep cell, alongside the images already saved to disk."""
    logs_batch:     Optional[list] = None   # [T] per-step batch-averaged dicts
    logs_per_image: Optional[list] = None   # [N][T] per-step per-image dicts


# ---------------------------------------------------------------------------
# Dataset

def load_imgnet(path_dir, n_entries=None, indices=None, seed=0):
    rng = random.Random(seed)
    if indices is not None:
        class_labels = list(indices)
    elif n_entries is not None:
        class_labels = list(range(n_entries))
    else:
        class_labels = list(range(1000))
    path_imgs = []
    for label in class_labels:
        class_dir = os.path.join(path_dir, str(label))
        imgs = sorted(os.listdir(class_dir), key=lambda x: int(os.path.splitext(x)[0]))
        path_imgs.append(os.path.join(class_dir, rng.choice(imgs)))
    return path_imgs, class_labels, list(range(len(class_labels)))


# ---------------------------------------------------------------------------
# Pipeline setup

def setup_pipeline(fixed):
    """Load EDM model, dynamics, and solver from fixed config."""
    device = "cuda"
    net_pkl = fixed.get("net_pkl",
        "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl")
    net, encoder = import_net_from_url(net_pkl, device=device)

    num_steps = fixed.get("num_steps", 32)
    dynamics = EDMDynamics(net=net, encoder=encoder)
    solver = EDMSolver(num_steps=num_steps)

    return dynamics, solver, encoder


# ---------------------------------------------------------------------------
# Distributed init

def init_distributed():
    """Init process group if launched via torchrun. Returns (rank, world_size)."""
    if "RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world_size = 0, 1
    return rank, world_size


# ---------------------------------------------------------------------------
# Projection matrix cache

class ProjectionCache:
    """Caches QR-decomposed projection matrices to avoid recomputing across cells."""

    def __init__(self):
        self._cache = {}   # (n_features, dim_in, orthonormal, seed) -> Q

    def get_matrix(self, n_features, dim_in, dim_out, orthonormal, seed=2):
        """Return (mat, mat_inv) for projecting dim_in -> dim_out."""
        if not orthonormal:
            return make_projection_matrix(
                dim_in, dim_out, n_features=n_features,
                orthonormal=False, seed=seed, device="cuda")

        key = (n_features, dim_in, orthonormal, seed)
        if key not in self._cache:
            self._cache.clear()
            g = torch.Generator(device="cuda").manual_seed(seed)
            if n_features > 1:
                raw = torch.randn((n_features, dim_in, dim_in), generator=g, device="cuda")
                Q, _ = torch.linalg.qr(raw)
            else:
                raw = torch.randn((dim_in, dim_in), generator=g, device="cuda")
                Q, _ = torch.linalg.qr(raw)
            self._cache[key] = Q

        Q = self._cache[key]
        if n_features > 1:
            Q_slice = Q[:, :, :dim_out]
            return Q_slice.transpose(-2, -1), Q_slice
        else:
            Q_slice = Q[:, :dim_out]
            return Q_slice.T, Q_slice


# ---------------------------------------------------------------------------
# Sweep runner

class SweepRunner:
    """Owns shared state for a sweep; exposes build() and run() for Gallery.generate()."""

    def __init__(self, dynamics, solver,
                 class_labels, paths_example, seeds,
                 projection_cache,
                 log_mode=None,        # None | "batch" | "per_image" | "both"
                 snapshot_steps=None): # list[int] of denoising step indices, or None
        self.dynamics         = dynamics
        self.solver           = solver
        self.class_labels     = class_labels
        self.paths_example    = paths_example
        self.seeds            = seeds
        self.proj_cache       = projection_cache
        self.log_mode         = log_mode
        self.snapshot_steps   = snapshot_steps

        if log_mode is not None:
            self.dynamics.logger = Logger()

    # ------------------------------------------------------------------

    def build(self, params):
        """Configure dynamics from a params dict (one grid cell)."""
        self.dynamics.ppg = None
        gc.collect()
        torch.cuda.empty_cache()

        nu         = float(params.get("nu"))
        gate_type  = params.get("gate_type", "quadratic")
        dim_out    = params.get("dim_out")
        n_features = int(params.get("n_features", 1))
        channel_idx = params.get("channel_idx")
        projection = params.get("projection", "orthonormal")
        scale      = float(params.get("scale", 1.0))
        mean_scale = params.get("mean_scale", "ve")
        rescale_combined = params.get("rescale_combined", True)

        if channel_idx is not None:
            channel_idx = int(channel_idx)

        vf_inner = create_sgdm(
            type_gate=gate_type, nu=nu,
            channeled=(n_features > 1), mean_scale=mean_scale)

        if dim_out is not None:
            dim_out = round(dim_out)
            if projection == "coordinate":
                g = torch.Generator().manual_seed(0)
                indices = torch.randperm(LATENT_DIM, generator=g)[:dim_out].tolist()
                ppg = create_ppg_linear(vf_inner=vf_inner, indices=indices,
                                        channel_idx=channel_idx, scale=scale, device="cuda")
            else:
                mat, mat_inv = self.proj_cache.get_matrix(
                    n_features, LATENT_DIM, dim_out,
                    orthonormal=(projection == "orthonormal"))
                ppg = create_ppg_linear(vf_inner=vf_inner, mat=mat, mat_inv=mat_inv,
                                        channel_idx=channel_idx, scale=scale, device="cuda")
        else:
            ppg = create_ppg_linear(vf_inner=vf_inner, channel_idx=channel_idx,
                                    scale=scale, device="cuda")

        self.dynamics.rescale_combined = rescale_combined
        self.dynamics.ppg = ppg

    # ------------------------------------------------------------------

    def run(self):
        """Run inference for all images. Returns (images, snapshots, RunResult | None)."""
        if self.dynamics.logger:
            self.dynamics.logger.reset()

        inputs = self._make_inputs()
        n = len(self.class_labels)
        n_snaps       = len(self.snapshot_steps) if self.snapshot_steps else 0
        all_images    = [None] * n
        all_snapshots = [[None] * n_snaps for _ in range(n)] if self.snapshot_steps else None

        for state in generate_images_local(self.solver, self.dynamics, inputs,
                                           verbose=True, max_batch_size=3,
                                           snapshot_steps=self.snapshot_steps):
            for img, idx in zip(state.images.detach().cpu().numpy(), state.indices):
                all_images[idx] = img
            if all_snapshots is not None:
                for s, snap_batch in enumerate(state.snapshots):
                    for img, idx in zip(snap_batch.detach().cpu().numpy(), state.indices):
                        all_snapshots[idx][s] = img

        del state
        gc.collect()
        torch.cuda.empty_cache()

        result = self._make_result()
        return all_images, all_snapshots, result

    # ------------------------------------------------------------------

    def _make_inputs(self):
        base = InputsIterable(seeds=self.seeds, device="cuda")
        return CombinedInputs(
            base,
            NoiseIterable(shape=(3, 64, 64), device="cuda"),
            LabelsIterable(labels=self.class_labels, num_classes=1000, device="cuda"),
            ExampleImagesIterable(self.paths_example, device="cuda"),
        )

    def _make_result(self):
        if self.log_mode is None:
            return None
        logger = self.dynamics.logger
        return RunResult(
            logs_batch     = logger.get_batch_logs()     if self.log_mode in ("batch", "both") else None,
            logs_per_image = logger.get_per_image_logs() if self.log_mode in ("per_image", "both") else None,
        )


# ---------------------------------------------------------------------------
# Log plot generation

def make_log_plots(logs_data, cell_dir):
    """Generate and save a score-diagnostics PNG for one sweep cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = logs_data.get("logs_batch")
    if not logs:
        return

    plot_path = os.path.join(cell_dir, "logs_plot.png")
    if os.path.exists(plot_path):
        return

    steps = range(len(logs))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.patch.set_facecolor("#16213e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#e0e0e0")
        ax.xaxis.label.set_color("#e0e0e0")
        ax.title.set_color("#e94560")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")

    for ax in axes:
        ax.grid()

    for key, lbl in [("norm_comb_mean", "combined"), ("norm_model_mean", "model"), ("norm_guide_mean", "guide")]:
        if key in logs[0]:
            axes[0].plot(steps, [d[key] for d in logs], label=lbl)
    axes[0].set_title("Score Norms"); axes[0].set_xlabel("Step"); axes[0].legend(fontsize=7)
    axes[0].set_yscale('log')

    for key, lbl in [("cos_model_guide_mean", "model/guide")]:
        if key in logs[0]:
            axes[1].plot(steps, [d[key] for d in logs], label=lbl)
    axes[1].set_title("Cosine Similarities"); axes[1].set_xlabel("Step"); axes[1].set_ylim(-1.1, 1.1); axes[1].legend(fontsize=7)

    for key, lbl in [("ratio_comb_model_mean", "‖comb‖/‖model‖"), ("ratio_guide_model_mean", "‖guide‖/‖model‖")]:
        if key in logs[0]:
            axes[2].plot(steps, [d[key] for d in logs], label=lbl)
    axes[2].set_title("Score Ratios"); axes[2].set_xlabel("Step"); axes[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Baseline

def _run_baseline(baseline_dir, meta, n_images, dynamics, solver, make_inputs_fn):
    """Run or load cached baseline images (no PPG). Returns list of relative paths."""
    meta_path = os.path.join(baseline_dir, "baseline_meta.json")
    images_exist = all(
        os.path.exists(os.path.join(baseline_dir, f"img_{i}.png"))
        for i in range(n_images)
    )

    # Check if cached baseline matches current config
    cached = False
    if images_exist and os.path.exists(meta_path):
        with open(meta_path) as f:
            saved_meta = json.load(f)
        if saved_meta == meta:
            cached = True
        else:
            print(f"Baseline config changed, regenerating ({baseline_dir}).")

    if cached:
        print(f"Baseline cached ({baseline_dir}), skipping generation.")
    else:
        print("Running baseline (no PPG)...")
        dynamics.ppg = None
        dynamics.logger = None
        inputs = make_inputs_fn()
        os.makedirs(baseline_dir, exist_ok=True)
        all_images = [None] * n_images
        for state in generate_images_local(solver, dynamics, inputs,
                                           verbose=True, max_batch_size=16):
            for img, idx in zip(state.images.detach().cpu().numpy(), state.indices):
                all_images[idx] = img
        for i, img in enumerate(all_images):
            Gallery._arr_to_pil(img).save(os.path.join(baseline_dir, f"img_{i}.png"))
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print("Baseline done.")

    return [os.path.join(baseline_dir, f"img_{i}.png") for i in range(n_images)]


# ---------------------------------------------------------------------------
# Main

def main():
    torch.set_grad_enabled(False)
    rank, world_size = init_distributed()

    config_path = sys.argv[1] if len(sys.argv) > 1 else "sweeper/configs/example_edm.yaml"
    gallery     = Gallery(config_path)
    config      = gallery.config
    fixed       = config.get("fixed", {})

    dynamics, solver, encoder = setup_pipeline(fixed)

    # Load dataset
    examples_cfg = config.get("examples", {})
    paths_example, class_labels, dataset_indices = load_imgnet(
        examples_cfg.get("dataset", "data/imgnet64"),
        n_entries=examples_cfg.get("n_entries"),
        indices=examples_cfg.get("indices"),
        seed=examples_cfg.get("seed", 0))
    seeds = dataset_indices if SEED_BY_DATASET_INDEX else list(range(len(class_labels)))

    # Logging and snapshot config
    log_cfg      = config.get("logging", {})
    snap_cfg     = config.get("snapshots", {})
    log_mode     = log_cfg.get("mode") if log_cfg else None
    snapshot_steps = snap_cfg.get("steps") if snap_cfg and snap_cfg.get("enabled", True) else None

    runner = SweepRunner(
        dynamics=dynamics, solver=solver,
        class_labels=class_labels, paths_example=paths_example,
        seeds=seeds,
        projection_cache=ProjectionCache(),
        log_mode=log_mode,
        snapshot_steps=snapshot_steps,
    )

    gallery.generate(runner.build, runner.run,
                     n_images=len(class_labels), rank=rank, world_size=world_size)

    # --- Baseline run (no PPG) ---
    if rank == 0:
        baseline_dir = config.get("baseline_dir")
        if baseline_dir is None:
            print("Warning: no baseline_dir in config, skipping baseline.")
            baseline_paths = None
        else:
            n = len(class_labels)
            baseline_meta = {"fixed": fixed, "examples": examples_cfg, "n_images": n}
            baseline_paths = _run_baseline(
                baseline_dir, baseline_meta, n,
                dynamics, solver, runner._make_inputs,
            )
    else:
        baseline_paths = None

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        if world_size > 1:
            gallery.merge_manifests(world_size)
        gallery.build_html(example_paths=paths_example, plot_fn=make_log_plots,
                           baseline_paths=baseline_paths)
    if world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
