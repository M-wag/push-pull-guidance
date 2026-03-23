import gc
import hashlib
import numpy as np
import os
import torch
import yaml

from diffusers import StableDiffusionPipeline, DDIMScheduler

from diagnostics import DiagnosticsReport
from util import load_images
from ppg.ppg import create_sgdm, create_ppg, create_ppg_linear, ChannelSelectMap, FlattenMap, RandomLinearLatentMap, PullbackLinear, PushPullVF
from generate import (StableDiffusionDynamics, DDIMSolver, generate_images, ddim_invert,
                       InputsIterable, NoiseIterable, PrecomputedNoiseIterable,
                       TextEmbeddingIterable, ExampleImagesIterable, CombinedInputs)


def load_wildti2i(path_dir, n_entries=None):
    with open(os.path.join(path_dir, "wild-ti2i-real.yaml")) as f:
        data = yaml.safe_load(f)
    if n_entries is not None:
        data = data[:n_entries]
    path_imgs = [os.path.join(path_dir, "data", os.path.basename(entry["init_img"])) for entry in data]
    target_prompts = [entry["target_prompts"][0] for entry in data]
    return path_imgs, target_prompts

def reconstruct_images(encoder, images):
    images = images.to(torch.float32) / 127.5 - 1
    latents = encoder.encode(images).latent_dist.sample()
    recons = encoder.decode(latents).sample
    return (recons.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)
    
if __name__ == "__main__":
    with torch.no_grad():
        # Setup hf diffusers pipeline
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        # Setup solver and dynamics
        solver = DDIMSolver(scheduler=pipe.scheduler, num_inference_steps=50)
        dynamics = StableDiffusionDynamics(unet=pipe.unet, vae=pipe.vae, guidance_scale=7.5, scheduler=pipe.scheduler)
        # Import examples
        paths_example, prompts = load_wildti2i("data/wild-ti2i", n_entries=3)
        examples = load_images(paths_example, rescale=False, dtype=torch.uint8, device="cuda")

        
        def get_noise_levels():
            alphas = solver.scheduler.alphas_cumprod
            t_idxs = solver.scheduler.timesteps
            alphas_at_t = alphas[t_idxs]
            noise_levels = (1 - alphas_at_t).sqrt()
            return noise_levels

        # Precompute DDIM-inverted noise from examples
        example_latents = dynamics.encoder.encode(
            load_images(paths_example, device="cuda", rescale=True))
        text_emb_iter = TextEmbeddingIterable(prompts, pipe.tokenizer, pipe.text_encoder, device="cuda")
        text_embeddings = text_emb_iter._encode(prompts)
        inverted_noise = ddim_invert(example_latents, pipe.unet, pipe.scheduler,
                                     text_embeddings=text_embeddings)

        use_ddim_inv = True  # toggle between random noise and DDIM-inverted noise

        def make_inputs():
            base = InputsIterable(seeds=range(0, len(prompts)), device="cuda")
            if use_ddim_inv:
                noise_ext = PrecomputedNoiseIterable(inverted_noise)
            else:
                noise_ext = NoiseIterable(shape=(4, 64, 64), device="cuda")
            return CombinedInputs(
                base,
                noise_ext,
                TextEmbeddingIterable(prompts, pipe.tokenizer, pipe.text_encoder, device="cuda"),
                ExampleImagesIterable(paths_example, dynamics.encoder, device="cuda"),
            )

        def run_and_report(header):
            report.add_header(header)
            for state in generate_images(solver, dynamics, make_inputs(), verbose=True):
                pass
            report.add_image_row(state.images.detach().cpu().numpy(), captions=prompts)
            return state.images


        def run_images():
            for state in generate_images(solver, dynamics, make_inputs(), verbose=True):
                pass
            images = [img for img in state.images.detach().cpu().numpy()]
            del state
            gc.collect()
            torch.cuda.empty_cache()
            return images

        def sweep_grid(report, row_name, row_values, col_name, col_values, build_fn,
                       cache_dir="sweeps/cache"):
            """Run a 2D sweep and add one image grid per batch element.

            build_fn(row_val, col_val) should configure dynamics (e.g. set dynamics.ppg).
            Completed cells are cached to disk so crashed runs can resume.
            """
            os.makedirs(cache_dir, exist_ok=True)
            row_labels = [f"{row_name}={v}" for v in row_values]
            col_labels = [f"{col_name}={v}" for v in col_values]

            def _cache_path(ri, ci):
                key = f"{row_name}={row_values[ri]}_{col_name}={col_values[ci]}"
                h = hashlib.md5(key.encode()).hexdigest()[:8]
                return os.path.join(cache_dir, f"{ri}_{ci}_{h}.npy")

            # results[r][c] = list of images (one per batch element)
            results = [[None] * len(col_values) for _ in row_values]
            for ri, rv in enumerate(row_values):
                for ci, cv in enumerate(col_values):
                    path = _cache_path(ri, ci)
                    if os.path.exists(path):
                        print(f"[cache hit] {row_name}={rv}, {col_name}={cv}")
                        cached = np.load(path)
                        results[ri][ci] = [cached[i] for i in range(len(cached))]
                    else:
                        print(f"[running]   {row_name}={rv}, {col_name}={cv}")
                        build_fn(rv, cv)
                        imgs = run_images()
                        np.save(path, np.stack(imgs))
                        results[ri][ci] = imgs

            n_images = len(results[0][0])
            for img_idx in range(n_images):
                report.add_header(f"Image {img_idx}", level=3)
                grid = [[results[ri][ci][img_idx] for ci in range(len(col_values))]
                        for ri in range(len(row_values))]
                report.add_image_grid(grid, row_labels=row_labels, col_labels=col_labels)

        # Init report
        report = DiagnosticsReport("SD 1.5 Guidance in Pixel Space")
        report.add_header("Examples")
        report.add_image_row(paths_example, captions=prompts)

        # No PPG baseline
        report.add_header("No PPG (baseline)")
        no_ppg_dir = "data/images/no-ppg"
        no_ppg_paths = sorted(os.path.join(no_ppg_dir, f) for f in os.listdir(no_ppg_dir) if os.path.isfile(os.path.join(no_ppg_dir, f)))
        report.add_image_row(no_ppg_paths)

        nus = get_noise_levels().tolist()[4::5]
        # nus = [1.0, get_noise_levels().tolist()[-1]] 
        
        _cached_mat = {}

        def _make_mat(n_features, dim_out, orthonormal=True):
            """Generate random matrix (and inverse), cached by (n_features, dim_out)."""
            n_features = int(n_features)
            dim_out = int(dim_out)
            key = (n_features, dim_out, orthonormal)
            if key not in _cached_mat:
                _cached_mat.clear()
                g = torch.Generator().manual_seed(2)
                if orthonormal:
                    if n_features > 1:
                        raw = torch.randn((n_features, 16384, dim_out), generator=g).to(device="cuda")
                        Q, _ = torch.linalg.qr(raw)
                        mat = Q.transpose(-2, -1)
                        mat_inv = Q
                    else:
                        raw = torch.randn((16384, dim_out), generator=g).to(device="cuda")
                        Q, _ = torch.linalg.qr(raw)
                        mat = Q.T
                        mat_inv = Q
                else:
                    if n_features > 1:
                        mat = torch.randn((n_features, dim_out, 16384), generator=g).to(device="cuda")
                    else:
                        mat = torch.randn((dim_out, 16384), generator=g).to(device="cuda")
                    mat_inv = torch.linalg.pinv(mat)
                _cached_mat[key] = (mat, mat_inv)
            return _cached_mat[key]

        def _build(nu, channel_idx=None, n_features=None, dim_out=None):
            dynamics.ppg = None
            gc.collect()
            torch.cuda.empty_cache()
            nf = n_features or 1
            channeled = nf > 1
            vf_inner = create_sgdm(type_gate="quadratic", nu=nu, channeled=channeled)
            if dim_out is not None:
                mat, mat_inv = _make_mat(nf, dim_out, orthonormal=False)
                ppg = create_ppg_linear(vf_inner=vf_inner, mat=mat, mat_inv=mat_inv,
                                        channel_idx=channel_idx, device="cuda")
            else:
                ppg = create_ppg_linear(vf_inner=vf_inner, channel_idx=channel_idx, device="cuda")
            dynamics.ppg = ppg

        # --- Sweep: n_features vs nu (fixed total_dim=6000) ---
        # total_dim = 2024
        # n_features_list = [1, 2, 4, 8]
        #
        # report.add_header(f"Sweep: n_features vs nu (total_dim={total_dim})")
        # sweep_grid(report, "n_features", n_features_list, "$\\nu$", [round(n, 3) for n in nus],
        #            lambda nf, nu: _build(nu, n_features=nf, dim_out=total_dim // nf))

        # --- Sweep: dim_out vs nu (fixed n_features=1) ---
        dim_outs = np.linspace(250, 8000, 10).astype(int)
        report.add_header("Sweep: dim_out vs nu")
        sweep_grid(report, "dim_out", dim_outs, "$\\nu$", [round(n, 3) for n in nus],
                   lambda dim_out, nu: _build(nu, dim_out=dim_out))

        # --- Sweep: channel vs nu (ambient map per channel) ---
        # channels = [0, 1, 2, 3]
        # report.add_header("Sweep: channel vs nu")
        # sweep_grid(report, "channel", channels, "$\\nu$", [round(n, 3) for n in nus],
        #            lambda ch, nu: _build(nu, channel_idx=ch))

        # Save report
        report.save(f"sweeps/old/dimout_ddim_quadratic_old.html")

