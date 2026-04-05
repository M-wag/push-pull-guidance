"""
Sweep script: run parameter grids over PPG settings and build an HTML viewer.

Usage:
    python sweep.py sweeper/configs/example.yaml                   # single GPU
    torchrun --nproc_per_node=N sweep.py sweeper/configs/example.yaml  # multi-GPU
"""

import gc
import os
import sys
import torch
import torch.distributed
import yaml

from dataclasses import dataclass
from typing import Optional
from diffusers import StableDiffusionPipeline, DDIMScheduler

from util import load_images, Logger
from ppg.ppg import create_sgdm, create_ppg_linear, make_projection_matrix
from generate import (StableDiffusionDynamics, DDIMSolver, EDMSolver, NoiseScheduleMap,
                       VEDynamicsWrapper, generate_images_local,
                       ddim_invert, InputsIterable, NoiseIterable, PrecomputedNoiseIterable,
                       TextEmbeddingIterable, ExampleImagesIterable, CombinedInputs)
from sweeper import Gallery

SEED_BY_DATASET_INDEX = True


# ---------------------------------------------------------------------------
# RunResult

@dataclass
class RunResult:
    """Auxiliary outputs from one sweep cell, alongside the images already saved to disk."""
    logs_batch:     Optional[list] = None   # [T] per-step batch-averaged dicts
    logs_per_image: Optional[list] = None   # [N][T] per-step per-image dicts


# ---------------------------------------------------------------------------
# Dataset

def load_wildti2i(path_dir, n_entries=None, indices=None):
    with open(os.path.join(path_dir, "wild-ti2i-real.yaml")) as f:
        data = yaml.safe_load(f)
    if indices is not None:
        dataset_indices = list(indices)
        data = [data[i] for i in indices]
    elif n_entries is not None:
        dataset_indices = list(range(n_entries))
        data = data[:n_entries]
    else:
        dataset_indices = list(range(len(data)))
    path_imgs = [os.path.join(path_dir, "data", os.path.basename(e["init_img"])) for e in data]
    target_prompts = [e["target_prompts"][0] for e in data]
    return path_imgs, target_prompts, dataset_indices


# ---------------------------------------------------------------------------
# Pipeline setup

def setup_pipeline(fixed):
    """Load SD pipeline, dynamics, and solver from fixed config."""
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    num_steps      = fixed.get("num_inference_steps", 50)
    guidance_scale = fixed.get("guidance_scale", 7.5)

    vp_dynamics = StableDiffusionDynamics(
        unet=pipe.unet, vae=pipe.vae,
        guidance_scale=guidance_scale, scheduler=pipe.scheduler)

    sampler = fixed.get("sampler", "ddim")
    if sampler == "edm":
        schedule_map = NoiseScheduleMap(pipe.scheduler.alphas_cumprod).to("cuda")
        solver   = EDMSolver(num_steps=num_steps, sigma_max=schedule_map.sigma_max)
        dynamics = VEDynamicsWrapper(vp_dynamics, schedule_map)
    else:
        solver   = DDIMSolver(scheduler=pipe.scheduler, num_inference_steps=num_steps)
        dynamics = vp_dynamics

    return pipe, dynamics, solver, num_steps


# ---------------------------------------------------------------------------
# DDIM inversion precompute

def precompute_ddim(pipe, dynamics, paths_example, prompts, batch_size=4, num_inference_steps=50):
    """Precompute DDIM-inverted noise from example images. Returns tensor on CUDA."""
    if not isinstance(pipe.scheduler, DDIMScheduler):
        raise ValueError(f"DDIM inversion requires DDIMScheduler, got {type(pipe.scheduler).__name__}")
    example_latents  = dynamics.encoder.encode(load_images(paths_example, device="cuda", rescale=True))
    text_embeddings  = TextEmbeddingIterable(
        prompts, pipe.tokenizer, pipe.text_encoder, device="cuda")._encode(prompts)
    return ddim_invert(
        example_latents, pipe.unet, pipe.scheduler,
        text_embeddings=text_embeddings, batch_size=batch_size,
        num_inference_steps=num_inference_steps)


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

    def __init__(self, pipe, dynamics, solver,
                 prompts, paths_example, seeds, inverted_noise,
                 projection_cache,
                 log_mode=None,        # None | "batch" | "per_image" | "both"
                 snapshot_steps=None): # list[int] of denoising step indices, or None
        self.pipe             = pipe
        self.dynamics         = dynamics
        self.solver           = solver
        self.prompts          = prompts
        self.paths_example    = paths_example
        self.seeds            = seeds
        self.inverted_noise   = inverted_noise
        self.proj_cache       = projection_cache
        self.log_mode         = log_mode
        self.snapshot_steps   = snapshot_steps
        self.noise_source     = "random"

        if log_mode is not None:
            self.dynamics.logger = Logger()

    # ------------------------------------------------------------------

    def build(self, params):
        """Configure dynamics from a params dict (one grid cell)."""
        self.dynamics.ppg = None
        gc.collect()
        torch.cuda.empty_cache()

        self.noise_source = params.get("noise_source", "random")

        nu         = float(params.get("nu"))
        gate_type  = params.get("gate_type", "quadratic")
        eta        = float(params.get("eta", 0.0))
        dim_out    = params.get("dim_out")
        n_features = int(params.get("n_features", 1))
        channel_idx = params.get("channel_idx")
        projection = params.get("projection", "orthonormal")
        scale      = float(params.get("scale", 1.0))
        mean_scale = params.get("mean_scale", "vp")
        rescale_combined = params.get("rescale_combined", True)

        if channel_idx is not None:
            channel_idx = int(channel_idx)
        if hasattr(self.solver, "eta"):
            self.solver.eta = eta

        vf_inner = create_sgdm(
            type_gate=gate_type, nu=nu,
            channeled=(n_features > 1), mean_scale=mean_scale)

        if dim_out is not None:
            dim_out = round(dim_out)
            if projection == "coordinate":
                g = torch.Generator().manual_seed(0)
                indices = torch.randperm(16384, generator=g)[:dim_out].tolist()
                ppg = create_ppg_linear(vf_inner=vf_inner, indices=indices,
                                        channel_idx=channel_idx, scale=scale, device="cuda")
            else:
                mat, mat_inv = self.proj_cache.get_matrix(
                    n_features, 16384, dim_out,
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
        """Run inference for all prompts. Returns (images, snapshots, RunResult | None)."""
        if self.dynamics.logger:
            self.dynamics.logger.reset()

        inputs = self._make_inputs()
        n = len(self.prompts)
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
        noise_ext = (PrecomputedNoiseIterable(self.inverted_noise)
                     if self.noise_source == "ddim_inversion"
                     else NoiseIterable(shape=(4, 64, 64), device="cuda"))
        return CombinedInputs(
            base,
            noise_ext,
            TextEmbeddingIterable(self.prompts, self.pipe.tokenizer,
                                  self.pipe.text_encoder, device="cuda"),
            ExampleImagesIterable(self.paths_example, self.dynamics.encoder, device="cuda"),
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
    """Generate and save a score-diagnostics PNG for one sweep cell.

    Called by Gallery._generate_plots for each cell that has a logs.json.
    Saves logs_plot.png into cell_dir; skips if already present.
    """
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
# Main

def main():
    torch.set_grad_enabled(False)
    rank, world_size = init_distributed()

    config_path = sys.argv[1] if len(sys.argv) > 1 else "sweeper/configs/example.yaml"
    gallery     = Gallery(config_path)
    config      = gallery.config
    fixed       = config.get("fixed", {})

    pipe, dynamics, solver, num_steps = setup_pipeline(fixed)

    # Load dataset
    examples_cfg = config.get("examples", {})
    paths_example, prompts, dataset_indices = load_wildti2i(
        examples_cfg.get("dataset", "data/wild-ti2i"),
        n_entries=examples_cfg.get("n_entries"),
        indices=examples_cfg.get("indices"))
    seeds = dataset_indices if SEED_BY_DATASET_INDEX else range(len(prompts))

    # Precompute DDIM inversion if any cell needs it
    noise_sources   = config.get("axes", {}).get("noise_source", {}).get("values", [fixed.get("noise_source")])
    inverted_noise  = None
    if "ddim_inversion" in noise_sources:
        if rank == 0:
            inverted_noise = precompute_ddim(pipe, dynamics, paths_example, prompts,
                                             num_inference_steps=num_steps)
        if world_size > 1:
            if rank == 0:
                shape_tensor = torch.tensor(list(inverted_noise.shape), device="cuda")
            else:
                shape_tensor = torch.zeros(4, dtype=torch.long, device="cuda")
            torch.distributed.broadcast(shape_tensor, src=0)
            if rank != 0:
                inverted_noise = torch.zeros(*shape_tensor.tolist(), device="cuda")
            torch.distributed.broadcast(inverted_noise, src=0)
            torch.distributed.barrier()

    # Logging and snapshot config
    log_cfg      = config.get("logging", {})
    snap_cfg     = config.get("snapshots", {})
    log_mode     = log_cfg.get("mode") if log_cfg else None
    snapshot_steps = snap_cfg.get("steps") if snap_cfg and snap_cfg.get("enabled", True) else None

    runner = SweepRunner(
        pipe=pipe, dynamics=dynamics, solver=solver,
        prompts=prompts, paths_example=paths_example,
        seeds=seeds, inverted_noise=inverted_noise,
        projection_cache=ProjectionCache(),
        log_mode=log_mode,
        snapshot_steps=snapshot_steps,
    )

    gallery.generate(runner.build, runner.run,
                     n_images=len(prompts), rank=rank, world_size=world_size)

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        if world_size > 1:
            gallery.merge_manifests(world_size)
        gallery.build_html(example_paths=paths_example, prompts=prompts, plot_fn=make_log_plots)
    if world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
