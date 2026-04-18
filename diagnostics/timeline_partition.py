"""
Diagnostic: denoising timeline using ScorePartitionGuided as the PPG inner score.

For each DDIM step, runs the sampler and captures the decoded x0_pred so we can
visually inspect what the partition-guided score is steering toward over time.

Mirrors ``timeline_frequency.py`` in structure; swaps ``create_sgdm`` for
``create_spg`` so the inner score is the new partition-guided one.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diagnostics import DiagnosticsReport
from ppg.ppg import create_spg, create_ppg_linear
from generate import (StableDiffusionDynamics, DDIMSolver, VAEEncoder,
                       InputsIterable, NoiseIterable, TextEmbeddingIterable,
                       ExampleImagesIterable, CombinedInputs)


LATENT_SHAPE = (4, 64, 64)
LATENT_DIM = int(np.prod(LATENT_SHAPE))   # 16384


def make_vis_inputs(base, encoder, prompt, path_example, pipe):
    extensions = [
        NoiseIterable(shape=LATENT_SHAPE, device="cuda"),
        TextEmbeddingIterable(prompt, pipe.tokenizer, pipe.text_encoder, device="cuda"),
    ]
    if path_example is not None:
        extensions.append(ExampleImagesIterable(path_example, encoder, device="cuda"))
    return CombinedInputs(base, *extensions)


def run_with_x0_snapshots(desc, use_net, ppg_module, snapshot_at,
                          dynamics, encoder, base, prompt, path_example, pipe,
                          num_vis_steps):
    """Run denoising and decode x0_pred at each snapshot step."""
    inputs = make_vis_inputs(base, encoder, prompt, path_example, pipe)
    inputs.rank_batches = [np.arange(len(inputs.seeds))]

    dynamics.use_net = use_net
    dynamics.ppg = ppg_module

    for state in inputs:
        dynamics.update(state)
        noise = state.noise

    scheduler_copy = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler_copy.set_timesteps(num_vis_steps)
    alphas_cumprod = scheduler_copy.alphas_cumprod

    latents = noise
    x0_snapshots = []
    for step_i, t_idx in tqdm(enumerate(scheduler_copy.timesteps), total=num_vis_steps, desc=desc):
        score = dynamics(latents, t_idx)
        sigma = dynamics.sigma(t_idx)
        noise_pred = -sigma * score

        if step_i in snapshot_at:
            alpha_bar = alphas_cumprod[t_idx].float()
            sqrt_alpha = alpha_bar.sqrt()
            x0_pred = (latents - sigma * noise_pred) / sqrt_alpha
            x0_snapshots.append(encoder.decode(x0_pred)[0].detach().cpu().numpy())

        latents = scheduler_copy.step(noise_pred, t_idx, latents).prev_sample

    return x0_snapshots


def build_partition_ppg(nu, mean_scale, basis="ambient", basis_kwargs=None,
                        type_gate="quadratic", k_min=0, gate_n=3, scale=1.0,
                        device="cuda"):
    """Build a PushPullVF with ScorePartitionGuided at the inner latent dim."""
    vf_inner = create_spg(
        type_gate=type_gate, nu=nu, d=LATENT_DIM,
        mean_scale=mean_scale, k_min=k_min, gate_n=gate_n,
        basis=basis, basis_kwargs=basis_kwargs, device=device,
    )
    return create_ppg_linear(vf_inner=vf_inner, scale=scale, device=device)


def make_timeline_report(save_path="timeline_partition.html"):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    num_vis_steps = 50
    snapshot_indices = list(range(0, num_vis_steps))
    step_labels = [f"Step {i}" for i in snapshot_indices]

    solver = DDIMSolver(scheduler=pipe.scheduler, num_inference_steps=num_vis_steps)
    dynamics = StableDiffusionDynamics(
        net=pipe.unet, encoder=VAEEncoder(pipe.vae),
        guidance_scale=7.5, scheduler=pipe.scheduler, 
        )
    encoder = dynamics.encoder
    
    img_id = 1
    if os.path.exists("data/wild-ti2i"):
        import yaml
        with open("data/wild-ti2i/wild-ti2i-real.yaml") as f:
            data = yaml.safe_load(f)
        path_example = [os.path.join("data/wild-ti2i", "data",
                                     os.path.basename(data[img_id]["init_img"]))]
        prompt = [data[img_id]["target_prompts"][0]]

    base = InputsIterable(seeds=[img_id], device="cuda")

    report = DiagnosticsReport("Denoising Timeline — ScorePartitionGuided (x0 predictions)")

    shared = dict(snapshot_at=snapshot_indices, dynamics=dynamics, encoder=encoder,
                  base=base, prompt=prompt, path_example=path_example, pipe=pipe,
                  num_vis_steps=num_vis_steps)

    # Conditions to sweep — feel free to edit.
    # Each config selects a partition basis ("ambient" | "orthonormal" | "frequency")
    # plus the usual NoiseGate / mean-scale knobs.
    configs = [
        dict(label="ambient, nu=0.0, vp",
             nu=0.0, mean_scale="vp", basis="ambient"),
        dict(label="orthonormal, nu=0.0, vp",
             nu=0.0, mean_scale="vp", basis="orthonormal", basis_kwargs=dict(seed=0)),
        dict(label="frequency, nu=0.0, vp",
             nu=0.0, mean_scale="vp", basis="frequency",
             basis_kwargs=dict(shape=LATENT_SHAPE)),
        # dict(label="ambient, nu=0.5, vp",
        #      nu=0.5, mean_scale="vp", basis="ambient"),
        # dict(label="orthonormal, nu=0.5, vp",
        #      nu=0.5, mean_scale="vp", basis="orthonormal", basis_kwargs=dict(seed=0)),
        # dict(label="frequency, nu=0.5, vp",
        #      nu=0.5, mean_scale="vp", basis="frequency",
        #      basis_kwargs=dict(shape=LATENT_SHAPE)),
    ]

    with torch.no_grad():
        sd = run_with_x0_snapshots("SD", use_net=True, ppg_module=None, **shared)

        results = []
        for cfg in configs:
            ppg_module = build_partition_ppg(
                nu=cfg["nu"], mean_scale=cfg["mean_scale"],
                basis=cfg.get("basis", "ambient"),
                basis_kwargs=cfg.get("basis_kwargs"),
            )
            ppg_only = run_with_x0_snapshots(
                f"{cfg['label']} — PPG only", use_net=False, ppg_module=ppg_module, **shared)
            both = run_with_x0_snapshots(
                f"{cfg['label']} — SD + PPG", use_net=True, ppg_module=ppg_module, **shared)
            results.append((cfg["label"], ppg_only, both))

        dynamics.use_net = True
        dynamics.ppg = None

    report.add_header("Decoded x0 predictions per step (ScorePartitionGuided)")
    report.add_note(
        "Partition-guided score with uniform-rank scheduler tied to a quadratic noise gate. "
        "k_min=0 so the gate may fully deactivate guidance at low σ. "
        "Basis varies per row: ambient (identity), random orthonormal, and "
        "per-channel 2D DCT (frequency, zig-zag ordered)."
    )

    for label, ppg_only, both in results:
        report.add_note(f"**{label}**")
        report.add_image_slider(
            [sd, ppg_only, both],
            row_labels=["SD", "PPG only", "SD + PPG"],
            step_labels=step_labels,
        )

    report.save(save_path)
    print(f"Timeline report saved to {save_path}")


if __name__ == "__main__":
    make_timeline_report("timeline_partition.html")
