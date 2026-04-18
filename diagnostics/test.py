"""
Diagnostic: compare the denoising timescales of Stable Diffusion vs PPG.

Plots the score magnitudes and noise-gate values over the DDIM timestep schedule
so we can see whether the two processes are aligned in time.
Also runs actual denoising with SD-only and PPG-only to visualize the processes.
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
from ppg.ppg import NoiseGate, create_sgdm, create_ppg_linear
from generate import (StableDiffusionDynamics, DDIMSolver, VAEEncoder,
                       InputsIterable, NoiseIterable, TextEmbeddingIterable,
                       ExampleImagesIterable, CombinedInputs, generate_images_local)
from util import load_images, Logger

def make_timeline_report(save_path="timeline.html"):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler = pipe.scheduler

    num_steps = 50
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps  # descending from ~999 to 0
    alphas_cumprod = scheduler.alphas_cumprod

    # Compute VP sigma and EDM sigma at each DDIM timestep
    alpha_bars = alphas_cumprod[timesteps].float()
    sigma_vp = (1 - alpha_bars).sqrt()          # √(1-ᾱ), range [0, 1)
    sigma_edm = ((1 - alpha_bars) / alpha_bars).sqrt()  # √((1-ᾱ)/ᾱ)
    sqrt_alpha = alpha_bars.sqrt()
    step_idx = np.arange(num_steps)

    # Noise gates with different nu values
    nus = [1e-4, 0.01, 0.05, 0.1, 0.3, 0.5]

    report = DiagnosticsReport("Denoising Timeline: SD vs PPG")

    report.add_header("Denoising Process: Mean Scale + Rescale Comparison + CoodinateProjection")
    # Setup dynamics and solver
    num_vis_steps = 50
    snapshot_indices = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49]
    solver = DDIMSolver(scheduler=pipe.scheduler, num_inference_steps=num_vis_steps)
    dynamics = StableDiffusionDynamics(
        unet=pipe.unet, vae=pipe.vae, guidance_scale=7.5, scheduler=pipe.scheduler)
    encoder = dynamics.encoder

    dynamics.logger = Logger()

    # Check if example images exist
    if os.path.exists("data/wild-ti2i"):
        import yaml
        with open("data/wild-ti2i/wild-ti2i-real.yaml") as f:
            data = yaml.safe_load(f)
        path_example = [os.path.join("data/wild-ti2i", "data", os.path.basename(data[0]["init_img"]))]
        prompt = [data[0]["target_prompts"][0]]

    base = InputsIterable(seeds=[0], device="cuda")

    def make_vis_inputs():
        extensions = [
            NoiseIterable(shape=(4, 64, 64), device="cuda"),
            TextEmbeddingIterable(prompt, pipe.tokenizer, pipe.text_encoder, device="cuda"),
        ]
        if path_example is not None:
            extensions.append(ExampleImagesIterable(path_example, encoder, device="cuda"))
        return CombinedInputs(base, *extensions)

    def run_with_snapshots(desc, use_unet, ppg_module, snapshot_at):
        """Run denoising and capture decoded snapshots at specified step indices.

        Returns (snapshots, logs) where logs is the per-step score diagnostics.
        """
        inputs = make_vis_inputs()
        inputs.rank_batches = [np.arange(len(inputs.seeds))]

        dynamics.use_unet = use_unet
        dynamics.ppg = ppg_module
        dynamics.logger.reset()

        for state in inputs:
            dynamics.update(state)
            noise = state.noise

        scheduler_copy = DDIMScheduler.from_config(pipe.scheduler.config)
        scheduler_copy.set_timesteps(num_vis_steps)

        latents = noise
        snapshots = []
        for step_i, t_idx in tqdm(enumerate(scheduler_copy.timesteps), total=num_vis_steps, desc=desc):
            score = dynamics(latents, t_idx)
            sigma = dynamics.sigma(t_idx)
            noise_pred = -sigma * score
            latents = scheduler_copy.step(noise_pred, t_idx, latents).prev_sample
            ### REMOVE THIS LATER
            if dynamics.use_unet and (dynamics.ppg is not None):
                nu = dynamics.ppg.nu
                sigma_add = torch.sqrt(sigma**4 / (2*sigma**2 + nu**2))
                # latents = latents + torch.randn_like(latents) * sigma_add
            ### REMOVE THIS LATER
            if step_i in snapshot_at:
                img = encoder.decode(latents)
                snapshots.append(img[0].detach().cpu().numpy())

        return snapshots, dynamics.logger.get_batch_logs()

    def run_condition(label, mean_scale, rescale_combined):
        """Run PPG-only and SD+PPG for a given condition. Returns (ppg, ppg_logs, both, both_logs)."""
        g = torch.Generator().manual_seed(0)
        indices = torch.randperm(16384, generator=g)[:0].tolist()
        ppg_module = create_ppg_linear(
            vf_inner=create_sgdm(type_gate="quadratic", nu=nu, mean_scale=mean_scale),
            scale=1.0)

        dynamics.rescale_combined = rescale_combined

        ppg,  ppg_logs  = run_with_snapshots(f"{label} — PPG only", use_unet=False, ppg_module=ppg_module, snapshot_at=snapshot_indices)
        both, both_logs = run_with_snapshots(f"{label} — SD + PPG",  use_unet=True,  ppg_module=ppg_module, snapshot_at=snapshot_indices)
        return ppg, ppg_logs, both, both_logs

    nu = 0.2
    step_labels = [f"Step {i}" for i in snapshot_indices]

    with torch.no_grad():
        sd,   sd_logs               = run_with_snapshots(f"SD", use_unet=True, ppg_module=None, snapshot_at=snapshot_indices)
        ppg, ppg_logs, both, both_logs = run_condition("VP", mean_scale="vp", rescale_combined=True)

        # Restore
        dynamics.use_unet = True
        dynamics.ppg = None
        dynamics.rescale_combined = True

    report.add_note("$s(t) = 1$ + variance rescaling — VP mean scale with product-of-Gaussians correction")
    report.add_image_slider(
        [sd, ppg, both],
        row_labels=["SD", "PPG only", "SD + PPG"],
        step_labels=step_labels,
    )

    report.add_header("Score Diagnostics")
    tmp_logger = Logger()
    for label, logs in [("SD", sd_logs), ("PPG only", ppg_logs), ("SD + PPG", both_logs)]:
        tmp_logger._step_batch = logs
        fig = tmp_logger.plot()
        if fig is not None:
            report.add_note(label)
            report.add_image(fig)
            plt.close(fig)

    report.save(save_path)
    print(f"Timeline report saved to {save_path}")


if __name__ == "__main__":
    make_timeline_report("sweeps/nu_is_0.2_with_variance_rescale.html")
