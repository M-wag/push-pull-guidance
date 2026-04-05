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
from util import load_images


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

    # # ── Section 1: Noise schedules ──
    # report.add_header("Noise Schedules over DDIM Steps")
    # report.add_note("VP sigma $\\sigma_{VP} = \\sqrt{1-\\bar\\alpha}$ vs "
    #                 "EDM sigma $\\sigma_{EDM} = \\sqrt{(1-\\bar\\alpha)/\\bar\\alpha}$. "
    #                 "PPG receives $\\sigma_{VP}$ when used with StableDiffusionDynamics.")
    #
    # fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    #
    # axes[0].plot(step_idx, sigma_vp.numpy(), label="$\\sigma_{VP}$", linewidth=2)
    # axes[0].plot(step_idx, sigma_edm.numpy(), label="$\\sigma_{EDM}$", linewidth=2, linestyle="--")
    # axes[0].set_xlabel("DDIM step (0 = noisiest)")
    # axes[0].set_ylabel("$\\sigma$")
    # axes[0].set_title("Noise Level")
    # axes[0].legend()
    # axes[0].set_yscale("log")
    # axes[0].grid(True, alpha=0.3)
    #
    # axes[1].plot(step_idx, alpha_bars.numpy(), label="$\\bar\\alpha$", linewidth=2)
    # axes[1].plot(step_idx, sqrt_alpha.numpy(), label="$\\sqrt{\\bar\\alpha}$", linewidth=2, linestyle="--")
    # axes[1].set_xlabel("DDIM step")
    # axes[1].set_title("Signal Scale ($\\sqrt{\\bar\\alpha}$ = VP mean scale)")
    # axes[1].legend()
    # axes[1].grid(True, alpha=0.3)
    #
    # axes[2].plot(step_idx, timesteps.numpy(), linewidth=2)
    # axes[2].set_xlabel("DDIM step")
    # axes[2].set_ylabel("DDPM timestep index")
    # axes[2].set_title("DDPM Timestep")
    # axes[2].grid(True, alpha=0.3)
    #
    # fig.tight_layout()
    # report.add_image(fig)
    # plt.close(fig)
    #
    # # ── Section 2: Noise gate vs step ──
    # report.add_header("Noise Gate Activation (Quadratic)")
    # report.add_note("Gate $g(\\sigma) = \\sigma^2 / (\\sigma^2 + \\nu^2)$ evaluated at $\\sigma_{VP}$. "
    #                 "This controls when PPG guidance is active.")
    #
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    #
    # for nu in nus:
    #     gate = NoiseGate("quadratic", nu)
    #     gate_vals = torch.stack([gate(s) for s in sigma_vp]).numpy()
    #     axes[0].plot(step_idx, gate_vals, label=f"$\\nu$={nu}", linewidth=1.5)
    #
    # axes[0].set_xlabel("DDIM step (0 = noisiest)")
    # axes[0].set_ylabel("Gate value")
    # axes[0].set_title("Gate vs DDIM Step")
    # axes[0].legend(fontsize=8)
    # axes[0].grid(True, alpha=0.3)
    #
    # # Also plot gate as function of sigma_vp directly
    # sigma_range = torch.linspace(0, 1, 500)
    # for nu in nus:
    #     gate = NoiseGate("quadratic", nu)
    #     gate_vals = torch.stack([gate(s) for s in sigma_range]).numpy()
    #     axes[1].plot(sigma_range.numpy(), gate_vals, label=f"$\\nu$={nu}", linewidth=1.5)
    #
    # axes[1].set_xlabel("$\\sigma_{VP}$")
    # axes[1].set_ylabel("Gate value")
    # axes[1].set_title("Gate vs $\\sigma_{VP}$")
    # axes[1].legend(fontsize=8)
    # axes[1].grid(True, alpha=0.3)
    #
    # fig.tight_layout()
    # report.add_image(fig)
    # plt.close(fig)
    #
    # # ── Section 3: Score magnitude comparison ──
    # report.add_header("Score Magnitudes: SD vs PPG")
    # report.add_note("SD score $\\approx -\\varepsilon_\\theta / \\sigma_{VP}$. "
    #                 "PPG score $\\propto g(\\sigma) \\cdot (\\sqrt{\\bar\\alpha} \\cdot \\mu - x) / \\sigma^2$ (VP-corrected). "
    #                 "Assuming $||\\varepsilon_\\theta|| \\approx 1$ and $||\\mu - x|| \\approx 1$ for shape comparison.")
    #
    # # Approximate magnitudes (unit-norm signals for shape comparison)
    # sd_score_mag = 1.0 / sigma_vp       # |score_sd| ~ |eps| / sigma
    # ppg_score_mag_ve = torch.stack([
    #     NoiseGate("quadratic", 0.1)(s) / (s**2) for s in sigma_vp
    # ])  # VE: gate * 1/sigma^2
    # ppg_score_mag_vp = torch.stack([
    #     NoiseGate("quadratic", 0.1)(s) * sqrt_alpha[i] / (s**2)
    #     for i, s in enumerate(sigma_vp)
    # ])  # VP: gate * sqrt_alpha/sigma^2
    #
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    #
    # axes[0].plot(step_idx, sd_score_mag.numpy(), label="SD (1/$\\sigma$)", linewidth=2)
    # axes[0].plot(step_idx, ppg_score_mag_ve.numpy(), label="PPG VE ($g/\\sigma^2$)", linewidth=2, linestyle="--")
    # axes[0].plot(step_idx, ppg_score_mag_vp.numpy(), label="PPG VP ($g\\sqrt{\\bar\\alpha}/\\sigma^2$)", linewidth=2, linestyle=":")
    # axes[0].set_xlabel("DDIM step (0 = noisiest)")
    # axes[0].set_ylabel("Score magnitude (log)")
    # axes[0].set_title("Score Magnitude ($\\nu=0.1$)")
    # axes[0].set_yscale("log")
    # axes[0].legend(fontsize=8)
    # axes[0].grid(True, alpha=0.3)
    #
    # # Ratio: PPG / SD
    # ratio_ve = ppg_score_mag_ve / sd_score_mag
    # ratio_vp = ppg_score_mag_vp / sd_score_mag
    # axes[1].plot(step_idx, ratio_ve.numpy(), label="VE PPG / SD", linewidth=2)
    # axes[1].plot(step_idx, ratio_vp.numpy(), label="VP PPG / SD", linewidth=2, linestyle="--")
    # axes[1].set_xlabel("DDIM step (0 = noisiest)")
    # axes[1].set_ylabel("Ratio")
    # axes[1].set_title("PPG-to-SD Score Ratio ($\\nu=0.1$)")
    # axes[1].set_yscale("log")
    # axes[1].legend(fontsize=8)
    # axes[1].grid(True, alpha=0.3)
    # axes[1].axhline(1.0, color="white", alpha=0.3, linestyle="--")
    #
    # fig.tight_layout()
    # report.add_image(fig)
    # plt.close(fig)
    #
    # # ── Section 4: Effective PPG contribution per step for several nu ──
    # report.add_header("PPG Score Magnitude for Various $\\nu$")
    # report.add_note("Shows how $\\nu$ shifts where PPG is most active relative to the denoising schedule. "
    #                 "VP-corrected: $g(\\sigma) \\cdot \\sqrt{\\bar\\alpha} / \\sigma^2$.")
    #
    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    #
    # for nu in nus:
    #     gate = NoiseGate("quadratic", nu)
    #     mag = torch.stack([
    #         gate(s) * sqrt_alpha[i] / (s**2) for i, s in enumerate(sigma_vp)
    #     ]).numpy()
    #     # Normalize to peak=1 for shape comparison
    #     if mag.max() > 0:
    #         mag = mag / mag.max()
    #     ax.plot(step_idx, mag, label=f"$\\nu$={nu}", linewidth=1.5)
    #
    # ax.set_xlabel("DDIM step (0 = noisiest)")
    # ax.set_ylabel("Normalized PPG score magnitude")
    # ax.set_title("PPG Activity Profile (normalized)")
    # ax.legend(fontsize=8)
    # ax.grid(True, alpha=0.3)
    #
    # fig.tight_layout()
    # report.add_image(fig)
    # plt.close(fig)

    # ── Section 5: Actual denoising process ──
    report.add_header("Denoising Process: Mean Scale + Rescale Comparison")
    report.add_note("Compares SD+PPG with different mean_scale (VE vs VP) and rescale_combined (on/off). "
                    "VE: means unscaled. VP: means scaled by $\\sqrt{\\bar\\alpha}$. "
                    "Rescale: product-of-Gaussians precision correction $w = (\\sigma^2+\\nu^2)/(2\\sigma^2+\\nu^2)$.")

    # Setup dynamics and solver
    num_vis_steps = 50
    snapshot_indices = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49]
    solver = DDIMSolver(scheduler=pipe.scheduler, num_inference_steps=num_vis_steps)
    dynamics = StableDiffusionDynamics(
        unet=pipe.unet, vae=pipe.vae, guidance_scale=7.5, scheduler=pipe.scheduler)
    encoder = dynamics.encoder

    # Use a single prompt/seed for clarity
    prompt = ["a photo of a cat"]
    path_example = None
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
        """Run denoising and capture decoded snapshots at specified step indices."""
        inputs = make_vis_inputs()
        inputs.rank_batches = [np.arange(len(inputs.seeds))]

        dynamics.use_unet = use_unet
        dynamics.ppg = ppg_module

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

            if step_i in snapshot_at:
                img = encoder.decode(latents)
                snapshots.append(img[0].detach().cpu().numpy())

        return snapshots

    def run_condition(label, mean_scale, rescale_combined):
        """Run SD, PPG-only, and SD+PPG for a given condition. Returns (sd, ppg, both)."""
        ppg_module = create_ppg_linear(
            vf_inner=create_sgdm(type_gate="quadratic", nu=nu, mean_scale=mean_scale),
            scale=1.0)
        dynamics.rescale_combined = rescale_combined

        ppg  = run_with_snapshots(f"{label} — PPG only", use_unet=False, ppg_module=ppg_module,  snapshot_at=snapshot_indices)
        both = run_with_snapshots(f"{label} — SD + PPG", use_unet=True,  ppg_module=ppg_module,  snapshot_at=snapshot_indices)
        return ppg, both

    nu = 0.5
    step_labels = [f"Step {i}" for i in snapshot_indices]

    with torch.no_grad():
        sd = run_with_snapshots(f"SD", use_unet=True,  ppg_module=None, snapshot_at=snapshot_indices)
        ppg_ve,  both_ve  = run_condition("VE",            mean_scale="ve", rescale_combined=False)
        ppg_ver, both_ver = run_condition("VE + rescale",  mean_scale="ve", rescale_combined=True)
        ppg_vp,  both_vp  = run_condition("VP",            mean_scale="vp", rescale_combined=False)
        ppg_vpr, both_vpr = run_condition("VP + rescale",  mean_scale="vp", rescale_combined=True)

        # Restore
        dynamics.use_unet = True
        dynamics.ppg = None
        dynamics.rescale_combined = True

    report.add_note("$s(t) = 1$ — VE mean scale, no variance rescaling")
    report.add_image_slider(
        [sd, ppg_ve, both_ve],
        row_labels=["SD", "PPG only", "SD + PPG"],
        step_labels=step_labels,
    )

    report.add_note("$s(t) = 1$ + variance rescaling — VE mean scale with product-of-Gaussians correction")
    report.add_image_slider(
        [sd, ppg_ver, both_ver],
        row_labels=["SD", "PPG only", "SD + PPG"],
        step_labels=step_labels,
    )

    report.add_note("$s(t) = \\sqrt{\\bar\\alpha}$ — VP mean scale, no variance rescaling")
    report.add_image_slider(
        [sd, ppg_vp, both_vp],
        row_labels=["SD", "PPG only", "SD + PPG"],
        step_labels=step_labels,
    )

    report.add_note("$s(t) = \\sqrt{\\bar\\alpha}$ + variance rescaling — VP mean scale with product-of-Gaussians correction")
    report.add_image_slider(
        [sd, ppg_vpr, both_vpr],
        row_labels=["SD", "PPG only", "SD + PPG"],
        step_labels=step_labels,
    )


    report.save(save_path)
    print(f"Timeline report saved to {save_path}")


if __name__ == "__main__":
    make_timeline_report("timeline.html")
