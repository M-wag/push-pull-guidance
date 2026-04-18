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


def make_vis_inputs(base, encoder, prompt, path_example, pipe):
    extensions = [
        NoiseIterable(shape=(4, 64, 64), device="cuda"),
        TextEmbeddingIterable(prompt, pipe.tokenizer, pipe.text_encoder, device="cuda"),
    ]
    if path_example is not None:
        extensions.append(ExampleImagesIterable(path_example, encoder, device="cuda"))
    return CombinedInputs(base, *extensions)


def run_with_snapshots(desc, use_net, ppg_module, snapshot_at, dynamics, encoder, base, prompt, path_example, pipe, num_vis_steps):
    """Run denoising and capture decoded snapshots at specified step indices."""
    inputs = make_vis_inputs(base, encoder, prompt, path_example, pipe)
    inputs.rank_batches = [np.arange(len(inputs.seeds))]

    dynamics.use_net = use_net
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


def fft2_log_power(x):
    """2D log power spectrum. x: [C, H, W] tensor -> [C, H, W] numpy array."""
    F = torch.fft.fftshift(torch.fft.fft2(x.float()))
    return torch.log1p(F.abs().pow(2)).cpu().numpy()


def freq_snapshot(score):
    """Render log power spectrum of score for all 4 channels. Returns (H, W, 3) uint8."""
    score_power = fft2_log_power(score[0])  # [4, H, W]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for c in range(4):
        axes[c].imshow(score_power[c], cmap="inferno")
        axes[c].set_title(f"ch{c}", fontsize=12)
        axes[c].axis("off")
    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = buf[:, :, :3].copy()
    plt.close(fig)
    return img


def run_with_freq_snapshots(desc, use_net, ppg_module, snapshot_at, dynamics, encoder, base, prompt, path_example, pipe, num_vis_steps):
    """Run denoising and capture frequency-space and decoded x0_pred snapshots at each step.
    Returns (freq_snapshots, x0_snapshots).
    """
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
    freq_snapshots = []
    x0_snapshots = []
    for step_i, t_idx in tqdm(enumerate(scheduler_copy.timesteps), total=num_vis_steps, desc=desc):
        score = dynamics(latents, t_idx)
        sigma = dynamics.sigma(t_idx)
        noise_pred = -sigma * score

        if step_i in snapshot_at:
            alpha_bar = alphas_cumprod[t_idx].float()
            sqrt_alpha = alpha_bar.sqrt()
            x0_pred = (latents - sigma * noise_pred) / sqrt_alpha

            freq_snapshots.append(freq_snapshot(score))
            x0_snapshots.append(encoder.decode(x0_pred)[0].detach().cpu().numpy())

        latents = scheduler_copy.step(noise_pred, t_idx, latents).prev_sample

    return freq_snapshots, x0_snapshots


def run_condition(label, mean_scale, rescale_combined, nu, snapshot_at, dynamics, encoder, base, prompt, path_example, pipe, num_vis_steps):
    """Run PPG-only and SD+PPG for a given condition. Returns (ppg, both)."""
    ppg_module = create_ppg_linear(
        vf_inner=create_sgdm(type_gate="quadratic", nu=nu, mean_scale=mean_scale),
        scale=1.0)
    dynamics.rescale_combined = rescale_combined

    kwargs = dict(snapshot_at=snapshot_at, dynamics=dynamics, encoder=encoder,
                  base=base, prompt=prompt, path_example=path_example, pipe=pipe,
                  num_vis_steps=num_vis_steps)
    ppg  = run_with_snapshots(f"{label} — PPG only", use_net=False, ppg_module=ppg_module,  **kwargs)
    both = run_with_snapshots(f"{label} — SD + PPG",  use_net=True,  ppg_module=ppg_module,  **kwargs)
    return ppg, both


def make_timeline_report(save_path="timeline.html"):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler = pipe.scheduler

    num_steps = 50
    scheduler.set_timesteps(num_steps)

    report = DiagnosticsReport("Denoising Timeline: SD vs PPG")
    report.add_header("")

    num_vis_steps = 50
    snapshot_indices = list(range(0, num_vis_steps))
    solver = DDIMSolver(scheduler=pipe.scheduler, num_inference_steps=num_vis_steps)
    dynamics = StableDiffusionDynamics(
        net=pipe.unet, encoder=VAEEncoder(pipe.vae), guidance_scale=7.5, scheduler=pipe.scheduler)
    encoder = dynamics.encoder

    prompt = ["a photo of a cat"]
    path_example = None
    if os.path.exists("data/wild-ti2i"):
        import yaml
        with open("data/wild-ti2i/wild-ti2i-real.yaml") as f:
            data = yaml.safe_load(f)
        path_example = [os.path.join("data/wild-ti2i", "data", os.path.basename(data[0]["init_img"]))]
        prompt = [data[0]["target_prompts"][0]]

    base = InputsIterable(seeds=[0], device="cuda")
    step_labels = [f"Step {i}" for i in snapshot_indices]

    shared = dict(snapshot_at=snapshot_indices, dynamics=dynamics, encoder=encoder,
                  base=base, prompt=prompt, path_example=path_example, pipe=pipe,
                  num_vis_steps=num_vis_steps)

    ppg_module = create_ppg_linear(
        vf_inner=create_sgdm(type_gate="quadratic", nu=0.5, mean_scale="vp"),
        scale=1.0)

    with torch.no_grad():
        sd_freq,   sd_x0   = run_with_freq_snapshots("SD",      use_net=True,  ppg_module=None,       **shared)
        ppg_freq,  ppg_x0  = run_with_freq_snapshots("PPG",     use_net=False, ppg_module=ppg_module,  **shared)
        both_freq, both_x0 = run_with_freq_snapshots("SD+PPG",  use_net=True,  ppg_module=ppg_module,  **shared)
        dynamics.use_net = True
        dynamics.ppg = None

    report.add_note("Predicted x0")
    report.add_image_slider(
        [sd_x0, ppg_x0, both_x0],
        row_labels=["SD", "PPG", "SD+PPG"],
        step_labels=step_labels,
    )

    report.add_note("Score power spectrum")
    report.add_image_slider(
        [sd_freq, ppg_freq, both_freq],
        row_labels=["SD", "PPG", "SD+PPG"],
        step_labels=step_labels,
        img_width="800px",
    )

    report.save(save_path)
    print(f"Timeline report saved to {save_path}")


if __name__ == "__main__":
    make_timeline_report("frequency.html")
