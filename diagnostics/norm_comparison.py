"""
Diagnostic: compare ||x_t|| over the DDIM denoising trajectory for PPG-only
runs using SGDM (ScoreGatedDiracMixture) vs ScorePartitionGuided (SPG).

All runs use PPG-only (no SD net) so the norm trajectory is determined
entirely by the guidance score.  A reference run with the SD net and no PPG
is included for context.

Run from the repo root:
    python diagnostics/norm_comparison.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

from diagnostics import DiagnosticsReport
from ppg.ppg import create_sgdm, create_spg, create_ppg_linear
from generate import (StableDiffusionDynamics, VAEEncoder,
                      InputsIterable, NoiseIterable, TextEmbeddingIterable,
                      ExampleImagesIterable, CombinedInputs)

LATENT_SHAPE = (4, 64, 64)
LATENT_DIM   = int(np.prod(LATENT_SHAPE))   # 16384
NUM_STEPS    = 50


# ── Input helpers ─────────────────────────────────────────────────────────────

def make_vis_inputs(base, encoder, prompt, path_example, pipe):
    exts = [
        NoiseIterable(shape=LATENT_SHAPE, device="cuda"),
        TextEmbeddingIterable(prompt, pipe.tokenizer, pipe.text_encoder, device="cuda"),
    ]
    if path_example is not None:
        exts.append(ExampleImagesIterable(path_example, encoder, device="cuda"))
    return CombinedInputs(base, *exts)


# ── Core runner ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_norm_trace(desc, ppg_module, use_net,
                   dynamics, base, prompt, path_example, pipe):
    """Run DDIM with the given PPG and record ||x_t||_F at every step.

    Returns (norms, image) where norms is a list of length NUM_STEPS+1
    (includes the initial noise) and image is a decoded HWC uint8 numpy array.
    """
    inputs = make_vis_inputs(base, dynamics.encoder, prompt, path_example, pipe)
    inputs.rank_batches = [np.arange(len(inputs.seeds))]

    dynamics.use_net = use_net
    dynamics.ppg     = ppg_module

    for state in inputs:
        dynamics.update(state)
        noise = state.noise

    sched = DDIMScheduler.from_config(pipe.scheduler.config)
    sched.set_timesteps(NUM_STEPS)

    latents = noise.clone()
    norms   = [latents.norm().item()]

    for t_idx in tqdm(sched.timesteps, desc=desc, leave=False):
        score      = dynamics(latents, t_idx)
        sigma      = dynamics.sigma(t_idx)
        noise_pred = -sigma * score
        latents    = sched.step(noise_pred, t_idx, latents).prev_sample
        norms.append(latents.norm().item())

    image = dynamics.encoder.decode(latents)[0].detach().cpu().numpy()
    return norms, image


# ── Plot helpers ───────────────────────────────────────────────────────────────

_PALETTE = [
    "#e94560",   # accent red
    "#4fc3f7",   # sky blue
    "#81c784",   # green
    "#ffb74d",   # orange
    "#ce93d8",   # purple
    "#80deea",   # cyan
    "#f48fb1",   # pink
    "#a5d6a7",   # light green
]

def make_norm_plot(traces, title):
    """
    traces: list of (label, norms, style_dict)
    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#16213e")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#e0e0e0")
    ax.xaxis.label.set_color("#e0e0e0")
    ax.yaxis.label.set_color("#e0e0e0")
    ax.title.set_color("#e94560")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a4a")
    ax.grid(color="#2a2a4a", linewidth=0.7)

    steps = np.arange(len(traces[0][1]))
    for (label, norms, style), palette_color in zip(traces, _PALETTE):
        style = dict(style)  # don't mutate the caller's dict
        color = style.pop("color", palette_color)
        ax.plot(steps, norms, label=label, color=color, **style)

    ax.set_xlabel("DDIM step", color="#e0e0e0")
    ax.set_ylabel(r"$\|x_t\|_F$", color="#e0e0e0")
    ax.set_title(title, color="#e94560")
    legend = ax.legend(fontsize=8, framealpha=0.3,
                       labelcolor="#e0e0e0", facecolor="#1a1a2e",
                       edgecolor="#2a2a4a")
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def make_norm_report(save_path="norm_comparison.html"):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    encoder  = VAEEncoder(pipe.vae)
    dynamics = StableDiffusionDynamics(
        net=pipe.unet, encoder=encoder,
        guidance_scale=7.5, scheduler=pipe.scheduler)

    # ── Dataset entry ─────────────────────────────────────────────────────────
    img_id       = 1
    path_example = None
    prompt       = ["a photo of a cat"]
    if os.path.exists("data/wild-ti2i"):
        import yaml
        with open("data/wild-ti2i/wild-ti2i-real.yaml") as f:
            data = yaml.safe_load(f)
        path_example = [os.path.join("data/wild-ti2i", "data",
                                     os.path.basename(data[img_id]["init_img"]))]
        prompt = [data[img_id]["target_prompts"][0]]

    base = InputsIterable(seeds=[img_id], device="cuda")

    shared = dict(dynamics=dynamics, base=base, prompt=prompt,
                  path_example=path_example, pipe=pipe)

    # ── Nu values to sweep ────────────────────────────────────────────────────
    nus = [0.993, 0.8515, 0.6412, 0.3271, 0.0413]
    type_gate = "hill"

    # ── Collect traces ────────────────────────────────────────────────────────
    # Build frequency basis once so it isn't recomputed per nu
    from ppg.ppg import _build_partition_basis_map
    basis_map_freq = _build_partition_basis_map(
        "frequency", LATENT_DIM, {"shape": LATENT_SHAPE}, device="cuda")

    with torch.no_grad():

        sgdm_traces        = []
        spg_ambient_traces = []
        spg_freq_traces    = []

        for nu in nus:
            # SGDM
            ppg_sgdm = create_ppg_linear(
                vf_inner=create_sgdm(type_gate=type_gate, nu=nu, mean_scale="vp"),
                scale=1.0)
            sgdm_traces.append(run_norm_trace(
                f"SGDM ν={nu}", ppg_module=ppg_sgdm, use_net=False, **shared))

            # SPG — ambient basis
            ppg_spg_amb = create_ppg_linear(
                vf_inner=create_spg(type_gate=type_gate, nu=nu, d=LATENT_DIM,
                                    mean_scale="vp", basis="ambient", device="cuda"),
                scale=1.0)
            spg_ambient_traces.append(run_norm_trace(
                f"SPG-ambient ν={nu}", ppg_module=ppg_spg_amb, use_net=False, **shared))

            # SPG — frequency basis (reuse pre-built basis_map)
            ppg_spg_freq = create_ppg_linear(
                vf_inner=create_spg(type_gate=type_gate, nu=nu, d=LATENT_DIM,
                                    mean_scale="vp", basis_map=basis_map_freq,
                                    device="cuda"),
                scale=1.0)
            spg_freq_traces.append(run_norm_trace(
                f"SPG-freq ν={nu}", ppg_module=ppg_spg_freq, use_net=False, **shared))

    # Unpack (norms, image) tuples
    sgdm_norms,        sgdm_images        = zip(*sgdm_traces)
    spg_ambient_norms, spg_ambient_images = zip(*spg_ambient_traces)
    spg_freq_norms,    spg_freq_images    = zip(*spg_freq_traces)

    # ── Build report ──────────────────────────────────────────────────────────
    report = DiagnosticsReport("Latent norm trace: SGDM vs ScorePartitionGuided")
    report.add_note(
        "PPG-only runs (no SD net). Each row shows the final decoded image and "
        "$\\|x_t\\|_F$ over the 50-step DDIM schedule, starting from the same noise sample."
    )

    solid  = dict(linewidth=1.8)
    dashed = dict(linewidth=1.4, linestyle="--")
    dotted = dict(linewidth=1.2, linestyle=":")

    for i, nu in enumerate(nus):
        report.add_header(f"ν = {nu}", level=3)

        # Row of final images
        report.add_image_row(
            [sgdm_images[i], spg_ambient_images[i], spg_freq_images[i]],
            captions=["SGDM", "SPG ambient", "SPG freq"],
        )

        # Norm plot
        traces = [
            ("SGDM",        sgdm_norms[i],        solid),
            ("SPG ambient", spg_ambient_norms[i],  dashed),
            ("SPG freq",    spg_freq_norms[i],     dotted),
        ]
        fig = make_norm_plot(
            [(lbl, norms, style) for lbl, norms, style in traces],
            title=f"‖xₜ‖_F  —  ν = {nu}, {type_gate} gate, vp mean-scale",
        )
        report.add_image(fig)
        plt.close(fig)

    # ── Summary: all nus on one plot per method pair ──────────────────────────
    report.add_header("Summary: SGDM vs SPG-ambient across ν", level=2)
    report.add_note("Solid = SGDM, dashed = SPG-ambient.")
    traces_summary = []
    for i, nu in enumerate(nus):
        c = _PALETTE[i % len(_PALETTE)]
        traces_summary.append((f"SGDM ν={nu}",        sgdm_norms[i],        dict(linewidth=1.8, color=c)))
        traces_summary.append((f"SPG-ambient ν={nu}", spg_ambient_norms[i], dict(linewidth=1.4, color=c, linestyle="--")))

    fig_summary = make_norm_plot(traces_summary, "‖xₜ‖_F  —  all ν, SGDM vs SPG-ambient")
    report.add_image(fig_summary, width="900px")
    plt.close(fig_summary)

    report.add_header("Summary: SGDM vs SPG-frequency across ν", level=2)
    report.add_note("Solid = SGDM, dotted = SPG-freq.")
    traces_freq_summary = []
    for i, nu in enumerate(nus):
        c = _PALETTE[i % len(_PALETTE)]
        traces_freq_summary.append((f"SGDM ν={nu}",     sgdm_norms[i],     dict(linewidth=1.8, color=c)))
        traces_freq_summary.append((f"SPG-freq ν={nu}", spg_freq_norms[i], dict(linewidth=1.4, color=c, linestyle=":")))

    fig_freq_summary = make_norm_plot(traces_freq_summary, "‖xₜ‖_F  —  all ν, SGDM vs SPG-freq")
    report.add_image(fig_freq_summary, width="900px")
    plt.close(fig_freq_summary)

    report.save(save_path)
    print(f"Norm comparison report saved to {save_path}")


if __name__ == "__main__":
    make_norm_report("norm_comparison.html")
