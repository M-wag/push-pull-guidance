"""
Sweep script: run parameter grids over PPG settings and build an HTML viewer.

Usage:
    python sweep.py sweeper/configs/example_sd.yaml                        # single GPU
    torchrun --nproc_per_node=N sweep.py sweeper/configs/example_sd.yaml   # multi-GPU
"""

import argparse
import gc
import json
import os
import torch
import torch.distributed
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional

from util import Logger
from ppg.ppg import (create_sgdm, create_spg, create_ppg_linear, make_projection_matrix,
                     create_ppg_nonlinear, registry_maps, create_ppg_composed)
from generate import (generate_images_local, InputsIterable, NoiseIterable,
                      ExampleImagesIterable, CombinedInputs)
from sweeper import Gallery, load_sweep_config
from sweeper.schema import (SweepConfig, SDModelConfig, EDMModelConfig,
                             LinearMapConfig, SpgMapConfig, IdentityMapConfig,
                             NonlinearMapConfig, MapConfig)

SEED_BY_DATASET_INDEX = True
RAISE_ERRORS = True


# ---------------------------------------------------------------------------
# RunResult

@dataclass
class RunResult:
    logs_batch:     Optional[list] = None
    logs_per_image: Optional[list] = None


# ---------------------------------------------------------------------------
# Model-specific constants

MODEL_DEFAULTS = {
    "sd": {
        "latent_dim": 16384,
        "noise_shape": (4, 64, 64),
        "max_batch_size": 3,
    },
    "edm": {
        "latent_dim":     3 * 64 * 64,
        "noise_shape":    (3, 64, 64),
        "max_batch_size": 128,
        "vae_latent_dim": 4 * (64 // 8) * (64 // 8),  # SD VAE: C+1 channels, H/8 x W/8
    },
}


# ---------------------------------------------------------------------------
# Seed expansion

def repeat_each(lst, n):
    if lst is None:
        return None
    return [x for x in lst for _ in range(n)]


def expand_seeds(base_seeds, n_seeds):
    return [bs * 1000 + s for bs in base_seeds for s in range(n_seeds)]


# ---------------------------------------------------------------------------
# Datasets

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


def load_imgnet(path_dir, n_entries=None, indices=None):
    with open(os.path.join(path_dir, "imgnet_labels.yaml")) as f:
        label_names = yaml.safe_load(f)
    label_names = {int(k): str(v).split(",")[0] for k, v in label_names.items()}

    with open(os.path.join(path_dir, "my_samples.yaml")) as f:
        entries = yaml.safe_load(f)

    if indices is not None:
        dataset_indices = list(indices)
        entries = [entries[i] for i in indices]
    elif n_entries is not None:
        dataset_indices = list(range(n_entries))
        entries = entries[:n_entries]
    else:
        dataset_indices = list(range(len(entries)))

    path_imgs = [e["init_img"] for e in entries]
    class_labels = [e["target_label"] for e in entries]
    prompts = [
        f"{label_names.get(e['source_label'], e['source_label'])} -> "
        f"{label_names.get(e['target_label'], e['target_label'])}"
        for e in entries
    ]
    return path_imgs, class_labels, dataset_indices, prompts


# ---------------------------------------------------------------------------
# Pipeline setup

def setup_pipeline_sd(model_cfg: SDModelConfig, solver_cfg):
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from generate import StableDiffusionDynamics, DDIMSolver, VAEEncoder

    pipe = StableDiffusionPipeline.from_pretrained(
        model_cfg.checkpoint, torch_dtype=torch.float32).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    encoder  = VAEEncoder(pipe.vae)
    dynamics = StableDiffusionDynamics(
        net=pipe.unet, encoder=encoder,
        guidance_scale=solver_cfg.guidance_scale, scheduler=pipe.scheduler)
    solver = DDIMSolver(scheduler=pipe.scheduler,
                        num_inference_steps=solver_cfg.num_steps,
                        ddim_eta=solver_cfg.ddim_eta)
    return pipe, dynamics, solver


def setup_pipeline_edm(model_cfg: EDMModelConfig, solver_cfg):
    from edm.dnnlib.util import import_net_from_url
    from generate import EDMDynamics, EDMSolver

    net, encoder = import_net_from_url(model_cfg.net_pkl, device="cuda")
    dynamics = EDMDynamics(net=net, encoder=encoder)
    solver   = EDMSolver(num_steps=solver_cfg.num_steps)
    return dynamics, solver, encoder


PIPELINE_SETUP = {
    "sd":  setup_pipeline_sd,
    "edm": setup_pipeline_edm,
}


# ---------------------------------------------------------------------------
# DDIM inversion precompute (SD only)

def precompute_ddim(pipe, dynamics, paths_example, prompts, num_inference_steps):
    from diffusers import DDIMScheduler
    from generate import TextEmbeddingIterable, ddim_invert
    from util import load_images

    example_latents = dynamics.encoder.encode(load_images(paths_example, device="cuda", rescale=False))
    text_embeddings = TextEmbeddingIterable(
        prompts, pipe.tokenizer, pipe.text_encoder, device="cuda")._encode(prompts)
    return ddim_invert(
        example_latents, pipe.unet, pipe.scheduler,
        text_embeddings=text_embeddings, batch_size=4,
        num_inference_steps=num_inference_steps)


# ---------------------------------------------------------------------------
# Distributed init

def init_distributed():
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
    def __init__(self):
        self._cache            = {}
        self._lowpass_cache    = {}
        self._dct_basis_cache  = {}
        self._spg_basis_cache  = {}
        self._map_instance_cache = {}

    def get_map_instance(self, map_type, map_kwargs):
        key = (map_type, tuple(sorted((map_kwargs or {}).items())))
        if key not in self._map_instance_cache:
            self._map_instance_cache[key] = registry_maps[map_type](**(map_kwargs or {})).to("cuda")
        return self._map_instance_cache[key]

    def _get_dct_basis(self, H, W):
        from ppg.ppg import _dct2_matrix
        key = (int(H), int(W))
        if key not in self._dct_basis_cache:
            D_h = _dct2_matrix(H).cuda()
            D_w = _dct2_matrix(W).cuda()
            D_2d = torch.kron(D_h, D_w)
            order = sorted(range(H * W), key=lambda i: (i // W + i % W, i // W))
            order_t = torch.tensor(order, dtype=torch.long, device="cuda")
            self._dct_basis_cache[key] = D_2d[order_t]
        return self._dct_basis_cache[key]

    def get_lowpass(self, shape, dim_out):
        key = (tuple(shape), int(dim_out))
        if key not in self._lowpass_cache:
            C, H, W = shape
            D_trunc = self._get_dct_basis(H, W)[:int(dim_out)]
            mat = torch.kron(torch.eye(C, device="cuda"), D_trunc)
            self._lowpass_cache[key] = (mat, mat.T)
        return self._lowpass_cache[key]

    def get_spg_basis(self, basis, d, basis_kwargs):
        from ppg.ppg import _build_partition_basis_map
        kwargs_key = tuple(sorted((basis_kwargs or {}).items()))
        key = (basis, int(d), kwargs_key)
        if key not in self._spg_basis_cache:
            print(f"Building SPG basis: {basis}, d={d}, kwargs={basis_kwargs}")
            self._spg_basis_cache[key] = _build_partition_basis_map(
                basis, d, basis_kwargs, device="cuda")
        return self._spg_basis_cache[key]

    def get_matrix(self, n_features, dim_in, dim_out, orthonormal, seed=2):
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
# Map builder

def build_map_layers(cfg: MapConfig, vf_inner, proj_cache: ProjectionCache, defaults: dict):
    """Return (maps, pullbacks, updated_vf_inner) for a single map config.

    SpgMapConfig replaces vf_inner with an SPG-adapted field; all other types
    leave vf_inner unchanged.  Maps are applied in the order returned.
    """
    from ppg.ppg import PullbackLinear, FunctionMap, registry_pullback, create_linear_maps

    latent_dim = defaults["latent_dim"]

    if isinstance(cfg, IdentityMapConfig):
        maps = create_linear_maps(device="cuda")
        pullbacks = [PullbackLinear()] * len(maps)
        return maps, pullbacks, vf_inner

    if isinstance(cfg, SpgMapConfig):
        d = round(cfg.dim_out) if cfg.dim_out is not None else latent_dim
        basis_kwargs = dict(cfg.basis_kwargs or {})
        if cfg.basis == "frequency" and "shape" not in basis_kwargs:
            basis_kwargs["shape"] = defaults["noise_shape"]
        basis_map = proj_cache.get_spg_basis(cfg.basis, d, basis_kwargs or None)
        vf_spg = create_spg(
            type_gate=vf_inner.type_gate, nu=vf_inner.nu, d=d,
            mean_scale=vf_inner.mean_scale, gate_n=vf_inner.gate_n,
            k_min=cfg.k_min, basis_map=basis_map, device="cuda")
        maps = create_linear_maps(device="cuda")
        pullbacks = [PullbackLinear()] * len(maps)
        return maps, pullbacks, vf_spg

    if isinstance(cfg, LinearMapConfig):
        dim_out    = round(cfg.dim_out) if cfg.dim_out is not None else None
        n_features = int(cfg.n_features)
        if dim_out is None:
            maps = create_linear_maps(device="cuda")
        elif cfg.projection == "coordinate":
            maps = create_linear_maps(
                projection="coordinate",
                projection_kwargs={"dim_in": latent_dim, "dim_out": dim_out},
                device="cuda")
        elif cfg.projection == "lowpass":
            mat, mat_inv = proj_cache.get_lowpass(defaults["noise_shape"], dim_out)
            maps = create_linear_maps(mat=mat, mat_inv=mat_inv, device="cuda")
        else:
            mat, mat_inv = proj_cache.get_matrix(
                n_features, latent_dim, dim_out,
                orthonormal=(cfg.projection == "orthonormal"), seed=cfg.seed)
            maps = create_linear_maps(mat=mat, mat_inv=mat_inv, device="cuda")
        pullbacks = [PullbackLinear()] * len(maps)
        return maps, pullbacks, vf_inner

    if isinstance(cfg, NonlinearMapConfig):
        map_instance = proj_cache.get_map_instance(cfg.map_type, cfg.map_kwargs)
        pb_kwargs = cfg.pullback_kwargs or {}
        maps = [FunctionMap(map_instance.forward, map_instance.inv)]
        pullbacks = [registry_pullback[cfg.pullback](pb_kwargs)]
        return maps, pullbacks, vf_inner

    raise ValueError(f"Unknown map config type: {type(cfg)}")


def build_map(cfg: MapConfig, vf_inner, proj_cache: ProjectionCache, defaults: dict):
    maps, pullbacks, vf_inner = build_map_layers(cfg, vf_inner, proj_cache, defaults)
    return create_ppg_composed(vf_inner, maps, pullbacks, scale=cfg.scale, device="cuda")


# ---------------------------------------------------------------------------
# Sweep runner

class SweepRunner:
    def __init__(self, config: SweepConfig, dynamics, solver,
                 paths_example, seeds, projection_cache,
                 pipe=None, prompts=None, inverted_noise=None, class_labels=None):
        self.config        = config
        self.dynamics      = dynamics
        self.solver        = solver
        self.paths_example = paths_example
        self.seeds         = seeds
        self.proj_cache    = projection_cache
        self.pipe          = pipe
        self.prompts       = prompts
        self.inverted_noise = inverted_noise
        self.class_labels  = class_labels

        self.noise_source     = config.noise_source
        self.defaults         = MODEL_DEFAULTS[config.model.type]
        self._example_tensors = None

        log_mode = config.logging.get("mode") if config.logging else None
        self.log_mode = log_mode
        snap_cfg = config.snapshots
        self.snapshot_steps = (snap_cfg.get("steps")
                               if snap_cfg and snap_cfg.get("enabled", True) else None)

        if log_mode is not None:
            self.dynamics.logger = Logger()

    @property
    def n_images(self):
        if isinstance(self.config.model, SDModelConfig):
            return len(self.prompts)
        return len(self.class_labels)

    def build(self, cell: SweepConfig):
        self.noise_source = cell.noise_source
        if hasattr(self.solver, "ddim_eta"):
            self.solver.ddim_eta = cell.solver.ddim_eta

        self.dynamics.ppg = None
        if cell.ppg is None:
            return 

        ppg_cfg  = cell.ppg
        gate_cfg = ppg_cfg.gate

        vf_inner = create_sgdm(
            type_gate=gate_cfg.type,
            nu=gate_cfg.nu,
            gate_n=int(gate_cfg.n),
            mean_scale=ppg_cfg.mean_scale,
            channeled=False,
        )

        map_cfgs = cell.maps or [IdentityMapConfig(type="identity")]
        all_maps, all_pullbacks = [], []
        cur_vf_inner = vf_inner
        cur_latent_dim = self.defaults["latent_dim"]
        for cfg in map_cfgs:
            effective_defaults = {**self.defaults, "latent_dim": cur_latent_dim}
            layers, pullbacks, cur_vf_inner = build_map_layers(cfg, cur_vf_inner, self.proj_cache, effective_defaults)
            all_maps.extend(layers)
            all_pullbacks.extend(pullbacks)
            if isinstance(cfg, NonlinearMapConfig):
                cur_latent_dim = self.defaults.get("vae_latent_dim", cur_latent_dim)
            elif isinstance(cfg, (LinearMapConfig, SpgMapConfig)) and cfg.dim_out is not None:
                cur_latent_dim = round(cfg.dim_out)
        composed_scale = 1.0
        for cfg in map_cfgs:
            composed_scale *= cfg.scale
        ppg = create_ppg_composed(cur_vf_inner, all_maps, all_pullbacks,
                                  scale=composed_scale, device="cuda")

        self.dynamics.ppg                = ppg
        self.dynamics.normalize_variance = ppg_cfg.normalize_variance
        self.dynamics.use_net = (
            (lambda sigma: sigma <= ppg_cfg.use_net_below)
            if ppg_cfg.use_net_below is not None else True
        )


    def run(self):
        logger = self.dynamics.logger

        inputs = self._make_inputs()
        n = self.n_images
        n_snaps       = len(self.snapshot_steps) if self.snapshot_steps else 0
        all_images    = [None] * n
        all_snapshots = [[None] * n_snaps for _ in range(n)] if self.snapshot_steps else None

        per_image_logs   = [None] * n if logger and self.log_mode in ("per_image", "both") else None
        batch_logs_accum = [] if logger and self.log_mode in ("batch", "both") else None

        for state in generate_images_local(self.solver, self.dynamics, inputs,
                                           verbose=True,
                                           max_batch_size=self.defaults["max_batch_size"],
                                           snapshot_steps=self.snapshot_steps):
            for img, idx in zip(state.images.detach().cpu().numpy(), state.indices):
                all_images[idx] = img
            if all_snapshots is not None:
                for s, snap_batch in enumerate(state.snapshots):
                    for img, idx in zip(snap_batch.detach().cpu().numpy(), state.indices):
                        all_snapshots[idx][s] = img
            if logger:
                if per_image_logs is not None:
                    for logs, idx in zip(logger.get_per_image_logs(), state.indices):
                        per_image_logs[idx] = logs
                if batch_logs_accum is not None:
                    batch_logs_accum.append(logger.get_batch_logs())
                logger.reset()

        del state
        gc.collect()
        torch.cuda.empty_cache()

        result = self._collect_result(per_image_logs, batch_logs_accum)
        return all_images, all_snapshots, result

    def _get_example_tensors(self):
        if self._example_tensors is None:
            from util import load_images
            encoder = getattr(self.dynamics, "encoder", None)
            if encoder is not None:
                examples = load_images(self.paths_example, device="cuda", rescale=False)
                examples = encoder.encode(examples)
            else:
                examples = load_images(self.paths_example, device="cuda", rescale=True)
            self._example_tensors = examples
        return self._example_tensors

    def _make_inputs(self):
        base       = InputsIterable(seeds=self.seeds, device="cuda")
        noise_shape = self.defaults["noise_shape"]
        precomputed = self._get_example_tensors()

        if isinstance(self.config.model, SDModelConfig):
            from generate import TextEmbeddingIterable, PrecomputedNoiseIterable
            noise_ext = (PrecomputedNoiseIterable(self.inverted_noise)
                         if self.noise_source == "ddim_inversion"
                         else NoiseIterable(shape=noise_shape, device="cuda"))
            return CombinedInputs(
                base, noise_ext,
                TextEmbeddingIterable(self.prompts, self.pipe.tokenizer,
                                      self.pipe.text_encoder, device="cuda"),
                ExampleImagesIterable(self.paths_example, precomputed=precomputed),
            )
        else:
            from generate import LabelsIterable
            return CombinedInputs(
                base,
                NoiseIterable(shape=noise_shape, device="cuda"),
                LabelsIterable(labels=self.class_labels, num_classes=1000, device="cuda"),
                ExampleImagesIterable(self.paths_example, precomputed=precomputed),
            )

    def _collect_result(self, per_image_logs, batch_logs_accum):
        if self.log_mode is None:
            return None
        merged_batch = None
        if batch_logs_accum:
            n_steps = len(batch_logs_accum[0])
            merged_batch = []
            for t in range(n_steps):
                step_dicts = [bl[t] for bl in batch_logs_accum]
                keys = step_dicts[0].keys()
                merged_batch.append({k: sum(d[k] for d in step_dicts) / len(step_dicts) for k in keys})
        return RunResult(logs_batch=merged_batch, logs_per_image=per_image_logs)


# ---------------------------------------------------------------------------
# Log plots

_log_plot_fig  = None
_log_plot_axes = None

def make_log_plots(logs_data, cell_dir):
    global _log_plot_fig, _log_plot_axes

    plot_path = os.path.join(cell_dir, "logs_plot.png")
    if os.path.exists(plot_path):
        return

    logs = logs_data.get("logs_batch")
    if not logs and logs_data.get("logs_per_image"):
        per_image = logs_data["logs_per_image"]
        n_steps = len(per_image[0])
        logs = []
        for t in range(n_steps):
            step_dicts = [img_logs[t] for img_logs in per_image]
            keys = step_dicts[0].keys()
            logs.append({k + "_mean": sum(d[k] for d in step_dicts) / len(step_dicts) for k in keys})
    if not logs:
        return

    if _log_plot_fig is None:
        _log_plot_fig, _log_plot_axes = plt.subplots(1, 3, figsize=(12, 3))
        _log_plot_fig.patch.set_facecolor("#16213e")
    fig, axes = _log_plot_fig, _log_plot_axes
    for ax in axes:
        ax.cla()
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#e0e0e0")
        ax.xaxis.label.set_color("#e0e0e0")
        ax.title.set_color("#e94560")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
        ax.grid()

    steps = range(len(logs))
    for key, lbl in [("norm_comb_mean", "combined"), ("norm_model_mean", "model"), ("norm_guide_mean", "guide")]:
        if key in logs[0]:
            axes[0].plot(steps, [d[key] for d in logs], label=lbl)
    axes[0].set_title("Score Norms"); axes[0].set_xlabel("Step"); axes[0].legend(fontsize=7)
    axes[0].set_yscale("log")

    for key, lbl in [("cos_model_guide_mean", "model/guide")]:
        if key in logs[0]:
            axes[1].plot(steps, [d[key] for d in logs], label=lbl)
    axes[1].set_title("Cosine Similarities"); axes[1].set_xlabel("Step")
    axes[1].set_ylim(-1.1, 1.1); axes[1].legend(fontsize=7)

    for key, lbl in [("ratio_comb_model_mean", "‖comb‖/‖model‖"), ("ratio_guide_model_mean", "‖guide‖/‖model‖")]:
        if key in logs[0]:
            axes[2].plot(steps, [d[key] for d in logs], label=lbl)
    axes[2].set_title("Score Ratios"); axes[2].set_xlabel("Step"); axes[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight", dpi=100, facecolor=fig.get_facecolor())


# ---------------------------------------------------------------------------
# Baseline

def _run_baseline(baseline_dir, meta, n_images, dynamics, solver, make_inputs_fn, max_batch_size):
    meta_path    = os.path.join(baseline_dir, "baseline_meta.json")
    images_exist = all(os.path.exists(os.path.join(baseline_dir, f"img_{i}.png"))
                       for i in range(n_images))

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
        dynamics.ppg    = None
        dynamics.use_net = True
        dynamics.logger  = None
        inputs = make_inputs_fn()
        os.makedirs(baseline_dir, exist_ok=True)
        all_images = [None] * n_images
        for state in generate_images_local(solver, dynamics, inputs,
                                           verbose=True, max_batch_size=max_batch_size):
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
    parser = argparse.ArgumentParser(description="PPG parameter sweep")
    parser.add_argument("config", help="Path to sweep config YAML")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    rank, world_size = init_distributed()

    config  = load_sweep_config(args.config)
    gallery = Gallery(config)

    # --- Pipeline setup ---
    pipeline = PIPELINE_SETUP[config.model.type](config.model, config.solver)
    if isinstance(config.model, SDModelConfig):
        pipe, dynamics, solver = pipeline
    else:
        dynamics, solver, _encoder = pipeline
        pipe = None

    # --- Dataset ---
    examples_cfg = config.examples
    n_seeds = int(examples_cfg.get("n_seeds", 1))
    if isinstance(config.model, SDModelConfig):
        paths_example, prompts, dataset_indices = load_wildti2i(
            examples_cfg.get("dataset", "data/wild-ti2i"),
            n_entries=examples_cfg.get("n_entries"),
            indices=examples_cfg.get("indices"))
        class_labels = None
        base_seeds = dataset_indices if SEED_BY_DATASET_INDEX else list(range(len(prompts)))
    else:
        paths_example, class_labels, dataset_indices, prompts = load_imgnet(
            path_dir=examples_cfg.get("dataset", "data/imgnet64"),
            n_entries=examples_cfg.get("n_entries"),
            indices=examples_cfg.get("indices"))
        base_seeds = dataset_indices if SEED_BY_DATASET_INDEX else list(range(len(class_labels)))

    # Expand everything example-major: image index i => example i//n_seeds, seed i%n_seeds
    paths_example_ex = repeat_each(paths_example, n_seeds)
    prompts_ex       = repeat_each(prompts, n_seeds)
    class_labels_ex  = repeat_each(class_labels, n_seeds)
    seeds = expand_seeds(base_seeds, n_seeds)

    # --- DDIM inversion precompute (SD only) ---
    inverted_noise = None
    if isinstance(config.model, SDModelConfig) and config.noise_source == "ddim_inversion":
        if rank == 0:
            inverted_noise = precompute_ddim(pipe, dynamics, paths_example, prompts,
                                             num_inference_steps=config.solver.num_steps)
            if n_seeds > 1:
                inverted_noise = inverted_noise.repeat_interleave(n_seeds, dim=0)
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

    # --- Runner ---
    runner = SweepRunner(
        config=config,
        dynamics=dynamics, solver=solver,
        paths_example=paths_example_ex, seeds=seeds,
        projection_cache=ProjectionCache(),
        pipe=pipe, prompts=prompts_ex,
        inverted_noise=inverted_noise,
        class_labels=class_labels_ex,
    )

    gallery.generate(runner.build, runner.run,
                     n_images=runner.n_images, rank=rank, world_size=world_size,
                     raise_errors=RAISE_ERRORS)

    # --- Baseline ---
    baseline_paths = None
    if rank == 0:
        if config.baseline_dir is None:
            print("Warning: no baseline_dir in config, skipping baseline.")
        else:
            n = runner.n_images
            baseline_meta = {"examples": examples_cfg, "n_images": n,
                             "solver": config.solver.model_dump()}
            baseline_paths = _run_baseline(
                config.baseline_dir, baseline_meta, n,
                dynamics, solver, runner._make_inputs,
                max_batch_size=runner.defaults["max_batch_size"],
            )

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        if world_size > 1:
            gallery.merge_manifests(world_size)
        gallery.build_html(example_paths=paths_example, prompts=prompts,
                           plot_fn=make_log_plots, n_seeds=n_seeds,
                           **({"baseline_paths": baseline_paths} if baseline_paths else {}))
    if world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
