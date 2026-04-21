import os
import numpy as np
import torch
import PIL.Image

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Tuple
from tqdm import tqdm
from diffusers import DDIMScheduler
from torch_utils import distributed as dist
from util import EasyDict, load_images

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# DDIM inversion from clean latent to noise by reversing the DDIM step.

@torch.no_grad()
def ddim_invert(latents, unet, scheduler, text_embeddings=None, guidance_scale=0.0,
                num_inference_steps=None, batch_size=None):

    sched = DDIMScheduler.from_config(scheduler.config)
    sched.set_timesteps(num_inference_steps or scheduler.num_inference_steps)
    alphas_cumprod = sched.alphas_cumprod
    timesteps_rev = list(reversed(sched.timesteps))

    # Precompute (t, alpha_cur, alpha_noisier) pairs
    # alpha_cur: where we are (cleaner), alpha_noisier: where we're stepping to
    alpha_pairs = []
    for i, t in enumerate(timesteps_rev):
        alpha_cur = alphas_cumprod[timesteps_rev[i - 1]] if i > 0 else sched.final_alpha_cumprod
        alpha_noisier = alphas_cumprod[t]
        alpha_pairs.append((t, alpha_cur, alpha_noisier))

    # Process in chunks if batch_size is specified
    if batch_size is not None and latents.shape[0] > batch_size:
        chunks = latents.split(batch_size)
        if text_embeddings is not None:
            # text_embeddings is (2B, 77, D) for CFG or (B, 77, D) without
            if guidance_scale > 0:
                uncond_emb, cond_emb = text_embeddings.chunk(2)
                emb_chunks = [(torch.cat([u, c]) )
                              for u, c in zip(uncond_emb.split(batch_size),
                                              cond_emb.split(batch_size))]
            else:
                cond_emb = text_embeddings.chunk(2)[1]
                emb_chunks = cond_emb.split(batch_size)
        else:
            emb_chunks = [None] * len(chunks)

        results = []
        for chunk, emb in zip(chunks, emb_chunks):
            if emb is not None and guidance_scale > 0:
                full_emb = emb  # already (2*bs, 77, D)
            elif emb is not None:
                full_emb = torch.cat([torch.zeros_like(emb), emb])  # rebuild for chunk
            else:
                full_emb = None
            results.append(ddim_invert(chunk, unet, scheduler,
                                       text_embeddings=full_emb,
                                       guidance_scale=guidance_scale,
                                       num_inference_steps=num_inference_steps,
                                       batch_size=None))
        return torch.cat(results)

    for t, alpha_cur, alpha_noisier in tqdm(alpha_pairs, desc="DDIM Inversion"):

        # Predict noise
        if text_embeddings is not None and guidance_scale > 0:
            latents_input = torch.cat([latents] * 2)
            noise_pred = unet(latents_input, t, encoder_hidden_states=text_embeddings).sample
            uncond, cond = noise_pred.chunk(2)
            eps = uncond + guidance_scale * (cond - uncond)
        else:
            if text_embeddings is not None:
                cond_emb = text_embeddings.chunk(2)[1]  # (B, 77, D)
            else:
                cond_emb = None
            eps = unet(latents, t, encoder_hidden_states=cond_emb).sample

        # DDIM inversion step: estimate x0 from current (cleaner) alpha, step to noisier alpha
        pred_x0 = (latents - (1 - alpha_cur).sqrt() * eps) / alpha_cur.sqrt()
        latents = alpha_noisier.sqrt() * pred_x0 + (1 - alpha_noisier).sqrt() * eps

    return latents

#----------------------------------------------------------------------------
# Abstract base classes

class Encoder(ABC):
    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images (B, C, H, W) -> latents (B, Z, H', W')"""

    @abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents (B, Z, H', W') -> images (B, C, H, W) in [0, 1]"""


class InputsIterable:
    """Base iterable that owns seeds and rank_batches. Yields one EasyDict per batch.

    Extensions (NoiseIterable, TextEmbeddingIterable, etc.) add data to each
    batch via their enrich(state, indices) method. Compose them with CombinedInputs.
    """

    def __init__(self, seeds: List[int], device: str = "cuda"):
        self.rank_batches = None  # injected by generate_images
        self.seeds  = seeds
        self.device = device

    def __iter__(self):
        for batch_idx, indices in enumerate(self.rank_batches):
            batch_seeds = [self.seeds[i] for i in indices]
            yield EasyDict(
                seeds       = batch_seeds,
                indices     = list(indices),
                batch_idx   = batch_idx,
                num_batches = len(self.rank_batches),
            )

    def __len__(self) -> int:
        return len(self.rank_batches)


class Dynamics(ABC):
    encoder: Encoder    # subclasses must assign; used by ImageIterable for decode

    @abstractmethod
    def update(self, state) -> None:
        """Capture per-batch context (e.g. text embeddings) from state before solver runs."""

    @abstractmethod
    def __call__(self, latents: torch.Tensor, t: int) -> torch.Tensor:
        """Return noise prediction for latents at timestep t."""


class Solver(ABC):
    @abstractmethod
    def __call__(
        self,
        dynamics: Dynamics,
        noise: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Any]:
        """Integrate dynamics from t=T to t=0. Returns (xs, aux); xs[-1] is final latent."""


#----------------------------------------------------------------------------
# Iterable which applies diffusion process to an InputsIterable.

class ImageIterable:
    def __init__(self, solver: Solver, dynamics: Dynamics, verbose=False,
                 snapshot_steps: Optional[List[int]] = None,
                 snapshot_as_x0: bool = True):
        self.solver = solver
        self.dynamics = dynamics
        self.verbose = verbose
        self.snapshot_steps = snapshot_steps
        self.snapshot_as_x0 = snapshot_as_x0

    def __call__(self, iter_state: InputsIterable) -> Iterable:
        self.solver.verbose = self.verbose
        for state in iter_state:
            yield self._process_batch(state)

    def _process_batch(self, state):
        if len(state.seeds) > 0:
            self.dynamics.update(state)
            xs, x0s = self.solver(self.dynamics, state.noise)
            state.images = self.dynamics.encoder.decode(xs[-1])
            if self.snapshot_steps is not None:
                snap_src = (x0s if (self.snapshot_as_x0 and x0s) else xs)
                state.snapshots = [
                    self.dynamics.encoder.decode(snap_src[s])
                    for s in self.snapshot_steps if s < len(snap_src)
                ]
            return state

#----------------------------------------------------------------------------
# Wrapper around iterables that saves images to disk.

class SavingIterable:
    def __init__(self, dir_save):
        self.dir_save = dir_save

    def __call__(self, iterable):
        for states in iterable:
            if self.dir_save is not None:
                self.save(states)
            yield states

    def save(self, states):
        images = (states.images.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        os.makedirs(self.dir_save, exist_ok=True)
        for image, seed in zip(images, states.seeds):
            PIL.Image.fromarray(image, 'RGB').save(os.path.join(self.dir_save, f"{seed}.png"))

#----------------------------------------------------------------------------
# HuggingFace Diffusers implementations

class VAEEncoder(Encoder):
    """Wraps HF AutoencoderKL to satisfy the Encoder interface."""

    def __init__(self, vae):
        self.vae   = vae
        self.scale = vae.config.scaling_factor  # 0.18215 for SD1.x, 0.13025 for SDXL

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images in [0, 255] -> scaled latents"""
        images = images.to(torch.float32) / 127.5 - 1
        return self.vae.encode(images).latent_dist.mean * self.scale

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """scaled latents -> images in [0, 1]"""
        images = self.vae.decode(latents / self.scale).sample
        return (images / 2 + 0.5).clamp(0, 1)


#----------------------------------------------------------------------------
# Composable input iterables

class NoiseIterable:
    """Extension that adds seeded random noise to each batch state."""

    def __init__(self, shape: Tuple[int, ...], device: str = "cuda"):
        self.shape  = shape
        self.device = device

    def enrich(self, state: EasyDict) -> None:
        rnd = StackedRandomGenerator(self.device, state.seeds)
        state.noise = rnd.randn([len(state.seeds), *self.shape], device=self.device)


class TextEmbeddingIterable:
    """Extension that adds text_embeddings and prompts to each batch state."""

    def __init__(self, prompts: List[str], tokenizer, text_encoder,
                 negative_prompt: str = "", device: str = "cuda"):
        self.prompts         = prompts
        self.tokenizer       = tokenizer
        self.text_encoder    = text_encoder
        self.negative_prompt = negative_prompt
        self.device          = device

    def enrich(self, state: EasyDict) -> None:
        batch_prompts = [self.prompts[i] for i in state.indices]
        state.prompts         = batch_prompts
        state.text_embeddings = self._encode(batch_prompts)

    @torch.no_grad()
    def _encode(self, prompts: List[str]) -> torch.Tensor:
        tok = self.tokenizer
        cond = self.text_encoder(
            tok(prompts,
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
        )[0]
        uncond = self.text_encoder(
            tok([self.negative_prompt] * len(prompts),
                padding="max_length",
                max_length=tok.model_max_length,
                return_tensors="pt"
            ).input_ids.to(self.device)
        )[0]
        return torch.cat([uncond, cond])


class ExampleImagesIterable:
    """Extension that adds example images to each batch state."""

    def __init__(self, paths_example: List[str], encoder: Optional[Encoder] = None,
                 device: str = "cuda", precomputed: Optional[torch.Tensor] = None):
        self.paths_example  = paths_example
        self.device         = device
        self.encoder        = encoder
        self.precomputed    = precomputed  # (N, ...) tensor; skips load+encode if set

    def enrich(self, state: EasyDict) -> None:
        if self.precomputed is not None:
            state.examples = self.precomputed[list(state.indices)]
            return
        batch_paths = [self.paths_example[i] for i in state.indices]
        if self.encoder is not None:
            examples = load_images(batch_paths, device=self.device, rescale=False)
            if examples is not None:
                state.examples = self.encoder.encode(examples)
        else:
            examples = load_images(batch_paths, device=self.device, rescale=True)
            if examples is not None:
                state.examples = examples


class DDIMInversionIterable:
    """Extension that produces noise via DDIM inversion of example images.

    Requires state.examples (encoded latents) and state.text_embeddings.
    Order extensions: ExampleImagesIterable, TextEmbeddingIterable, DDIMInversionIterable.
    """

    def __init__(self, unet, scheduler, guidance_scale: float = 0.0,
                 num_inference_steps: Optional[int] = None):
        self.unet                = unet
        self.scheduler           = scheduler
        self.guidance_scale      = guidance_scale
        self.num_inference_steps = num_inference_steps

    def enrich(self, state: EasyDict) -> None:
        state.noise = ddim_invert(
            state.examples, self.unet, self.scheduler,
            text_embeddings=state.text_embeddings,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
        )

class LabelsIterable:
    """Extension that adds one-hot class label tensors to each batch state.

    Converts integer class indices to one-hot vectors of shape (B, num_classes),
    matching the conditioning format expected by EDM class-conditional networks.
    """

    def __init__(self, labels: List[int], num_classes: int, device: str = "cuda"):
        self.labels      = labels
        self.num_classes = num_classes
        self.device      = device

    def enrich(self, state: EasyDict) -> None:
        batch_labels = torch.tensor([self.labels[i] for i in state.indices], device=self.device)
        state.labels = torch.nn.functional.one_hot(batch_labels, self.num_classes).float()


class MetadataIterable:
    """Extension that adds arbitrary per-image metadata fields to each batch state.

    Pass any number of keyword arguments mapping field names to lists or tensors
    of length N (one entry per seed). Each field is sliced by state.indices and
    written onto the state under the same name.

    Example:
        MetadataIterable(class_id=class_labels, example_id=example_ids)
        # -> state.class_id and state.example_id available in every batch
    """

    def __init__(self, **fields):
        self.fields = fields  # name -> list | Tensor, length N

    def enrich(self, state: EasyDict) -> None:
        for name, values in self.fields.items():
            if isinstance(values, torch.Tensor):
                state[name] = values[list(state.indices)]
            else:
                state[name] = [values[i] for i in state.indices]


class PrecomputedNoiseIterable:
    """Extension that injects a precomputed noise tensor into each batch state."""

    def __init__(self, noise: torch.Tensor):
        self.noise = noise

    def enrich(self, state: EasyDict) -> None:
        state.noise = self.noise[state.indices]


class CombinedInputs(InputsIterable):
    """Combines a base InputsIterable with extensions that enrich each batch state.

    Extensions are applied in order, so dependencies are resolved by ordering:
      CombinedInputs(base, NoiseIterable(...), TextEmbeddingIterable(...), ...)
      CombinedInputs(base, ExampleImagesIterable(...), DDIMInversionIterable(...), ...)
    """

    def __init__(self, base: InputsIterable, *extensions):
        self.base       = base
        self.extensions = extensions

    @property
    def seeds(self):
        return self.base.seeds

    @property
    def rank_batches(self):
        return self.base.rank_batches

    @rank_batches.setter
    def rank_batches(self, value):
        self.base.rank_batches = value

    def __iter__(self):
        for state in self.base:
            for ext in self.extensions:
                (ext.enrich if hasattr(ext, "enrich") else ext)(state)
            yield state

    def __len__(self) -> int:
        return len(self.base)


#----------------------------------------------------------------------------
# Abstract base for EDM / SD dynamics with shared PPG and logging logic.

class DiffusionDynamics(Dynamics):
    """Shared infrastructure for all diffusion dynamics.

    Subclasses must implement:
      sigma(t)              -- convert solver-native time to noise level
      score_net(latents, t) -- neural network score estimate

    Solver-native time conventions:
      EDMSolver  passes sigma (float scalar)  → sigma(t) = t
      DDIMSolver passes t_idx (int)           → sigma(t_idx) = sqrt(1 - ᾱ_{t_idx})
    """

    def __init__(self, net, encoder: Encoder, ppg=None, use_net=True,
                 normalize_variance: str = "decomposed", logger=None):
        self.net                = net
        self.encoder            = encoder
        self.ppg                = ppg
        self.use_net            = use_net       # bool or callable(sigma) -> bool
        self.normalize_variance = normalize_variance  # None, "split", "global", or "decomposed"
        self.logger             = logger
        self._text_embeddings   = None

    @abstractmethod
    def sigma(self, t) -> torch.Tensor:
        """Convert the solver's time representation to noise level sigma."""

    @abstractmethod
    def signal_scale(self, t) -> torch.Tensor:
        """Signal scaling factor s(t) where x(t) = s(t)·x₀ + σ(t)·ε."""

    @abstractmethod
    def score_net(self, latents: torch.Tensor, t) -> torch.Tensor:
        """Neural network score estimate. t is solver-native time."""

    @abstractmethod
    def update(self, state) -> None:
        """Capture per-batch context from state before the solver runs."""

    @staticmethod
    def noise_pred_to_score(noise_pred: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """ε → score:  s = -ε / σ"""
        return -noise_pred / sigma

    @staticmethod
    def score_to_noise_pred(score: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """score → ε:  ε = -σ · s"""
        return -sigma * score

    def score_to_x0(self, score: torch.Tensor, x: torch.Tensor, t) -> torch.Tensor:
        """score → x₀:  x₀ = (x + σ²·score) / s(t)"""
        sigma = self.sigma(t)
        return (x + sigma ** 2 * score) / self.signal_scale(t)

    def normalize_variance_score(self, score_from_net: torch.Tensor,
                                  score_ppg: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Variance-corrected rescaling for product of Gaussians.

        Combined precision is 1/σ² + 1/(σ²+ν²), giving effective variance
        σ_c² = σ²(σ²+ν²) / (2σ²+ν²). Rescale to restore original noise level.

        "split":  rescale only the guided subspace; leave P⊥ untouched.
        "global": single scalar rescaling via the determinant of the
                  anisotropic covariance: w_τ(α) = (σ_c²/σ_t²)^α
                  where α = r/d is the fraction of guided dimensions.
        """
        # w = σ_c² / σ_t² (per-dimension weight for the guided subspace)
        score_ppg = score_ppg.to(score_from_net.dtype)
        w = (1 / (1 + self.ppg.noise_gate_inner(sigma.float()))).to(score_from_net.dtype)
        score_sum = score_from_net + score_ppg

        if self.normalize_variance == "global":
            alpha = self._guided_fraction()
            return w ** alpha * score_sum

        # "split": block-diagonal rescaling
        if self.ppg.is_projection and not self.ppg.is_ambient:
            score_proj = self.ppg.project(score_from_net.float()).to(score_from_net.dtype)
            score_orth = score_from_net - score_proj
            return score_orth + w * (score_proj + score_ppg)
        else:
            return w * score_sum

    def _guided_fraction(self) -> float:
        """Return α = r/d, the fraction of guided dimensions. 1.0 for ambient (full-rank)."""
        if self.ppg.is_ambient or not self.ppg.is_projection:
            return 1.0
        for map_ in self.ppg.maps:
            if hasattr(map_, 'projection_ratio') and map_.projection_ratio is not None:
                return 1.0 / map_.projection_ratio
        return 1.0

    def decomposed_score(self, score_from_net: torch.Tensor,
                         latents: torch.Tensor, t) -> torch.Tensor:
        """Variance-preserving combination via orthogonal decomposition.

        Evaluates PPG at s(t)·x̂₀ instead of the noisy x, giving:
          s_c = score_net + G_t · P(s(t)·μ_ex − s(t)·x̂₀) / σ²
        which preserves the total precision τ_t·I.
        """
        sigma = self.sigma(t)
        x0 = self.score_to_x0(score_from_net, latents, t)
        scaled_x0 = self.signal_scale(t) * x0
        score_ppg = self.ppg(scaled_x0.float(), sigma.float()).to(score_from_net.dtype)
        return score_from_net + score_ppg

    def __call__(self, latents: torch.Tensor, t) -> torch.Tensor:
        sigma = self.sigma(t)
        use_net = self.use_net(sigma) if callable(self.use_net) else self.use_net

        if use_net:
            score_from_net = self.score_net(latents, t)
        else:
            score_from_net = torch.zeros_like(latents)
        score_combined = score_from_net

        if self.ppg:
            if self.normalize_variance == "decomposed" and use_net:
                score_combined = self.decomposed_score(score_from_net, latents, t)
                score_ppg = score_combined - score_from_net  # for logging
            else:
                score_ppg = self.ppg(latents.float(), sigma.float()).to(score_from_net.dtype)
                if self.normalize_variance not in (None, "none") and use_net:
                    score_combined = self.normalize_variance_score(score_from_net, score_ppg, sigma)
                else:
                    score_combined = score_from_net + score_ppg

        if self.logger:
            score_ppg_log = score_ppg if self.ppg else torch.zeros_like(score_from_net)
            self.logger.record_scores(score_combined, score_from_net, score_ppg_log)

        return score_combined


#----------------------------------------------------------------------------
# HuggingFace Diffusers implementations

class NoiseScheduleMap:
    """Precomputed bidirectional mapping between EDM sigma and DDPM t_idx / ᾱ.

    VP forward process:  x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
    VE forward process:  x_σ = x₀ + σ · ε

    The EDM-equivalent sigma for a DDPM timestep t is: σ_edm(t) = √((1-ᾱ_t) / ᾱ_t)
    """

    def __init__(self, alphas_cumprod: torch.Tensor):
        self._alphas = alphas_cumprod.float()
        self._edm_sigmas = ((1 - self._alphas) / self._alphas).sqrt()

    @property
    def sigma_max(self) -> float:
        return self._edm_sigmas[-1].item()

    @property
    def sigma_min(self) -> float:
        return self._edm_sigmas[0].item()

    def to(self, device):
        self._alphas = self._alphas.to(device)
        self._edm_sigmas = self._edm_sigmas.to(device)
        return self

    def sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """EDM sigma → continuous DDPM float timestep in [0, T-1]."""
        edm_sigmas = self._edm_sigmas.to(sigma.device)
        above = (edm_sigmas <= sigma)
        i = (above.long().sum() - 1).clamp(0, len(edm_sigmas) - 2)
        lo, hi = edm_sigmas[i], edm_sigmas[i + 1]
        frac = ((sigma - lo) / (hi - lo)).clamp(0, 1)
        return (i.float() + frac).clamp(0, len(edm_sigmas) - 1)

    def sigma_to_alpha(self, sigma: torch.Tensor) -> torch.Tensor:
        """EDM sigma → ᾱ (cumulative alpha).  ᾱ = 1 / (1 + σ²)."""
        return 1.0 / (1.0 + sigma ** 2)

    def t_to_sigma(self, t_idx) -> torch.Tensor:
        """DDPM integer timestep → VP sigma √(1-ᾱ_t)."""
        alpha = self._alphas.to(t_idx.device)[t_idx]
        return (1 - alpha).sqrt()


class StableDiffusionDynamics(DiffusionDynamics):
    """VP / DDPM dynamics (HuggingFace UNet). Solver-native time is integer DDPM timestep.

    For use with EDMSolver, wrap with VEDynamicsWrapper.
    """

    def __init__(self, net, encoder: Encoder, scheduler, guidance_scale: float = 7.5,
                 ppg=None, use_net: bool = True, normalize_variance: str = "split", logger=None):
        super().__init__(net, encoder, ppg, use_net, normalize_variance, logger)
        self.scheduler      = scheduler
        self.guidance_scale = guidance_scale
        self._text_embeddings = None

    def sigma(self, t_idx) -> torch.Tensor:
        """VP noise level: σ_vp = √(1-ᾱ_t)."""
        alpha = self.scheduler.alphas_cumprod[t_idx]
        return (1 - alpha).sqrt()

    def signal_scale(self, t_idx) -> torch.Tensor:
        """VP signal scale: s(t) = √ᾱ_t."""
        alpha = self.scheduler.alphas_cumprod[t_idx]
        return alpha.sqrt()

    def score_net(self, latents: torch.Tensor, t_idx) -> torch.Tensor:
        latents_input = torch.cat([latents] * 2)
        noise_pred = self.net(
            latents_input, t_idx,
            encoder_hidden_states=self._text_embeddings,
        ).sample
        uncond, cond = noise_pred.chunk(2)
        eps = uncond + self.guidance_scale * (cond - uncond)
        return self.noise_pred_to_score(eps, self.sigma(t_idx))

    def update(self, state) -> None:
        self._text_embeddings = state.text_embeddings  # (2B, 77, D)
        if self.ppg:
            self.ppg.update(state)

class EDMDynamics(DiffusionDynamics):
    """VE / EDM dynamics. Solver-native time is sigma (noise level) directly.

    The EDM denoiser D(x; σ) is called as net(x, sigma) and returns the
    denoised image. Score: s = (D(x;σ) - x) / σ².
    """

    def __init__(self, net, encoder: Encoder, ppg=None, use_net: bool = True,
                 normalize_variance: str = "split", logger=None):
        super().__init__(net, encoder, ppg, use_net, normalize_variance, logger)
        self._class_labels = None

    @property
    def sigma_min(self) -> Optional[float]:
        return getattr(self.net, 'sigma_min', None)

    @property
    def sigma_max(self) -> Optional[float]:
        return getattr(self.net, 'sigma_max', None)

    def sigma(self, t) -> torch.Tensor:
        return t  # EDMSolver passes sigma directly

    def signal_scale(self, t):
        return 1.0

    def score_net(self, latents: torch.Tensor, sigma) -> torch.Tensor:
        denoised = self.net(latents, sigma, self._class_labels)
        return (denoised - latents) / sigma ** 2

    def update(self, state) -> None:
        self._class_labels = getattr(state, 'labels', None)
        if self.ppg:
            self.ppg.update(state)

#----------------------------------------------------------------------------
# EDM Solver (Heun's method)
#
# Dynamics returns the score ∇_x log p(x; σ) where σ is the noise level.
# PF ODE : d = (x - D(x;σ))/σ = -σ · score.

class EDMSolver(Solver):

    def __init__(
        self,
        num_steps:        int   = 50,
        sigma_min:        float = 0.002,
        sigma_max:        float = 80.0,
        rho:              float = 7,
        S_churn:          float = 0,
        S_min:            float = 0,
        S_max:            float = float('inf'),
        S_noise:          float = 1,
        apply_2nd_order:  bool  = True,
        verbose:          bool  = False,
    ):
        self.num_steps       = num_steps
        self.sigma_min       = sigma_min
        self.sigma_max       = sigma_max
        self.rho             = rho
        self.S_churn         = S_churn
        self.S_min           = S_min
        self.S_max           = S_max
        self.S_noise         = S_noise
        self.apply_2nd_order = apply_2nd_order
        self.verbose         = verbose

    @torch.no_grad()
    def __call__(
        self,
        dynamics: Dynamics,
        noise: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Any]:

        num_steps = self.num_steps
        sigma_min, sigma_max, rho = self.sigma_min, self.sigma_max, self.rho

        # Clamp to the range the network actually supports
        if getattr(dynamics, 'sigma_min', None) is not None:
            sigma_min = max(sigma_min, dynamics.sigma_min)
        if getattr(dynamics, 'sigma_max', None) is not None:
            sigma_max = min(sigma_max, dynamics.sigma_max)

        # EDM sigma schedule (float64 for numerical precision)
        step_indices = torch.arange(num_steps, device=noise.device, dtype=torch.float64)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
                   (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, t_steps.new_zeros(1)])

        # Initialize: x = σ_max · ε
        x_next = noise.to(torch.float64) * t_steps[0]

        xs = []
        x0s = []
        for i, (sigma_cur, sigma_next) in tqdm(
            list(enumerate(zip(t_steps[:-1], t_steps[1:]))),
            disable=not self.verbose,
        ):
            x_cur = x_next

            # Stochastic churn: temporarily increase noise
            gamma = min(self.S_churn / num_steps, np.sqrt(2) - 1) if (self.S_min <= sigma_cur <= self.S_max) else 0
            sigma_hat = sigma_cur + gamma * sigma_cur
            x_hat = x_cur + (sigma_hat ** 2 - sigma_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # PF ODE d = -σ · score
            score = dynamics(x_hat, sigma_hat)
            d_cur = -sigma_hat * score

            # Predicted x0 at current sigma
            if hasattr(dynamics, 'score_to_x0'):
                x0s.append(dynamics.score_to_x0(score, x_hat, sigma_hat))

            # Euler step
            x_next = x_hat + d_cur * (sigma_next - sigma_hat)

            # Heun's 2nd-order correction
            if self.apply_2nd_order and i < num_steps - 1:
                score_next = dynamics(x_next, sigma_next)
                d_prime = -sigma_next * score_next
                x_next = x_hat + (sigma_next - sigma_hat) * (0.5 * d_cur + 0.5 * d_prime)

            xs.append(x_next)

        return xs, (x0s if x0s else None)


class DDIMSolver(Solver):
    """Integrates dynamics from t=T to t=0 using a DDIM scheduler.

    Expects dynamics to return the score ∇_x log p(x_t).
    Converts to noise prediction internally for scheduler.step().
    """

    def __init__(self, scheduler: DDIMScheduler, num_inference_steps: int = 50,
                 ddim_eta: float = 0.0, eta_seed: int = 0, verbose: bool = False):
        self.scheduler           = scheduler
        self.num_inference_steps = num_inference_steps
        self.ddim_eta            = ddim_eta
        self.eta_seed            = eta_seed
        self.verbose             = verbose
        self.scheduler.set_timesteps(num_inference_steps)

    def __call__(
        self,
        dynamics: Dynamics,
        noise: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], None]:

        self.scheduler.set_timesteps(self.num_inference_steps)

        step_kwargs = {}
        if self.ddim_eta > 0:
            step_kwargs['eta'] = self.ddim_eta
            step_kwargs['generator'] = torch.Generator(device=noise.device).manual_seed(self.eta_seed)

        latents = noise
        xs = []
        x0s = []
        for t_idx in tqdm(self.scheduler.timesteps, disable=not self.verbose):
            score = dynamics(latents, t_idx)
            sigma = dynamics.sigma(t_idx)
            noise_pred = -sigma * score
            if hasattr(dynamics, 'score_to_x0'):
                x0s.append(dynamics.score_to_x0(score, latents, t_idx))
            latents = self.scheduler.step(noise_pred, t_idx, latents, **step_kwargs).prev_sample
            xs.append(latents)
        return xs, (x0s if x0s else None)

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# EasyDict(images, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    solver:         Solver,
    dynamics:       Dynamics,
    inputs:         InputsIterable,

    max_batch_size: int          = 32,
    dir_out:        Optional[str] = None,
    verbose:        bool          = False,
    device                       = torch.device("cuda"),
) -> Iterable:

    # Initialize torch distributed
    if not torch.distributed.is_initialized():
        dist.init()

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Compute and inject rank batches into inputs
    num_batches = max((len(inputs.seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    inputs.rank_batches = np.array_split(np.arange(len(inputs.seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]

    image_iter = ImageIterable(solver, dynamics, verbose)(inputs)

    if dir_out:
        image_iter = SavingIterable(dir_out)(image_iter)

    return image_iter


#----------------------------------------------------------------------------
# Single-process image generation. No distributed logic.

def generate_images_local(
    solver:         Solver,
    dynamics:       Dynamics,
    inputs:         InputsIterable,

    max_batch_size:  int                   = 32,
    verbose:         bool                  = False,
    snapshot_steps:  Optional[List[int]]   = None,
    snapshot_as_x0:  bool                  = True,
) -> Iterable:

    num_batches = max((len(inputs.seeds) - 1) // max_batch_size + 1, 1)
    inputs.rank_batches = np.array_split(np.arange(len(inputs.seeds)), num_batches)

    image_iter = ImageIterable(solver, dynamics, verbose,
                               snapshot_steps=snapshot_steps,
                               snapshot_as_x0=snapshot_as_x0)(inputs)
    return image_iter

