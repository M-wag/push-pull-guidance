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
# Abstract base classes

class Encoder(ABC):
    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images (B, C, H, W) -> latents (B, Z, H', W')"""

    @abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents (B, Z, H', W') -> images (B, C, H, W) in [0, 1]"""


class InputsIterable(ABC):
    seeds: List[int]    # required attribute; used by generate_images for rank splitting
    rank_batches = None # set to None at construction; injected by generate_images

    @abstractmethod
    def __iter__(self):
        """Yield one EasyDict per batch, driven by self.rank_batches."""

    @abstractmethod
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
    def __init__(self, solver: Solver, dynamics: Dynamics, verbose=False):
        self.solver = solver
        self.dynamics = dynamics
        self.verbose = verbose

    def __call__(self, iter_state: InputsIterable) -> Iterable:
        self.solver.verbose = self.verbose
        for state in iter_state:
            yield self._process_batch(state)

    def _process_batch(self, state):
        if len(state.seeds) > 0:
            # Update dynamics
            self.dynamics.update(state)
            # Generate images
            xs, _ = self.solver(self.dynamics, state.noise)
            state.images = self.dynamics.encoder.decode(xs[-1])
            # Yield results.
            torch.distributed.barrier() # keep the ranks in sync
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
        """images in [-1, 1] -> scaled latents"""
        return self.vae.encode(images).latent_dist.mean * self.scale

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """scaled latents -> images in [0, 1]"""
        images = self.vae.decode(latents / self.scale).sample
        return (images / 2 + 0.5).clamp(0, 1)


#----------------------------------------------------------------------------
# Composable input iterables

class NoiseIterable(InputsIterable):
    """Base iterable that generates seeded noise. Owns seeds and rank_batches."""

    def __init__(self, seeds: List[int], shape: Tuple[int, ...], device: str = "cuda"):
        self.rank_batches = None  # injected by generate_images
        self.seeds  = seeds
        self.shape  = shape
        self.device = device

    def __iter__(self):
        for batch_idx, indices in enumerate(self.rank_batches):
            batch_seeds = [self.seeds[i] for i in indices]
            rnd   = StackedRandomGenerator(self.device, batch_seeds)
            noise = rnd.randn([len(batch_seeds), *self.shape], device=self.device)
            yield EasyDict(
                seeds       = batch_seeds,
                indices     = list(indices),
                batch_idx   = batch_idx,
                num_batches = len(self.rank_batches),
                noise       = noise,
            )

    def __len__(self) -> int:
        return len(self.rank_batches)


class TextEmbeddingIterable:
    """Extension that adds text_embeddings and prompts to each batch state."""

    def __init__(self, prompts: List[str], tokenizer, text_encoder,
                 negative_prompt: str = "", device: str = "cuda"):
        self.prompts         = prompts
        self.tokenizer       = tokenizer
        self.text_encoder    = text_encoder
        self.negative_prompt = negative_prompt
        self.device          = device

    def enrich(self, state: EasyDict, indices: list) -> None:
        batch_prompts = [self.prompts[i] for i in indices]
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

    def __init__(self, paths_example: List[str], encoder: Optional[Encoder] = None, device: str = "cuda"):
        self.paths_example  = paths_example
        self.device         = device
        self.encoder        = encoder

    def enrich(self, state: EasyDict, indices: list) -> None:
        batch_paths = [self.paths_example[i] for i in indices]
        examples = load_images(batch_paths, device=self.device, rescale=True)
        if examples is not None:
            if self.encoder:
                examples = self.encoder.encode(examples)
            state.examples = examples


class CombinedInputs(InputsIterable):
    """Combines a base NoiseIterable with extensions that enrich each batch state."""

    def __init__(self, base: NoiseIterable, *extensions):
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
                ext.enrich(state, state.indices)
            yield state

    def __len__(self) -> int:
        return len(self.base)


#----------------------------------------------------------------------------
# HuggingFace Diffusers implementations

class StableDiffusionDynamics(Dynamics):
    """ Dynamics for Stable Diffusion: UNet forward pass with classifier-free guidance. """

    def __init__(self, unet, vae, scheduler, guidance_scale: float = 7.5, ppg=None, use_unet: bool = True):
        self.unet               = unet
        self.encoder            = VAEEncoder(vae)
        self.scheduler          = scheduler
        self.guidance_scale     = guidance_scale
        self.ppg                = ppg
        self.use_unet           = use_unet
        self._text_embeddings   = None

    def update(self, state) -> None:
        self._text_embeddings = state.text_embeddings  # (2B, 77, D)
        if self.ppg:
            self.ppg.update(state)

    def __call__(self, latents: torch.Tensor, t_idx: int) -> torch.Tensor:
        """CFG noise prediction."""
        if self.use_unet:
            latents_input = torch.cat([latents] * 2)
            noise_pred = self.unet(
                latents_input, t_idx,
                encoder_hidden_states=self._text_embeddings,
            ).sample
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + self.guidance_scale * (cond - uncond)
        else:
            noise_pred = torch.zeros_like(latents)

        if self.ppg:
            alpha = self.scheduler.alphas_cumprod[t_idx]
            noise = ((1 - alpha) / alpha).sqrt()
            score = self.ppg(latents, noise) # ∇log p(c | x)
            noise_pred += -noise * score

        return noise_pred


class DDIMSolver(Solver):
    """
    Integrates StableDiffusionDynamics from t=T to t=0 using a DDIM scheduler.
    """

    def __init__(self, scheduler: DDIMScheduler, num_inference_steps: int = 50, verbose: bool = False):
        self.scheduler           = scheduler
        self.num_inference_steps = num_inference_steps
        self.verbose             = verbose
        self.scheduler.set_timesteps(num_inference_steps)

    def __call__(
        self,
        dynamics: Dynamics,
        noise: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], None]:

        latents = noise
        xs = []
        for t_idx in tqdm(self.scheduler.timesteps, disable=not self.verbose):
            noise_pred = dynamics(latents, t_idx)
            latents = self.scheduler.step(noise_pred, t_idx, latents).prev_sample
            xs.append(latents)
        return xs, None

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

