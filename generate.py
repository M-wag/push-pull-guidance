import os
import numpy as np
import torch
import PIL.Image

from abc import ABC, abstractmethod
from diffusers import DDIMScheduler
from torch_utils import distributed as dist
from tqdm import tqdm
from typing import Any, Iterable, List, Optional, Tuple

#----------------------------------------------------------------------------
# Convenience class that behaves like a dict but allows access with the attribute syntax. x = d.key

class EasyDict(dict):

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

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


class TextConditionedInputsIterable(InputsIterable):
    """ 
    Yields per-batch initial conditions for text-to-image diffusion. 
    Text encoding is performed here as it is part of the initial condition, not the dynamics.
    """

    def __init__(
        self,
        seeds: List[int],
        shape: Tuple[int, ...],     # e.g. (4, 64, 64) for 512px SD — set explicitly by caller
        prompts: List[str],         # one per seed
        tokenizer,
        text_encoder,
        negative_prompt: str = "",
        device: str = "cuda",
    ):
        self.rank_batches    = None  # injected by generate_images
        self.seeds           = seeds
        self.shape           = shape
        self.prompts         = prompts
        self.tokenizer       = tokenizer
        self.text_encoder    = text_encoder
        self.negative_prompt = negative_prompt
        self.device          = device

    def __iter__(self):
        for batch_idx, indices in enumerate(self.rank_batches):
            batch_seeds   = [self.seeds[i]  for i in indices]
            batch_prompts = [self.prompts[i] for i in indices]

            rnd   = StackedRandomGenerator(self.device, batch_seeds)
            noise = rnd.randn([len(batch_seeds), *self.shape], device=self.device)

            yield EasyDict(
                seeds           = batch_seeds,
                indices         = list(indices),
                batch_idx       = batch_idx,
                num_batches     = len(self.rank_batches),
                noise           = noise,
                text_embeddings = self._encode(batch_prompts),  # (2B, 77, D)
                prompts         = batch_prompts,
            )

    def __len__(self) -> int:
        return len(self.rank_batches)

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


class StableDiffusionDynamics(Dynamics):
    """ Dynamics for Stable Diffusion: UNet forward pass with classifier-free guidance. """

    def __init__(self, unet, vae, guidance_scale: float = 7.5, controller=None):
        self.unet             = unet
        self.encoder          = VAEEncoder(vae)  
        self.guidance_scale   = guidance_scale
        self.controller       = controller
        self._text_embeddings = None

    def update(self, state) -> None:
        self._text_embeddings = state.text_embeddings  # (2B, 77, D)
        if self.controller is not None:
            self.controller.reset()

    def __call__(self, latents: torch.Tensor, t: int) -> torch.Tensor:
        """CFG noise prediction."""
        latents_input = torch.cat([latents] * 2)
        noise_pred = self.unet(
            latents_input, t,
            encoder_hidden_states=self._text_embeddings,
        ).sample
        uncond, cond = noise_pred.chunk(2)
        return uncond + self.guidance_scale * (cond - uncond)


class DDIMSolver(Solver):
    """
    Integrates StableDiffusionDynamics from t=T to t=0 using a DDIM scheduler.
    Satisfies the Solver interface required by ImageIterable.
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
        for t in tqdm(self.scheduler.timesteps, disable=not self.verbose):
            noise_pred = dynamics(latents, t)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
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

