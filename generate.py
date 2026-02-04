# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import numpy as np
import torch
import PIL.Image
import dnnlib
import dblib

from torch_utils import distributed as dist
from typing import Iterable

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
# Iterable representing initial conditions.

class InitialConditionIterable:
    def __init__(
        self, 
        rank_batches, 
        seeds, 
        shape,
        label_dim,
        class_idx=None, 
        example_idx_range=None,
        dir_template=None,
        device = "cpu",
    ):

        self.rank_batches = rank_batches
        self.seeds = seeds
        self.shape = shape
        self.label_dim = label_dim
        self.dir_template = dir_template
        self.class_idx = class_idx
        self.example_idx_range = example_idx_range
        self.device = device
    
    def __iter__(self):
        for batch_idx, indices in enumerate(self.rank_batches):
            batch_seeds = [self.seeds[idx] for idx in indices]
            r = dnnlib.EasyDict(
                seeds=batch_seeds,
                indices=indices,
                batch_idx=batch_idx,
                num_batches=len(self.rank_batches),
                labels=None,
                noise=None,
                example_paths=[],
                example_idx=[],
                examples=None,
            )
            
            if len(batch_seeds) > 0:
                rnd = StackedRandomGenerator(self.device, batch_seeds)
                r.noise = rnd.randn([len(batch_seeds), *self.shape], device=self.device)
                if self.label_dim:
                    r.labels = torch.eye(self.label_dim, device=self.device)[
                        rnd.randint(self.label_dim, size=[len(batch_seeds)], device=self.device)
                    ]
                    if self.class_idx is not None:
                        r.labels[:, :] = 0
                        r.labels[:, self.class_idx] = 1
                    
                    if self.dir_template is not None:
                        for seed, label in zip(batch_seeds, torch.argmax(r.labels, axis=1)):
                            example_idx = self._sample_example_idx(self.dir_template, seed, label, idx_range=self.example_idx_range)
                            example_path = os.path.join(self.dir_template, str(int(label)), f"{example_idx}.png")
                            r.example_idx.append(example_idx)
                            r.example_paths.append(example_path)
                            r.examples = dnnlib.util.load_templates_batch(r.example_paths)
            yield r

    def __len__(self):
        return len(self.rank_batches)
    
    def _sample_example_idx(self, template_dir, seed, class_, idx_range=None):
        class_dir = os.path.join(template_dir, str(int(class_)))
        if idx_range:
            low, high = idx_range
        else:
            low = 0
            high = len(os.listdir(class_dir))
        g = torch.Generator().manual_seed(seed)
        return torch.randint(low, high, (), generator=g).item()
    

#----------------------------------------------------------------------------
# Iterable which adds encoded template latents to the noise.

class NoisyExamplesIterable:
    """Wraps an iterable and adds encoded template latents to the noise."""

    def __init__(self, encoder, sigma_max):
        self.encoder = encoder
        self.sigma_max = sigma_max

    def __call__(self, iterable):
        for state in iterable:
            if state.examples is not None:
                latents_example = self.encoder.encode(state.examples).to(state.noise.device)
                state.noise += (latents_example / self.sigma_max)
            yield state


#----------------------------------------------------------------------------
# Iterable which applies diffusion process to initial condition iterable.

class ImageIterable:
    def __init__(self, solver, dynamics, verbose=False):
        self.solver = solver
        self.dynamics = dynamics
        self.verbose = verbose

    def __call__(self, iter_state: InitialConditionIterable) -> Iterable:
        for state in iter_state:
            yield self._process_batch(state)

    def _process_batch(self, state):
        if len(state.seeds) > 0:
            # Update dynamics
            self.dynamics.update(state)
            # Generate images
            xs, _ = self.solver(self.dynamics, state.noise, state.labels)
            state.images = self.dynamics.encoder.decode(xs[-1])
            # Yield results.
            torch.distributed.barrier() # keep the ranks in sync
            return state


#----------------------------------------------------------------------------
# Logs experiment metadata to a SQLite database.

class ExperimentLogger:
    def __init__(self, db, experiment_factory):
        """
        Args:
            db: SQLite database connection
            experiment_factory: Callable (state_prms, result_path) -> Experiment
        """
        self.db = db
        self.experiment_factory = experiment_factory

    def log(self, state_prms, result_path):
        experiment = self.experiment_factory(state_prms, result_path)
        with self.db:
            dblib.util.save_experiment(self.db.cursor(), experiment)

    def get_last_id(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT id FROM experiments ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return int(row[0]) if row else 0


#----------------------------------------------------------------------------
# Wrapper around iterables that saves images to disk.

class SavingIterable:
    def __init__(self, dir_save, start_id=0, logger=None, use_subdirs=True, filename_fn=None):
        self.dir_save = dir_save
        self.logger = logger
        self.use_subdirs = use_subdirs
        self.filename_fn = filename_fn
        self._current_id = start_id if logger is None else logger.get_last_id()

    def __call__(self, iterable):
        for states in iterable:
            if self.dir_save is not None:
                self.save(states)
            yield states

    def save(self, states):
        images = states.images.permute(0, 2, 3, 1).detach().cpu().numpy()
        ids_example = [int(path.split("/")[-1].split(".")[0]) for path in states.example_paths]
        ids_class = [int(path.split("/")[-2]) for path in states.example_paths]

        for (image, id_example, id_class) in zip(images, ids_example, ids_class):
            self._current_id += 1

            if self.filename_fn is not None:
                filename = self.filename_fn(id_class, id_example, self._current_id)
                result_path = os.path.join(self.dir_save, filename)
                dir_image = self.dir_save
            elif self.use_subdirs:
                dir_image = os.path.join(self.dir_save, f'{self._current_id // 1000 * 1000:06d}')
                result_path = os.path.join(dir_image, f"{self._current_id:06d}.png")
            else:
                dir_image = self.dir_save
                result_path = os.path.join(dir_image, f"{self._current_id:06d}.png")

            # Log to database if logger is configured
            if self.logger is not None:
                state_prms = {
                    "id_example": id_example,
                    "id_class": id_class,
                }
                self.logger.log(state_prms, result_path)

            # Save image
            os.makedirs(dir_image, exist_ok=True)
            PIL.Image.fromarray(image, 'RGB').save(result_path)

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, examples, batch_idx, num_batches, indices, seeds)

def generate_images(
    solver,
    dynamics,

    class_idx           = None,                 # Class label. None = select randomly.
    use_noisy_examples  = False,                # Whether to use noisy version of latents of examples for x_T
    example_idx_range   = None,                 # Indicates a range (low, high) of the example indices you want to sample
    seeds               = range(16, 24),        # List of random seeds.

    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    dir_out             = None,                 # If passed, where images are stored.
    verbose             = False,                # Enable status prints?
    device              = torch.device("cuda")  # Which compute device to use.
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
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]

    # Setup intial states
    shape = (dynamics.img_channels, dynamics.img_resolution, dynamics.img_resolution)
    label_dim = dynamics.label_dim
    dir_template="data/images/examples"
    states = InitialConditionIterable(
            rank_batches=rank_batches,
            seeds=seeds,
            shape=shape,
            label_dim=label_dim,
            dir_template=dir_template,
            device=device,
            ##
            class_idx=class_idx,
            example_idx_range=example_idx_range,
    )

    # Optionally add noisy examples
    if use_noisy_examples:
        states = NoisyExamplesIterable(dynamics.encoder, solver.sigma_max)(states)

    # Map to image transformation
    image_iter = ImageIterable(solver, dynamics, verbose)(states)

    # SavingIterable
    if dir_out:
       image_iter = SavingIterable(dir_out)(image_iter)

    return image_iter


