# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from importlib import reload
from torch_utils import distributed as dist
from training.networks import update_EDM
from mylib.diffusion import EDMSampler,  load_templates_batch
from mylib.gvf import create_gvf

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
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gvf_args            = None,                 # Arguments to initialize GuidanceVectorfield. None = lambda x : 0
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    template_dir        = None,                 # Where templates are stored
    sampler_kwargs      = None,                 # Additional arguments for the sampler function.
    gradient_kwargs     = None,                 # Arguments defining the type of gradient used in sampler
    live_editing        = False,                # Allow live-editing of the code 
    ddim_inversion      = False,                # Whether to use DDIM inversion to generate initial noise 
    use_noisy_examples  = True,                 # Whether to use noisy version of latents of examples for x_T
):
    
    import mylib.diffusion
    if live_editing:
        reload(mylib.diffusion)
    from mylib.diffusion import EDMSampler

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    if isinstance(net, str):
        dist.print0(f'Loading network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema']
        net = update_EDM(net).to(device) # Update EDM code
        encoder = data.get('encoder', None)
        if encoder is None:
            encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None


    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Create guidance vectorfield
    gvf = None
    if gvf_args:
        if verbose:
            dist.print0(f'Creating Guidance Vectorfield from args ...')
        gvf = create_gvf(**gvf_args).to(device)
        gradient_kwargs["gvf"] = gvf

    # Setup sampler 
    edm_sampler = EDMSampler(time_disc = "edm", gradient_kwargs=gradient_kwargs)

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)
        
        def _sample_example_idx(self, template_dir, seed, class_, idx_range=None):
            # Determien directory for specific class
            class_dir = os.path.join(template_dir, str(int(class_)))

            # If no custom range, sample from all files in class directory
            if idx_range:
                low, high = idx_range
            else:
                low = 0
                high = len(os.listdir(class_dir))

            g = torch.Generator().manual_seed(seed)
            return torch.randint(low , high, (), generator=g).item()
        
        def _update_examples_gvf(self, gvf, paths):
            examples = load_templates_batch(paths).unsqueeze(1).to(device)  # [B, N, C, H, W] TODO : will this mess up for N > 1
            examples = encoder.encode_latents(examples)
            gvf.set_features_template(examples) 
            gvf.setup_score()
        
        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, examples=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                r.example_paths = []
                r.example_idx = []
                # Randomly pick class index
                if len(r.seeds) > 0:
                    # Pick labels and corresponding examples.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    r.labels = None
                    if net.label_dim > 0:
                        r.labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(r.seeds)], device=device)]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1

                        # For each label, pick a random example and save its path.
                        for seed, label in zip(r.seeds, torch.argmax(r.labels, axis=1)): 
                            example_idx = self._sample_example_idx(template_dir, seed, label)
                            example_path = os.path.join(template_dir, str(int(label)), f"{example_idx}.png")
                            r.example_idx.append(example_idx)
                            r.example_paths.append(example_path)

                    # Compute latents for the example
                    latents_example  = encoder.encode_latents(load_templates_batch(r.example_paths)).to(device)
                    # Whether to use DDIM inversion
                    if ddim_inversion:
                        print("Inverting examples")
                        xTs, activations_by_t = edm_sampler.edm_inversion(
                                net, images=latents_example, labels=r.labels, device=device, disable_tqdm=True, **sampler_kwargs)
                        r.noise = xTs.to(device)

                        # Update bottleneck features 
                        if "h_exam_per_t" in gradient_kwargs:
                            gradient_kwargs["h_exam_per_t"] = {activations_by_t["bottleneck"][t] for t in activations_by_t["bottleneck"].keys()}
                    
                    # Initialize SDEdit
                    if use_noisy_examples:
                        r_noise = latents_example / net.sigma_max + r.noise

                    # Update gvf to use examples of current batch
                    if gradient_kwargs.get("gvf"):
                        self._update_examples_gvf(gradient_kwargs["gvf"], r.example_paths)

                    # Update gradient kwargs for sampler
                    edm_sampler.init_gradient(gradient_kwargs)

                    # Generate images
                    xs, _ = edm_sampler(net, noise=r.noise, labels=r.labels, device=device, disable_tqdm=True, **sampler_kwargs)
                    r.images = encoder.decode(xs[-1])

                    # Save images.
                    if outdir is not None:
                        for seed, image in zip(r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:06d}.png'))

                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()
