""" Read the configs from a JSON and run images without reloading PyTorch models"""

import os
import copy
import pickle
import torch
import tqdm
import dnnlib
import generate
import myconfig

from PIL import Image
from typing import Literal
from torch_utils import distributed as dist
from mylib.helpers import update_EDM
from myconfig import sampler_args
from myconfig import gvf_args as gvf_args_og

#----------------------------------------------------------------------------
# Visualization functions for composing images together.
# With possibility for batching.

def compose_images(images : list[Image.Image], axis=Literal["h", "v"]) -> Image.Image:
    """Takes a list of PIL Images and produces a horizontal composition"""
    if not images:
        raise ValueError("Empty image list provided")

    if axis == "h":
        height = max(image.height for image in images)
        width = sum(image.width for image in images)
        comp = Image.new("RGB", (width, height))

        cum_width = 0
        for image in images:
            comp.paste(image, (cum_width, 0))
            cum_width += image.width

    elif axis == "v":
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        comp = Image.new("RGB", (width, height))

        cum_height = 0
        for image in images:
            comp.paste(image, (0, cum_height))
            cum_height += image.height

    else:
        raise ValueError(f"Invalid axis {axis}, must be 'h' or 'v'")

    return comp
    
def compose_images_batched(image_lists: list[list[Image.Image]], axis=Literal["h", "v"]) -> list[Image.Image]:
    """Batched version of compose_images"""
    if not image_lists:
        raise ValueError("Empty batch list provided")

    # Transpose the 2D list: [[img1a, img1b], [img2a, img2b]] -> [[img1a, img2a], [img1b, img2b]]
    transposed = list(zip(*image_lists))
    return [compose_images(list(group), axis) for group in transposed]

#----------------------------------------------------------------------------
# Continously runs a network for gennerating images 
# GVF and sampler args are passed from myconfig.py

def main():
    dist.init()
    
    # Configuration
    num_images = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = range(0, num_images)
    outdir = ".temp/last"
    template_dir = "data/templates_per_classid"

    # Load Model 
    if dist.get_rank() == 0:
        dist.print0('Loading network...')
    net_pkl = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl"
    with dnnlib.util.open_url(net_pkl, verbose=True) as f:
        data = pickle.load(f)
    net = update_EDM(data['ema']).to(device)
    # Load encoder
    encoder = data.get('encoder', None)
    if encoder is None:
        encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')


    gvf_argss = []
    for nu in [31.21, 8.28, 2.17, 0.41, 0.02]: # intial sweep
    # for nu in [8.28, 6.46, 5.0, 3.83, 2.9, 2.17]: # zoom template reconstruction 
    # for nu in [80.0, 66.93, 55.74, 46.19, 38.07, 31.22, 25.45, 20.61, 16.59, 13.26, 10.52, 8.28, 6.46, 5.0, 3.83, 2.9, 2.17, 1.60]: # zoom before 8.28 reconstruction 
    # for nu in [1.6086, 1.1743, 0.8446, 0.5977, 0.4154, 0.283, 0.1886, 0.1226, 0.0774, 0.0474, 0.0279, 0.0158, 0.0085, 0.0043, 0.002]: # all the way at the end
        clone = copy.deepcopy(gvf_args_og)
        clone["vectorfield"]["noise_gate"] = copy.deepcopy(gvf_args_og["vectorfield"]["noise_gate"])
        clone["vectorfield"]["noise_gate"]["nu"] = nu
        gvf_argss.append(clone)

    # Iterate through differnet configs
    edit_per_param = [] 
    for gvf_args in gvf_argss:
        # Refernce non-serializbles
        gvf_args["args_references"]["network"] = net

        # Generate images
        if dist.get_rank() == 0:
            dist.print0("Generating images...")
        
        image_iter = generate.generate_images(
            net,
            encoder=encoder,
            gvf_args=gvf_args,
            seeds=seeds,
            verbose=(dist.get_rank() == 0),
            device=device,
            template_dir=template_dir,
            sampler_kwargs=sampler_args,
            gradient_kwargs=myconfig.gradient_kwargs,
            live_editing=False,
            ddim_inversion=False,
            use_noisy_examples=False,
        )


        # Get paths from all batches, not just last
        results = []
        for r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
            results.append(r)
        
        # Only run composition on rank 0
        if dist.get_rank() == 0:
            # Get path from unedited images
            path_original = [f".temp/uncond-32steps-1storder/{i:06d}.png" for i in range(0, num_images)]    
            # Get paths from all batches, not just last
            path_examples = []
            edited_images = []
            for batch_result in results:
                path_examples.extend(batch_result.example_paths)
                edited_images.extend([img for img in batch_result.images])

            # Convert to PIL 
            examples = [Image.open(path) for path in path_examples] 
            original = [Image.open(path) for path in path_original]
            edit_per_param.append([Image.fromarray(arr.permute(1, 2, 0).cpu().numpy(), "RGB") for arr in edited_images])

    # Compose templates together 
    og = compose_images(original, "v")
    # Compose examples together
    ex = compose_images(examples, "v")
    # Compose edited images together
    edited = compose_images(compose_images_batched(edit_per_param, "h"), "v")
    # Put togetogher
    comp = compose_images([og, edited, ex], "h")
    comp.save(os.path.join(outdir, "batch.png"))  



    torch.distributed.barrier()
    if dist.get_rank() == 0:
        print("Exiting...")

if __name__ == "__main__":
    main()
