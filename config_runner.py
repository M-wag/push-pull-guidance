""" Read the configs from a JSON and run images without reloading PyTorch models"""

import os
import pickle
import importlib
import traceback
import torch
import tqdm
import dnnlib
import generate

from PIL import Image
from torch_utils import distributed as dist
from mylib.helpers import update_EDM

#----------------------------------------------------------------------------
# Visualization functions for composing images together.
# With possibility for batching.

def compose_images(images : list[Image.Image]) -> Image.Image:
    """Takes a list of PIL Images and produces a horizontal composition"""
    if not images:
        raise ValueError("Empty image list provided")

    height = max(image.height for image in images)
    width = sum([image.width for image in images])

    comp = Image.new('RGB', (width, height))

    cum_width = 0
    for image in images:
        comp.paste(image, (cum_width, 0))
        cum_width += image.width

    return comp
    
def compose_images_batched(image_lists: list[list[Image.Image]]) -> list[Image.Image]:
    """Batched version of compose_images"""
    if not image_lists:
        raise ValueError("Empty batch list provided")

    # Transpose the 2D list: [[img1a, img1b], [img2a, img2b]] -> [[img1a, img2a], [img1b, img2b]]
    transposed = list(zip(*image_lists))
    return [compose_images(list(group)) for group in transposed]

#----------------------------------------------------------------------------
# Continously runs a network for gennerating images 
# GVF and sampler args are passed from myconfig.py

def main():
    dist.init()
    
    # Configuration
    num_images = 20
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

    # Main interactive loop
    while True:
        try:
            if dist.get_rank() == 0:
                user_input = input(">>> ").strip().lower()
            if user_input == 'quit':
                break
                
            # Reload configuration
            import myconfig
            importlib.reload(myconfig)
            importlib.reload(generate)
            from myconfig import gvf_args

            # Refernce non-serializbles
            gvf_args["args_references"]["network"] = net

            # Generate images
            if dist.get_rank() == 0:
                dist.print0("Generating images...")
            
            image_iter = generate.generate_images(
                net,
                encoder=encoder,
                gvf_args=myconfig.gvf_args,
                seeds=seeds,
                verbose=(dist.get_rank() == 0),
                device=device,
                template_dir=template_dir,
                sampler_kwargs=myconfig.sampler_args,
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
                edited = [Image.fromarray(arr.permute(1, 2, 0).cpu().numpy(), "RGB") for arr in edited_images]

                # Compose and save
                compositions = compose_images_batched([original, edited, examples])
                
                for i, comp in enumerate(compositions):
                    comp.save(os.path.join(outdir, f"{i:06d}.png"))  # Fixed typo

        except KeyboardInterrupt:
            if dist.get_rank() == 0:
                print("\nGeneration interrupted. Ready for new config.")
            continue
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"\nError occurred: {str(e)}")
                traceback.print_exc()
            continue

    # Cleanup
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        print("Exiting...")

if __name__ == "__main__":
    main()
