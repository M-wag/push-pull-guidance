""" Read the configs from a JSON and run images without reloading PyTorch models"""

import os
import pickle
import traceback
import torch
import dnnlib
import tqdm

from PIL import Image
from run_metrics import ExperimentRunner
from torch_utils import distributed as dist
from training.networks import update_EDM
from einops import rearrange
from mylib.diffusion import time_steps_edm

#----------------------------------------------------------------------------
# Continously runs a network for gennerating images 

def main():
    dist.init()
    
    # Load Model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # Setup runner 
    num_images = 9
    outdir = "data/images/config_runner/"

    if True:
        paths = {
            "config"    : "configs/config_runner.py",
            "templates" : "data/images/examples",
            "out"       : None,
        }

        # Setup runner and generate images
        runner = ExperimentRunner(paths, num_images=num_images)
        image_iter = runner.generate_images(net=net, encoder=encoder)

        # Get paths from all batches, not just last
        results = []
        for r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
            results.append(r)
        torch.distributed.barrier()

        # Only run composition on rank 0
        if dist.get_rank() == 0:
            img = rearrange(results[0].images, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=3)
            Image.fromarray(img.cpu().numpy()).save(os.path.join(outdir, f"out.png"))

    # Cleanup
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        print("Exiting...")

if __name__ == "__main__":
    main()
