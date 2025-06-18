import numpy as np
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os 
import torch
import pickle
from einops import rearrange, repeat
from dataclasses import replace
from mylib.diffusion import edm_sampler, ConfigSimulation, ConfigSampler, ConfigGuidanceVF, load_templates_batch, create_vf, schedule_diffusion
from mylib.visual import visualize_from_path
from training.networks import EDMPrecond
from torch_utils import misc
import math 
import shutil
import dnnlib

MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'

VF_PIXEL= ConfigGuidanceVF(
        type_latent = "pixel",
        decay_rate = 1.0,
        v_0 = [40, 20, 10 , 5],
        scale = 0.1,
        template_path = "data/data/cat_1.jpg",
        )

VF_PIXEL_SCALE_AND_V0 = ConfigGuidanceVF(
        type_latent = "pixel",
        decay_rate = 1.0,
        v_0 = [15, 30],
        scale = [0.1, 0.5],
        template_path = "data/input/cat_1.jpg",
        )

VF_VAE_JVP = ConfigGuidanceVF(
        type_latent = "hf",
        type_eval = "jvp",
        hf_url = "stabilityai/sd-turbo",
        v_0 = [10, 20, 40],
        template_path = "data/data/cat_1.jpg",
        threshold_weight = 0.5,
        )

VF_VAE_NUMDIFF = VF_VAE_JVP(type_eval="numdiff")

VF_LINEAR = ConfigGuidanceVF(
        type_latent = "linear",
        decay_rate = 1.0,
        v_0 = [45, 30, 25, 20, 15],
        scale = 1.0,
        template_path = "data/input/",
        seed_mat = 0,
        n_features = 5,
        dim_feature = 32,
        T = 1.0,
        )

VF_LINEAR_HF = ConfigGuidanceVF(
        type_latent = "hf-linear",
        type_eval = "numdiff",
        hf_url = "stabilityai/sd-turbo",
        decay_rate = 1.0,
        v_0 = [45, 30, 15],
        scale = 1.0,
        template_path = "data/input/",
        seed_mat = 0,
        n_features = 32,
        dim_feature = 64,
        T = 1.0,
        )

VF_UNET = ConfigGuidanceVF(
        type_latent = "unet",
        type_eval = "numdiff",
        v_0 = 40,
        # scale = 0.2,
        # scale = 0.0,
        template_path = "data/data/cat_1.jpg",
        n_skips = 1,
        )

if __name__ == "__main__":
    cnfg = ConfigSimulation(
        network_pkl   = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl',
        device        = "cuda" if torch.cuda.is_available() else "cpu",
        seed          = 0,
        input_shape   = (3, 64, 64),
        # guidance_vf   = VF_VAE_NUMDIFF.split()[0],
        guidance_vf   = VF_UNET.split()[0],
        # guidance_vf   = VF_PIXEL.split()[0],
        diffusion     = ConfigSampler(
            num_steps=32, 
            class_idx=281,
            batch_size=9,
        ),
    )

    # Create network
    with dnnlib.util.open_url(cnfg.network_pkl) as f:
        net_old = pickle.load(f)['ema'].to(cnfg.device)
    net = EDMPrecond(*net_old.init_args, **net_old.init_kwargs).to(cnfg.device)
    net.model.save_skips = True
    net.eval()
    misc.copy_params_and_buffers(net_old, net, require_all=True)
    
    # Create guidance vectorfield
    templates = load_templates_batch([cnfg.guidance_vf.template_path] * cnfg.diffusion.batch_size, device=cnfg.device, dtype=torch.float64)
    vf = create_vf(cnfg.guidance_vf, templates, net=net, cnfg_sim=cnfg)

    # Run sampler
    with torch.no_grad():
        xs, _ = edm_sampler(net, vf, seed=cnfg.seed, device=cnfg.device, **cnfg.diffusion.to_dict())

    plt.imshow(rearrange(xs[-1].detach().numpy(), "(b1 b2) c h w -> (b1 h) (b2 w) c ", b1=int(np.sqrt(cnfg.diffusion.batch_size))))
    plt.show()
               


