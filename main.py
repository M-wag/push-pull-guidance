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
from mylib.diffusion import edm_sampler, ConfigSimulation, ConfigSampler, ConfigGuidanceVF, load_templates, create_vf
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
        v_0 = [40, 20, 10],
        template_path = "data/input/cat_1.jpg",
        threshold_weight = 0.1,
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


if __name__ == "__main__":
    cnfg = ConfigSimulation(
        network_pkl   = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl',
        device        = "cuda" if torch.cuda.is_available() else "cpu",
        seed          = 0,
        input_shape   = (3, 64, 64),
        guidance_vf   = VF_PIXEL.split()[0],
        diffusion     = ConfigSampler(num_steps=5, class_idx=282),
    )

    # Create network
    with dnnlib.util.open_url(cnfg.network_pkl) as f:
        net_old = pickle.load(f)['ema']
    net = EDMPrecond(*net_old.init_args, **net_old.init_kwargs).to(cnfg.device)
    misc.copy_params_and_buffers(net_old, net, require_all=True)
    

    for a, b in zip(net.state_dict().keys(), net_old.state_dict().keys()):
        assert a == b
    
    for x in net_old.init_kwargs.items():
        print(x)

    # Create guidance vectorfield
    templates = load_templates(cnfg.guidance_vf.template_path)
    vf = create_vf(cnfg.guidance_vf, templates)

    # Run sampler
    edm_sampler(net_old, vf, seed=None, device=cnfg.device, **cnfg.diffusion.to_dict())
    # print(net.model.saved_skips)

