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
from mylib.diffusion import schedule_diffusion, ConfigSimulation, ConfigDiffusion, ConfigGuidanceVF, load_templates
from mylib.visual import visualize_from_path
import math 
import shutil

MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'

VF_PIXEL= ConfigGuidanceVF(
        type_latent = "pixel",
        decay_rate = 1.0,
        v_0 = [40, 20, 10 , 5],
        scale_template_score = 1.0,
        template_path = "data/input/cat.jpg",
        )

VF_PIXEL_SCALE_AND_V0 = ConfigGuidanceVF(
        type_latent = "pixel",
        decay_rate = 1.0,
        v_0 = [15, 30],
        scale_template_score = [0.1, 0.5],
        template_path = "data/input/cat.jpg",
        )

VF_VAE_JVP = ConfigGuidanceVF(
        type_latent = "hf",
        type_eval = "jvp",
        hf_url = "stabilityai/sd-turbo",
        decay_rate = 1.0,
        v_0 = [40, 20, 10 , 5],
        scale_template_score = 1.0,
        template_path = "data/input/cat.jpg",
        )

VF_VAE_NUMDIFF = VF_VAE_JVP(type_eval="numdiff")



def run_no_guidance(cnfg, path_exp):
    # Pass to scheduler
    raw_data = schedule_diffusion(cnfg(guidance_vf=None))
    # Save result
    raw_data_path = os.path.join(path_exp, "raw_data_og.pkl")

    with open(raw_data_path, "wb") as f:
        pickle.dump(raw_data, f)

    return path_exp

if __name__ == "__main__":
    exp_name = "run_and_plot_check"

    guidance_configs = [
        VF_PIXEL_SCALE_AND_V0,
        VF_VAE_JVP,
        VF_VAE_NUMDIFF,
    ]

    for i, guidance_vf in enumerate(guidance_configs):
        cnfg_sim = ConfigSimulation(
            network_pkl   = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl',
            device        = "cuda" if torch.cuda.is_available() else "cpu",
            seed          = 0,
            input_shape   = (3, 64, 64),
            guidance_vf   = guidance_vf(v_0=10.0),
            diffusion     = ConfigDiffusion(num_steps=24),
        )

        # Set up the per-experiment directory
        path_exp = os.path.join(os.getcwd(), "data", "output", exp_name, str(i))
        if os.path.exists(path_exp):
            shutil.rmtree(path_exp)      # remove old directory entirely
        os.makedirs(path_exp, exist_ok=True)
        print(f"[{i}] Created directory: {path_exp}")

        # Run the diffusion schedule and save results
        raw_data = schedule_diffusion(cnfg_sim)
        with open(os.path.join(path_exp, "raw_data.pkl"), "wb") as f:
            pickle.dump(raw_data, f)
        with open(os.path.join(path_exp, "cnfg_sim.pkl"), "wb") as f:
            pickle.dump(cnfg_sim, f)

        if i == 0:
            run_no_guidance(cnfg_sim, path_exp)
        else:
            shutil.copy(os.path.join(os.getcwd(), "data", "output", exp_name, str(0), "raw_data_og.pkl"), 
                        os.path.join(os.getcwd(), "data", "output", exp_name, str(i), "raw_data_og.pkl"))

        # Visualize
        visualize_from_path(path_exp, exp_name)
    plt.show()


