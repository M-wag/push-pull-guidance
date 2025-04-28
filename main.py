import matplotlib.pyplot as plt
import os 
import torch
import pickle
from einops import rearrange
from dataclasses import replace
from mylib.diffusion import schedule_diffusion, ConfigSimulation, ConfigDiffusion, ConfigGuidanceVF


MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
CNFG_NO_GUIDANCE = ConfigSimulation(
        network_pkl     = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl',
        device          = "cuda" if torch.cuda.is_available() else "cpu",
        seed            = 0,
        input_shape     = (3, 64, 64),
        guidance_vf     = None,
        diffusion       = ConfigDiffusion(num_steps=32),
        )

def run_no_guidance(exp_name, cnfg):
    # Set output destination
    path_exp = os.path.join(os.getcwd(), "data", "output", exp_name)
    if os.path.exists(path_exp):
        i = 1
        while True:
            path_exp = os.path.join(os.getcwd(), "data", "output", f"{exp_name}_{i}")
            if not os.path.exists(path_exp):
                break
            i += 1
    os.makedirs(path_exp)
    print(f"Making directory {path_exp}")

    # Pass to scheduler
    raw_data = schedule_diffusion(cnfg)

    # Save result
    raw_data_path = os.path.join(path_exp, "raw_data.pkl")
    cnfg_sim_path = os.path.join(path_exp, "cnfg_sim.pkl")

    with open(raw_data_path, "wb") as f:
        pickle.dump(raw_data, f)

    with open(cnfg_sim_path, "wb") as f:
        pickle.dump(cnfg, f)

    return path_exp

if __name__ == "__main__":
    cnfg = CNFG_NO_GUIDANCE(
            seed = 1,
            diffusion=CNFG_NO_GUIDANCE.diffusion(num_steps=32, batch_size=1))
    path_exp =  run_no_guidance("no_guidance", cnfg)

    raw_data_path = os.path.join(path_exp, "raw_data.pkl")
    cnfg_sim_path = os.path.join(path_exp, "cnfg_sim.pkl")
    with open(raw_data_path, "rb") as f:
        raw_data = pickle.load(f)

    data = raw_data[0, 0, -1]
    data_reshaped = rearrange(data, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=1)
    plt.imshow(data_reshaped)
    plt.show()

    
    

