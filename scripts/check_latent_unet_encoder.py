import numpy as np
import matplotlib.pyplot as plt
import os 
import torch
import pickle
import dnnlib
from mylib.diffusion import edm_sampler, ConfigSimulation, ConfigSampler, ConfigGuidanceVF, load_templates_batch, create_vf, schedule_diffusion
from training.networks import EDMPrecond
from mylib.gvf import ConfigGVFUnet, ConfigGVFAmbient, BuidlerUNetGVF
from torch_utils import misc
from einops import rearrange

MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'

if __name__ == "__main__":

    cfg = ConfigGVFUnet(
        type_eval = "numdiff",
        idx_skips = (15, ),
        vf_latent = ConfigGVFAmbient()
    )

    device= "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    templates = load_templates_batch(["data/data/cat_1.jpg"], device=device, dtype=dtype) 
    network_pkl   = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl'

    with dnnlib.util.open_url(network_pkl) as f:
        net_old = pickle.load(f)['ema'].to(device)
    net = EDMPrecond(*net_old.init_args, **net_old.init_kwargs).to(device)
    net.model.save_skips = True
    net.eval()
    misc.copy_params_and_buffers(net_old, net, require_all=True)

    sigma = torch.tensor(1e-1).to(device).to(dtype)
    y = net(templates, sigma)
    y  = (y * 127.5 + 128) / 255

    builder = BuidlerUNetGVF(cfg, templates, device=device, dtype=dtype, net=net)
    builder._setup_latents()

    y_test = builder.latent_inv_fn(builder.latent_fn(templates))
    y_test  = (y_test * 127.5 + 128) / 255

    assert torch.all(torch.isclose(y, y_test))
    assert torch.equal(y, y_test)
    import matplotlib.pyplot as plt 
    
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(rearrange(y[-1].detach().cpu(), "C H W -> H W C"))
    axes[1].imshow(rearrange(y_test[-1].detach().cpu(), "C H W -> H W C"))
    plt.show()




    

