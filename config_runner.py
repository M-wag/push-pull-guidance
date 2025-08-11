""" Read the configs from a JSON and run images without reloading PyTorch models"""

import pickle
import importlib
import traceback
import torch
import dnnlib
import generate
import tqdm  
from torch_utils import distributed as dist
from mylib.helpers import update_EDM

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
            
            # Generate images
            if dist.get_rank() == 0:
                dist.print0("Generating images...")
            
            image_iter = generate.generate_images(
                net,
                encoder=encoder,
                gvf_args=myconfig.gvf_args,
                outdir=outdir,
                subdirs=True,
                seeds=seeds,
                verbose=(dist.get_rank() == 0),
                device=device,
                template_dir=template_dir,
                sampler_kwargs=myconfig.sampler_args,
            )

            for _ in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
                pass

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
