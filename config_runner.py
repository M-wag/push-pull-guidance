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
from imgviz import compose_images, compose_images_batched, draw_title, pad_top

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
    num_images = 5
    paths = {
        "config"    : "configs/config_runner.py",
        "templates" : "data/images/examples",
        "out"       : "data/images/config_runner"
    }
    runner = ExperimentRunner(paths, num_images=num_images)

    # Main interactive loop
    while True:
        try:
            if dist.get_rank() == 0:
                user_input = input(">>> ").strip().lower()
            if user_input == 'quit':
                break
                
            # Reload configuration and run
            runner.set_config(paths["config"])
            image_iter = runner.generate_images(net=net, encoder=encoder)

            # Get paths from all batches, not just last
            results = []
            for r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
                results.append(r)
            torch.distributed.barrier()

            # Only run composition on rank 0
            if dist.get_rank() == 0:
                # Get path from unedited images
                path_original = [f"data/images/uncond-32steps-2ndorder/000000/{i:06d}.png" for i in range(0, num_images)]
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
                compositions = compose_images_batched([edited], "h")
                compositions = compose_images(compositions, "v")
                compositions.save(os.path.join(paths["out"], f"out.png"))  
                

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
