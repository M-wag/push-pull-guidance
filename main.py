import os
import torch
import yaml

from diffusers import StableDiffusionPipeline, DDIMScheduler
from ppg.ppg import create_sgdm, create_ppg
from generate import (StableDiffusionDynamics, DDIMSolver, generate_images,
                       NoiseIterable, TextEmbeddingIterable, ExampleImagesIterable, CombinedInputs)


def load_wildti2i(path_dir, n_entries=None):
    with open(os.path.join(path_dir, "wild-ti2i-real.yaml")) as f:
        data = yaml.safe_load(f)
    if n_entries is not None:
        data = data[:n_entries]
    path_imgs = [os.path.join(path_dir, "data", os.path.basename(entry["init_img"])) for entry in data]
    target_prompts = [entry["target_prompts"][0] for entry in data]
    return path_imgs, target_prompts
    
@torch.no_grad()
def main():
    # Setup push pull guidance
    sgdm = create_sgdm(type_gate="quadratic", nu=0.5)
    ppg = create_ppg(vf_inner=sgdm)

    # Setup hf diffusers pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load prompts and images for Wild-TI2I dataset
    paths_example, prompts = load_wildti2i("data/wild-ti2i", n_entries=3)

    # Setup solver and dynamics
    solver   = DDIMSolver(scheduler=pipe.scheduler, num_inference_steps=50)
    dynamics = StableDiffusionDynamics(unet=pipe.unet, vae=pipe.vae, guidance_scale=7.5, scheduler=pipe.scheduler, ppg=ppg)
    # Setup input iterables
    inputs = CombinedInputs(
        NoiseIterable(seeds=range(0, len(prompts)), shape=(4, 64, 64), device="cuda"),
        TextEmbeddingIterable(prompts, pipe.tokenizer, pipe.text_encoder, device="cuda"),
        ExampleImagesIterable(paths_example, dynamics.encoder, device="cuda"),
    )

    # Generate
    for state in generate_images(solver, dynamics, inputs, verbose=True, dir_out="outputs/"):
        print(state.seeds, state.images.shape)

if __name__ == "__main__":
    main()
