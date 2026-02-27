import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from generate import StableDiffusionDynamics, TextConditionedInputsIterable, DDIMSolver, generate_images

@torch.no_grad()
def main():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    dynamics = StableDiffusionDynamics(unet=pipe.unet, vae=pipe.vae, guidance_scale=7.5)
    solver   = DDIMSolver(scheduler=pipe.scheduler, num_inference_steps=50)
    inputs = TextConditionedInputsIterable(
        seeds           = [42, 42],
        shape           = (4, 64, 64),
        prompts         = ["a cat wearing a hat", "man made of milk"],
        tokenizer       = pipe.tokenizer,
        text_encoder    = pipe.text_encoder,
        device          = "cuda",
    )

    for state in generate_images(solver, dynamics, inputs, verbose=True, dir_out="outputs/"):
        print(state.seeds, state.images.shape)

if __name__ == "__main__":
    main()

