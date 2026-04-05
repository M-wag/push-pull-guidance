import os
import torch
import yaml

from diffusers import StableDiffusionPipeline, DDIMScheduler

from diagnostics import DiagnosticsReport
from util import load_images
from ppg.ppg import create_sgdm, create_ppg
from generate import (StableDiffusionDynamics, DDIMSolver, EDMSolver, NoiseScheduleMap,
                       VEDynamicsWrapper, generate_images, ddim_invert,
                       InputsIterable, NoiseIterable, PrecomputedNoiseIterable,
                       TextEmbeddingIterable, ExampleImagesIterable, CombinedInputs)


def load_wildti2i(path_dir, n_entries=None):
    with open(os.path.join(path_dir, "wild-ti2i-real.yaml")) as f:
        data = yaml.safe_load(f)
    if n_entries is not None:
        data = data[:n_entries]
    path_imgs = [os.path.join(path_dir, "data", os.path.basename(entry["init_img"])) for entry in data]
    target_prompts = [entry["target_prompts"][0] for entry in data]
    return path_imgs, target_prompts

def reconstruct_images(encoder, images):
    images = images.to(torch.float32) / 127.5 - 1
    latents = encoder.encode(images).latent_dist.sample()
    recons = encoder.decode(latents).sample
    return (recons.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)
    
if __name__ == "__main__":
    with torch.no_grad():
        # Setup hf diffusers pipeline
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        # Init report 
        report = DiagnosticsReport("SD 1.5 Guidance in Pixel Space")

        # Show inputs of path 
        report.add_header("Inputs")
        paths_example, prompts = load_wildti2i("data/wild-ti2i", n_entries=3)
        examples = load_images(paths_example, rescale=False, dtype=torch.uint8, device="cuda")
        report.add_image_row(paths_example, captions=prompts)

        # Loaded Examples 
        report.add_header("Loaded  Examples")
        report.add_image_row(examples.cpu().detach().numpy())

        #  reconstructed
        report.add_header("Reconstructed Examples")
        reconstructed_examples = reconstruct_images(pipe.vae, examples).cpu().detach().numpy()
        report.add_image_row(reconstructed_examples)

        solver = DDIMSolver(scheduler=pipe.scheduler, num_inference_steps=50)
        dynamics = StableDiffusionDynamics(unet=pipe.unet, vae=pipe.vae, guidance_scale=7.5, scheduler=pipe.scheduler)


        base = InputsIterable(seeds=range(0, len(prompts)), device="cuda")

        def make_inputs():
            return CombinedInputs(
                base,
                NoiseIterable(shape=(4, 64, 64), device="cuda"),
                TextEmbeddingIterable(prompts, pipe.tokenizer, pipe.text_encoder, device="cuda"),
                ExampleImagesIterable(paths_example, dynamics.encoder, device="cuda"),

            )

        def run_and_report(header, inputs_fn=make_inputs):
            report.add_header(header)
            for state in generate_images(solver, dynamics, inputs_fn(), verbose=True):
                pass
            report.add_image_row(state.images.detach().cpu().numpy(), captions=prompts)
            return state.images


            # Precompute DDIM-inverted noise
        example_latents = dynamics.encoder.encode(
            load_images(paths_example, device="cuda", rescale=True))
        text_emb_ext = TextEmbeddingIterable(prompts, pipe.tokenizer, pipe.text_encoder, device="cuda")
        text_embeddings = text_emb_ext._encode(prompts)
        inverted_noise = ddim_invert(example_latents, pipe.unet, pipe.scheduler,
                                     text_embeddings=text_embeddings)

        def make_inputs_ddim_inv():
            return CombinedInputs(
                base,
                PrecomputedNoiseIterable(inverted_noise),
                TextEmbeddingIterable(prompts, pipe.tokenizer, pipe.text_encoder, device="cuda"),
                ExampleImagesIterable(paths_example, dynamics.encoder, device="cuda"),
            )

        # Show DDIM-inverted noise (decoded back to pixel space)
        report.add_header("DDIM-Inverted Noise (decoded)")
        decoded_noise = dynamics.encoder.decode(inverted_noise)
        report.add_image_row(decoded_noise.detach().cpu().numpy(), captions=prompts)


        # No dynamics at all
        dynamics.use_unet = False
        images_no_dyn = run_and_report("No Dynamics")

        # ppg + no-unet
        dynamics.use_unet = False
        dynamics.ppg = create_ppg(vf_inner=create_sgdm(type_gate="quadratic", nu=1e-4), scale=1.0)
        images_ppg_only = run_and_report("PPG without U-Net")

        # no ppg
        dynamics.use_unet = True
        dynamics.ppg = None
        images_sd = run_and_report("SD with No PPG")

        # no ppg (with noise)
        dynamics.use_unet = True
        dynamics.ppg = None
        solver.eta = 1.0
        images_sd = run_and_report("SD with No PPG eta=1.0")
        solver.eta = 0.0

        # SD with DDIM-inverted noise + CFG (should approximately reconstruct)
        dynamics.use_unet = True
        dynamics.ppg = None
        images_ddim_inv = run_and_report("SD with DDIM-Inverted Noise (CFG=7.5)", inputs_fn=make_inputs_ddim_inv)

        # SD with DDIM-inverted noise, no CFG (should reconstruct examples closely)
        dynamics.use_unet = True
        dynamics.ppg = None
        dynamics.guidance_scale = 0.0
        images_ddim_inv_nocfg = run_and_report("SD with DDIM-Inverted Noise (no CFG)", inputs_fn=make_inputs_ddim_inv)
        dynamics.guidance_scale = 7.5  # restore

        report.save("sanity_check.html")
