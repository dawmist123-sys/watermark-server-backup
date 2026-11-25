"""Stable Diffusion generation and image-to-latent detection for HVS-GM."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline

from core import PROTOCOL, Watermark

VARIANTS = ("full", "no_fixed_mask", "no_window", "no_mcr", "no_mcre",
            "fixed_1_75", "fixed_equal_energy")


class Pipeline:
    def __init__(self, checkpoint, device, checkpoint_variant=None, spec=PROTOCOL):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use --device cpu for small integration tests.")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if PROTOCOL["inversion_guidance"] != 1.0:
            raise ValueError("This implementation uses one empty-prompt branch (guidance 1)")
        checkpoint = Path(checkpoint)
        if checkpoint_variant is None and (checkpoint/"unet"/"diffusion_pytorch_model.fp16.safetensors").is_file() and not (checkpoint/"unet"/"diffusion_pytorch_model.safetensors").is_file():
            checkpoint_variant = "fp16"
        self.checkpoint_variant = checkpoint_variant
        self.pipe = StableDiffusionPipeline.from_pretrained(
            str(checkpoint), torch_dtype=self.dtype, local_files_only=True,
            safety_checker=None, requires_safety_checker=False,
            variant=checkpoint_variant).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        inverse_config = dict(self.pipe.scheduler.config)
        inverse_config.pop("skip_prk_steps", None)  # PNDM setting carried by some checkpoints
        self.inverse = DDIMInverseScheduler.from_config(inverse_config)
        self.pipe.set_progress_bar_config(disable=True)
        self.empty = self.text("")
        self.core = Watermark(self.noise(PROTOCOL["key_seed"]).float().cpu().numpy()[0], spec)

    def noise(self, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn((1, *PROTOCOL["latent_shape"]), generator=generator)

    @torch.inference_mode()
    def text(self, prompt):
        ids = self.pipe.tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt").input_ids
        return self.pipe.text_encoder(ids.to(self.device))[0]

    @torch.inference_mode()
    def prepass(self, z, prompt, updates=None):
        updates = PROTOCOL["preliminary_updates"] if updates is None else updates
        if not 1 <= updates <= PROTOCOL["denoising_steps"]:
            raise ValueError("Prepass updates must fall within the DDIM schedule")
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(PROTOCOL["denoising_steps"], device=self.device)
        feature = z.clone() * scheduler.init_noise_sigma
        text = torch.cat((self.empty, self.text(prompt)))
        for t in scheduler.timesteps[:updates]:
            inputs = scheduler.scale_model_input(torch.cat((feature, feature)), t)
            uncond, cond = self.pipe.unet(inputs, t, encoder_hidden_states=text).sample.chunk(2)
            prediction = uncond + PROTOCOL["generation_guidance"] * (cond - uncond)
            feature = scheduler.step(prediction, t, feature, eta=0).prev_sample
        return feature

    @torch.inference_mode()
    def generate(self, prompt, seed, variant="full"):
        initial = self.noise(seed).to(self.device, self.dtype)
        if variant != "clean":
            feature = self.prepass(initial, prompt)
            perturbation = self.core.modulation(
                initial[0].float().cpu().numpy(), feature[0].float().cpu().numpy(), variant)
            initial = (initial.float() + torch.as_tensor(
                perturbation[None], dtype=torch.float32, device=self.device)).to(self.dtype)
        return self.pipe(prompt, latents=initial,
                         num_inference_steps=PROTOCOL["denoising_steps"],
                         guidance_scale=PROTOCOL["generation_guidance"],
                         height=512, width=512, eta=0).images[0]

    @torch.inference_mode()
    def recover(self, image, prompt=""):
        image = image.convert("RGB")
        if image.size != (512, 512):
            raise ValueError("Provide a 512 × 512 image. Apply any test resizing explicitly.")
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0)
        pixels = pixels.to(self.device, self.dtype) / 127.5 - 1
        initial = self.pipe.vae.encode(pixels).latent_dist.mode() * self.pipe.vae.config.scaling_factor
        self.inverse.set_timesteps(PROTOCOL["denoising_steps"], device=self.device)
        text = self.empty if not prompt else self.text(prompt)
        for t in self.inverse.timesteps:
            model_input = self.inverse.scale_model_input(initial, t)
            noise = self.pipe.unet(model_input, t, encoder_hidden_states=text).sample
            initial = self.inverse.step(noise, t, initial).prev_sample
        return initial[0].float().cpu().numpy()

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-variant", help="Optional Diffusers weight variant, for example fp16")
    mode = parser.add_subparsers(dest="mode", required=True)
    gen = mode.add_parser("generate")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--variant", choices=(*VARIANTS, "clean"), default="full")
    gen.add_argument("--output", type=Path, required=True)
    det = mode.add_parser("detect")
    det.add_argument("image", type=Path)
    det.add_argument("--score", choices=("pearson", "negative_l1"), default="pearson")
    det.add_argument("--variant", choices=VARIANTS, default="full")
    args = parser.parse_args()
    pipe = Pipeline(args.checkpoint, args.device, args.checkpoint_variant)
    pipe.sync()
    start = time.perf_counter()
    if args.mode == "generate":
        image = pipe.generate(args.prompt, args.seed, args.variant)
        pipe.sync()
        seconds = time.perf_counter() - start
        image.save(args.output)
        result = {"generation_seconds": seconds, "seed": args.seed, "variant": args.variant}
    else:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        pipe.sync()
        start = time.perf_counter()
        score = pipe.core.score(pipe.recover(image), args.score, args.variant)
        pipe.sync()
        result = {"score": score, "scoring_rule": args.score, "detection_seconds": time.perf_counter() - start}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
