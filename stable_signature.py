"""Benchmark a Stable Signature VAE and its matched watermark extractor."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

from attacks import MAIN, EXTRA, attack
from benchmark import roc, test_rows, timed
from pipeline import Pipeline


def bit_agreement(image, extractor, target):
    rgb = np.array(image.convert("RGB"))
    x = torch.from_numpy(rgb).to(target.device, torch.float32).permute(2, 0, 1) / 255
    mean = x.new_tensor([.485, .456, .406])[:, None, None]
    std = x.new_tensor([.229, .224, .225])[:, None, None]
    logits = extractor(((x-mean)/std)[None]).flatten()
    if logits.shape != target.shape or not torch.isfinite(logits).all():
        raise ValueError("The extractor must return one finite logit per key bit")
    return float(((logits > 0) == target).float().mean().item())


def load_marked_vae(clean_vae, path, variant, device, dtype):
    if path.is_dir():
        return AutoencoderKL.from_pretrained(
            str(path), local_files_only=True, torch_dtype=dtype, variant=variant
        ).to(device).eval()

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("ldm_decoder", checkpoint)
    converted = convert_ldm_vae_checkpoint(state, clean_vae.config)
    decoder = {key.removeprefix("decoder."): value for key, value in converted.items()
               if key.startswith("decoder.")}
    marked = deepcopy(clean_vae)
    marked.decoder.load_state_dict(decoder, strict=True)
    if marked.post_quant_conv is not None and "post_quant_conv.weight" in converted:
        post_quant = {key.removeprefix("post_quant_conv."): value
                      for key, value in converted.items() if key.startswith("post_quant_conv.")}
        marked.post_quant_conv.load_state_dict(post_quant, strict=True)
    return marked.eval()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--marked-vae", type=Path, required=True)
    p.add_argument("--vae-variant")
    p.add_argument("--extractor", type=Path, required=True)
    p.add_argument("--key", required=True, help="Binary key matched to the VAE")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", choices=MAIN+EXTRA, default=MAIN)
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if len(a.key) != 48 or set(a.key) - {"0", "1"}:
        p.error("Stable Signature requires a 48-bit binary key")
    rows = test_rows(a.manifest, a.limit)
    pipe = Pipeline(a.checkpoint, a.device)
    target = torch.tensor([bit == "1" for bit in a.key], device=pipe.device)
    extractor = torch.jit.load(str(a.extractor), map_location=pipe.device).eval()
    clean_vae = pipe.pipe.vae
    marked_vae = load_marked_vae(clean_vae, a.marked_vae, a.vae_variant, pipe.device, pipe.dtype)
    for key in ("latent_channels", "block_out_channels", "scaling_factor"):
        if getattr(clean_vae.config, key) != getattr(marked_vae.config, key):
            raise ValueError(f"Marked VAE differs from the base VAE in {key}")
    a.output.mkdir(parents=True, exist_ok=False)
    scores = {name: ([], []) for name in a.conditions}
    generation, detection = [], []
    with (a.output/"scores.jsonl").open("x", encoding="utf8") as stream:
        for row in rows:
            seed, prompt = int(row["seed"]), row["caption"]
            clean = pipe.generate(prompt, seed, "clean")
            pipe.pipe.vae = marked_vae
            try:
                marked, seconds = timed(pipe, pipe.generate, prompt, seed, "clean")
            finally:
                pipe.pipe.vae = clean_vae
            generation.append(seconds)
            images = a.output/"images"/row["image_id"]
            images.mkdir(parents=True)
            clean.save(images/"clean.png")
            marked.save(images/"marked.png")
            for name, pair in scores.items():
                for label, source in enumerate((clean, marked)):
                    value, seconds = timed(pipe, bit_agreement, attack(source, name, seed), extractor, target)
                    pair[label].append(value)
                    if name == "clean" and label == 1:
                        detection.append(seconds)
                    stream.write(json.dumps(dict(image_id=int(row["image_id"]), condition=name,
                                                 label=label, score=value))+"\n")
    report = dict(pairs=len(rows), checkpoint=str(a.checkpoint), marked_vae=str(a.marked_vae),
                  extractor=str(a.extractor), conditions={name:roc(*pair) for name,pair in scores.items()})
    for name, values in (("generation",generation), ("detection",detection)):
        if values:
            report[name+"_seconds"] = dict(mean=float(np.mean(values)),
                                            sd=float(np.std(values, ddof=1)) if len(values)>1 else None)
    (a.output/"summary.json").write_text(json.dumps(report,indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
