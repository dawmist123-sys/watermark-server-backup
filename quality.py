"""FID, raw CLIP cosine similarity, and paired LPIPS for saved generations."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from benchmark import test_rows


def image_tensor(image, device):
    tensor = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)
    return tensor[None].to(device, torch.float32)/127.5 - 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True, help="benchmark.py output images directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int)
    p.add_argument("--clip-model", required=True, help="OpenCLIP architecture")
    p.add_argument("--clip-weights", type=Path, required=True, help="Local OpenCLIP weights")
    p.add_argument("--lpips-network", choices=("alex", "vgg", "squeeze"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import lpips
    import open_clip
    from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3

    if not a.clip_weights.is_file():
        p.error("Provide a local CLIP weight file")
    rows = test_rows(a.manifest, a.limit)
    if len(rows) < 2:
        p.error("FID requires at least two pairs")
    clean, marked = ([str(a.images/row["image_id"]/name) for row in rows]
                     for name in ("clean.png","marked.png"))
    loss = lpips.LPIPS(net=a.lpips_network).to(a.device).eval()
    clip, _, transform = open_clip.create_model_and_transforms(
        a.clip_model, pretrained=str(a.clip_weights), device=a.device)
    clip.eval()
    tokenizer = open_clip.get_tokenizer(a.clip_model)
    distances, similarities = [], []
    with torch.inference_mode():
        for row, before, after in zip(rows, clean, marked):
            with Image.open(before) as ref, Image.open(after) as result:
                distances.append(float(loss(image_tensor(ref,a.device),image_tensor(result,a.device)).item()))
                x = clip.encode_image(transform(result.convert("RGB"))[None].to(a.device), normalize=True)
                y = clip.encode_text(tokenizer([row["caption"]]).to(a.device), normalize=True)
                similarities.append(float((x*y).sum().item()))
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(a.device).eval()
    stats = [calculate_activation_statistics(paths,inception,batch_size=16,dims=2048,
                                             device=a.device,num_workers=0) for paths in (clean,marked)]
    report = dict(pairs=len(rows), clip_model=a.clip_model, lpips_network=a.lpips_network,
                  fid=float(calculate_frechet_distance(*stats[0],*stats[1])),
                  clip_score=float(np.mean(similarities)), lpips=float(np.mean(distances)))
    a.output.write_text(json.dumps(report,indent=2),encoding="utf8")


if __name__ == "__main__":
    main()
