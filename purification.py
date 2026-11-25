"""DiffPure t=50 and t=100 with the resizing control in section 4.2."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import torch

from benchmark import roc, test_rows
from pipeline import Pipeline
from quality import image_tensor


@torch.inference_mode()
def purify(image, step, seed, index, purifier, settings):
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    if step is not None:
        settings.t = step
        torch.manual_seed(seed)
        sample = purifier.image_editing_sample(image_tensor(image,"cuda"),bs_id=index)
        x = ((sample[0].clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy()
        image = Image.fromarray(x)
    return image.resize((512, 512), Image.Resampling.LANCZOS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diffpure-source", type=Path, required=True)
    p.add_argument("--diffpure-weights", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--score", choices=("pearson","negative_l1"), required=True)
    p.add_argument("--limit", type=int, default=100)
    a = p.parse_args()
    sys.path.insert(0, str(a.diffpure_source.resolve()))
    from runners.diffpure_guided import GuidedDiffusion
    import lpips
    import yaml
    config = yaml.safe_load((a.diffpure_source/"configs"/"imagenet.yml").read_text(encoding="utf8"))
    config = SimpleNamespace(**{k:SimpleNamespace(**v) if isinstance(v,dict) else v for k,v in config.items()})
    rows = test_rows(a.manifest,a.limit)
    a.output.mkdir(parents=True,exist_ok=False)
    settings = SimpleNamespace(log_dir=str(a.output/"diffpure"),sample_step=1,t=50)
    Path(settings.log_dir).mkdir()
    purifier = GuidedDiffusion(settings, config, model_dir=str(a.diffpure_weights.resolve()))
    engine = Pipeline(a.checkpoint,"cuda")
    loss = lpips.LPIPS(net="vgg").cuda().eval()
    conditions = {"clean":None,"resize256_only":None,"DiffPure_t50":50,"DiffPure_t100":100}
    scores = {k:([],[]) for k in conditions}
    distances = {k:[] for k in conditions}
    with torch.inference_mode():
        for j,row in enumerate(rows):
            seed = int(row["seed"])
            original = [engine.generate(row["caption"],seed,variant) for variant in ("clean","full")]
            for name, step in conditions.items():
                for label,image in enumerate(original):
                    attacked = image if name=="clean" else purify(image,step,seed,j+2,purifier,settings)
                    value = engine.core.score(engine.recover(attacked),a.score)
                    scores[name][label].append(value)
                    if label==1:
                        distances[name].append(float(loss(image_tensor(image,"cuda"),image_tensor(attacked,"cuda")).item()))
    report = {k:{**roc(*pair),"lpips_to_input":sum(distances[k])/len(rows)}
              for k,pair in scores.items()}
    (a.output/"summary.json").write_text(
        json.dumps(dict(test_pairs=len(rows),score_rule=a.score,conditions=report),indent=2),encoding="utf8")


if __name__ == "__main__":
    main()
