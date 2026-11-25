"""Probe texture correspondence and compare the three inversion conditions."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from scipy.stats import spearmanr
import torch

from benchmark import roc, test_rows
from core import PROTOCOL
from pipeline import Pipeline


def texture(image):
    array = np.asarray(image, dtype=np.float64)
    width = PROTOCOL["local_window"]
    w = (1, width, width) if array.ndim == 3 else width
    mean = uniform_filter(array, w, mode="reflect")
    result = np.sqrt(np.maximum(uniform_filter(array*array, w, mode="reflect")-mean*mean, 0))
    return result.mean(axis=0) if array.ndim == 3 else result


def compare(a,b):
    x,y = a.ravel(),b.ravel()
    k = len(x)//4
    peak = np.argsort(x, kind="stable")[-k:]
    ref = np.argsort(y, kind="stable")[-k:]
    return float(spearmanr(x,y).statistic), float(np.intersect1d(peak,ref).size/k)


def probe(pipe, rows, steps):
    stats = {step:[] for step in steps}
    shuffled = {step:[] for step in steps}
    for row in rows:
        seed, prompt = int(row["seed"]), row["caption"]
        z = pipe.noise(seed).to(pipe.device, pipe.dtype)
        image = pipe.generate(prompt, seed, "clean").convert("L").resize((64,64), Image.Resampling.BOX)
        target = texture(image)
        rng = np.random.default_rng(seed)
        for step in steps:
            source = texture(pipe.prepass(z,prompt,step)[0].float().cpu().numpy())
            stats[step].append(compare(source,target))
            shuffled[step].append(compare(rng.permutation(source.ravel()).reshape(64,64),target))
    return {step:{"mean_spearman_and_top_quartile_overlap": np.mean(stats[step],axis=0).tolist(),
                  "shuffled_control":np.mean(shuffled[step],axis=0).tolist()} for step in steps}


def inversion(pipe, rows, score):
    scores = {x:([],[]) for x in ("direct","true_prompt","empty_prompt")}
    errors = {"true_prompt":[], "empty_prompt":[]}
    for row in rows:
        seed, prompt = int(row["seed"]), row["caption"]
        z = pipe.noise(seed).to(pipe.device,pipe.dtype)
        feature = pipe.prepass(z,prompt)
        base = z[0].float().cpu().numpy()
        perturbation = pipe.core.modulation(base,feature[0].float().cpu().numpy())
        actual = (z.float()+torch.as_tensor(perturbation[None],device=pipe.device,dtype=torch.float32)).to(pipe.dtype)
        clean = pipe.generate(prompt,seed,"clean")
        marked = pipe.generate(prompt,seed,"full")
        for label,(image,initial) in enumerate(zip((clean,marked),(z,actual))):
            direct = initial[0].float().cpu().numpy()
            scores["direct"][label].append(pipe.core.score(direct,score))
            for name,condition in (("true_prompt",prompt),("empty_prompt","")):
                rec = pipe.recover(image,condition)
                scores[name][label].append(pipe.core.score(rec,score))
                if label==1:
                    errors[name].append(float(np.sqrt(np.mean((rec-direct)**2))))
    return {k:{**roc(*v),**({"mean_marked_rmse":float(np.mean(errors[k]))} if k in errors else {})} for k,v in scores.items()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    sub=ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("texture").add_argument("--steps", nargs="+", type=int, required=True)
    sub.add_parser("inversion").add_argument("--score", choices=("pearson","negative_l1"), required=True)
    a=ap.parse_args()
    rows=test_rows(a.manifest,a.limit)
    pipe=Pipeline(a.checkpoint,a.device)
    if a.mode=="texture":
        value=probe(pipe,rows,a.steps)
    else:
        value=inversion(pipe,rows,a.score)
    print(json.dumps(dict(pairs=len(rows),diagnostic=a.mode,results=value),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
