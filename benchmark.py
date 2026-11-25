"""HVS-GM benchmark for geometric edits, component ablations and timing."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from attacks import EXTRA, MAIN, attack
from core import PROTOCOL
from pipeline import Pipeline, VARIANTS


def roc(negative, positive):
    if not negative or len(negative) != len(positive):
        raise ValueError("Use matched nonempty negative and positive scores")
    score = np.asarray(negative + positive, dtype=float)
    if not np.isfinite(score).all():
        raise ValueError("Non-finite scores")
    label = np.repeat([0, 1], len(negative))
    fp, tp, _ = roc_curve(label, score, drop_intermediate=False)
    return {"auc": float(roc_auc_score(label, score)),
            "tpr_at_fpr_le_1pct": float(tp[fp <= .01].max())}


def test_rows(manifest, limit):
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = [x for x in csv.DictReader(f) if x.get("split", "test") == "test"]
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    rows = rows[:limit]
    if not rows or len({int(x["image_id"]) for x in rows}) != len(rows):
        raise ValueError("Test rows must have unique image IDs")
    return rows


def timed(pipe, operation, *args):
    pipe.sync()
    start = time.perf_counter()
    value = operation(*args)
    pipe.sync()
    return value, time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variant", choices=VARIANTS, default="full")
    ap.add_argument("--score", choices=("pearson", "negative_l1", "negative_complex_l1"), required=True)
    ap.add_argument("--conditions", nargs="+", choices=MAIN+EXTRA, default=MAIN)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--sigma-mask", type=float)
    ap.add_argument("--sigma-window", type=float)
    ap.add_argument("--sigmoid-coefficient", type=float)
    ap.add_argument("--strength-range", nargs=2, type=float)
    a = ap.parse_args()
    if len(set(a.conditions)) != len(a.conditions):
        ap.error("Attack conditions must be unique")
    rows = test_rows(a.manifest, a.limit)
    spec = dict(PROTOCOL)
    for key, value in (("sigma_M", a.sigma_mask), ("sigma_w", a.sigma_window),
                       ("sigmoid_coefficient", a.sigmoid_coefficient)):
        if value is not None:
            if not np.isfinite(value) or value <= 0:
                ap.error(f"{key} must be positive and finite")
            spec[key] = value
    if a.strength_range is not None:
        lo, hi = a.strength_range
        if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
            ap.error("Use finite, nonnegative strength bounds in increasing order")
        spec.update(strength_min=lo, strength_max=hi)
    pipe = Pipeline(a.checkpoint, a.device, spec=spec)
    a.output.mkdir(parents=True, exist_ok=False)
    scores = {name: ([], []) for name in a.conditions}
    timings = []
    with (a.output/"scores.csv").open("x", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(("image_id","condition","unwatermarked","watermarked"))
        for row in rows:
            image_id, seed, prompt = int(row["image_id"]), int(row["seed"]), row["caption"]
            clean = pipe.generate(prompt, seed, "clean")
            marked, generation = timed(pipe, pipe.generate, prompt, seed, a.variant)
            if a.save_images:
                folder = a.output/"images"/str(image_id)
                folder.mkdir(parents=True)
                clean.save(folder/"clean.png")
                marked.save(folder/"marked.png")
            for name, (negative, positive) in scores.items():
                z_clean = pipe.recover(attack(clean, name, seed))
                marked_input = attack(marked, name, seed)
                def score(image):
                    return pipe.core.score(pipe.recover(image), a.score, a.variant)
                yes, detection = timed(pipe, score, marked_input)
                no = pipe.core.score(z_clean, a.score, a.variant)
                writer.writerow((image_id, name, no, yes))
                negative.append(no)
                positive.append(yes)
                if name == "clean":
                    timings.append((generation, detection))
    result = {"count":len(rows), "variant":a.variant, "score_rule":a.score, "protocol":spec,
              "conditions":{name:roc(*pair) for name,pair in scores.items()}}
    if timings:
        times = np.asarray(timings)
        result["timing_seconds"] = {
            label:{"mean":float(value.mean()),"std":float(value.std(ddof=1)) if len(value)>1 else None}
            for label,value in zip(("generation","detection"),times.T)
        }
    (a.output/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
