"""Table 6 analysis from measured scores, not from values copied out of the paper."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from core import PROTOCOL


def evaluate(calibration, watermarked, unwatermarked, resamples, seed):
    if any(a.ndim != 1 for a in (calibration, watermarked, unwatermarked)):
        raise ValueError("Expected one-dimensional score arrays")
    if not calibration.size or watermarked.shape != unwatermarked.shape or not watermarked.size:
        raise ValueError("Calibration negatives and paired test scores are required")
    if not all(np.isfinite(a).all() for a in (calibration, watermarked, unwatermarked)):
        raise ValueError("All scores must be finite")
    threshold = np.quantile(calibration, 0.99, method="higher")
    labels = np.repeat([0, 1], len(watermarked))

    def roc_metrics(positive, negative):
        scores = np.concatenate([negative, positive])
        fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
        return roc_auc_score(labels, scores), float(tpr[fpr <= 0.01].max())

    auc, tpr = roc_metrics(watermarked, unwatermarked)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        index = rng.integers(0, len(watermarked), len(watermarked))
        samples.append(roc_metrics(watermarked[index], unwatermarked[index]))
    ci = np.percentile(samples, [2.5, 97.5], axis=0)
    return dict(auc=auc, roc_tpr_at_fpr_le_1pct=tpr, auc_ci=ci[:, 0].tolist(),
                roc_tpr_ci=ci[:, 1].tolist(), threshold=float(threshold),
                fixed_tpr=float(np.mean(watermarked > threshold)),
                actual_fpr=float(np.mean(unwatermarked > threshold)))


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path, help="CSV: image_id,score")
    parser.add_argument("test", type=Path, help="CSV: condition,image_id,watermarked,unwatermarked")
    parser.add_argument("--smoke", action="store_true", help="Allow small measured score files")
    args = parser.parse_args()
    cal = rows(args.calibration)
    test = rows(args.test)
    cal_ids = {x["image_id"] for x in cal}
    if len(cal_ids) != len(cal) or cal_ids.intersection(x["image_id"] for x in test):
        raise ValueError("Calibration IDs must be unique and disjoint from test IDs")
    conditions = sorted({x["condition"] for x in test})
    if not args.smoke and set(conditions) != set(PROTOCOL["table6_conditions"]):
        raise ValueError("Expected all six conditions from the Table 6 protocol")
    ids = None
    result = {}
    for condition in conditions:
        group = [x for x in test if x["condition"] == condition]
        current_ids = {x["image_id"] for x in group}
        if len(group) != len(current_ids) or ids is not None and current_ids != ids:
            raise ValueError("Every condition must contain the same unique paired IDs")
        ids = current_ids
        if not args.smoke and (len(cal) != PROTOCOL["calibration_negatives"] or len(group) != PROTOCOL["test_pairs"]):
            raise ValueError("Table 6 protocol requires 1,000 negatives and 1,000 pairs")
        group.sort(key=lambda row: row["image_id"])
        result[condition] = evaluate(
            np.array([float(x["score"]) for x in cal]),
            np.array([float(x["watermarked"]) for x in group]),
            np.array([float(x["unwatermarked"]) for x in group]),
            20 if args.smoke else PROTOCOL["bootstrap_resamples"], PROTOCOL["bootstrap_seed"])
    if not result:
        raise ValueError("No test conditions")
    print(json.dumps(dict(smoke=args.smoke, calibration_negatives=len(cal), test_pairs=len(ids),
                          conditions=result), indent=2))


if __name__ == "__main__":
    main()
