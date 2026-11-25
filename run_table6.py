"""Generate Table 6 protocol inputs and write measured paired and calibration scores."""
import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path


from attacks import attack
from core import PROTOCOL
from pipeline import Pipeline


CONDITIONS = tuple(PROTOCOL["table6_conditions"])


@contextmanager
def score_file(path, header):
    with path.open("x", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(header)
        yield out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-variant", help="Optional Diffusers weight variant, for example fp16")
    p.add_argument("--smoke-per-split", type=int, help="Limit both splits for an integration check")
    a = p.parse_args()
    with a.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    cal = [row for row in manifest if row["split"] == "calibration"]
    test = [row for row in manifest if row["split"] == "test"]
    if len(manifest) != len(cal) + len(test):
        raise ValueError("Unexpected split in manifest")
    if len({int(row["image_id"]) for row in manifest}) != len(manifest):
        raise ValueError("Expected unique, disjoint image IDs")
    for group, split, count in [(cal, "calibration", PROTOCOL["calibration_negatives"]),
                                (test, "test", PROTOCOL["test_pairs"])]:
        start = PROTOCOL[f"{split}_generation_seed_start"]
        if len(group) != count or [int(row["seed"]) for row in group] != list(range(start, start+count)):
            raise ValueError(f"Unexpected size or seed sequence for {split}")
    if a.smoke_per_split is not None:
        if not 0 < a.smoke_per_split < min(len(cal), len(test)):
            raise ValueError("Invalid smoke size")
        cal, test = cal[:a.smoke_per_split], test[:a.smoke_per_split]
    engine = Pipeline(a.checkpoint, a.device, a.checkpoint_variant)
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = {"scoring_rule": "negative_l1", "smoke": a.smoke_per_split is not None,
                "checkpoint": str(a.checkpoint.resolve()), "device": a.device,
                "checkpoint_variant": engine.checkpoint_variant,
                "calibration_ids": len(cal), "test_ids": len(test), "conditions": CONDITIONS,
                "status": "running"}
    run_file = a.output / "run.json"
    run_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def detect(image):
        return engine.core.score(engine.recover(image), "negative_l1")

    with score_file(a.output/"calibration.csv", ("image_id", "score")) as scores:
        for row in cal:
            image = engine.generate(row["caption"], int(row["seed"]), "clean")
            scores.writerow((row["image_id"], detect(image)))

    with score_file(a.output/"test.csv", ("condition","image_id","watermarked","unwatermarked")) as scores:
        for row in test:
            prompt, seed = row["caption"], int(row["seed"])
            clean = engine.generate(prompt, seed, "clean")
            watermarked = engine.generate(prompt, seed, "full")
            for condition in CONDITIONS:
                scores.writerow((condition, row["image_id"],
                                 detect(attack(watermarked, condition)),
                                 detect(attack(clean, condition))))
    metadata["status"] = "complete"
    run_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
