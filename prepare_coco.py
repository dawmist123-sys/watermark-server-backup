"""Create independent Table 6 COCO manifests from the official caption JSON files."""
import argparse
import csv
import json
from pathlib import Path
import random

from core import PROTOCOL


def captions(path):
    records = {}
    for row in json.loads(path.read_text(encoding="utf-8"))["annotations"]:
        records.setdefault(int(row["image_id"]), row["caption"])
    return records


def sample(data, count, exclude, seed_start, rng, split):
    eligible = sorted(set(data).difference(exclude))
    if len(eligible) < count:
        raise ValueError(f"Not enough eligible IDs for {split}")
    return [(split, image_id, data[image_id], seed_start + i)
            for i, image_id in enumerate(rng.sample(eligible, count))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--exclude-test-ids", type=Path, required=True,
                   help="JSON array of known historical test IDs")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    train, val = captions(a.train), captions(a.validation)
    rng = random.Random(PROTOCOL["sampling_seed"])
    cal = sample(train, PROTOCOL["calibration_negatives"], set(), PROTOCOL["calibration_generation_seed_start"], rng, "calibration")
    test = sample(val, PROTOCOL["test_pairs"], {x[1] for x in cal} |
                  {int(i) for i in json.loads(a.exclude_test_ids.read_text(encoding="utf-8"))},
                  PROTOCOL["test_generation_seed_start"], rng, "test")
    with a.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("split", "image_id", "caption", "seed"))
        writer.writerows(cal + test)


if __name__ == "__main__":
    main()
