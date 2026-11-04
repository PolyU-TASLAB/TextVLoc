"""Combine aligned single-frame opposite-view surroundings experiments."""

import argparse
import csv
from pathlib import Path

BASE = Path("outputs/text_depth_ablation_v1/three_datasets_text8_visual12/summary.csv")
OUT = Path("outputs/opposite_view_surroundings_comparison_v1")
RUNS = [
    ("UrbanV2X", "six_view_no_cam1", Path("outputs/opposite_view_urban_493_six_view_no_cam1/summary.csv"), "paired_cam6_queries"),
    ("SYNTHIA dawn→dawn", "four_view", Path("outputs/opposite_view_synthia_four_view_surroundings/summary.csv"), None),
    ("Oxford RobotCar", "four_view_fp16", Path("outputs/opposite_view_robotcar_four_view_surroundings_fp16/summary.csv"), None),
]
METHODS = {"netvlad", "megaloc", "cosplace", "eigenplaces", "TextVLoc_dual_lora"}
FIELDS = ["dataset", "method", "K", "queries", "database", "R@5m", "R@10m", "R@20m"]


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    rows = []
    available = {name for name, _, path, _ in RUNS if path.is_file()}
    for row in read(args.baseline):
        if row["dataset"] in available and row["method"] in METHODS:
            rows.append({key: row[key] for key in FIELDS})
    for dataset, method, path, setting in RUNS:
        if not path.is_file():
            print(f"PENDING {dataset} {method}: {path}")
            continue
        for row in read(path):
            if setting is not None and row["setting"] != setting:
                continue
            rows.append({"dataset": dataset, "method": method,
                         **{key: row[key] for key in FIELDS[2:]}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
