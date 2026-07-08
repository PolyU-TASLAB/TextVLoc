#!/usr/bin/env python3
"""Summarize SYNTHIA Sequence-02 opposite-view retrieval results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "experiment",
    "method",
    "sequence_length",
    "center_stride",
    "Top1 R5/R10/R20",
    "Top5 R5/R10/R20",
    "Top10 R5/R10/R20",
    "Top20 R5/R10/R20",
    "Top20 mean",
]


def read_summary(path: Path, method_alias: str | None = None) -> dict[str, dict[int, dict[str, float]]]:
    by_method: dict[str, dict[int, dict[str, float]]] = {}
    if not path.exists():
        return by_method
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = method_alias or row["method"]
            k = int(row["K"])
            by_method.setdefault(method, {})[k] = {
                "R5": float(row.get("R@5m", row.get("R5", 0.0))),
                "R10": float(row.get("R@10m", row.get("R10", 0.0))),
                "R20": float(row.get("R@20m", row.get("R20", 0.0))),
                "mean": float(row["mean_best_distance_m"]),
                "sequence_length": row.get("sequence_length", "1"),
                "center_stride": row.get("center_stride", "1"),
            }
    return by_method


def format_method(experiment: str, method: str, ks: dict[int, dict[str, float]]) -> dict[str, str]:
    row = {
        "experiment": experiment,
        "method": method,
        "sequence_length": str(ks[1].get("sequence_length", "1")),
        "center_stride": str(ks[1].get("center_stride", "1")),
    }
    for k in [1, 5, 10, 20]:
        metrics = ks[k]
        row[f"Top{k} R5/R10/R20"] = "{:.2f}/{:.2f}/{:.2f}".format(
            metrics["R5"], metrics["R10"], metrics["R20"]
        )
    row["Top20 mean"] = "{:.2f}m".format(ks[20]["mean"])
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/SYNTHIA/seq02_lost_eval"))
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["dawn_back_to_dawn_front", "fall_back_to_fall_front", "fall_back_to_dawn_front"],
    )
    parser.add_argument("--clip-method", default="clip_front_oracle_text_no_robotcar_epoch8")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for experiment in args.experiments:
        image_summary = args.eval_dir / experiment / "summary_image_vpr.csv"
        methods = read_summary(image_summary)
        clip_summary = args.eval_dir / experiment / args.clip_method / "summary.csv"
        methods.update(read_summary(clip_summary, method_alias="CLIP-FrontOracleText"))
        for method, ks in methods.items():
            if all(k in ks for k in [1, 5, 10, 20]):
                rows.append(format_method(experiment, method, ks))

    output_csv = args.output_csv or (args.eval_dir / "summary_synthia_seq02_lost_compact.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(output_csv)
    print("| " + " | ".join(FIELDS) + " |")
    print("|" + "|".join(["---"] * len(FIELDS)) + "|")
    for row in rows:
        print("| " + " | ".join(row[field] for field in FIELDS) + " |")


if __name__ == "__main__":
    main()
