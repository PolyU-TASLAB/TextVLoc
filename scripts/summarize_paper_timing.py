"""Aggregate local per-query measurements into a timing summary."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    stages = ("global_extraction", "retrieval", "coarse", "local_extraction",
              "matching_and_correspondence", "pose_estimation", "end_to_end")
    methods = ("text_only", "text_image", "hloc")
    summary, table = {}, []
    for method in methods:
        records = [r for r in rows if r["method"] == method and r["warmup"].lower() == "false"]
        if not records or len({r["query"] for r in records}) != len(records):
            raise ValueError(f"Missing or duplicate measured queries for {method}")
        summary[method] = {}
        for stage in stages:
            if not records[0].get(stage):
                continue
            values = np.array([float(r[stage]) for r in records])
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid timings: {method}/{stage}")
            summary[method][stage] = {"mean_ms": float(values.mean()),
                                     "median_ms": float(np.median(values)),
                                     "p95_ms": float(np.percentile(values, 95)), "n": len(values)}
        table.append({"method": method, "queries": len(records),
                      **{stage: round(values["mean_ms"], 2) for stage, values in summary[method].items()},
                      "throughput_hz": round(1000/summary[method]["end_to_end"]["mean_ms"], 2)})
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
