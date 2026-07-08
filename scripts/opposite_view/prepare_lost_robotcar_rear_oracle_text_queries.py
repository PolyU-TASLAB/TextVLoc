#!/usr/bin/env python3
"""Build CLIP-RearOracleText queries for LoST-style RobotCar data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_poses(path: Path) -> dict[str, dict[str, float | int | str]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["image_name"]] = {
                "timestamp": int(row["image_timestamp"]),
                "easting": float(row["easting"]),
                "northing": float(row["northing"]),
            }
    return rows


def load_captions(path: Path) -> dict[str, str]:
    captions: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            desc = (row.get("description") or "").strip()
            key = (row.get("image_name") or row.get("image_path") or "").strip()
            if not key or not desc:
                continue
            p = Path(key)
            candidates = {
                key,
                p.name,
                p.stem,
                f"mono_rear/{p.name}",
            }
            for candidate in candidates:
                captions[candidate] = desc
    return captions


def xy(row: dict[str, float | int | str]) -> np.ndarray:
    return np.array([float(row["easting"]), float(row["northing"])], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km_rectified"),
    )
    parser.add_argument("--rear-captions-csv", type=Path, required=True)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km_rectified/clip_rear_oracle_text_queries.csv"),
    )
    parser.add_argument("--max-oracle-distance-m", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_names = read_list(args.prepared_dir / "queries_front.txt")
    db_names = read_list(args.prepared_dir / "database_rear.txt")
    poses = load_poses(args.prepared_dir / "poses.csv")
    captions = load_captions(args.rear_captions_csv)

    db_xy = np.stack([xy(poses[name]) for name in db_names], axis=0)
    db_ts = np.asarray([int(poses[name]["timestamp"]) for name in db_names], dtype=np.int64)

    rows = []
    missing = 0
    too_far = 0
    for query in query_names:
        q_pose = poses[query]
        dists = np.linalg.norm(db_xy - xy(q_pose), axis=1)
        best_idx = int(np.argmin(dists))
        rear_name = db_names[best_idx]
        desc = captions.get(rear_name) or captions.get(Path(rear_name).name) or captions.get(Path(rear_name).stem)
        if not desc:
            missing += 1
            continue
        oracle_distance = float(dists[best_idx])
        if oracle_distance > args.max_oracle_distance_m:
            too_far += 1
        rows.append(
            {
                "image_name": query,
                "description": desc,
                "oracle_rear_image": rear_name,
                "oracle_distance_m": f"{oracle_distance:.3f}",
                "oracle_time_diff_ms": f"{abs(int(q_pose['timestamp']) - int(db_ts[best_idx])) / 1000.0:.3f}",
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "description",
                "oracle_rear_image",
                "oracle_distance_m",
                "oracle_time_diff_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"queries={len(query_names)} database_rear={len(db_names)}")
    print(f"captions={len(captions)} rows={len(rows)} missing_captions={missing}")
    print(f"oracle_distance_gt_{args.max_oracle_distance_m:g}m={too_far}")
    print(f"out_csv={args.out_csv}")


if __name__ == "__main__":
    main()
