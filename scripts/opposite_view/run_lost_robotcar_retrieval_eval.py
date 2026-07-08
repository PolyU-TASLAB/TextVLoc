#!/usr/bin/env python3
"""Run global retrieval on prepared LoST-style RobotCar opposite-view data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

from hloc import extract_features


TOPKS = [1, 5, 10, 20]
THRESHOLDS = [5.0, 10.0, 20.0, 40.0, 80.0]


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_poses(path: Path) -> dict[str, np.ndarray]:
    poses = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            poses[row["image_name"]] = np.array(
                [float(row["easting"]), float(row["northing"])],
                dtype=np.float64,
            )
    return poses


def extract_global_features(
    conf_name: str,
    image_root: Path,
    names: list[str],
    feature_path: Path,
    overwrite: bool,
) -> Path:
    conf = extract_features.confs[conf_name]
    return extract_features.main(
        conf,
        image_root,
        feature_path=feature_path,
        image_list=names,
        as_half=True,
        overwrite=overwrite,
    )


def load_descriptors(feature_path: Path, names: list[str]) -> np.ndarray:
    descs = []
    with h5py.File(feature_path, "r") as h5:
        missing = [name for name in names if name not in h5]
        if missing:
            raise KeyError(f"{feature_path} missing {len(missing)} descriptors, first={missing[0]}")
        for name in names:
            desc = np.asarray(h5[name]["global_descriptor"], dtype=np.float32).reshape(-1)
            descs.append(desc)
    mat = np.stack(descs, axis=0)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-12)
    return mat


def evaluate(
    method: str,
    query_names: list[str],
    query_desc: np.ndarray,
    db_names: list[str],
    db_desc: np.ndarray,
    poses: dict[str, np.ndarray],
    out_dir: Path,
) -> list[dict[str, float | int | str]]:
    scores = query_desc @ db_desc.T
    max_k = max(TOPKS)
    order = np.argsort(-scores, axis=1)[:, :max_k]

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / f"pairs-query-{method}{max_k}.txt"
    with pairs_path.open("w", encoding="utf-8") as f:
        for qi, query in enumerate(query_names):
            for db_idx in order[qi]:
                f.write(f"{query} {db_names[int(db_idx)]}\n")

    rows = []
    for topk in TOPKS:
        best_distances = []
        missing = 0
        for qi, query in enumerate(query_names):
            qpos = poses.get(query)
            if qpos is None:
                missing += 1
                best_distances.append(float("inf"))
                continue
            dists = []
            for db_idx in order[qi, :topk]:
                db = db_names[int(db_idx)]
                dpos = poses.get(db)
                if dpos is not None:
                    dists.append(float(np.linalg.norm(qpos - dpos)))
            if dists:
                best_distances.append(min(dists))
            else:
                missing += 1
                best_distances.append(float("inf"))
        arr = np.asarray(best_distances, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        row: dict[str, float | int | str] = {
            "method": method,
            "K": topk,
            "queries": len(query_names),
            "database": len(db_names),
            "missing_queries": missing,
            "mean_best_distance_m": float(np.mean(finite)) if len(finite) else float("inf"),
            "pairs": str(pairs_path),
        }
        for threshold in THRESHOLDS:
            row[f"R@{int(threshold)}m"] = float(np.mean(arr <= threshold) * 100.0)
        rows.append(row)
    return rows


def write_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", type=Path, default=Path("datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/SensorKit_hloc/lost_robotcar_overcast_autumn_2m_2km"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["netvlad", "megaloc"],
        choices=["netvlad", "megaloc", "cosplace", "eigenplaces", "dir", "openibl"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_root = args.prepared_dir / "images"
    query_names = read_list(args.prepared_dir / "queries_front.txt")
    db_names = read_list(args.prepared_dir / "database_rear.txt")
    poses = load_poses(args.prepared_dir / "poses.csv")

    all_rows = []
    for method in args.methods:
        features_dir = args.out_dir / "features" / method
        query_feature_path = features_dir / f"{method}_front.h5"
        db_feature_path = features_dir / f"{method}_rear.h5"
        extract_global_features(method, image_root, query_names, query_feature_path, args.overwrite)
        extract_global_features(method, image_root, db_names, db_feature_path, args.overwrite)
        query_desc = load_descriptors(query_feature_path, query_names)
        db_desc = load_descriptors(db_feature_path, db_names)
        rows = evaluate(method, query_names, query_desc, db_names, db_desc, poses, args.out_dir / "results")
        all_rows.extend(rows)
        for row in rows:
            if row["K"] in {5, 10, 20}:
                print(
                    f"{method} K={row['K']} "
                    f"R5={row['R@5m']:.2f} R10={row['R@10m']:.2f} R20={row['R@20m']:.2f} "
                    f"R40={row['R@40m']:.2f} R80={row['R@80m']:.2f} "
                    f"mean={row['mean_best_distance_m']:.2f}"
                )

    write_summary(args.out_dir / "summary.csv", all_rows)
    print(f"summary={args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
