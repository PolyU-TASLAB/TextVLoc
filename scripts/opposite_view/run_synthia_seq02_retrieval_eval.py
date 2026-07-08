#!/usr/bin/env python3
"""Run image-based global retrieval on LoST-style SYNTHIA Sequence-02 tests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

from hloc import extract_features
from sequence_retrieval_utils import sequence_scores, write_pairs


TOPKS = [1, 5, 10, 20]
THRESHOLDS = [5.0, 10.0, 20.0]


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
    sequence_length: int,
    center_stride: int,
) -> list[dict[str, float | int | str]]:
    frame_scores = query_desc @ db_desc.T
    scores, query_centers, db_centers = sequence_scores(
        frame_scores,
        sequence_length=sequence_length,
        center_stride=center_stride,
    )
    eval_query_names = [query_names[int(i)] for i in query_centers]
    eval_db_names = [db_names[int(i)] for i in db_centers]
    order = np.argsort(-scores, axis=1)[:, : max(TOPKS)]

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if sequence_length == 1 and center_stride == 1 else f"_seq{sequence_length}_stride{center_stride}"
    pairs_path = out_dir / f"pairs-query-{method}{suffix}20.txt"
    write_pairs(pairs_path, eval_query_names, eval_db_names, order)

    rows = []
    for topk in TOPKS:
        best_distances = []
        missing = 0
        for qi, query in enumerate(eval_query_names):
            qpos = poses.get(query)
            if qpos is None:
                missing += 1
                best_distances.append(float("inf"))
                continue
            dists = []
            for db_idx in order[qi, :topk]:
                dpos = poses.get(eval_db_names[int(db_idx)])
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
            "queries": len(eval_query_names),
            "database": len(eval_db_names),
            "frame_queries": len(query_names),
            "frame_database": len(db_names),
            "sequence_length": sequence_length,
            "center_stride": center_stride,
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
    parser.add_argument("--data-root", type=Path, default=Path("datasets/SYNTHIA/SEQS-02"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("outputs/SYNTHIA/seq02_lost"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/SYNTHIA/seq02_lost_eval"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["netvlad", "megaloc", "cosplace", "eigenplaces"],
        choices=["netvlad", "megaloc", "cosplace", "eigenplaces", "dir", "openibl"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--center-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = args.prepared_dir / args.experiment
    query_names = read_list(exp_dir / "queries.txt")
    db_names = read_list(exp_dir / "database.txt")
    poses = load_poses(args.prepared_dir / "poses.csv")

    all_rows = []
    for method in args.methods:
        features_dir = args.out_dir / args.experiment / "features" / method
        query_feature_path = features_dir / f"{method}_query.h5"
        db_feature_path = features_dir / f"{method}_database.h5"
        extract_global_features(method, args.data_root, query_names, query_feature_path, args.overwrite)
        extract_global_features(method, args.data_root, db_names, db_feature_path, args.overwrite)
        query_desc = load_descriptors(query_feature_path, query_names)
        db_desc = load_descriptors(db_feature_path, db_names)
        rows = evaluate(
            method,
            query_names,
            query_desc,
            db_names,
            db_desc,
            poses,
            args.out_dir / args.experiment / "results",
            args.sequence_length,
            args.center_stride,
        )
        all_rows.extend(rows)
        for row in rows:
            print(
                f"{args.experiment} {method} K={row['K']} "
                f"R5={row['R@5m']:.2f} R10={row['R@10m']:.2f} R20={row['R@20m']:.2f} "
                f"mean={row['mean_best_distance_m']:.2f}"
            )

    summary_path = args.out_dir / args.experiment / "summary_image_vpr.csv"
    write_summary(summary_path, all_rows)
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
