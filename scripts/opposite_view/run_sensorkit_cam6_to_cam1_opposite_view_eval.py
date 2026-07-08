#!/usr/bin/env python3
"""LoST-style opposite-view retrieval on SensorKit cam6 -> cam1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hloc import extract_features
from hloc.extract_clip_features import load_model_and_processor
from sequence_retrieval_utils import sequence_scores, write_pairs


TOPKS = [1, 5, 10, 20]
THRESHOLDS = [5.0, 10.0, 20.0]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def image_names(root: Path, cam: str) -> list[str]:
    return sorted(f"{cam}/{p.name}" for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def timestamp_ns(name: str) -> int:
    return int(Path(name).stem)


def load_poses(paths: list[Path]) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                poses[f"{row['camera_id']}/{Path(row['image_id']).name}"] = np.array(
                    [float(row["X_enu"]), float(row["Y_enu"]), float(row["Z_enu"])],
                    dtype=np.float64,
                )
    return poses


def load_captions(path: Path) -> dict[str, str]:
    captions: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("image_name") or "").strip()
            text = (row.get("description") or "").strip()
            if name and text:
                captions[Path(name).name] = text
                captions[f"cam1/{Path(name).name}"] = text
    return captions


def nearest_cam1_text_rows(
    query_names: list[str],
    cam1_names: list[str],
    captions: dict[str, str],
) -> tuple[list[str], list[dict[str, str | int | float]]]:
    available = [name for name in cam1_names if name in captions]
    if not available:
        raise ValueError("No cam1 captions matched the database image names.")
    cam1_ts = np.asarray([timestamp_ns(name) for name in available], dtype=np.int64)
    texts = []
    rows = []
    for query in query_names:
        qts = timestamp_ns(query)
        idx = int(np.argmin(np.abs(cam1_ts - qts)))
        source = available[idx]
        source_ts = int(cam1_ts[idx])
        time_diff_ns = abs(qts - source_ts)
        text = captions[source]
        texts.append(text)
        rows.append(
            {
                "cam6_query_image": query,
                "cam6_query_timestamp_ns": qts,
                "cam1_text_source_image": source,
                "cam1_text_source_timestamp_ns": source_ts,
                "time_diff_ns": time_diff_ns,
                "time_diff_ms": time_diff_ns / 1e6,
                "description": text,
            }
        )
    return texts, rows


def write_text_source_map(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_global(method: str, image_root: Path, names: list[str], feature_path: Path, overwrite: bool) -> Path:
    conf = extract_features.confs[method]
    return extract_features.main(
        conf,
        image_root,
        feature_path=feature_path,
        image_list=names,
        as_half=True,
        overwrite=overwrite,
    )


def load_global_matrix(path: Path, names: list[str], dataset_name: str = "global_descriptor") -> np.ndarray:
    descs = []
    with h5py.File(path, "r") as h5:
        missing = [name for name in names if name not in h5]
        if missing:
            raise KeyError(f"{path} missing {len(missing)} descriptors, first={missing[0]}")
        for name in names:
            obj = h5[name]
            desc = obj[dataset_name] if isinstance(obj, h5py.Group) else obj
            descs.append(np.asarray(desc, dtype=np.float32).reshape(-1))
    mat = np.stack(descs, axis=0)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-12)
    return mat


def load_clip_image_matrix(path: Path, names: list[str]) -> np.ndarray:
    descs = []
    with h5py.File(path, "r") as h5:
        for name in names:
            descs.append(np.asarray(h5[name], dtype=np.float32).reshape(-1))
    mat = np.stack(descs, axis=0)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-12)
    return mat


def load_text_matrix(path: Path, names: list[str]) -> np.ndarray:
    descs = []
    with h5py.File(path, "r") as h5:
        missing = [name for name in names if name not in h5]
        if missing:
            raise KeyError(f"{path} missing {len(missing)} text descriptors, first={missing[0]}")
        for name in names:
            descs.append(np.asarray(h5[name], dtype=np.float32).reshape(-1))
    mat = np.stack(descs, axis=0)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-12)
    return mat


@torch.no_grad()
def encode_texts(
    names: list[str],
    texts: list[str],
    feature_path: Path,
    *,
    model,
    processor,
    device: str,
    batch_size: int,
    overwrite: bool,
) -> np.ndarray:
    if feature_path.exists() and not overwrite:
        return load_text_matrix(feature_path, names)

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_path.exists():
        feature_path.unlink()

    with h5py.File(feature_path, "w", libver="latest") as h5:
        for start in tqdm(range(0, len(names), batch_size), desc="Encoding T-ReLoc texts"):
            batch_names = names[start : start + batch_size]
            batch_texts = texts[start : start + batch_size]
            tokens = processor(batch_texts).to(device)
            feats = F.normalize(model.encode_text(tokens), dim=-1)
            for name, feat in zip(batch_names, feats.cpu().numpy().astype(np.float32)):
                h5[name] = feat.reshape(-1)
    return load_text_matrix(feature_path, names)


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

    suffix = f"_seq{sequence_length}_stride{center_stride}"
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
            distances = []
            for db_idx in order[qi, :topk]:
                dpos = poses.get(eval_db_names[int(db_idx)])
                if dpos is not None:
                    distances.append(float(np.linalg.norm(qpos - dpos)))
            if distances:
                best_distances.append(min(distances))
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
            "median_best_distance_m": float(np.median(finite)) if len(finite) else float("inf"),
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
    parser.add_argument("--dataset", type=Path, default=Path("datasets/SensorKit"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs/SensorKit_hloc/cam6_to_cam1_opposite_view"))
    parser.add_argument("--existing-feature-root", type=Path, default=Path("outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025"))
    parser.add_argument(
        "--gt",
        type=Path,
        nargs="+",
        default=[
            Path("datasets/SensorKit/0620_gt/enu_all_mapping_gt.csv"),
            Path("datasets/SensorKit/0620_gt/enu_all_query_gt.csv"),
        ],
    )
    parser.add_argument("--captions-csv", type=Path, default=Path("outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/qwen3_query_prompts_qwen-vl_77clip10.csv"))
    parser.add_argument("--clip-db-features", type=Path, default=Path("outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/clip_features_cam1_datacomp_ft.h5"))
    parser.add_argument("--lora-ckpt", type=Path, default=Path("datasets/SensorKit/clipdata/finetune_output/qwen3_clipdata_text_sparse03_remove/lora_epoch_22.pt"))
    parser.add_argument("--methods", nargs="+", default=["netvlad", "megaloc", "cosplace", "eigenplaces"])
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--center-stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_root = args.dataset / "allcams"
    query_names = image_names(image_root / "cam6", "cam6")
    db_names = image_names(image_root / "cam1", "cam1")
    poses = load_poses(args.gt)
    out_dir = args.outputs / f"seq{args.sequence_length}_stride{args.center_stride}"
    feature_dir = out_dir / "features"
    result_dir = out_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    rows_all = []
    existing = {
        "netvlad": args.existing_feature_root / "global-feats-netvlad.h5",
        "megaloc": args.existing_feature_root / "global-feats-megaloc.h5",
    }
    for method in args.methods:
        if method in existing and existing[method].exists() and not args.overwrite:
            feature_path = existing[method]
        else:
            feature_path = feature_dir / method / f"{method}_cam6_cam1.h5"
            extract_global(method, image_root, query_names + db_names, feature_path, args.overwrite)
        query_desc = load_global_matrix(feature_path, query_names)
        db_desc = load_global_matrix(feature_path, db_names)
        rows = evaluate(method, query_names, query_desc, db_names, db_desc, poses, result_dir, args.sequence_length, args.center_stride)
        rows_all.extend(rows)
        for row in rows:
            print(
                f"{method} K={row['K']} R5={row['R@5m']:.2f} R10={row['R@10m']:.2f} "
                f"R20={row['R@20m']:.2f} median={row['median_best_distance_m']:.2f}"
            )

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    captions = load_captions(args.captions_csv)
    texts, text_source_rows = nearest_cam1_text_rows(query_names, db_names, captions)
    text_source_map_path = out_dir / "cam6_query_to_cam1_front_text_qwen-vl.csv"
    write_text_source_map(text_source_map_path, text_source_rows)
    print(f"text_source_map={text_source_map_path}")
    model, processor, _ = load_model_and_processor(lora_ckpt_path=args.lora_ckpt, device=device)
    query_desc = encode_texts(
        query_names,
        texts,
        feature_dir / "treloc" / "cam6_query_front_text_lora22.h5",
        model=model,
        processor=processor,
        device=device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    db_desc = load_clip_image_matrix(args.clip_db_features, db_names)
    rows = evaluate("T-ReLoc", query_names, query_desc, db_names, db_desc, poses, result_dir, args.sequence_length, args.center_stride)
    rows_all.extend(rows)
    for row in rows:
        print(
            f"T-ReLoc K={row['K']} R5={row['R@5m']:.2f} R10={row['R@10m']:.2f} "
            f"R20={row['R@20m']:.2f} median={row['median_best_distance_m']:.2f}"
        )

    summary_path = out_dir / "summary_sensorkit_cam6_to_cam1_opposite_view.csv"
    write_summary(summary_path, rows_all)
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
