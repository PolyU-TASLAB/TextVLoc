#!/usr/bin/env python3
"""Evaluate CLIP text-to-image retrieval on LoST-style RobotCar data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from hloc.extract_clip_features import load_model_and_processor


TOPKS = [1, 5, 10, 20]
THRESHOLDS = [5.0, 10.0, 20.0]


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_poses(path: Path) -> dict[str, np.ndarray]:
    poses = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            poses[row["image_name"]] = np.array([float(row["easting"]), float(row["northing"])], dtype=np.float64)
    return poses


def load_text_queries(path: Path) -> tuple[list[str], list[str]]:
    names = []
    texts = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("image_name") or "").strip()
            text = (row.get("description") or "").strip()
            if name and text:
                names.append(name)
                texts.append(text)
    return names, texts


def load_h5_matrix(path: Path, names: list[str]) -> np.ndarray:
    descs = []
    with h5py.File(path, "r") as h5:
        missing = [name for name in names if name not in h5]
        if missing:
            raise KeyError(f"{path} missing {len(missing)} descriptors, first={missing[0]}")
        for name in names:
            desc = np.asarray(h5[name], dtype=np.float32).reshape(-1)
            descs.append(desc)
    mat = np.stack(descs, axis=0)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-12)
    return mat


@torch.no_grad()
def encode_images(
    names: list[str],
    image_root: Path,
    feature_path: Path,
    *,
    model,
    processor,
    preprocess,
    device: str,
    overwrite: bool,
) -> np.ndarray:
    if feature_path.exists() and not overwrite:
        return load_h5_matrix(feature_path, names)

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_path.exists():
        feature_path.unlink()

    is_openclip = hasattr(model, "encode_image")
    with h5py.File(feature_path, "w", libver="latest") as h5:
        for name in tqdm(names, desc="Encoding CLIP images"):
            image = Image.open(image_root / name).convert("RGB")
            if is_openclip:
                tensor = preprocess(image).unsqueeze(0).to(device)
                feat = F.normalize(model.encode_image(tensor), dim=-1)
            else:
                inputs = processor(images=image, return_tensors="pt").to(device)
                feat = F.normalize(model.get_image_features(**inputs), dim=-1)
            h5[name] = feat.cpu().numpy().astype(np.float32).reshape(-1)
    return load_h5_matrix(feature_path, names)


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
        return load_h5_matrix(feature_path, names)

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_path.exists():
        feature_path.unlink()

    is_openclip = hasattr(model, "encode_text")
    with h5py.File(feature_path, "w", libver="latest") as h5:
        for start in tqdm(range(0, len(names), batch_size), desc="Encoding CLIP texts"):
            batch_names = names[start : start + batch_size]
            batch_texts = texts[start : start + batch_size]
            if is_openclip:
                tokens = processor(batch_texts).to(device)
                feats = F.normalize(model.encode_text(tokens), dim=-1)
            else:
                inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                feats = F.normalize(model.get_text_features(**inputs), dim=-1)
            for name, feat in zip(batch_names, feats.cpu().numpy().astype(np.float32)):
                h5[name] = feat.reshape(-1)
    return load_h5_matrix(feature_path, names)


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
    order = np.argsort(-scores, axis=1)[:, : max(TOPKS)]

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / f"pairs-query-{method}20.txt"
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
            dists = []
            if qpos is None:
                missing += 1
                best_distances.append(float("inf"))
                continue
            for db_idx in order[qi, :topk]:
                dpos = poses.get(db_names[int(db_idx)])
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
            row[f"R{int(threshold)}"] = float(np.mean(arr <= threshold) * 100.0)
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
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km_rectified"),
    )
    parser.add_argument("--query-text-csv", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/SensorKit_hloc/lost_robotcar_overcast_autumn_2m_2km_rectified/clip_rear_oracle_text_no_robotcar_epoch8"),
    )
    parser.add_argument(
        "--lora-ckpt",
        type=Path,
        default=Path("datasets/SensorKit/clipdata/finetune_output/qwen3_clipdata_text_sparse03_remove_no_robotcar/lora_epoch_8.pt"),
    )
    parser.add_argument("--method", default="clip_rear_oracle_text_no_robotcar_epoch8")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    image_root = args.prepared_dir / "images"
    db_names = read_list(args.prepared_dir / "database_rear.txt")
    query_names, query_texts = load_text_queries(args.query_text_csv)
    poses = load_poses(args.prepared_dir / "poses.csv")

    model, processor, preprocess = load_model_and_processor(lora_ckpt_path=args.lora_ckpt, device=device)
    query_desc = encode_texts(
        query_names,
        query_texts,
        args.out_dir / "features" / "query_text.h5",
        model=model,
        processor=processor,
        device=device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    db_desc = encode_images(
        db_names,
        image_root,
        args.out_dir / "features" / "database_rear_image.h5",
        model=model,
        processor=processor,
        preprocess=preprocess,
        device=device,
        overwrite=args.overwrite,
    )

    rows = evaluate(args.method, query_names, query_desc, db_names, db_desc, poses, args.out_dir / "results")
    write_summary(args.out_dir / "summary.csv", rows)
    for row in rows:
        print(
            f"{row['method']} K={row['K']} "
            f"R5={row['R5']:.2f} R10={row['R10']:.2f} R20={row['R20']:.2f} "
            f"mean={row['mean_best_distance_m']:.2f}"
        )
    print(f"summary={args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
