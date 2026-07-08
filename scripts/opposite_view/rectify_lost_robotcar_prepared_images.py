#!/usr/bin/env python3
"""Rectify sampled Oxford RobotCar images used by the LoST-style experiment."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def add_sdk_path(sdk_dir: Path) -> None:
    python_dir = sdk_dir / "python"
    if not python_dir.exists():
        raise FileNotFoundError(f"RobotCar SDK python dir not found: {python_dir}")
    sys.path.insert(0, str(python_dir))


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def demosaic(raw: np.ndarray, rel_name: str) -> np.ndarray:
    if rel_name.startswith("stereo_centre/"):
        return cv2.cvtColor(raw, cv2.COLOR_BayerGB2RGB)
    if rel_name.startswith("mono_rear/"):
        return cv2.cvtColor(raw, cv2.COLOR_BayerRG2RGB)
    raise ValueError(f"Unknown RobotCar camera for {rel_name}")


def rectify_one(src: Path, dst: Path, rel_name: str, model) -> None:
    raw = np.asarray(Image.open(src))
    rgb = demosaic(raw, rel_name)
    rectified = model.undistort(rgb)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rectified.astype(np.uint8), mode="RGB").save(dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km_rectified"),
    )
    parser.add_argument(
        "--sdk-dir",
        type=Path,
        default=Path("datasets/SensorKit/LoST_Robotcar/robotcar-dataset-sdk"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Debug limit for quick checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_sdk_path(args.sdk_dir)
    from camera_model import CameraModel

    src_root = args.prepared_dir / "images"
    dst_root = args.out_dir / "images"
    query_names = read_list(args.prepared_dir / "queries_front.txt")
    db_names = read_list(args.prepared_dir / "database_rear.txt")
    names = query_names + db_names
    if args.limit is not None:
        names = names[: args.limit]

    models_dir = args.sdk_dir / "models"
    models = {
        "stereo_centre": CameraModel(str(models_dir), "stereo/centre"),
        "mono_rear": CameraModel(str(models_dir), "mono_rear"),
    }

    processed = 0
    skipped = 0
    for rel_name in names:
        src = src_root / rel_name
        dst = dst_root / rel_name
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.exists() and dst.stat().st_size > 0 and not args.overwrite:
            skipped += 1
            continue
        camera_key = rel_name.split("/", 1)[0]
        rectify_one(src, dst, rel_name, models[camera_key])
        processed += 1
        if processed % 100 == 0:
            print(f"processed={processed} skipped={skipped}")

    if args.limit is None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.prepared_dir / "queries_front.txt", args.out_dir / "queries_front.txt")
        shutil.copy2(args.prepared_dir / "database_rear.txt", args.out_dir / "database_rear.txt")
        shutil.copy2(args.prepared_dir / "poses.csv", args.out_dir / "poses.csv")

    print(f"prepared_dir={args.prepared_dir}")
    print(f"out_dir={args.out_dir}")
    print(f"images={len(names)} processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
