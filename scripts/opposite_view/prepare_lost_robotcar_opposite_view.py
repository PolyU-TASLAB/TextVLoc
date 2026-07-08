#!/usr/bin/env python3
"""Prepare LoST-style Oxford RobotCar opposite-view data.

This extracts only sampled frames from RobotCar tar archives. The default setup
matches the LoST Overcast Autumn experiment: front stereo centre queries against
rear monocular database, sampled from the start of the traversal every 2 meters.
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import tarfile
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ImageMember:
    timestamp: int
    tar_path: Path
    member_name: str
    rel_name: str


@dataclass(frozen=True)
class Pose:
    timestamp: int
    easting: float
    northing: float
    down: float
    roll: float
    pitch: float
    yaw: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.easting, self.northing], dtype=np.float64)


@dataclass(frozen=True)
class Sample:
    image: ImageMember
    pose: Pose
    distance_from_start_m: float


def list_tar_images(tar_paths: list[Path], view_dir: str) -> list[ImageMember]:
    images: list[ImageMember] = []
    for tar_path in tar_paths:
        with tarfile.open(tar_path) as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".png"):
                    continue
                if f"/{view_dir}/" not in member.name:
                    continue
                timestamp = int(Path(member.name).stem)
                rel_name = f"{view_dir.replace('/', '_')}/{timestamp}.png"
                images.append(ImageMember(timestamp, tar_path, member.name, rel_name))
    return sorted(images, key=lambda x: x.timestamp)


def load_ins(gps_tar: Path, sequence: str) -> list[Pose]:
    target = f"{sequence}/gps/ins.csv"
    poses: list[Pose] = []
    with tarfile.open(gps_tar) as tar:
        f = tar.extractfile(target)
        if f is None:
            raise FileNotFoundError(f"{target} not found in {gps_tar}")
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
        for row in reader:
            if row.get("ins_status") != "INS_SOLUTION_GOOD":
                continue
            poses.append(
                Pose(
                    timestamp=int(row["timestamp"]),
                    easting=float(row["easting"]),
                    northing=float(row["northing"]),
                    down=float(row["down"]),
                    roll=float(row["roll"]),
                    pitch=float(row["pitch"]),
                    yaw=float(row["yaw"]),
                )
            )
    return sorted(poses, key=lambda x: x.timestamp)


def nearest_pose(image_ts: int, poses: list[Pose], pose_timestamps: list[int], max_dt_us: int) -> Pose | None:
    i = bisect_left(pose_timestamps, image_ts)
    candidates = []
    if i < len(poses):
        candidates.append(poses[i])
    if i > 0:
        candidates.append(poses[i - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda p: abs(p.timestamp - image_ts))
    if abs(best.timestamp - image_ts) > max_dt_us:
        return None
    return best


def attach_poses(images: list[ImageMember], poses: list[Pose], max_dt_us: int) -> list[tuple[ImageMember, Pose]]:
    pose_timestamps = [p.timestamp for p in poses]
    out = []
    for image in images:
        pose = nearest_pose(image.timestamp, poses, pose_timestamps, max_dt_us)
        if pose is not None:
            out.append((image, pose))
    return out


def sample_by_distance(
    images_with_pose: list[tuple[ImageMember, Pose]],
    spacing_m: float,
    max_distance_m: float | None,
) -> list[Sample]:
    samples: list[Sample] = []
    if not images_with_pose:
        return samples

    previous_pose = images_with_pose[0][1]
    last_selected_pose: Pose | None = None
    traveled = 0.0
    for idx, (image, pose) in enumerate(images_with_pose):
        if idx > 0:
            traveled += float(np.linalg.norm(pose.xy - previous_pose.xy))
        previous_pose = pose
        if max_distance_m is not None and traveled > max_distance_m:
            break
        if last_selected_pose is None or float(np.linalg.norm(pose.xy - last_selected_pose.xy)) >= spacing_m:
            samples.append(Sample(image=image, pose=pose, distance_from_start_m=traveled))
            last_selected_pose = pose
    return samples


def extract_samples(samples: list[Sample], output_image_root: Path) -> None:
    output_image_root.mkdir(parents=True, exist_ok=True)
    by_tar: dict[Path, list[Sample]] = {}
    for sample in samples:
        dst = output_image_root / sample.image.rel_name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        by_tar.setdefault(sample.image.tar_path, []).append(sample)

    for tar_path, tar_samples in by_tar.items():
        wanted = {sample.image.member_name: sample for sample in tar_samples}
        with tarfile.open(tar_path) as tar:
            for member in tar:
                sample = wanted.get(member.name)
                if sample is None:
                    continue
                src = tar.extractfile(member)
                if src is None:
                    continue
                dst = output_image_root / sample.image.rel_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                with dst.open("wb") as f:
                    shutil.copyfileobj(src, f)


def write_list(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sample.image.rel_name for sample in samples) + "\n", encoding="utf-8")


def write_poses(path: Path, samples_by_split: dict[str, list[Sample]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split",
            "image_name",
            "image_timestamp",
            "pose_timestamp",
            "easting",
            "northing",
            "down",
            "roll",
            "pitch",
            "yaw",
            "distance_from_start_m",
            "source_tar",
            "source_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split, samples in samples_by_split.items():
            for sample in samples:
                writer.writerow(
                    {
                        "split": split,
                        "image_name": sample.image.rel_name,
                        "image_timestamp": sample.image.timestamp,
                        "pose_timestamp": sample.pose.timestamp,
                        "easting": f"{sample.pose.easting:.6f}",
                        "northing": f"{sample.pose.northing:.6f}",
                        "down": f"{sample.pose.down:.6f}",
                        "roll": f"{sample.pose.roll:.9f}",
                        "pitch": f"{sample.pose.pitch:.9f}",
                        "yaw": f"{sample.pose.yaw:.9f}",
                        "distance_from_start_m": f"{sample.distance_from_start_m:.3f}",
                        "source_tar": sample.image.tar_path.name,
                        "source_member": sample.image.member_name,
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SensorKit/LoST_Robotcar"))
    parser.add_argument("--sequence", default="2014-12-09-13-21-02")
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km"))
    parser.add_argument("--spacing-m", type=float, default=2.0)
    parser.add_argument("--max-distance-m", type=float, default=2000.0)
    parser.add_argument("--max-pose-dt-us", type=int, default=100_000)
    parser.add_argument("--no-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rear_tars = sorted(args.data_dir.glob(f"{args.sequence}_mono_rear_*.tar"))
    front_tars = sorted(args.data_dir.glob(f"{args.sequence}_stereo_centre_*.tar"))
    gps_tar = args.data_dir / f"{args.sequence}_gps.tar"
    if not rear_tars:
        raise FileNotFoundError(f"No rear tar files found in {args.data_dir}")
    if not front_tars:
        raise FileNotFoundError(f"No stereo centre tar files found in {args.data_dir}")
    if not gps_tar.exists():
        raise FileNotFoundError(gps_tar)

    poses = load_ins(gps_tar, args.sequence)
    front_images = list_tar_images(front_tars, "stereo/centre")
    rear_images = list_tar_images(rear_tars, "mono_rear")

    front_with_pose = attach_poses(front_images, poses, args.max_pose_dt_us)
    rear_with_pose = attach_poses(rear_images, poses, args.max_pose_dt_us)
    front_samples = sample_by_distance(front_with_pose, args.spacing_m, args.max_distance_m)
    rear_samples = sample_by_distance(rear_with_pose, args.spacing_m, args.max_distance_m)

    image_root = args.out_dir / "images"
    if not args.no_extract:
        extract_samples(front_samples + rear_samples, image_root)

    write_list(args.out_dir / "queries_front.txt", front_samples)
    write_list(args.out_dir / "database_rear.txt", rear_samples)
    write_poses(args.out_dir / "poses.csv", {"query_front": front_samples, "database_rear": rear_samples})

    print(f"sequence={args.sequence}")
    print(f"spacing_m={args.spacing_m:g}")
    print(f"max_distance_m={args.max_distance_m:g}")
    print(f"front_raw={len(front_images)} front_with_pose={len(front_with_pose)} front_samples={len(front_samples)}")
    print(f"rear_raw={len(rear_images)} rear_with_pose={len(rear_with_pose)} rear_samples={len(rear_samples)}")
    print(f"image_root={image_root}")
    print(f"queries={args.out_dir / 'queries_front.txt'}")
    print(f"database={args.out_dir / 'database_rear.txt'}")
    print(f"poses={args.out_dir / 'poses.csv'}")


if __name__ == "__main__":
    main()
