#!/usr/bin/env python3
"""Prepare LoST-style SYNTHIA Sequence-02 front/rear tests.

The script is intentionally conservative about directory names because public
SYNTHIA mirrors are not fully consistent after extraction. It searches for Dawn
and Fall RGB image folders, then maps front/rear images by frame id.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_EXPERIMENTS = [
    # name, query_condition, query_view, db_condition, db_view, query_text_condition, query_text_view
    ("dawn_back_to_dawn_front", "dawn", "rear", "dawn", "front", "dawn", "front"),
    ("fall_back_to_fall_front", "fall", "rear", "fall", "front", "fall", "front"),
    ("fall_back_to_dawn_front", "fall", "rear", "dawn", "front", "fall", "front"),
]


@dataclass(frozen=True)
class Frame:
    condition: str
    view: str
    frame_id: int
    rel_path: str


def normalize_condition(text: str) -> str | None:
    low = text.lower()
    if "dawn" in low:
        return "dawn"
    if "fall" in low:
        return "fall"
    return None


def normalize_view(text: str) -> str | None:
    low = text.lower()
    if "front" in low or low in {"omni_f", "front_f"}:
        return "front"
    if "rear" in low or "back" in low or low in {"omni_b", "back_b"}:
        return "rear"
    return None


def frame_id_from_path(path: Path) -> int | None:
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        return None
    return int(numbers[-1])


def scan_frames(data_root: Path) -> list[Frame]:
    frames: list[Frame] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(data_root)
        parts = rel.parts
        if "RGB" not in parts or "Stereo_Left" not in parts:
            continue
        cond = None
        view = None
        for part in parts:
            cond = cond or normalize_condition(part)
            view = view or normalize_view(part)
        if cond is None or view is None:
            continue
        frame_id = frame_id_from_path(path)
        if frame_id is None:
            continue
        frames.append(Frame(cond, view, frame_id, rel.as_posix()))
    return frames


def write_manifest(frames: list[Frame], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "view", "frame_id", "image_name"])
        writer.writeheader()
        for frame in frames:
            writer.writerow(
                {
                    "condition": frame.condition,
                    "view": frame.view,
                    "frame_id": frame.frame_id,
                    "image_name": frame.rel_path,
                }
            )


def write_poses(frames: list[Frame], output: Path, frame_stride_m: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "easting", "northing", "frame_id"])
        writer.writeheader()
        for frame in frames:
            writer.writerow(
                {
                    "image_name": frame.rel_path,
                    "easting": f"{frame.frame_id * frame_stride_m:.6f}",
                    "northing": "0.000000",
                    "frame_id": frame.frame_id,
                }
            )


def by_condition_view(frames: list[Frame]) -> dict[tuple[str, str], list[Frame]]:
    grouped: dict[tuple[str, str], list[Frame]] = {}
    for frame in frames:
        grouped.setdefault((frame.condition, frame.view), []).append(frame)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda x: x.frame_id)
    return grouped


def frames_by_id(frames: list[Frame]) -> dict[int, Frame]:
    return {frame.frame_id: frame for frame in frames}


def sparse_frames(
    frames: list[Frame],
    *,
    sample_every: int,
    min_distance_m: float | None,
    frame_stride_m: float,
) -> list[Frame]:
    if sample_every < 1:
        raise ValueError("--sample-every must be >= 1")
    if not frames:
        return frames

    selected = frames[::sample_every]
    if min_distance_m is None or min_distance_m <= 0:
        return selected

    min_frame_delta = max(1, math.ceil(min_distance_m / frame_stride_m))
    sparse = []
    last_frame_id: int | None = None
    for frame in selected:
        if last_frame_id is None or frame.frame_id - last_frame_id >= min_frame_delta:
            sparse.append(frame)
            last_frame_id = frame.frame_id
    return sparse


def nearest_by_frame_id(index: dict[int, Frame], frame_id: int) -> tuple[Frame | None, int | None]:
    if not index:
        return None, None
    if frame_id in index:
        return index[frame_id], 0
    ids = sorted(index)
    pos = np.searchsorted(ids, frame_id)
    candidates = []
    if pos < len(ids):
        candidates.append(ids[pos])
    if pos > 0:
        candidates.append(ids[pos - 1])
    if not candidates:
        return None, None
    best = min(candidates, key=lambda x: abs(x - frame_id))
    return index[best], abs(best - frame_id)


def write_experiment_lists(
    grouped: dict[tuple[str, str], list[Frame]],
    out_dir: Path,
    max_frames: int | None,
    frame_stride_m: float,
    data_root: Path,
    sample_every: int,
    min_distance_m: float | None,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for exp, q_cond, q_view, db_cond, db_view, text_cond, text_view in DEFAULT_EXPERIMENTS:
        queries = sparse_frames(
            grouped.get((q_cond, q_view), []),
            sample_every=sample_every,
            min_distance_m=min_distance_m,
            frame_stride_m=frame_stride_m,
        )
        db = sparse_frames(
            grouped.get((db_cond, db_view), []),
            sample_every=sample_every,
            min_distance_m=min_distance_m,
            frame_stride_m=frame_stride_m,
        )
        text_sources = sparse_frames(
            grouped.get((text_cond, text_view), []),
            sample_every=sample_every,
            min_distance_m=min_distance_m,
            frame_stride_m=frame_stride_m,
        )
        if max_frames is not None:
            queries = queries[:max_frames]
            db = db[:max_frames]
            text_sources = text_sources[:max_frames]
        db_by_id = frames_by_id(db)
        text_by_id = frames_by_id(text_sources)
        exp_dir = out_dir / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        query_list = exp_dir / "queries.txt"
        db_list = exp_dir / "database.txt"
        text_source_list = exp_dir / "query_text_source_images.txt"
        text_source_path_list = exp_dir / "query_text_source_image_paths.txt"
        text_map_csv = exp_dir / "query_text_source_map.csv"
        query_list.write_text("\n".join(f.rel_path for f in queries) + ("\n" if queries else ""), encoding="utf-8")
        db_list.write_text("\n".join(f.rel_path for f in db) + ("\n" if db else ""), encoding="utf-8")

        mapped_text_sources = []
        with text_map_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "query_image_name",
                    "query_frame_id",
                    "text_source_image_name",
                    "text_source_frame_id",
                    "text_source_delta_m",
                    "target_db_image_name",
                    "target_db_frame_id",
                    "target_db_delta_m",
                ],
            )
            writer.writeheader()
            for query in queries:
                text_source, text_delta = nearest_by_frame_id(text_by_id, query.frame_id)
                target_db, db_delta = nearest_by_frame_id(db_by_id, query.frame_id)
                if text_source is not None:
                    mapped_text_sources.append(text_source.rel_path)
                writer.writerow(
                    {
                        "query_image_name": query.rel_path,
                        "query_frame_id": query.frame_id,
                        "text_source_image_name": text_source.rel_path if text_source else "",
                        "text_source_frame_id": text_source.frame_id if text_source else "",
                        "text_source_delta_m": f"{(text_delta or 0) * frame_stride_m:.6f}" if text_source else "",
                        "target_db_image_name": target_db.rel_path if target_db else "",
                        "target_db_frame_id": target_db.frame_id if target_db else "",
                        "target_db_delta_m": f"{(db_delta or 0) * frame_stride_m:.6f}" if target_db else "",
                    }
                )
        mapped_text_sources = sorted(set(mapped_text_sources))
        text_source_list.write_text(
            "\n".join(mapped_text_sources) + ("\n" if mapped_text_sources else ""),
            encoding="utf-8",
        )
        mapped_text_source_paths = [(data_root / rel).as_posix() for rel in mapped_text_sources]
        text_source_path_list.write_text(
            "\n".join(mapped_text_source_paths) + ("\n" if mapped_text_source_paths else ""),
            encoding="utf-8",
        )
        rows.append(
            {
                "experiment": exp,
                "query_condition": q_cond,
                "query_view": q_view,
                "db_condition": db_cond,
                "db_view": db_view,
                "query_text_condition": text_cond,
                "query_text_view": text_view,
                "queries": len(queries),
                "database": len(db),
                "query_text_sources": len(mapped_text_sources),
                "sample_every": sample_every,
                "min_distance_m": "" if min_distance_m is None else min_distance_m,
                "query_list": query_list.as_posix(),
                "database_list": db_list.as_posix(),
                "query_text_source_list": text_source_list.as_posix(),
                "query_text_source_path_list": text_source_path_list.as_posix(),
                "query_text_source_map": text_map_csv.as_posix(),
            }
        )
    return rows


def write_protocol(rows: list[dict[str, str | int]], output: Path) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/SYNTHIA/SEQS-02"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/SYNTHIA/seq02_lost"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride-m", type=float, default=2.0)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Keep one frame every N sorted frames for each condition/view.",
    )
    parser.add_argument(
        "--min-distance-m",
        type=float,
        default=None,
        help="Optional minimum along-track spacing after --sample-every, using frame_id * frame_stride_m.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.out_dir / "manifest.csv"
    frames = scan_frames(args.data_root)
    if not frames:
        raise SystemExit(
            f"No SYNTHIA Dawn/Fall front/rear images found under {args.data_root}. "
            "Extract SYNTHIA-SEQS-02-DAWN.rar and SYNTHIA-SEQS-02-FALL.rar first."
        )
    grouped = by_condition_view(frames)
    write_manifest(frames, manifest_path)
    write_poses(frames, args.out_dir / "poses.csv", args.frame_stride_m)
    rows = write_experiment_lists(
        grouped,
        args.out_dir,
        args.max_frames,
        args.frame_stride_m,
        args.data_root,
        args.sample_every,
        args.min_distance_m,
    )
    write_protocol(rows, args.out_dir / "protocol.csv")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote poses: {args.out_dir / 'poses.csv'}")
    print(f"Wrote protocol: {args.out_dir / 'protocol.csv'}")
    for row in rows:
        print(
            f"{row['experiment']}: queries={row['queries']} database={row['database']} "
            f"text_sources={row['query_text_sources']} "
            f"sample_every={row['sample_every']} min_distance_m={row['min_distance_m']} "
            f"({row['query_condition']} {row['query_view']} -> {row['db_condition']} {row['db_view']}; "
            f"text={row['query_text_condition']} {row['query_text_view']})"
        )


if __name__ == "__main__":
    main()
