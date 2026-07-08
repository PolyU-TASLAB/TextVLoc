from __future__ import annotations

from pathlib import Path

import numpy as np


def window_offsets(length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("sequence length must be >= 1")
    before = length // 2
    after = length - before - 1
    return np.arange(-before, after + 1, dtype=np.int64)


def sequence_scores(
    frame_scores: np.ndarray,
    *,
    sequence_length: int,
    center_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate same-direction diagonal scores for query/database windows.

    Returns:
        seq_scores: [valid_query_centers, valid_db_centers]
        query_centers: indices into the original query list
        db_centers: indices into the original database list
    """
    if center_stride < 1:
        raise ValueError("center_stride must be >= 1")
    offsets = window_offsets(sequence_length)
    q_count, db_count = frame_scores.shape
    first = int(-offsets.min())
    q_last = int(q_count - offsets.max())
    db_last = int(db_count - offsets.max())
    query_centers = np.arange(first, q_last, center_stride, dtype=np.int64)
    db_centers = np.arange(first, db_last, center_stride, dtype=np.int64)
    if len(query_centers) == 0 or len(db_centers) == 0:
        raise ValueError(
            f"No full sequence windows for length={sequence_length}, "
            f"query_frames={q_count}, db_frames={db_count}."
        )

    seq = np.zeros((len(query_centers), len(db_centers)), dtype=np.float32)
    for offset in offsets:
        seq += frame_scores[query_centers + offset][:, db_centers + offset]
    seq /= float(sequence_length)
    return seq, query_centers, db_centers


def write_pairs(
    path: Path,
    query_names: list[str],
    db_names: list[str],
    order: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qi, query in enumerate(query_names):
            for db_idx in order[qi]:
                f.write(f"{query} {db_names[int(db_idx)]}\n")
