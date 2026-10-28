"""Audit metric distances of UrbanV2X NetVLAD-positive evaluation pairs."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("datasets/SensorKit")
PAIRS = ROOT / "query/cam1_all/05_allcams_positive_pairs_for_eval.csv"
QUERY_POSES = ROOT / "0620_gt/enu_all_query_gt.csv"
REFERENCE_POSES = ROOT / "0620_gt/enu_all_mapping_gt.csv"
OUT = Path("outputs/opposite_view_urban_positive_pair_distance_audit")
FILTERED = ROOT / "query/cam1_all/05_noncam1_positive_pairs_netvlad05_enu3d20m_for_eval.csv"


def poses(path):
    with path.open(newline="") as handle:
        return {
            (row["camera_id"], row["image_id"]): np.array(
                [float(row["X_enu"]), float(row["Y_enu"]), float(row["Z_enu"])])
            for row in csv.DictReader(handle)
        }


def stats(rows):
    horizontal = np.asarray([row["distance_2d_m"] for row in rows], dtype=float)
    spatial = np.asarray([row["distance_3d_m"] for row in rows], dtype=float)
    return {
        "pairs": len(rows),
        "queries": len({row["query_image"] for row in rows}),
        "distance_2d_m": {
            "min": float(np.min(horizontal)), "median": float(np.median(horizontal)),
            "mean": float(np.mean(horizontal)),
            **{f"p{p}": float(np.percentile(horizontal, p)) for p in (75, 90, 95, 99)},
            "max": float(np.max(horizontal)),
            **{f"within_{m}m_pct": float(100 * np.mean(horizontal <= m))
               for m in (5, 10, 20, 50, 100)},
        },
        "distance_3d_m": {
            "median": float(np.median(spatial)),
            "p95": float(np.percentile(spatial, 95)),
            "max": float(np.max(spatial)),
            "within_20m_pct": float(100 * np.mean(spatial <= 20)),
        },
    }


def main():
    query_poses = poses(QUERY_POSES)
    reference_poses = poses(REFERENCE_POSES)
    rows = []
    retained_pairs = []
    skipped = Counter()
    with PAIRS.open(newline="") as handle:
        for pair in csv.DictReader(handle):
            query_image = Path(pair["anchor"]).name
            reference_image = Path(pair["positive"]).name
            camera = Path(pair["positive"]).parent.name
            if camera == "cam1":
                skipped["same_camera_not_in_reference_database"] += 1
                continue
            qpos = query_poses.get(("cam1", query_image))
            rpos = reference_poses.get((camera, reference_image))
            if qpos is None or rpos is None:
                skipped["missing_pose"] += 1
                continue
            delta = qpos - rpos
            rows.append({
                "query_image": query_image, "reference_camera": camera,
                "reference_image": reference_image,
                "distance_2d_m": float(np.linalg.norm(delta[:2])),
                "distance_3d_m": float(np.linalg.norm(delta)),
            })
            if rows[-1]["distance_3d_m"] <= 20.0:
                retained_pairs.append(pair)
    if not rows:
        raise ValueError("No positive pairs with positions")
    if skipped["missing_pose"]:
        raise ValueError(f"Missing ENU poses for {skipped['missing_pose']} cross-camera positives")
    groups = defaultdict(list)
    for row in rows:
        groups[row["reference_camera"]].append(row)
    result = {
        "positive_pair_rule": "NetVLAD similarity >= 0.5 (not a metric-distance rule)",
        "position_source": [str(QUERY_POSES), str(REFERENCE_POSES)],
        "pair_source": str(PAIRS),
        "excluded_or_missing": dict(skipped),
        "filtered_positive_csv": str(FILTERED),
        "filtered_pairs": len(retained_pairs),
        "removed_cross_camera_pairs_over_20m": len(rows) - len(retained_pairs),
        "filtered_queries": len({Path(pair["anchor"]).name for pair in retained_pairs}),
        "all_reference_cameras": stats(rows),
        "by_reference_camera": {camera: stats(group) for camera, group in sorted(groups.items())},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if result["filtered_queries"] != 493:
        raise ValueError("20 m filter leaves queries without positives")
    with FILTERED.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anchor", "positive"])
        writer.writeheader()
        writer.writerows(retained_pairs)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
