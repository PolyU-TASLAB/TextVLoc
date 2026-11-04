"""Recompute the paper fine-localization statistics, or rerun Top-20 HLoc."""
import argparse
import json
from pathlib import Path

import numpy as np
from hloc.utils.read_write_model import qvec2rotmat
from scripts.eval_paper_retrieval import ROOT, positions


def statistics(errors):
    distance = np.linalg.norm(errors, axis=1)
    if not len(distance):
        return {"count": 0}
    return {"count": len(distance), "mean_m": float(distance.mean()),
            "rmse_m": float(np.sqrt((distance**2).mean())),
            "median_m": float(np.median(distance)),
            "p90_m": float(np.percentile(distance, 90)),
            "p95_m": float(np.percentile(distance, 95)),
            "mean_horizontal_m": float(np.linalg.norm(errors[:, :2], axis=1).mean()),
            "mean_vertical_m": float(np.abs(errors[:, 2]).mean())}


def report(pose_file, retrieval_run, dataset_root):
    protocol = json.loads((retrieval_run / "manifest.json").read_text())
    names, database = protocol["query_order"], protocol["database_order"]
    qgt = positions(dataset_root / "0620_gt/enu_all_query_gt.csv")
    dgt = positions(dataset_root / "0620_gt/enu_all_mapping_gt.csv")
    actual = {}
    for line in pose_file.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != 8 or values[0] in actual:
            raise ValueError("Expected one name, quaternion and translation per pose")
        q = np.array([float(x) for x in values[1:5]])
        t = np.array([float(x) for x in values[5:8]])
        actual[values[0]] = -qvec2rotmat(q).T @ t
    ground_truth = np.array([qgt[f"cam1/{Path(n).name}"] for n in names])
    predicted = np.array([actual.get(n, [np.nan]*3) for n in names])
    delta = predicted-ground_truth
    valid = np.isfinite(delta).all(axis=1)
    success = valid & (np.linalg.norm(delta, axis=1) < 10)
    with np.load(retrieval_run / "rankings.npz") as archive:
        selected = archive["original_qwen__global"][:, 0]
    coarse_delta = np.array([dgt[database[j]] for j in selected])-ground_truth
    coarse_success = np.linalg.norm(coarse_delta, axis=1) < 10
    result = {"queries": len(names), "poses_returned": int(valid.sum()),
              "success_definition": f"APE < 10m over all {len(names)} queries; missing/nonfinite pose fails",
              "error_statistics": "successful estimates only; all returned poses also reported separately",
              "text_only": {"success_rate_pct": float(100*coarse_success.mean()),
                            "successful": statistics(coarse_delta[coarse_success])},
              "text_image": {"success_rate_pct": float(100*success.mean()),
                             "successful": statistics(delta[success]),
                             "all_returned": statistics(delta[valid])}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--pose-file", type=Path, help="Local HLoc poses for the check command")
    parser.add_argument("--retrieval-run", type=Path, required=True,
                        help="Local retrieval output containing manifest.json and rankings.npz")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/SensorKit")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-sfm", type=Path)
    parser.add_argument("--reference-features", type=Path)
    parser.add_argument("--query-features", type=Path)
    parser.add_argument("--query-dir", type=Path)
    parser.add_argument("--query-list", type=Path, help="Local query intrinsics list, required for run")
    args = parser.parse_args()
    if args.command == "run":
        if not args.output_dir or not args.reference_sfm or not args.reference_features or not args.query_list:
            parser.error("run requires --output-dir, --reference-sfm, --reference-features and --query-list")
        from hloc import extract_features, localize_sfm, match_features
        protocol = json.loads((args.retrieval_run / "manifest.json").read_text())
        args.output_dir.mkdir(parents=True, exist_ok=False)
        pairs = args.output_dir / "pairs.txt"
        with np.load(args.retrieval_run / "rankings.npz") as archive, pairs.open("w") as handle:
            for name, row in zip(protocol["query_order"], archive["original_qwen__global"]):
                for j in row:
                    handle.write(f"{name} {protocol['database_order'][j]}\n")
        query_features = args.query_features
        if query_features is None:
            query_features = extract_features.main(
                extract_features.confs["superpoint_aachen"],
                args.query_dir or args.dataset_root / "query/cam1_all/0_undistorted",
                args.output_dir, image_list=protocol["query_order"])
        matches = args.output_dir / "matches.h5"
        match_features.main(match_features.confs["superpoint+lightglue"], pairs,
                            query_features, features_ref=args.reference_features, matches=matches)
        args.pose_file = args.output_dir / "poses.txt"
        localize_sfm.main(args.reference_sfm, args.query_list, pairs,
                          query_features, matches, args.pose_file)
    if args.pose_file is None:
        parser.error("check requires --pose-file")
    result = report(args.pose_file, args.retrieval_run, args.dataset_root)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
