"""UrbanV2X six-camera captions excluding the cam1 retrieval database."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SOURCE = Path("outputs/text_depth_ablation_v1/three_datasets_text8_visual12")
OUT = Path("outputs/opposite_view_urban_493_six_view_no_cam1")
OLD_PROMPT_MANIFEST = None
CAMERAS = ["cam0", "cam2", "cam3", "cam4", "cam5", "cam6"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_info():
    runs = json.loads((SOURCE / "manifest.json").read_text())["runs"]
    run = next(run for run in runs if run["dataset"] == "UrbanV2X")
    run["checkpoint"] = "checkpoints/text8_visual12_epoch22.pt"
    return run


def query_groups():
    from hloc.generate_multiview_captions import build_groups

    groups = build_groups(
        multi_cam_root=Path("datasets/SensorKit/allcams"), cameras=CAMERAS,
        anchor_camera="cam0", anchor_dir=None, max_delta_ns=600_000_000,
        min_cameras=1)
    assert len(groups) == 493
    assert len({group["image_name"] for group in groups}) == 493
    assert all(not any(cam == "cam1" for cam, _ in group["views"])
               for group in groups)
    return groups


def generate(limit):
    import open_clip
    from hloc.generate_multiview_captions import QWEN_VL_MODEL_ID, load_qwen_multiview_captioner

    groups = query_groups()
    prompt = OLD_PROMPT_MANIFEST.read_text()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "captions.csv"
    manifest_path = OUT / "caption_manifest.json"
    manifest = {
        "model": QWEN_VL_MODEL_ID, "prompt": prompt,
        "source": "493 cam0-anchored groups using cam0,cam2,cam3,cam4,cam5,cam6 only",
        "excluded_reference_camera": "cam1", "max_delta_ns": 600_000_000,
        "max_pixels_per_image": 262144,
        "generation": {"max_new_tokens": 256, "do_sample": False},
        "query_names": [g["image_name"] for g in groups],
        "groups": [{"query": g["image_name"],
                    "views": [{"camera": cam, "path": str(path)} for cam, path in g["views"]]}
                   for g in groups],
    }
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    completed = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        completed = {row["image_name"]: row["description"] for row in rows}
        assert len(rows) == len(completed) and all(completed.values())
        assert set(completed).issubset(set(manifest["query_names"]))
    pending = [group for group in groups if group["image_name"] not in completed]
    if limit:
        pending = pending[:limit]
    print(f"No-cam1 captions: {len(completed)}/493 complete; {len(pending)} this run", flush=True)
    if not pending:
        return
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    captioner = load_qwen_multiview_captioner("cuda", max_pixels=262144)
    exists = csv_path.exists()
    with csv_path.open("a" if exists else "x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["image_name", "description"])
            handle.flush()
        for index, group in enumerate(pending, 1):
            description = ""
            for attempt in range(3):
                instruction = prompt if attempt == 0 else (
                    prompt + "\n\nWrite a shorter complete description of at most "
                    f"{50 - 10 * (attempt - 1)} words, keeping only the most distinctive visible landmarks.")
                description = captioner(group["views"], prompt=instruction, max_new_tokens=256).strip()
                if description and len(tokenizer.encode(description)) + 2 <= 77:
                    break
            else:
                raise RuntimeError(f"Cannot fit CLIP context: {group['image_name']}")
            writer.writerow([group["image_name"], description])
            handle.flush()
            if index % 25 == 0 or index == len(pending):
                print(f"Generated {index}/{len(pending)}", flush=True)


def evaluate():
    import open_clip
    import torch
    from run_sensorkit_cam6_to_cam1_opposite_view_eval import load_poses
    from hloc.clip_lora_checkpoint import load_openclip_lora_checkpoint

    run = run_info()
    groups = query_groups()
    q = [group["image_name"] for group in groups]
    db = run["database"]
    with (OUT / "captions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    captions = {row["image_name"]: row["description"] for row in rows}
    assert len(rows) == len(q) == len(captions)
    assert set(captions) == set(q)
    ckpt_path = Path(run["checkpoint"])
    model, tokenizer, _, _ = load_openclip_lora_checkpoint(ckpt_path, "cuda")
    text_features = []
    with torch.no_grad():
        for start in range(0, len(q), 32):
            emb = model.encode_text(tokenizer([captions[name] for name in q[start:start + 32]]).cuda())
            text_features.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
    text_features = np.concatenate(text_features)
    with np.load(SOURCE / "UrbanV2X_features.npz") as features:
        assert run["queries"] == features["query_names"].tolist()
        assert db == features["database_names"].tolist()
        db_features = features["database_features"]
    order = np.argsort(-(text_features @ db_features.T), axis=1)[:, :20]
    poses = load_poses([Path("datasets/SensorKit/0620_gt") / f"enu_all_{part}_gt.csv"
                        for part in ("mapping", "query")])
    qp = np.stack([poses[name] for name in q])
    dp = np.stack([poses[name] for name in db])
    distances = np.linalg.norm(qp[:, None] - dp[order], axis=-1)
    result = []
    for k in (1, 5, 10, 20):
        best = distances[:, :k].min(axis=1)
        row = {"setting": "all_cam0_anchors", "K": k, "queries": len(q), "database": len(db),
               **{f"R@{metres}m": 100 * float(np.mean(best <= metres))
                  for metres in (5, 10, 20)}}
        result.append(row)
        print(row, flush=True)

    # Match each existing cam6 visual query to the nearest cam0 timestamp.
    # This subset supports paired comparison with the archived cam6 image baselines.
    cam0_ts = np.asarray([int(Path(name).stem) for name in q], dtype=np.int64)
    cam6_queries = run["queries"]
    paired_indices = np.asarray([
        int(np.argmin(np.abs(cam0_ts - int(Path(name).stem))))
        for name in cam6_queries], dtype=np.int64)
    assert len(set(paired_indices.tolist())) == len(cam6_queries) == 491
    assert max(abs(int(Path(cam6).stem) - int(Path(q[idx]).stem))
               for cam6, idx in zip(cam6_queries, paired_indices)) <= 600_000_000
    paired_distances = np.linalg.norm(
        np.stack([poses[name] for name in cam6_queries])[:, None]
        - dp[order[paired_indices]], axis=-1)
    for k in (1, 5, 10, 20):
        best = paired_distances[:, :k].min(axis=1)
        row = {"setting": "paired_cam6_queries", "K": k,
               "queries": len(cam6_queries), "database": len(db),
               **{f"R@{metres}m": 100 * float(np.mean(best <= metres))
                  for metres in (5, 10, 20)}}
        result.append(row)
        print(row, flush=True)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    with (OUT / "per_query_493.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_image", "description", "input_camera_paths",
                         "top1_reference", "top1_distance_m", "best_top20_distance_m"])
        for index, (name, group) in enumerate(zip(q, groups)):
            writer.writerow([name, captions[name], ";".join(str(path) for _, path in group["views"]),
                             db[order[index, 0]], float(distances[index, 0]),
                             float(distances[index].min())])
    with (OUT / "per_query_paired_491.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cam6_query", "cam0_caption_anchor", "description",
                         "top1_reference", "top1_distance_m", "best_top20_distance_m"])
        for index, (name, anchor_idx) in enumerate(zip(cam6_queries, paired_indices)):
            writer.writerow([name, q[anchor_idx], captions[q[anchor_idx]],
                             db[order[anchor_idx, 0]], float(paired_distances[index, 0]),
                             float(paired_distances[index].min())])
    (OUT / "evaluation_manifest.json").write_text(json.dumps({
        "condition": "Six-camera Qwen description excluding reference camera cam1",
        "caption_count": len(q), "paired_cam6_queries": len(cam6_queries),
        "database": len(db), "cameras": CAMERAS,
        "view_counts": {str(count): sum(len(group["views"]) == count for group in groups)
                        for count in sorted({len(group["views"]) for group in groups})},
        "checkpoint": str(ckpt_path), "checkpoint_sha256": digest(ckpt_path),
        "caption_manifest_sha256": digest(OUT / "caption_manifest.json"),
        "caption_csv_sha256": digest(OUT / "captions.csv"),
        "reference_features": str(SOURCE / "UrbanV2X_features.npz"),
        "notes": "Cam0 anchors are query-side timestamps. No cam1 image is used to generate text. Two anchors lack all six views. All cameras and references still come from the same capture session.",
    }, indent=2) + "\n")


def main():
    global SOURCE, OLD_PROMPT_MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "evaluate"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source", type=Path, default=SOURCE, help="Local feature directory with manifest.json and dataset feature NPZ files")
    parser.add_argument("--prompt-file", type=Path, help="Local plain-text multi-view caption prompt; required for generation")
    args = parser.parse_args()
    SOURCE, OLD_PROMPT_MANIFEST = args.source, args.prompt_file
    if args.stage == "generate" and args.prompt_file is None:
        parser.error("generate requires --prompt-file")
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.stage == "generate":
        generate(args.limit)
    else:
        evaluate()


if __name__ == "__main__":
    main()
