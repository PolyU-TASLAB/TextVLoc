"""RobotCar front/rear/left/right surroundings text retrieval.

Place demosaiced and rectified mono_left/*.png, mono_right/*.png, and
mono_rear/*.png from the 2014-12-09 13:21:02 traversal under --side-root.
Rear frames are selected from the same traversal but exclude exact reference
database images. They remain geographically adjacent privileged observations.
Run in clip-hloc-dev.
"""

import argparse
import bisect
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

SOURCE = Path("outputs/text_depth_ablation_v1/three_datasets_text8_visual12")
PREPARED = Path("datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km_rectified")
OUT = Path("outputs/opposite_view_robotcar_four_view_surroundings_fp16")
PROMPT_MANIFEST = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_info():
    runs = json.loads((SOURCE / "manifest.json").read_text())["runs"]
    run = next(run for run in runs if run["dataset"] == "Oxford RobotCar")
    run["checkpoint"] = "checkpoints/text8_visual12_epoch22.pt"
    return run


def nearest_image(timestamp, images, tolerance_us):
    stamps = [int(path.stem) for path in images]
    at = bisect.bisect_left(stamps, timestamp)
    choices = images[max(0, at - 1):min(len(images), at + 1)]
    if not choices:
        return None
    closest = min(choices, key=lambda path: abs(int(path.stem) - timestamp))
    return closest if abs(int(closest.stem) - timestamp) <= tolerance_us else None


def groups(side_root, tolerance_us):
    run = run_info()
    side_images = {}
    for camera in ("mono_left", "mono_right", "mono_rear"):
        folder = side_root / camera
        images = sorted(folder.glob("*.png"), key=lambda path: int(path.stem)) if folder.is_dir() else []
        if camera == "mono_rear":
            reference_names = {Path(name).name for name in run["database"]}
            images = [path for path in images if path.name not in reference_names]
        if not images:
            raise FileNotFoundError(
                f"Missing rectified {folder}/*.png. Provide the 2014-12-09-13-21-02 "
                f"{camera} archives and run prepare_robotcar_side_views.py.")
        if Image.open(images[0]).mode != "RGB":
            raise ValueError(f"{folder} contains raw Bayer frames; demosaic and rectify first")
        side_images[camera] = images
    result = []
    for query in run["queries"]:
        assert query.startswith("stereo_centre/")
        front = PREPARED / "images" / query
        if not front.is_file():
            raise FileNotFoundError(front)
        timestamp = int(front.stem)
        views = [("stereo_centre", front)]
        for camera in ("mono_left", "mono_right", "mono_rear"):
            image = nearest_image(timestamp, side_images[camera], tolerance_us)
            if image is None:
                raise ValueError(f"No {camera} frame within {tolerance_us} us of {query}")
            views.append((camera, image))
        result.append((query, views))
    assert len(result) == 866
    return result


def generate(side_root, tolerance_us, limit, load_in_4bit):
    import open_clip
    from hloc.generate_multiview_captions import QWEN_VL_MODEL_ID, load_qwen_multiview_captioner

    items = groups(side_root, tolerance_us)
    prompt = PROMPT_MANIFEST.read_text()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path, manifest_path = OUT / "captions.csv", OUT / "caption_manifest.json"
    manifest = {
        "source": "RobotCar overcast autumn stereo_centre, mono_left, mono_right, mono_rear",
        "reference_direction_included": "mono_rear_non_database_frame",
        "model": QWEN_VL_MODEL_ID, "prompt": prompt,
        "load_in_4bit": load_in_4bit,
        "max_pixels_per_image": 262144,
        "generation": {"max_new_tokens": 256, "do_sample": False},
        "timestamp_tolerance_us": tolerance_us,
        "groups": [{"query": name, "views": [
            {"camera": camera, "path": str(path),
             "delta_us": int(path.stem) - int(Path(name).stem)} for camera, path in views]}
                   for name, views in items],
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            if csv_path.exists():
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    if any(csv.DictReader(handle)):
                        raise ValueError("Existing caption manifest has different input images or settings")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    completed = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        completed = {row["image_name"]: row["description"] for row in rows}
        assert len(rows) == len(completed) and all(completed.values())
        assert set(completed).issubset({name for name, _ in items})
    pending = [(name, views) for name, views in items if name not in completed]
    if limit:
        pending = pending[:limit]
    print(f"RobotCar four-view: {len(completed)}/866 complete; {len(pending)} this run", flush=True)
    if not pending:
        return
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    captioner = load_qwen_multiview_captioner("cuda", max_pixels=262144,
                                               load_in_4bit=load_in_4bit)
    exists = csv_path.exists()
    with csv_path.open("a" if exists else "x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["image_name", "description"])
            handle.flush()
        for index, (name, views) in enumerate(pending, 1):
            for attempt in range(3):
                instruction = prompt if attempt == 0 else (
                    prompt + "\n\nWrite a shorter complete description of at most "
                    f"{50 - 10 * (attempt - 1)} words using only visible details.")
                description = captioner(views, prompt=instruction, max_new_tokens=256).strip()
                if description and len(tokenizer.encode(description)) + 2 <= 77:
                    break
            else:
                raise RuntimeError(f"Caption exceeds CLIP context: {name}")
            writer.writerow([name, description])
            handle.flush()
            if index % 20 == 0 or index == len(pending):
                print(f"Generated {index}/{len(pending)}", flush=True)


def evaluate():
    import open_clip
    import torch
    from run_lost_robotcar_retrieval_eval import load_poses
    from hloc.clip_lora_checkpoint import load_openclip_lora_checkpoint

    run = run_info()
    q, db = run["queries"], run["database"]
    with (OUT / "captions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    captions = {row["image_name"]: row["description"] for row in rows}
    assert len(rows) == len(q) == len(captions) and set(captions) == set(q)
    ckpt_path = Path(run["checkpoint"])
    model, tokenizer, _, _ = load_openclip_lora_checkpoint(ckpt_path, "cuda")
    vectors = []
    with torch.no_grad():
        for start in range(0, len(q), 32):
            emb = model.encode_text(tokenizer([captions[name] for name in q[start:start + 32]]).cuda())
            vectors.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
    text_features = np.concatenate(vectors)
    with np.load(SOURCE / "Oxford_features.npz") as features:
        assert q == features["query_names"].tolist()
        assert db == features["database_names"].tolist()
        db_features = features["database_features"]
    order = np.argsort(-(text_features @ db_features.T), axis=1)[:, :20]
    poses = load_poses(PREPARED / "poses.csv")
    qp, dp = np.stack([poses[name] for name in q]), np.stack([poses[name] for name in db])
    distances = np.linalg.norm(qp[:, None] - dp[order], axis=-1)
    results = []
    for k in (1, 5, 10, 20):
        best = distances[:, :k].min(axis=1)
        row = {"K": k, "queries": len(q), "database": len(db),
               **{f"R@{metres}m": 100 * float(np.mean(best <= metres))
                  for metres in (5, 10, 20)}}
        results.append(row)
        print(row, flush=True)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (OUT / "per_query.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_image", "description", "top1_reference",
                         "top1_distance_m", "best_top20_distance_m"])
        for index, name in enumerate(q):
            writer.writerow([name, captions[name], db[order[index, 0]],
                             float(distances[index, 0]), float(distances[index].min())])
    (OUT / "evaluation_manifest.json").write_text(json.dumps({
        "source": "RobotCar stereo_centre, mono_left, mono_right, mono_rear; same-traversal reference direction included without exact database frames",
        "queries": len(q), "database": len(db),
        "checkpoint": str(ckpt_path), "checkpoint_sha256": sha(ckpt_path),
        "caption_csv_sha256": sha(OUT / "captions.csv"),
        "caption_manifest_sha256": sha(OUT / "caption_manifest.json"),
        "caption_model_load_in_4bit": json.loads((OUT / "caption_manifest.json").read_text())["load_in_4bit"],
        "notes": "All cameras and the rear observation come from the same traversal; exact database rear frames are excluded, but adjacent rear frames are visually and geographically close. This is a privileged complete-surroundings condition.",
    }, indent=2) + "\n")


def main():
    global SOURCE, PROMPT_MANIFEST
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "generate", "evaluate"])
    parser.add_argument("--side-root", type=Path, default=PREPARED / "images")
    parser.add_argument("--timestamp-tolerance-us", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--source", type=Path, default=SOURCE, help="Local feature directory with manifest.json and dataset feature NPZ files")
    parser.add_argument("--prompt-file", type=Path, help="Local plain-text multi-view caption prompt; required for generation")
    args = parser.parse_args()
    SOURCE, PROMPT_MANIFEST = args.source, args.prompt_file
    if args.stage == "generate" and args.prompt_file is None:
        parser.error("generate requires --prompt-file")
    OUT = args.out_dir
    if args.limit < 0 or args.timestamp_tolerance_us < 0:
        parser.error("--limit and --timestamp-tolerance-us must be nonnegative")
    if args.stage == "check":
        items = groups(args.side_root, args.timestamp_tolerance_us)
        database = set(run_info()["database"])
        maximum_delta_us = {
            camera: max(abs(int(path.stem) - int(Path(name).stem))
                        for name, views in items for direction, path in views
                        if direction == camera)
            for camera in ("mono_left", "mono_right", "mono_rear")
        }
        exact_rear_overlap = sum(
            f"mono_rear/{views[-1][1].name}" in database for _, views in items)
        report = {
            "groups": len(items), "database": len(database),
            "maximum_timestamp_delta_us": maximum_delta_us,
            "query_rear_frames_identical_to_database_frame": exact_rear_overlap,
            "note": "The rear observation is drawn from the same traversal, with exact database rear frames excluded.",
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "alignment_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    elif args.stage == "generate":
        generate(args.side_root, args.timestamp_tolerance_us, args.limit,
                 args.load_in_4bit)
    else:
        evaluate()


if __name__ == "__main__":
    main()
