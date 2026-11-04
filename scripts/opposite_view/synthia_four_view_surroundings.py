"""SYNTHIA front/back/left/right surroundings captions and retrieval."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SOURCE = Path("outputs/text_depth_ablation_v1/three_datasets_text8_visual12")
OUT = Path("outputs/opposite_view_synthia_four_view_surroundings")
ROOT = Path("datasets/SYNTHIA/SEQS-02")
EXTRACTED = Path("outputs/opposite_view_synthia_three_view_no_front/extracted")
OLD_PROMPT_MANIFEST = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_info():
    runs = json.loads((SOURCE / "manifest.json").read_text())["runs"]
    run = next(run for run in runs if run["dataset"] == "SYNTHIA dawn→dawn")
    run["checkpoint"] = "checkpoints/text8_visual12_epoch22.pt"
    return run


def groups():
    run = run_info()
    result = []
    for query in run["queries"]:
        assert "/Omni_B/" in query
        views = []
        for direction in ("Omni_B", "Omni_L", "Omni_R", "Omni_F"):
            base = EXTRACTED if direction in ("Omni_L", "Omni_R") else ROOT
            path = base / query.replace("/Omni_B/", f"/{direction}/")
            assert path.is_file() and path.stat().st_size > 0, path
            views.append((direction, path))
        result.append((query, views))
    assert len(result) == 95
    return result


def generate(limit):
    import open_clip
    from hloc.generate_multiview_captions import QWEN_VL_MODEL_ID, load_qwen_multiview_captioner

    items = groups()
    prompt = OLD_PROMPT_MANIFEST.read_text()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path, manifest_path = OUT / "captions.csv", OUT / "caption_manifest.json"
    manifest = {
        "source": "SYNTHIA Omni_B, Omni_L, Omni_R, Omni_F from the query sequence",
        "reference_direction_included": "Omni_F", "model": QWEN_VL_MODEL_ID,
        "prompt": prompt, "max_pixels_per_image": 262144,
        "generation": {"max_new_tokens": 256, "do_sample": False},
        "groups": [{"query": name,
                    "views": [{"direction": direction, "path": str(path)} for direction, path in views]}
                   for name, views in items],
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
    pending = [(name, views) for name, views in items if name not in completed]
    if limit:
        pending = pending[:limit]
    print(f"SYNTHIA four-view: {len(completed)}/95 complete; {len(pending)} this run", flush=True)
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
        for index, (name, views) in enumerate(pending, 1):
            for attempt in range(3):
                instruction = prompt if attempt == 0 else (
                    prompt + "\n\nWrite a shorter complete description of at most "
                    f"{50 - 10 * (attempt - 1)} words using only the most distinctive visible details.")
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
    with np.load(SOURCE / "SYNTHIA_features.npz") as features:
        assert q == features["query_names"].tolist()
        assert db == features["database_names"].tolist()
        db_features = features["database_features"]
    order = np.argsort(-(text_features @ db_features.T), axis=1)[:, :20]
    pose_csv = Path("outputs/SYNTHIA/seq02_lost/poses.csv")
    poses = load_poses(pose_csv)
    qp = np.stack([poses[name] for name in q])
    dp = np.stack([poses[name] for name in db])
    distances = np.linalg.norm(qp[:, None] - dp[order], axis=-1)
    result = []
    for k in (1, 5, 10, 20):
        best = distances[:, :k].min(axis=1)
        row = {"K": k, "queries": len(q), "database": len(db),
               **{f"R@{metres}m": 100 * float(np.mean(best <= metres))
                  for metres in (5, 10, 20)}}
        result.append(row)
        print(row, flush=True)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    with (OUT / "per_query.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_image", "description", "top1_reference",
                         "top1_distance_m", "best_top20_distance_m"])
        for index, name in enumerate(q):
            writer.writerow([name, captions[name], db[order[index, 0]],
                             float(distances[index, 0]), float(distances[index].min())])
    (OUT / "evaluation_manifest.json").write_text(json.dumps({
        "source": "SYNTHIA Omni_B, Omni_L, Omni_R, Omni_F; same-frame reference direction included",
        "queries": len(q), "database": len(db),
        "checkpoint": str(ckpt_path), "checkpoint_sha256": digest(ckpt_path),
        "caption_csv_sha256": digest(OUT / "captions.csv"),
        "caption_manifest_sha256": digest(OUT / "caption_manifest.json"),
        "notes": "Same-frame Omni_F imagery used for captions also appears in the reference database. This is a privileged complete-surroundings condition.",
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
