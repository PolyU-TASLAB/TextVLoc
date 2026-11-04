"""Reproduce the manuscript retrieval, coarse localization and query-text tables.

Run from the repository root with ``python -m scripts.eval_paper_retrieval``.
The check command validates local evaluation inputs without loading models.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("pretrained", "global", "rerank")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_captions(path, names):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {row["image_name"]: row["description"].strip() for row in rows}
    if len(rows) != len(by_name) or set(by_name) != set(names) or not all(by_name.values()):
        raise ValueError(f"Incomplete, duplicate or empty captions: {path}")
    return [by_name[name] for name in names]


def positive_map(path):
    result = defaultdict(set)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            result[Path(row["anchor"]).name].add(row["positive"].split("allcams/")[-1])
    return result


def positions(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return {f"{r['camera_id']}/{Path(r['image_id']).name}":
                [float(r[f"{axis}_enu"]) for axis in "XYZ"] for r in csv.DictReader(handle)}


def summarize(ranked, queries, database, positives, query_positions, db_positions):
    ranked = np.asarray(ranked)
    if ranked.shape != (len(queries), 20) or ranked.min() < 0 or ranked.max() >= len(database):
        raise ValueError("Expected valid Top-20 database indices for every query")
    if any(len(set(row)) != 20 for row in ranked):
        raise ValueError("Duplicate candidate in a Top-20 list")
    if any(Path(queries[i]).name == Path(database[j]).name
           for i, row in enumerate(ranked) for j in row):
        raise ValueError("Same-basename self match in ranking")
    hits = np.array([[database[j] in positives[Path(q).name] for j in row]
                     for q, row in zip(queries, ranked)], dtype=bool)
    distances = np.linalg.norm(db_positions[ranked] - query_positions[:, None], axis=-1)
    return [{"K": k, "queries": len(queries), "database": len(database),
             "positive_recall_pct": float(100 * hits[:, :k].any(axis=1).mean()),
             **{f"loc_{m}m_pct": float(100 * (distances[:, :k].min(axis=1) < m).mean())
                for m in (5, 10, 15)},
             "median_best_distance_m": float(np.median(distances[:, :k].min(axis=1)))}
            for k in (1, 5, 10, 20)]


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def context(protocol, positive_path):
    qgt = positions(Path(protocol["query_positions"]))
    dgt = positions(Path(protocol["reference_positions"]))
    queries, database = protocol["query_order"], protocol["database_order"]
    qp = np.array([qgt[f"cam1/{Path(q).name}"] for q in queries])
    dp = np.array([dgt[n] for n in database])
    positives = positive_map(positive_path)
    if not all(Path(q).name in positives for q in queries):
        raise ValueError("Positive list does not cover every query")
    return queries, database, positives, qp, dp


def check(args, protocol):
    queries, database, _, _, _ = context(protocol, args.positive_pairs)
    variants = args.variants or ["original_qwen"]
    paths = dict(protocol["caption_paths"])
    for item in args.caption:
        label, separator, path = item.partition("=")
        if not separator or not path:
            raise ValueError("--caption requires VARIANT=CSV")
        paths[label] = path
    for variant in variants:
        read_captions(Path(paths[variant]), queries)
    print(f"PASS: {len(queries)} queries, {len(database)} references, {len(variants)} caption sets")


def evaluate(args, protocol):
    import torch
    import torch.nn.functional as F
    import open_clip
    from PIL import Image
    from hloc.clip_lora_checkpoint import load_openclip_lora_checkpoint

    queries, database, positives, qp, dp = context(protocol, args.positive_pairs)
    if args.query_limit:
        queries, qp = queries[:args.query_limit], qp[:args.query_limit]
    if args.reference_limit:
        database, dp = database[:args.reference_limit], dp[:args.reference_limit]
    if len(database) < 20:
        raise ValueError("At least 20 references required")
    caption_paths = {v: ROOT / p for v, p in protocol["caption_paths"].items()}
    overridden = set()
    for item in args.caption:
        label, separator, path = item.partition("=")
        if not separator or not path or not label or label in overridden:
            raise ValueError("--caption requires a unique VARIANT=CSV")
        caption_paths[label] = Path(path)
        overridden.add(label)
    variants = args.variants or ["original_qwen"]
    if any(v not in caption_paths for v in variants):
        raise ValueError("Unknown caption variant")
    texts = {}
    for variant in variants:
        all_texts = read_captions(caption_paths[variant], protocol["query_order"])
        texts[variant] = all_texts[:len(queries)]
    out = args.output_dir
    if out is None:
        raise ValueError("evaluate requires --output-dir")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = args.device
    excluded = torch.tensor([[Path(q).name == Path(d).name for d in database]
                             for q in queries], device=device)
    rows, archives, model_info = [], {}, {}

    @torch.inference_mode()
    def image_features(model, preprocess, selected, *, regions=False, batch_size=32):
        batches = []
        for start in range(0, len(selected), batch_size):
            tensors = []
            for name in selected[start:start + batch_size]:
                with Image.open(args.dataset_root / "allcams" / name) as source:
                    image = source.convert("RGB")
                if regions:
                    width, height = image.size
                    images = [image.crop((x*(width//3), y*(height//3),
                                         width if x == 2 else (x+1)*(width//3),
                                         height if y == 2 else (y+1)*(height//3)))
                              for y in range(3) for x in range(3)]
                else:
                    images = [image]
                tensors.extend(preprocess(im) for im in images)
            batch = torch.stack(tensors).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=regions and str(device).startswith("cuda")):
                features = model.encode_image(batch)
            features = F.normalize(features.float(), dim=-1)
            if regions:
                features = features.reshape(-1, 9, features.shape[-1]).half()
            batches.append(features.cpu())
            if start % 320 == 0:
                print(f"{'Regions' if regions else 'References'}: {min(start+batch_size,len(selected))}/{len(selected)}", flush=True)
        return torch.cat(batches).numpy()

    with torch.inference_mode():
        for arm in ("pretrained", "global"):
            if arm == "pretrained" and arm not in args.methods:
                continue
            if arm == "global" and not set(args.methods) & {"global", "rerank"}:
                continue
            if arm == "pretrained":
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-L-14", pretrained="datacomp_xl_s13b_b90k", device=device)
                model.eval()
                tokenizer = open_clip.get_tokenizer("ViT-L-14")
            else:
                model, tokenizer, preprocess, ckpt = load_openclip_lora_checkpoint(args.checkpoint, device)
                model_info = {k: ckpt.get(k) for k in ("model_name", "pretrained", "epoch", "lora_target",
                              "lora_text_last_n", "lora_visual_last_n", "lora_r", "lora_alpha")}
            full = image_features(model, preprocess, database, batch_size=args.batch_size)
            np.savez_compressed(out / f"{arm}_database.npz", names=np.array(database), features=full)
            db = torch.from_numpy(full).to(device)
            candidates, text_features = {}, {}
            for variant, descriptions in texts.items():
                vectors = []
                for start in range(0, len(descriptions), 32):
                    tokens = tokenizer(descriptions[start:start+32]).to(device)
                    vectors.append(F.normalize(model.encode_text(tokens).float(), dim=-1))
                query = torch.cat(vectors)
                scores = (query @ db.T).masked_fill(excluded, -torch.inf)
                selected = scores.topk(20, dim=1).indices.cpu().numpy()
                candidates[variant], text_features[variant] = selected, query.cpu().numpy()
                if arm in args.methods:
                    archives[f"{variant}__{arm}"] = selected
                    rows.extend({"variant": variant, "method": arm, **r} for r in
                                summarize(selected, queries, database, positives, qp, dp))
            if arm == "global" and "rerank" in args.methods:
                pool = np.unique(np.concatenate(list(candidates.values())))
                selected_names = [database[j] for j in pool]
                # The manuscript recomputed candidate full images with batch size 4.
                rerank_full = image_features(model, preprocess, selected_names, batch_size=4)
                regions = image_features(model, preprocess, selected_names, regions=True, batch_size=4)
                np.savez_compressed(out / "candidate_features.npz", db_indices=pool, full=rerank_full, regions=regions)
                for variant in variants:
                    candidate, query = candidates[variant], text_features[variant]
                    index = np.searchsorted(pool, candidate)
                    global_scores = np.einsum("qd,qkd->qk", query, rerank_full[index])
                    local_scores = np.empty_like(global_scores)
                    for start in range(0, len(queries), 32):
                        stop = min(start+32, len(queries))
                        local_scores[start:stop] = np.einsum("qd,qkrd->qkr", query[start:stop],
                            regions[index[start:stop]].astype(np.float32)).max(axis=-1)
                    order = np.argsort(-(0.75*global_scores+0.25*local_scores), axis=1, kind="stable")
                    ranked = np.take_along_axis(candidate, order, axis=1)
                    archives[f"{variant}__rerank"] = ranked
                    rows.extend({"variant": variant, "method": "rerank", **r} for r in
                                summarize(ranked, queries, database, positives, qp, dp))
                    if args.alpha_sweep and variant == "original_qwen":
                        sweep = []
                        for alpha in (0, .25, .5, .75, 1):
                            order = np.argsort(-(alpha*global_scores+(1-alpha)*local_scores), axis=1, kind="stable")
                            rank = np.take_along_axis(candidate, order, axis=1)
                            sweep.extend({"alpha": alpha, **r} for r in summarize(rank, queries, database, positives, qp, dp))
                        write_csv(out / "alpha_sensitivity.csv", sweep)
            del model, db
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    write_csv(out / "summary.csv", rows)
    np.savez_compressed(out / "rankings.npz", **archives)
    for key, ranked in archives.items():
        with (out / f"{key}__hloc_pairs.txt").open("w") as handle:
            for name, rank in zip(queries, ranked):
                for j in rank:
                    handle.write(f"{name} {database[j]}\n")
    (out / "manifest.json").write_text(json.dumps({
        "query_order": queries, "database_order": database, "variants": variants,
        "methods": args.methods, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint), "model": model_info,
        "positive_pairs": str(args.positive_pairs), "positive_pairs_sha256": sha256(args.positive_pairs),
        "caption_paths": {v: str(caption_paths[v]) for v in variants},
        "caption_sha256": {v: sha256(caption_paths[v]) for v in variants},
        "smoke_only": bool(args.query_limit or args.reference_limit),
        "grid": 3, "region_reduce": "max", "global_weight": .75,
        "region_encoding": "fp16 autocast, normalized and stored fp16", "global_encoding": "fp32",
    }, indent=2) + "\n")
    print(f"Wrote {len(rows)} metric rows to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "evaluate"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--positive-pairs", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/SensorKit")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/text8_visual12_epoch22.pt")
    parser.add_argument("--query-captions", type=Path, default=ROOT / "outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/qwen3_query_prompts_qwen-vl_77clip10.csv")
    parser.add_argument("--query-positions", type=Path)
    parser.add_argument("--reference-positions", type=Path)
    parser.add_argument("--database-list", type=Path, help="Optional newline-separated reference names; otherwise use non-cam1 mapping poses")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--caption", action="append", default=[], metavar="VARIANT=CSV",
                        help="Use a local caption CSV for a variant")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-limit", type=int, default=0, help="Smoke check only; 0 = all 493")
    parser.add_argument("--reference-limit", type=int, default=0, help="Smoke check only; 0 = all 2952")
    parser.add_argument("--alpha-sweep", action="store_true", help="Report original-query alpha sensitivity; does not select a new default")
    args = parser.parse_args()
    if args.batch_size < 1 or args.query_limit < 0 or args.reference_limit < 0:
        parser.error("Batch size must be positive; limits must be nonnegative")
    args.positive_pairs = args.positive_pairs or args.dataset_root / "query/cam1_all/05_allcams_positive_pairs_for_eval.csv"
    args.query_positions = args.query_positions or args.dataset_root / "0620_gt/enu_all_query_gt.csv"
    args.reference_positions = args.reference_positions or args.dataset_root / "0620_gt/enu_all_mapping_gt.csv"
    with args.query_captions.open(newline="", encoding="utf-8-sig") as handle:
        names = [r["image_name"] for r in csv.DictReader(handle)]
    if not names or len(names) != len(set(names)):
        raise ValueError("Query names must be nonempty and unique")
    database = sorted(n for n in positions(args.reference_positions) if not n.startswith("cam1/"))
    if args.database_list:
        database = [line.strip() for line in args.database_list.read_text().splitlines() if line.strip()]
    if not database or len(database) != len(set(database)):
        raise ValueError("Reference names must be nonempty and unique")
    variants_root = ROOT / "outputs/query_text_generalization"
    caption_paths = {"original_qwen": str(args.query_captions)}
    caption_paths.update({v: str(variants_root / "qwen3_prompt_v1" / f"{v}.csv") for v in
                          ("natural_static", "third_person_static", "brief_static", "noun_phrases")})
    caption_paths.update({
        "synthetic_casual": str(variants_root / "qwen3_synthetic_user_v1/synthetic_casual.csv"),
        "llava_original_prompt": str(variants_root / "llava_mistral7b_qwen_prompt_v3/original_prompt.csv"),
        "claude_opus55": str(variants_root / "claude_opus55_qwen_prompt_v1/original_prompt.csv"),
        "deepseek_flash": str(ROOT / "outputs/cam1_deepseek_query_descriptions/descriptions.csv"),
        "gemma3_12b": str(ROOT / "outputs/cam1_gemma3_query_descriptions/descriptions.csv"),
        **{f"controlled_{v}": str(ROOT / f"outputs/paper_text_perturbations/controlled_{v}.csv") for v in
           ("incomplete", "color", "detail")},
    })
    protocol = {"query_order": names, "database_order": database, "caption_paths": caption_paths,
                "query_positions": str(args.query_positions), "reference_positions": str(args.reference_positions)}
    (check if args.command == "check" else evaluate)(args, protocol)


if __name__ == "__main__":
    main()
