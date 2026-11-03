"""Time each query continuously from supplied text/image to location or pose.

Reference CLIP/NetVLAD/SuperPoint descriptors and the SfM map are prepared
offline. Models, descriptors, and the map are loaded once before timing.
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import open_clip
import pycolmap
import torch
import torch.nn.functional as F

from hloc import extract_features, extractors, matchers
from hloc.extractors.netvlad import NetVLAD
from hloc.utils.base_model import dynamic_load
from hloc.utils.parsers import parse_image_lists
from scripts.paper_geometry import make_ref_input
from train_localclip import inject_lora_openclip


BASE = Path("outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025")
CLEAN = Path("outputs/SensorKit_hloc/netvlad_corruption_stress_full/clean")
DATASET = Path("datasets/SensorKit")
QUERY_IMAGES = DATASET / "query/cam1_all/0_undistorted"
TEXT_INDEX = Path("outputs/paper_retrieval/global_database.npz")
CHECKPOINT = Path("checkpoints/text8_visual12_epoch22.pt")
OUT = Path("outputs/text_depth_ablation_v1/online_end_to_end")


def sync():
    torch.cuda.synchronize()


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()),
            "max_ms": float(a.max()), "n": len(a)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=493)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rotate-method-order", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET)
    parser.add_argument("--map-dir", type=Path, default=BASE)
    parser.add_argument("--query-assets", type=Path, default=CLEAN)
    parser.add_argument("--text-index", type=Path, default=TEXT_INDEX)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    dataset = args.dataset_root
    base = args.map_dir
    clean = args.query_assets
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    with (base / "qwen3_query_prompts_qwen-vl_77clip10.csv").open(newline="") as f:
        captions = list(csv.DictReader(f))
    queries = parse_image_lists(clean / "queries_with_intrinsics.txt", with_intrinsics=True)
    assert len(captions) == len(queries) == 493
    assert [r["image_name"] for r in captions] == [name for name, _ in queries]
    assert 0 < args.limit <= len(queries) and 0 <= args.warmup < args.limit

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    clip, _, _ = open_clip.create_model_and_transforms(
        ckpt["model_name"], pretrained=ckpt["pretrained"], device="cuda")
    clip = inject_lora_openclip(
        clip, target="all", last_n_blocks=12, r=ckpt["lora_r"],
        alpha=ckpt["lora_alpha"], dropout=ckpt["lora_dropout"],
        text_last_n_blocks=ckpt["lora_text_last_n"],
        visual_last_n_blocks=ckpt["lora_visual_last_n"])
    loaded = clip.load_state_dict(ckpt["lora_state"], strict=False)
    assert not loaded.unexpected_keys
    assert not [key for key in loaded.missing_keys if "lora_" in key]
    clip.cuda().eval()
    tokenizer = open_clip.get_tokenizer(ckpt["model_name"])

    db = np.load(args.text_index)
    db_names = db["names"].tolist()
    text_db = torch.from_numpy(db["features"]).float().cuda()
    with h5py.File(base / "global-feats-netvlad.h5", "r") as h5:
        net_db = torch.from_numpy(np.stack(
            [h5[name]["global_descriptor"][:] for name in db_names])).float().cuda()
    net = NetVLAD({"name": "netvlad"}).eval().cuda()

    sp_conf = extract_features.confs["superpoint_aachen"]
    sp_cls = dynamic_load(extractors, sp_conf["model"]["name"])
    sp = sp_cls(sp_conf["model"]).eval().cuda()
    lg_cls = dynamic_load(matchers, "lightglue")
    lg = lg_cls({"features": "superpoint"}).eval().cuda()
    sp_images = extract_features.ImageDataset(
        (dataset / "query/cam1_all/0_undistorted"), sp_conf["preprocessing"], paths=[q[0] for q in queries])
    net_images = extract_features.ImageDataset(
        (dataset / "query/cam1_all/0_undistorted"), {"resize_max": 1024}, paths=[q[0] for q in queries])

    reconstruction = pycolmap.Reconstruction(base / "sfm_superpointmax+lightglue")
    ref_by_name = {image.name: image for image in reconstruction.images.values()}
    registered = torch.tensor([name in ref_by_name for name in db_names], device="cuda")
    ref_point_ids = {
        name: np.asarray([point.point3D_id if point.has_point3D() else -1
                          for point in ref_by_name[name].points2D], dtype=np.int64)
        for name in db_names if name in ref_by_name
    }
    geo = {}
    with (dataset / "0620_gt/enu_all_mapping_gt.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            geo[row["camera_id"] + "/" + row["image_id"]] = np.asarray(
                [float(row[f"{axis}_enu"]) for axis in "XYZ"])
    assert all(name in geo for name in db_names)

    def retrieve_text(description):
        t0 = time.perf_counter()
        tokens = tokenizer([description]).cuda()
        embedding = F.normalize(clip.encode_text(tokens).float(), dim=-1)
        sync()
        t1 = time.perf_counter()
        scores = embedding @ text_db.T
        chosen = scores.topk(20, dim=-1).indices[0].tolist()
        location = geo[db_names[chosen[0]]]
        sync()
        t2 = time.perf_counter()
        assert location.shape == (3,)
        return chosen, {"global_extraction": (t1-t0)*1000,
                        "retrieval": (t2-t1)*1000, "coarse": (t2-t0)*1000}

    def retrieve_netvlad(index):
        t0 = time.perf_counter()
        raw = net_images[index]
        image = torch.from_numpy(raw["image"]).unsqueeze(0).float().cuda()
        embedding = net({"image": image})["global_descriptor"]
        sync()
        t1 = time.perf_counter()
        scores = (embedding @ net_db.T).masked_fill(~registered[None, :], -torch.inf)
        chosen = scores.topk(20, dim=-1).indices[0].tolist()
        sync()
        t2 = time.perf_counter()
        return chosen, {"global_extraction": (t1-t0)*1000,
                        "retrieval": (t2-t1)*1000, "coarse": (t2-t0)*1000}

    def fine(index, chosen, query_camera, ref_h5):
        t0 = time.perf_counter()
        raw = sp_images[index]
        image = torch.from_numpy(raw["image"]).unsqueeze(0).cuda()
        pred = {key: value[0] for key, value in sp({"image": image}).items()}
        original_size = raw["original_size"]
        current_size = np.asarray(raw["image"].shape[-2:][::-1])
        scales = torch.tensor(original_size / current_size, dtype=torch.float32,
                              device="cuda")
        pred["keypoints"] = (pred["keypoints"] + 0.5) * scales - 0.5
        query_data = {key + "0": value.unsqueeze(0) for key, value in pred.items()}
        query_data["image0"] = torch.empty(
            (1, 1, int(original_size[1]), int(original_size[0])), device="cuda")
        sync()
        t1 = time.perf_counter()

        kp_to_ids = defaultdict(list)
        for db_idx in chosen:
            name = db_names[db_idx]
            if name not in ref_by_name:
                continue
            ref_ids = ref_point_ids[name]
            if not np.any(ref_ids != -1):
                continue
            data = {**query_data, **make_ref_input(ref_h5[name])}
            matches = lg(data)["matches0"][0].cpu().numpy()
            valid_q = np.flatnonzero(matches >= 0)
            valid_q = valid_q[ref_ids[matches[valid_q]] != -1]
            for qi in valid_q:
                pid = int(ref_ids[matches[qi]])
                if pid not in kp_to_ids[int(qi)]:
                    kp_to_ids[int(qi)].append(pid)
        sync()
        t2 = time.perf_counter()

        q_indices = [qi for qi, ids in kp_to_ids.items() for _ in ids]
        pids = [pid for ids in kp_to_ids.values() for pid in ids]
        if pids:
            kp = pred["keypoints"].cpu().numpy() + 0.5
            points2d = kp[q_indices]
            points3d = [reconstruction.points3D[pid].xyz for pid in pids]
            ret = pycolmap.estimate_and_refine_absolute_pose(
                points2d, points3d, query_camera,
                estimation_options={"ransac": {"max_error": 12}})
        else:
            ret = None
        first_registered = next((db_names[idx] for idx in chosen if db_names[idx] in ref_by_name), None)
        if ret is None and first_registered is None:
            raise RuntimeError("No registered reference among Top-20")
        pose = ret["cam_from_world"] if ret is not None else ref_by_name[first_registered].cam_from_world
        center = -pose.rotation.matrix().T @ pose.translation
        assert np.all(np.isfinite(center))
        t3 = time.perf_counter()
        return {"local_extraction": (t1-t0)*1000,
                "matching_and_correspondence": (t2-t1)*1000,
                "pose_estimation": (t3-t2)*1000,
                "fine": (t3-t0)*1000}, center, ret is not None

    rows = []
    with h5py.File(base / "feats-superpoint-n4096-r1024.h5", "r") as ref_h5:
        for index, (name, camera) in enumerate(queries[:args.limit]):
            methods = ("text_only", "text_image", "hloc")
            if args.rotate_method_order:
                methods = methods[index % len(methods):] + methods[:index % len(methods)]
            for method in methods:
                start = time.perf_counter()
                if method == "hloc":
                    chosen, stage = retrieve_netvlad(index)
                else:
                    chosen, stage = retrieve_text(captions[index]["description"])
                if method != "text_only":
                    fine_stage, center, pnp_success = fine(index, chosen, camera, ref_h5)
                    stage.update(fine_stage)
                    stage["pnp_success"] = pnp_success
                    stage["estimated_x"] = float(center[0])
                    stage["estimated_y"] = float(center[1])
                    stage["estimated_z"] = float(center[2])
                else:
                    stage["estimated_x"], stage["estimated_y"], stage["estimated_z"] = geo[db_names[chosen[0]]].tolist()
                sync()
                stage["end_to_end"] = (time.perf_counter()-start)*1000
                stage.update({"query": name, "method": method,
                              "warmup": index < args.warmup,
                              "top1": db_names[chosen[0]],
                              "top20": "|".join(db_names[j] for j in chosen)})
                rows.append(stage)
            if (index+1) % 10 == 0 or index+1 == args.limit:
                print(f"Completed {index+1}/{args.limit} queries", flush=True)
                with (args.output_dir / "per_query.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(
                        key for row in rows for key in row)))
                    writer.writeheader()
                    writer.writerows(rows)

    result = {"device": torch.cuda.get_device_name(0),
              "checkpoint": str(args.checkpoint), "queries_total": args.limit,
              "warmup_queries": args.warmup,
              "timed_queries_per_method": args.limit-args.warmup,
              "rotated_method_order": args.rotate_method_order,
              "offline": "models, reference global descriptors, SfM map and geo-tags loaded before timing; precomputed reference local-feature HDF5 opened once",
              "online": "existing text string tokenization/encoding; JPEG decode and feature extraction for query images; Top20 search; per-candidate HDF5 reference-feature reads; LightGlue matching; pooled 2D-3D correspondences; PyCOLMAP RANSAC pose",
              "excluded": "caption generation, speech recognition, model/map load, output-file writing",
              "methods": {}}
    for method in ("text_only", "text_image", "hloc"):
        subset = [row for row in rows if row["method"] == method and not row["warmup"]]
        keys = ("global_extraction", "retrieval", "coarse", "local_extraction",
                "matching_and_correspondence", "pose_estimation", "fine", "end_to_end")
        result["methods"][method] = {
            key: summary([row[key] for row in subset]) for key in keys if key in subset[0]}
        if method != "text_only":
            result["methods"][method]["pnp_success_count"] = sum(
                row["pnp_success"] for row in subset)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
