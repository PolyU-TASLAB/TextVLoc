import argparse
import bisect
import csv
import os
import textwrap
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from . import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

QWEN_VL_MODEL_ID = os.environ.get("QWEN_VL_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")

MULTIVIEW_NAV_PROMPT = textwrap.dedent(
    """
    You are a navigation assistant observing synchronized camera views from the same vehicle pose.
    Each image is a different camera direction around the vehicle.
    Write one concise scene description that summarizes the permanent/static surroundings visible across all views.
    Do not mention people, cars, buses, weather, sky, image quality, camera names, or that multiple images were provided.
    Focus on landmarks, buildings, road geometry, intersections, traffic signs/signals, vegetation, street furniture, pavement, and distinctive materials.
    Use first-person navigation language such as "I am near...", "ahead...", "to my left...", and "to my right..." when direction is clear.
    The entire description must be a single complete sentence under 77 words.
    """
).strip()


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _timestamp(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Image filename stem must be an integer timestamp: {path}") from exc


def _list_camera_images(camera_dir: Path) -> list[Path]:
    if not camera_dir.exists():
        return []
    return sorted(
        [p for p in camera_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=_timestamp,
    )


def _nearest_image(images: list[Path], timestamps: list[int], target: int, max_delta_ns: int) -> Path | None:
    if not images:
        return None
    idx = bisect.bisect_left(timestamps, target)
    candidates = []
    if idx < len(images):
        candidates.append(images[idx])
    if idx > 0:
        candidates.append(images[idx - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda p: abs(_timestamp(p) - target))
    if abs(_timestamp(best) - target) > max_delta_ns:
        return None
    return best


def build_groups(
    *,
    multi_cam_root: Path,
    cameras: list[str],
    anchor_camera: str,
    anchor_dir: Path | None,
    max_delta_ns: int,
    min_cameras: int,
) -> list[dict]:
    camera_images = {}
    camera_timestamps = {}
    for cam in cameras:
        imgs = _list_camera_images(multi_cam_root / cam)
        camera_images[cam] = imgs
        camera_timestamps[cam] = [_timestamp(p) for p in imgs]
        logger.info("Camera %s: %d images", cam, len(imgs))

    anchors = _list_camera_images(anchor_dir) if anchor_dir else camera_images.get(anchor_camera, [])
    if not anchors:
        raise ValueError("No anchor images found.")

    groups = []
    for anchor in anchors:
        anchor_ts = _timestamp(anchor)
        views = []
        for cam in cameras:
            if anchor_dir is not None and cam == anchor_camera and anchor.exists():
                # Keep the exact query image for the anchor view.
                candidate = anchor
            else:
                candidate = _nearest_image(
                    camera_images.get(cam, []),
                    camera_timestamps.get(cam, []),
                    anchor_ts,
                    max_delta_ns,
                )
            if candidate is not None:
                views.append((cam, candidate))
        if len(views) >= min_cameras:
            image_name = anchor.name if anchor_dir else f"{anchor_camera}/{anchor.name}"
            groups.append(
                {
                    "image_name": image_name,
                    "timestamp_ns": anchor_ts,
                    "views": views,
                }
            )
    return groups


def load_processed(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    processed = set()
    with output_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row.get("image_name")
            if image_name:
                processed.add(image_name)
    return processed


def load_qwen_multiview_captioner(device: str, *, max_pixels=None, load_in_4bit=False):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from transformers import BitsAndBytesConfig

    logger.info("Loading Qwen-VL model: %s", QWEN_VL_MODEL_ID)
    processor = AutoProcessor.from_pretrained(
        QWEN_VL_MODEL_ID, trust_remote_code=True,
        **({"max_pixels": max_pixels} if max_pixels is not None else {}))
    if max_pixels is not None:
        processor.image_processor.size = {
            "shortest_edge": min(65536, max_pixels), "longest_edge": max_pixels}
    model = AutoModelForVision2Seq.from_pretrained(
        QWEN_VL_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        **({"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )} if load_in_4bit else {}),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    tokenizer = processor.tokenizer
    eos_id = tokenizer.eos_token_id

    def caption(views: list[tuple[str, Path]], *, prompt=None, max_new_tokens=112) -> str:
        content = [] if prompt is not None else [
            {
                "type": "text",
                "text": (
                    "The following images are different camera views captured at the same vehicle pose. "
                    "Use all views to describe the surrounding place."
                ),
            }
        ]
        images = []
        for index, (cam, image_path) in enumerate(views, start=1):
            content.append({"type": "text", "text": f"View {index} ({cam}):"})
            content.append({"type": "image"})
            images.append(Image.open(image_path).convert("RGB"))
        content.append({"type": "text", "text": MULTIVIEW_NAV_PROMPT if prompt is None else prompt})

        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
            )
        new_ids = out[0, inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return caption


def main(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cameras = args.cameras or sorted(
        [p.name for p in args.multi_cam_root.iterdir() if p.is_dir() and p.name.startswith("cam")]
    )
    if args.anchor_camera not in cameras:
        cameras = [args.anchor_camera, *cameras]

    groups = build_groups(
        multi_cam_root=args.multi_cam_root,
        cameras=cameras,
        anchor_camera=args.anchor_camera,
        anchor_dir=args.anchor_dir,
        max_delta_ns=args.max_delta_ns,
        min_cameras=args.min_cameras,
    )
    if args.max_groups:
        groups = groups[: args.max_groups]

    processed = load_processed(args.output_csv)
    groups = [g for g in groups if g["image_name"] not in processed]
    logger.info("Remaining groups to caption: %d", len(groups))
    if not groups:
        return

    caption = load_qwen_multiview_captioner(device)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists()
    with args.output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writerow(["image_name", "timestamp_ns", "cameras", "image_paths", "description"])

        for idx, group in enumerate(tqdm(groups, desc="qwen-vl-multiview"), start=1):
            views = group["views"]
            try:
                desc = caption(views)
            except Exception as exc:
                logger.error("Failed %s: %s", group["image_name"], exc)
                continue
            if not desc:
                logger.warning("Empty caption: %s", group["image_name"])
                continue
            writer.writerow(
                [
                    group["image_name"],
                    group["timestamp_ns"],
                    ";".join(cam for cam, _ in views),
                    ";".join(str(path) for _, path in views),
                    desc,
                ]
            )
            f.flush()
            if idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate one Qwen-VL caption from synchronized multi-camera views.")
    parser.add_argument(
        "--multi_cam_root",
        type=Path,
        default=Path("datasets/SensorKit/allcams"),
        help="Root containing camera subdirectories such as cam0, cam1, ...",
    )
    parser.add_argument(
        "--anchor_dir",
        type=Path,
        help="Optional query image directory. Output image_name follows these anchor filenames.",
    )
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", help="Camera directories to use. Default: all cam* under multi_cam_root.")
    parser.add_argument("--anchor_camera", default="cam1")
    parser.add_argument("--max_delta_ns", type=int, default=80_000_000, help="Max timestamp mismatch per camera.")
    parser.add_argument("--min_cameras", type=int, default=2, help="Skip anchors with fewer matched views.")
    parser.add_argument("--max_groups", type=int, help="Debug limit.")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Default: auto.")
    main(parser.parse_args())
