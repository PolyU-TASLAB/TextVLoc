import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import csv
import textwrap
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    BlipForConditionalGeneration,
    BlipProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from . import logger

# Qwen3-VL：模型 ID 可用环境变量覆盖；prompt 与 clip 管线常用导航描述一致
QWEN_VL_MODEL_ID = os.environ.get("QWEN_VL_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
QWEN_VL_NAV_PROMPT = textwrap.dedent(
    """
    You are a navigation assistant observing this image.
    Describe the scene as if you're standing at the camera's viewpoint.
    Do **not** mention people, person, cars, bus, or other sentimental description (Sky, clouds, weather, feeling).
    Describe only permanent/static elements, Use first-person perspective ("I'm at...", "On my left...", etc.).
    Focus on:
    - Unique or permanent landmarks (e.g., logos, building, Pavement & Road Markings, Intersection Geometry)
    - Traffic Signs and Signals

    - Road structure or intersections
    - Vegetation & Street Furniture
    - Building materials or signs that help visual recognition

    Keep the description concise and natural. Avoid speculation or subjective opinions.
    **IMPORTANT: The entire description must be a single, complete sentence and strictly under 77 words.**
    """
).strip()

LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
LLAVA_PROMPT_TEMPLATE = """USER: <image>
You are a navigation assistant observing this image.
Describe the scene as if you're standing at the camera's viewpoint.
Use first-person perspective ("I'm at...", "On my left...", etc.).
Focus on:
- Unique or permanent landmarks (e.g., logos, building names, statues)
- Directional relationships (e.g., "on my right", "ahead", "behind me")
- Road structure or intersections
- Building materials or signs that help visual recognition

Keep the description concise and natural. Avoid speculation or subjective opinions.
**IMPORTANT: The entire description must be a single, complete sentence and strictly under 77 words.**
ASSISTANT:"""


def get_captioner(model_name: str, device: str, *, revision=None, load_in_4bit=True):
    """按模型名加载一次模型，返回 caption(image_path) -> str。"""
    logger.info(f"Loading model '{model_name}' onto device '{device}'...")

    if model_name == "blip":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float16,
        ).to(device).eval()

        def caption(image_path: Path) -> str:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=75)
            return processor.decode(out[0], skip_special_tokens=True).strip()

        return caption

    if model_name == "llava":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_ID, revision=revision)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID,
            revision=revision,
            **({"quantization_config": bnb_config} if load_in_4bit else {}),
            device_map=device,
            torch_dtype=torch.float16,
        ).eval()

        def caption(image_path: Path, *, prompt: str = QWEN_VL_NAV_PROMPT,
                    max_new_tokens: int = 96) -> str:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(
                text=text,
                images=image,
                return_tensors="pt",
            ).to(model.device, torch.float16)
            eos_token_id = processor.tokenizer.eos_token_id
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    eos_token_id=eos_token_id, pad_token_id=eos_token_id)
            new_ids = out[0, inputs["input_ids"].shape[-1]:]
            return processor.decode(new_ids, skip_special_tokens=True).strip()

        return caption

    if model_name == "qwen-vl":
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            processor = AutoProcessor.from_pretrained(QWEN_VL_MODEL_ID, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                QWEN_VL_MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            tok = processor.tokenizer
            eos_id = tok.eos_token_id

            def caption(image_path: Path, *, prompt: str = QWEN_VL_NAV_PROMPT) -> str:
                image = Image.open(image_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                inputs = processor(
                    text=[text], images=[image], return_tensors="pt"
                ).to(model.device)
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=96,
                        do_sample=False,
                        eos_token_id=eos_id,
                        pad_token_id=eos_id,
                    )
                new_ids = out[0, inputs["input_ids"].shape[-1] :]
                return tok.decode(new_ids, skip_special_tokens=True).strip()

            return caption

        except Exception as e:
            logger.error(
                f"Failed to load Qwen-VL ({QWEN_VL_MODEL_ID}): {e}. "
                "Check transformers, GPU memory, and model access."
            )
            return lambda _p: ""

    raise ValueError(f"Unsupported caption model: {model_name}")


def _load_image_list(list_file: Path) -> list[Path]:
    paths: list[Path] = []
    with open(list_file, encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                paths.append(Path(p))
    return paths


def main(args):
    """
    模式 A：--query_dir — 递归 *.jpg/*.png，CSV 列为 image_name, description（相对 query_dir）。
    模式 B：--image_list — 每行一个绝对/相对图像路径，CSV 列为 image_path, description（训练管线常用）。
    两种模式均支持断点续写。
    """
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        caption_func = get_captioner(args.model, device)
    except Exception as e:
        logger.error(f"Failed to load model '{args.model}': {e}")
        return

    image_list_path = getattr(args, "image_list", None)
    query_dir = getattr(args, "query_dir", None)

    if image_list_path:
        list_path = Path(image_list_path)
        all_image_paths = _load_image_list(list_path)
        id_key = lambda p: str(p.resolve())
        header = ["image_path", "description"]
        row_id = lambda p: str(p.resolve())
    else:
        if not query_dir:
            logger.error("Provide --query_dir or --image_list.")
            return
        qdir = Path(query_dir)
        all_image_paths = sorted(
            list(qdir.rglob("*.jpg")) + list(qdir.rglob("*.png"))
        )
        id_key = lambda p: p.relative_to(qdir).as_posix()
        header = ["image_name", "description"]
        row_id = lambda p: p.relative_to(qdir).as_posix()

    if not all_image_paths:
        logger.error("No images to process.")
        return

    output_csv_path = Path(args.output_csv)
    file_exists = output_csv_path.exists()
    processed: set[str] = set()

    if file_exists:
        logger.info(f"Resume: reading {output_csv_path} ...")
        with open(output_csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                next(reader)
                for row in reader:
                    if row:
                        processed.add(row[0])
            except StopIteration:
                logger.warning("Output file is empty.")
        logger.info(f"{len(processed)} rows already present.")

    remaining = [p for p in all_image_paths if id_key(p) not in processed]
    if not remaining:
        logger.info("Nothing left to do.")
        return

    logger.info(f"Captioning {len(remaining)} images...")

    mode = "a" if file_exists else "w"
    with open(output_csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(header)

        for i, img_path in enumerate(tqdm(remaining, desc=args.model), start=1):
            key = row_id(img_path)
            try:
                if not img_path.exists():
                    logger.warning(f"Missing file, skip: {img_path}")
                    continue
                desc = caption_func(img_path)
                if desc:
                    writer.writerow([key, desc])
                    f.flush()
                else:
                    logger.warning(f"Empty caption: {key}")
            except Exception as e:
                logger.error(f"Failed {key}: {e}")
            if i % 500 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(f"Done. Output: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image captions (BLIP / LLaVA / Qwen-VL).")
    parser.add_argument("--query_dir", type=Path, help="Directory to scan for *.jpg / *.png (mutual with --image_list).")
    parser.add_argument(
        "--image_list",
        type=Path,
        help="Text file: one image path per line (writes image_path column).",
    )
    parser.add_argument("--output_csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--model",
        type=str,
        default="blip",
        choices=["blip", "llava", "qwen-vl"],
        help="Captioning backend.",
    )
    parser.add_argument("--device", type=str, help="cuda | cpu (default: auto).")
    ns = parser.parse_args()
    if not ns.image_list and not ns.query_dir:
        parser.error("Provide --query_dir or --image_list.")
    main(ns)
