"""Generate 493 cam1 query descriptions with local Gemma 3 12B 4-bit.

Run in clip-hloc-dev after accepting the google/gemma-3-12b-it license and
authenticating Hugging Face in the container. No API request contains images.
"""

import argparse
import csv
import json
from pathlib import Path

from scripts.paper_caption_inputs import (
    PROMPT, QUERY_DIR, QUERY_LIST, input_names, sha256,
)


MODEL_ID = "google/gemma-3-12b-it"
OUT_DIR = Path("outputs/cam1_gemma3_query_descriptions")
OUT_CSV = OUT_DIR / "descriptions.csv"


def main():
    global QUERY_DIR, QUERY_LIST, OUT_DIR, OUT_CSV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", type=Path, default=QUERY_DIR)
    parser.add_argument("--query-list", type=Path, default=QUERY_LIST)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="First N missing images; 0 means all")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without model loading")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    QUERY_DIR, QUERY_LIST, OUT_DIR = args.query_dir, args.query_list, args.output_dir
    OUT_CSV = OUT_DIR / "descriptions.csv"
    names = input_names(QUERY_DIR, QUERY_LIST)
    prior = {}
    if OUT_CSV.exists():
        with OUT_CSV.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                assert row["image_name"] not in prior and row["description"].strip()
                prior[row["image_name"]] = row["description"]
        assert set(prior).issubset(names)
    pending = [name for name in names if name not in prior]
    if args.limit:
        pending = pending[:args.limit]
    print(f"Validated 493 cam1 images; {len(prior)} complete; {len(pending)} pending", flush=True)
    if args.dry_run or not pending:
        return

    import torch
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
            device_map="auto", torch_dtype=torch.bfloat16,
        ).eval()
    except OSError as error:
        raise SystemExit(
            "Gemma 3 weights unavailable. Accept the license at "
            "https://huggingface.co/google/gemma-3-12b-it and authenticate "
            "Hugging Face inside clip-hloc-dev before rerunning.") from error

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": MODEL_ID, "quantization": "bitsandbytes 4-bit",
        "generation": {"do_sample": False, "max_new_tokens": 120},
        "prompt": PROMPT,
        "source": "one cam1 query image per request; no reference image, caption, GPS, or result",
        "query_count": 493, "query_directory": str(QUERY_DIR),
        "query_list": str(QUERY_LIST), "query_list_sha256": sha256(QUERY_LIST),
        "human_authored": False, "generator": "Gemma 3 12B local 4-bit",
    }
    manifest_path = OUT_DIR / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    exists = OUT_CSV.exists()
    with OUT_CSV.open("a" if exists else "x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["image_name", "description"])
            handle.flush()
        for index, name in enumerate(pending, 1):
            with Image.open(QUERY_DIR / name) as source:
                image = source.convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=120, do_sample=False)
            description = processor.decode(tokens[0][input_len:], skip_special_tokens=True).strip()
            if not description:
                raise RuntimeError(f"Gemma 3 returned an empty description for {name}")
            writer.writerow([name, description])
            handle.flush()
            if index % 25 == 0 or index == len(pending):
                print(f"Gemma 3 generated {index}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
