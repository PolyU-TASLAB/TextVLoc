#!/usr/bin/env python3
"""Generate paired Qwen query variants without changing training or old captions.

Run as python -m scripts.generate_query_prompt_variants from the project root.
The baseline CSV fixes the complete query population. --limit only caps pending
images per variant, so a pilot can safely resume into a full run.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_queries(baseline, image_root):
    with baseline.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"image_name", "description"}.issubset(reader.fieldnames or []):
            raise ValueError("Baseline requires image_name and description columns")
        names = [row["image_name"].strip() for row in reader]
    if not names or len(names) != len(set(names)):
        raise ValueError("Baseline must contain a nonempty, unique query list")
    root = image_root.resolve()
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing image or path outside query root: {name}")
    return names


def read_completed(path, allowed):
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_name", "description"]:
            raise ValueError(f"Invalid CSV header: {path}")
        completed = set()
        for row in reader:
            name = row.get("image_name")
            if (None in row or name not in allowed or name in completed
                    or not (row.get("description") or "").strip()):
                raise ValueError(f"Invalid/duplicate/incomplete row in {path}: {name}")
            completed.add(name)
    return completed


def prepare_run(output, manifest, names, dry_run):
    metadata = output.with_suffix(".manifest.json")
    if metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Configuration changed; use a new output directory: {metadata}")
    elif output.exists():
        raise ValueError(f"Refusing to resume CSV without provenance: {output}")
    completed = read_completed(output, set(names))
    if not dry_run and not metadata.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with metadata.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return [name for name in names if name not in completed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-config", type=Path,
                        default=Path(__file__).with_name("query_prompt_variants.json"))
    parser.add_argument("--variants", nargs="+", help="Default: all prompt variants")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--limit", type=int, default=0,
                        help="Pending images per variant; 0 means all (pilot only, not evaluation subset)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")

    catalog = json.loads(args.prompt_config.read_text(encoding="utf-8"))
    variants = args.variants or list(catalog)
    if len(variants) != len(set(variants)):
        parser.error("Duplicate variants")
    if any(name not in catalog or Path(name).name != name for name in variants):
        parser.error("Unknown or unsafe variant name")
    names = read_queries(args.baseline_csv, args.query_dir)
    source = Path(__file__).resolve().parents[1] / "hloc/generate_captions.py"
    runs = []
    for variant in variants:
        spec = catalog[variant]
        manifest = {
            "schema_version": 1,
            "variant": variant,
            "category": spec["category"],
            "prompt": spec["prompt"],
            "model_id": args.model_id,
            "device": args.device,
            "generation": {"max_new_tokens": 96, "do_sample": False},
            "captioner_source_sha256": digest(source),
            "runner_source_sha256": digest(Path(__file__)),
            "baseline_csv": str(args.baseline_csv.resolve()),
            "baseline_sha256": digest(args.baseline_csv),
            "query_dir": str(args.query_dir.resolve()),
            "query_names": names,
            "text_source": "query_image_only",
            "human_written": False,
        }
        output = args.output_dir / f"{variant}.csv"
        pending = prepare_run(output, manifest, names, args.dry_run)
        selected = pending[:args.limit] if args.limit else pending
        print(f"{variant}: total={len(names)} completed={len(names)-len(pending)} "
              f"pending={len(pending)} this_run={len(selected)} output={output}", flush=True)
        runs.append((spec, output, selected))
    if args.dry_run or not any(selected for _, _, selected in runs):
        return

    # Set before importing the existing captioner; load the VLM once for all prompts.
    os.environ["QWEN_VL_MODEL_ID"] = args.model_id
    from hloc.generate_captions import get_captioner
    caption = get_captioner("qwen-vl", args.device)
    for spec, output, selected in runs:
        if not selected:
            continue
        exists = output.exists()
        with output.open("a" if exists else "x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not exists:
                writer.writerow(["image_name", "description"])
                handle.flush()
            for index, name in enumerate(selected, 1):
                description = caption(args.query_dir / name, prompt=spec["prompt"]).strip()
                if not description:
                    raise RuntimeError(f"Empty generation: {name}; stopping, not silently dropping query")
                writer.writerow([name, description])
                handle.flush()
                print(f"{output.stem} {index}/{len(selected)} {name}", flush=True)


if __name__ == "__main__":
    main()
