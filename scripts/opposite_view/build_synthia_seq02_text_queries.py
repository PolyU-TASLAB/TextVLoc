#!/usr/bin/env python3
"""Build CLIP text-query CSVs from SYNTHIA front-view captions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_captions(path: Path) -> dict[str, str]:
    captions: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            desc = (row.get("description") or "").strip()
            if not desc:
                continue
            keys = []
            for field in ["image_name", "image_path"]:
                value = (row.get(field) or "").strip()
                if value:
                    keys.append(value)
                    keys.append(Path(value).as_posix())
                    keys.append(Path(value).name)
            for key in keys:
                captions[key] = desc
    return captions


def find_caption(captions: dict[str, str], image_name: str) -> str:
    candidates = [image_name, Path(image_name).as_posix(), Path(image_name).name]
    for key in candidates:
        if key in captions:
            return captions[key]
    for key, desc in captions.items():
        if key.endswith(image_name):
            return desc
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--captions-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_csv = args.experiment_dir / "query_text_source_map.csv"
    output_csv = args.output_csv or (args.experiment_dir / "query_text_front_oracle.csv")
    captions = load_captions(args.captions_csv)

    rows = []
    missing = 0
    with map_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            query = row["query_image_name"]
            source = row["text_source_image_name"]
            desc = find_caption(captions, source)
            if not desc:
                missing += 1
                continue
            rows.append(
                {
                    "image_name": query,
                    "text_source_image_name": source,
                    "target_db_image_name": row.get("target_db_image_name", ""),
                    "description": desc,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_name", "text_source_image_name", "target_db_image_name", "description"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"output={output_csv}")
    print(f"queries={len(rows)}")
    print(f"missing_captions={missing}")


if __name__ == "__main__":
    main()
