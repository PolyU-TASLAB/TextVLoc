"""Generate 493 cam1 query descriptions with DeepSeek Flash vision.

Run in clip-hloc-dev. Only the query image and fixed prompt are sent; no
reference image, original caption, GPS, or retrieval result is used.
"""

import argparse
import base64
import csv
import json
import os
import random
import time
from pathlib import Path

import requests

from scripts.paper_caption_inputs import (
    PROMPT, QUERY_DIR, QUERY_LIST, input_names, sha256,
)


OUT_DIR = Path("outputs/cam1_deepseek_query_descriptions")
OUT_CSV = OUT_DIR / "descriptions.csv"
API_URL = "https://api.deepseek.com/chat/completions"
KEY_FILE = Path("/run/textvloc/deepseek_api_key")


def caption(session, key, image_path):
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    request = {
        "model": "deepseek-flash",
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}", "detail": "original"}},
        ]}],
        "max_tokens": 200,
    }
    for attempt in range(6):
        try:
            response = session.post(
                API_URL, json=request,
                headers={"Authorization": f"Bearer {key}"}, timeout=120)
        except requests.RequestException as error:
            if attempt == 5:
                raise RuntimeError(f"DeepSeek connection failed: {type(error).__name__}") from error
            time.sleep(min(2 ** attempt + random.random(), 30))
            continue
        if response.status_code == 200:
            data = response.json()
            value = data["choices"][0]["message"].get("content", "").strip()
            if not value:
                raise RuntimeError("DeepSeek returned an empty description")
            return value, data.get("usage", {})
        if response.status_code in (408, 409, 429, 500, 502, 503, 504) and attempt < 5:
            time.sleep(min(2 ** attempt + random.random(), 30))
            continue
        if response.status_code == 402:
            raise RuntimeError("DeepSeek API HTTP 402: account balance is insufficient")
        try:
            error = response.json().get("error", {})
            error_code = error.get("code") or error.get("type") or "unknown"
        except ValueError:
            error_code = "non_json_error"
        raise RuntimeError(f"DeepSeek API HTTP {response.status_code}: {error_code}")
    raise AssertionError("unreachable")


def main():
    global QUERY_DIR, QUERY_LIST, OUT_DIR, OUT_CSV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", type=Path, default=QUERY_DIR)
    parser.add_argument("--query-list", type=Path, default=QUERY_LIST)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="First N missing images; 0 means all")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without API calls")
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
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key and KEY_FILE.is_file():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise SystemExit(f"No DEEPSEEK_API_KEY or {KEY_FILE} inside clip-hloc-dev; no requests sent")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": "deepseek-flash", "endpoint": API_URL,
        "thinking": "disabled", "image_detail": "original", "max_tokens": 200,
        "prompt": PROMPT,
        "source": "one cam1 query image per request; no reference image, caption, GPS, or result",
        "query_count": 493, "query_directory": str(QUERY_DIR),
        "query_list": str(QUERY_LIST), "query_list_sha256": sha256(QUERY_LIST),
        "human_authored": False, "generator": "DeepSeek Flash vision API",
    }
    manifest_path = OUT_DIR / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    exists = OUT_CSV.exists()
    with requests.Session() as session, OUT_CSV.open(
            "a" if exists else "x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["image_name", "description"])
            handle.flush()
        for index, name in enumerate(pending, 1):
            description, usage = caption(session, key, QUERY_DIR / name)
            writer.writerow([name, description])
            handle.flush()
            if index % 25 == 0 or index == len(pending):
                print(f"DeepSeek generated {index}/{len(pending)}; "
                      f"last tokens {usage.get('total_tokens', 'unknown')}", flush=True)


if __name__ == "__main__":
    main()
