"""Shared inputs and prompt for the paper DeepSeek/Gemma caption generators."""

import csv
import hashlib
from pathlib import Path

QUERY_DIR = Path('datasets/SensorKit/query/cam1_all/0_undistorted')

QUERY_LIST = Path('outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/qwen3_query_prompts_qwen-vl_77clip10.csv')

PROMPT = 'You are at the camera position in this single street-view photograph. Give a natural first-person description that would help another person identify this place.\n\nDescribe only what is visibly supported by THIS photograph. Choose two to four distinctive, relatively persistent cues: building facades, road or intersection geometry, signs, storefronts, pavement markings, vegetation, and their spatial relationships. Mention a place name, logo, sign text, color, or left/right direction only if clear in the image. Do not invent unseen surroundings or infer a location from outside knowledge. Ignore moving vehicles and people. Write one or two fluent English sentences, 25–55 words, without mentioning the photograph or your task. Output only the description.'

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def input_names(query_dir=QUERY_DIR, query_list=QUERY_LIST):
    with query_list.open(newline='', encoding='utf-8-sig') as handle:
        names = [row['image_name'] for row in csv.DictReader(handle)]
    assert len(names) == len(set(names)) == 493
    assert all(((query_dir / name).is_file() for name in names))
    return names
