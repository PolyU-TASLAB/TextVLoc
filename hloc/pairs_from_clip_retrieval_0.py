import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import h5py
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from . import logger

# =======================================================================
#  全局常量和设备设置
# =======================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =======================================================================
#  辅助函数
# =======================================================================

def try_import_lora():
    try:
        import loralib as lora
        return lora
    except ImportError:
        raise ImportError("loralib is not installed. Please run: pip install loralib")

def load_all_features(h5_path: Path) -> (List[str], np.ndarray):
    """高效地从 HDF5 文件中加载所有 CLIP 特征。"""
    logger.info(f"Loading CLIP features from {h5_path}...")
    start_time = time.time()
    names, features = [], []
    with h5py.File(h5_path, 'r') as f:
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                feat = obj[()]
                if feat.ndim == 1:
                    features.append(feat)
                    names.append(name)
        f.visititems(collect)
    if not features:
        raise RuntimeError(f"No valid CLIP features found in {h5_path}")
    logger.info(f"Loaded {len(names)} features in {time.time() - start_time:.2f}s.")
    return names, np.stack(features)

def create_positive_map(csv_path: Path) -> Dict[str, Set[str]]:
    """从 CSV 文件创建用于计算召回率的 positive_map。"""
    if not csv_path or not csv_path.exists():
        return {}
    positive_map = defaultdict(set)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) >= 2:
                query_name = Path(row[0]).name
                db_name = Path(row[1]).name
                positive_map[query_name].add(db_name)
    logger.info(f"Created positive_map with {len(positive_map)} queries from {csv_path}.")
    return dict(positive_map)

# =======================================================================
#  主函数
# =======================================================================
@torch.no_grad()
def main(
    clip_features_path: Path,
    text_query_or_csv_path: Path,
    num_loc: int,
    loc_pairs: Path,
    model_instance=None,
    processor_instance=None,
    query_list_path: Optional[Path] = None,
    positive_pairs_csv: Optional[Path] = None,
):
    t_total_start = time.time()
    logger.info("Starting CLIP retrieval pipeline...")

    # --- 1. 加载 CLIP 模型 ---
    if model_instance and processor_instance:
        model, processor = model_instance, processor_instance
    else:
        logger.warning("No pre-loaded CLIP model provided, loading from scratch.")
        model, _, processor = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="datacomp_xl_s13b_b90k", device=DEVICE
        )

    # --- 2. 读取和解析文本查询 ---
    is_single_prompt = isinstance(text_query_or_csv_path, str) and not str(text_query_or_csv_path).endswith('.csv')
    if is_single_prompt:
        if not query_list_path:
            raise ValueError("--query_list_path is required when using a single text prompt.")
        with open(query_list_path, 'r') as f:
            query_names = [line.strip().split()[0] for line in f if line.strip()]
        text_queries = [text_query_or_csv_path] * len(query_names)
    else:
        df = pd.read_csv(text_query_or_csv_path)
        text_queries = df['description'].tolist()
        query_names = df['image_name'].tolist()
    logger.info(f"Loaded {len(query_names)} queries.")

    # --- 3. 编码所有文本查询 ---
    logger.info("Encoding text queries...")
    if isinstance(processor, open_clip.tokenizer.SimpleTokenizer):
        inputs = processor(text_queries).to(DEVICE)
        text_features = model.encode_text(inputs)
    else: # Hugging Face Processor
        inputs = processor(text=text_queries, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        text_features = model.get_text_features(**inputs)
    text_features = F.normalize(text_features, p=2, dim=-1)
    logger.info("Text query encoding complete.")

    # --- 4. 加载数据库图像特征 ---
    db_names, db_features = load_all_features(clip_features_path)
    db_features_tensor = torch.from_numpy(db_features).to(DEVICE)

    # --- 5. 计算相似度并进行初始 Top-K 检索 ---
    similarities = db_features_tensor @ text_features.T
    top_k_clip_scores, top_k_db_indices = torch.topk(similarities, k=min(num_loc, len(db_names)), dim=0)

    # --- 6. 写入配对文件 ---
    top_k_db_indices = top_k_db_indices[:num_loc, :]
    top_k_clip_scores = top_k_clip_scores[:num_loc, :]

    with open(loc_pairs, "w") as f:
        for i, query_name in enumerate(tqdm(query_names, desc="Writing final pairs")):
            retrieved_indices = top_k_db_indices.T[i].cpu().numpy()
            retrieved_scores = top_k_clip_scores.T[i].cpu().numpy()
            retrieved_names = [db_names[j] for j in retrieved_indices]
            line_parts = [query_name] + [item for pair in zip(retrieved_names, retrieved_scores) for item in (pair[0], f"{pair[1]:.6f}")]
            f.write(" ".join(line_parts) + "\n")

    logger.info(f"Successfully saved Top-{num_loc} pairs to {loc_pairs}")
    logger.info(f"Total pipeline time: {time.time() - t_total_start:.2f} seconds")

    # --- 7. 召回率计算 ---
    if positive_pairs_csv and positive_pairs_csv.exists():
        logger.info(f"Calculating recall using ground truth from {positive_pairs_csv}...")
        positive_map = create_positive_map(positive_pairs_csv)
        
        # We need to calculate recall based on the final sorted indices
        final_top_k_indices = top_k_db_indices.T.cpu().numpy()
        
        recalls = {}
        k_values = [1, 5, 10, min(20, num_loc)]
        total_queries_with_gt = 0
        
        for i, query_name in enumerate(query_names):
            query_basename = Path(query_name).name
            ground_truth_names = positive_map.get(query_basename)
            if not ground_truth_names:
                continue
            
            total_queries_with_gt += 1
            retrieved_db_names = [db_names[j] for j in final_top_k_indices[i]]
            
            for k in k_values:
                if k not in recalls: recalls[k] = 0
                retrieved_top_k_basenames = {Path(p).name for p in retrieved_db_names[:k]}
                if not ground_truth_names.isdisjoint(retrieved_top_k_basenames):
                    recalls[k] += 1
        
        if total_queries_with_gt > 0:
            logger.info("--- Retrieval Recall Stats ---")
            for k in sorted(recalls.keys()):
                recall_percent = (recalls[k] / total_queries_with_gt) * 100
                logger.info(f"Recall@{k}: {recall_percent:.2f}%")
            logger.info("------------------------------")
        else:
            logger.warning("Could not calculate recall: No queries found in the ground truth file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve and optionally rerank image pairs based on CLIP text-image similarity.")
    # --- 基本参数 ---
    parser.add_argument("--clip_features_path", type=Path, required=True, help="Path to the HDF5 file with DB CLIP features.")
    parser.add_argument("--text_input", type=str, required=True, help="A single text prompt or a path to a CSV file.")
    parser.add_argument("--num_loc", type=int, default=20, help="Number of top database images to retrieve for each query.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output pairs file.")
    parser.add_argument("--query_list_path", type=Path, help="Required only when using a single text prompt.")
    parser.add_argument("--positive_pairs_csv", type=Path, help="Optional CSV file with (anchor, positive) pairs for recall calculation.")
    
    args = parser.parse_args()

    text_input_path = Path(args.text_input)
    input_arg = text_input_path if text_input_path.is_file() and args.text_input.endswith('.csv') else args.text_input

    main(
        clip_features_path=args.clip_features_path,
        text_query_or_csv_path=input_arg,
        num_loc=args.num_loc,
        loc_pairs=args.output,
        query_list_path=args.query_list_path,
        positive_pairs_csv=args.positive_pairs_csv,
    )
