from PIL import Image
from typing import List, Dict
import argparse
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from transformers import CLIPModel as HfCLIPModel, CLIPProcessor, CLIPTokenizer
from pathlib import Path
import torch
import torch.nn as nn  # <-- Add this line
import h5py
import time
import numpy as np
from tqdm import tqdm
import pandas as pd # 1. 导入 pandas 用于读取 CSV
from .utils.read_write_model import read_images_binary
import open_clip
from . import logger
from .utils.parsers import parse_image_lists
from collections import defaultdict # <-- 1. 导入 defaultdict
import re
import spacy
import csv
from groundingdino.util.inference import predict as predict_grounding_dino, load_image # <-- 修改这一行


try:
    from transformers import CLIPProcessor, CLIPModel, SiglipProcessor, SiglipModel
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_ID = "openai/clip-vit-base-patch32"
# MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# MODEL_ID = "zer0int/LongCLIP-GmP-ViT-L-14" 
# MODEL_ID = "google/siglip-base-patch16-224" 
MODEL_ID = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"

# =======================================================================
#  辅助函数 (用于加载特征和模型)
# =======================================================================
def load_all_features(h5_path: Path):
    """
    高效地从 HDF5 文件中加载所有 CLIP 特征。
    这个版本避免了缓慢的 visititems 递归遍历。
    """
    logger.info(f"正在从 {h5_path} 高效加载 CLIP 特征...")
    start_time = time.time()
    
    names = []
    features = []
    
    with h5py.File(h5_path, 'r') as f:
        # 假设特征存储在顶层或简单的层级下
        # 我们可以直接迭代顶层键
        for key in f.keys():
            # 假设每个键对应一个图像的特征数据集
            if isinstance(f[key], h5py.Dataset):
                names.append(key)
                features.append(f[key][()])
            # 如果特征存储在更深的层级，例如 'cam0/123.jpg'
            # h5py 的 keys() 可能不会直接暴露，但我们可以尝试另一种方式
            # 这里我们先假设是简单结构，如果不行再调整
    
    # 如果上面的方法因为层级问题没读到，使用一个更鲁棒但仍比 visititems 快的方法
    if not names:
        logger.warning("简单加载失败，尝试另一种方法...")
        def find_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(name)
        f = h5py.File(h5_path, 'r')
        f.visititems(find_datasets) # 先找到所有名字
        features = [f[n][()] for n in names] # 然后批量读取
        f.close()

    if not features:
        raise RuntimeError(f"在 {h5_path} 中未找到有效的 CLIP 特征")

    logger.info(f"成功加载 {len(names)} 个特征，耗时 {time.time() - start_time:.2f} 秒。")
    return names, np.stack(features)


def try_import_lora():
    try:
        import loralib as lora
        return lora
    except ImportError:
        raise ImportError("loralib 未安装，请运行: pip install loralib")

def _replace_linear_with_lora_linear(mod: nn.Linear, r=8, alpha=16, dropout=0.0):
    lora = try_import_lora()
    new = lora.Linear(
        mod.in_features, mod.out_features,
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias=(mod.bias is not None)
    )
    with torch.no_grad():
        new.weight.copy_(mod.weight)
        if mod.bias is not None:
            new.bias.copy_(mod.bias)
    return new

def inject_lora_openclip(model, last_n_blocks=4, r=8, alpha=16, dropout=0.0):
    lora = try_import_lora()
    # Vision tower
    v_blocks = model.visual.transformer.resblocks
    for blk in v_blocks[-last_n_blocks:]:
        blk.attn.out_proj = _replace_linear_with_lora_linear(blk.attn.out_proj, r, alpha, dropout)
        blk.mlp.c_fc      = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
        blk.mlp.c_proj    = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)
    # Text tower
    t_blocks = model.transformer.resblocks
    for blk in t_blocks[-last_n_blocks:]:
        blk.attn.out_proj = _replace_linear_with_lora_linear(blk.attn.out_proj, r, alpha, dropout)
        blk.mlp.c_fc      = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
        blk.mlp.c_proj    = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)
    return model

# --- 核心修复：添加缺失的 merge_lora_weights 函数 ---
def merge_lora_weights(model):
    lora = try_import_lora()
    for n, m in model.named_modules():
        if isinstance(m, lora.LoRALayer) and hasattr(m, 'merge'):
            logger.info(f"Merging LoRA weights for layer: {n}")
            m.merge()
# --- 修复结束 ---

# --- 1. 从 train_clip.py 复制 Projection 类 ---
class Projection(nn.Module):
    def __init__(self, dim: int, hidden: int = 0):
        super().__init__()
        if hidden and hidden > 0:
            self.proj = nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, dim)
            )
        else:
            self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.proj(x)
        return torch.nn.functional.normalize(x, dim=-1)

# --- 2. 包装模型以应用线性探针头 ---
class ProbedCLIP(nn.Module):
    def __init__(self, clip_model, img_head, txt_head, logit_scale):
        super().__init__()
        self.clip_model = clip_model
        self.img_head = img_head
        self.txt_head = txt_head
        self.logit_scale = nn.Parameter(logit_scale)

    def encode_image(self, x):
        with torch.no_grad():
            img_feat = self.clip_model.encode_image(x)
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
        return self.img_head(img_feat)

    def encode_text(self, x):
        with torch.no_grad():
            txt_feat = self.clip_model.encode_text(x)
            txt_feat = torch.nn.functional.normalize(txt_feat, dim=-1)
        return self.txt_head(txt_feat)





# # 3. 新增：计算召回率的核心函数
# @torch.no_grad()
# def calculate_recall_at_k(query_texts, query_names, db_feats, db_names, model, processor, device, k_values=(1, 5, 10, 20)):
#     """
#     计算给定文本查询的图像检索召回率。
#     """
#     logger.info(f"Calculating Recall@{k_values} for {len(query_texts)} queries...")
    
#     db_names_list = list(db_names)
#     # 关键修复：将数据库名称列表转换为仅包含文件名，以匹配查询名称的格式
#     db_basenames_list = [Path(p).name for p in db_names]
#     # print("Database basenames sample:", db_basenames_list[:5])
#     db_feats_gpu = torch.from_numpy(db_feats).to(device).float()
#     db_feats_gpu = torch.nn.functional.normalize(db_feats_gpu, p=2, dim=-1)

#     hits_at_k = {k: 0 for k in k_values}
#     total_queries = 0

#     for query_text, query_name in tqdm(zip(query_texts, query_names), total=len(query_texts), desc="Recall@K"):
#         try:
#             # 真值就是与文本描述同名的那张图在数据库中的索引
#             # 使用处理过的 basename 列表进行查找
#             gt_index = db_basenames_list.index(query_name)
#         except ValueError:
#             logger.warning(f"Query image '{query_name}' not found in the database names. Skipping for recall calculation.")
#             continue
        
#         total_queries += 1
        
#         # 编码文本 (兼容 transformers 和 open_clip)
#         if isinstance(processor, CLIPProcessor):
#             inputs = processor(text=query_text, return_tensors="pt", padding=True, truncation=True).to(device)
#             text_feat = model.get_text_features(**inputs)
#         else: # open_clip tokenizer
#             inputs = processor([query_text]).to(device)
#             text_feat = model.encode_text(inputs)
        
#         text_feat = torch.nn.functional.normalize(text_feat, p=2, dim=-1)
        
#         # 计算相似度并找到 Top-K
#         similarities = (text_feat @ db_feats_gpu.T).squeeze()
#         top_k_indices = torch.topk(similarities, max(k_values)).indices

#         # 检查命中
#         for k in k_values:
#             if gt_index in top_k_indices[:k]:
#                 hits_at_k[k] += 1
    
#     if total_queries == 0:
#         logger.error("No valid queries found for recall calculation. Check if query names match database names.")
#         return {k: 0.0 for k in k_values}

#     # 计算并打印结果
#     recall_results = {k: (hits_at_k[k] / total_queries) * 100 for k in k_values}
#     logger.info("--- Retrieval Recall Stats ---")
#     for k, recall in recall_results.items():
#         logger.info(f"Recall@{k}: {recall:.2f}%")
#     logger.info("------------------------------")
    
#     return recall_results





@torch.no_grad()
def calculate_recall_at_k(query_names, db_names, top_k_indices, k_values, positive_map=None):
    """
    计算给定查询的图像检索召回率。
    """
    if positive_map is None:
        logger.error("未提供 positive_map，无法计算召回率。")
        return {k: 0.0 for k in k_values}

    recalls = {k: 0 for k in k_values}
    total_queries_with_gt = 0
    db_names_set = set(db_names)

    for i, query_name in enumerate(tqdm(query_names, desc="Calculating recall")):
        # --- 核心修复：使用 basename 进行查找 ---
        # 无论 query_name 是 'cam0/123.jpg' 还是 '123.jpg'，我们都只用 '123.jpg' 作为键
        query_basename = Path(query_name).name
        
        # 在 positive_map 中查找时，也需要考虑 key 可能是 'cam0/123.jpg' 的情况
        # 我们假设 positive_map 的键是相对路径
        ground_truth_names = positive_map.get(query_basename)
        if ground_truth_names is None:
            # 如果用 basename 找不到，尝试用原始名再找一次，以增加兼容性
            ground_truth_names = positive_map.get(query_name)

        if ground_truth_names is None or not ground_truth_names:
            # 如果在真值文件中找不到这个查询，就跳过
            continue
        # --- 修复结束 ---
        
        total_queries_with_gt += 1
        
        top_k_for_query = top_k_indices[i]
        top_k_names = [db_names[j] for j in top_k_for_query]

        for k in k_values:
            # 检查前k个检索结果中是否有任何一个属于正确答案集合
            retrieved_top_k = set(top_k_names[:k])
            if not retrieved_top_k.isdisjoint(ground_truth_names):
                recalls[k] += 1

    if total_queries_with_gt == 0:
        logger.error("无法计算召回率，因为没有一个查询图像在数据库中有对应的正确答案。")
        return {k: 0.0 for k in k_values}

    for k in k_values:
        recalls[k] = (recalls[k] / total_queries_with_gt) * 100
    
    return recalls




def load_finetuned_model(ckpt_path: Path, device: str):
    """
    加载一个经过 LoRA 或 Linear Probe 微调的 open_clip 模型用于推理。
    """
    logger.info(f"正在从检查点加载微调模型: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # --- 核心修复：让模式检测更健壮 ---
    mode = ckpt.get('mode')
    if mode is None:
        # 如果 'mode' 不存在，则根据内容进行猜测
        if 'lora_state' in ckpt:
            logger.warning("检查点中未找到 'mode' 键，但检测到 'lora_state'。假设为 'lora' 模式。")
            mode = 'lora'
        elif 'img_head' in ckpt:
            logger.warning("检查点中未找到 'mode' 键，但检测到 'img_head'。假设为 'linear' 模式。")
            mode = 'linear'
    # --- 修复结束 ---
    
    # 加载基础模型 (对两种模式都通用)
    model_name = ckpt['model_name']
    pretrained = ckpt['pretrained']
    logger.info(f"  - 基础模型: {model_name} ({pretrained})")
    base_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device='cpu'
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    if mode == 'lora':
        lora = try_import_lora()
        logger.info("  - 检测到 LoRA 模式。正在注入和合并权重...")
        model = inject_lora_openclip(
            base_model,
            last_n_blocks=ckpt['lora_last_n'], r=ckpt['lora_r'],
            alpha=ckpt['lora_alpha'], dropout=ckpt['lora_dropout']
        )
        model.load_state_dict(ckpt['lora_state'], strict=False)
        merge_lora_weights(model)
        logger.info("✅ LoRA 模型加载并准备就绪！")

    elif mode == 'linear':
        logger.info("  - 检测到 Linear Probe 模式。正在加载投影头...")
        dim = base_model.text_projection.shape[-1]
        # 假设 proj_hidden=0，如果您的训练设置不同，需要修改这里
        img_head = Projection(dim, hidden=0)
        txt_head = Projection(dim, hidden=0)
        
        img_head.load_state_dict(ckpt['img_head'])
        txt_head.load_state_dict(ckpt['txt_head'])
        
        # 使用包装器来应用头
        model = ProbedCLIP(base_model, img_head, txt_head, ckpt['logit_scale'])
        logger.info("✅ Linear Probe 模型加载并准备就绪！")
        
    else:
        raise ValueError(f"不支持的检查点模式: '{mode}'。只支持 'lora' 或 'linear'。")

    model.to(device).eval()
    return model, tokenizer, preprocess










def load_all_features(h5_path: Path):
    """
    递归读取所有 feature（支持多级分层结构），返回 (图像名, 特征向量) 列表。
    """
    features = []
    names = []

    # 在 hloc/pairs_from_clip_retrieval.py 中
    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            feat = obj[()]
            # --- 关键修改 ---
            # 允许 512 (base) 或 768 (long) 维度的特征 或 1024 (large) 维度的特征
            if feat.ndim == 1 and feat.shape[0] in (512, 768, 1024):
                features.append(feat)
                names.append(name)
            else:
                # (可选) 增加一个警告，以防未来出现其他维度
                logger.warning(f"Skipping feature with unexpected shape {feat.shape} at {name}")

    with h5py.File(h5_path, 'r') as f:
        f.visititems(collect)

    if not features:
        raise RuntimeError(f"No valid CLIP features found in {h5_path}")

    return names, np.stack(features)



# @torch.no_grad()
# def main(
#     clip_features_path: Path,
#     text_query: str,
#     num_matches: int,
#     output_path: Path,
#     query_list_path: Path,
#     model_id: str = MODEL_ID,
# ):

#     """
#     根据 CLIP 文本描述检索最相关的图像，并为每个查询图像生成相同的 top-k 配对。
#     """
#     logger.info(f"Generating pairs from CLIP text retrieval...")
#     logger.info(f"Text query: '{text_query}'")

#     # 1. 加载模型
#     model = CLIPModel.from_pretrained(model_id).to(DEVICE)
#     processor = CLIPProcessor.from_pretrained(model_id)

#     # 2. 加载数据库特征
#     db_names, db_features = load_all_features(clip_features_path)
#     db_features_tensor = torch.from_numpy(db_features).to(DEVICE)
#     logger.info(f"Loaded {len(db_names)} features from {clip_features_path}")

#     # 3. 编码文本
#     inputs = processor(text=text_query, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
#     text_features = model.get_text_features(**inputs)
#     text_features /= text_features.norm(p=2, dim=-1, keepdim=True)  # L2 normalize

#     # 4. 相似度计算 + Top-K
#     similarities = (db_features_tensor @ text_features.T).squeeze()  # cosine similarity
#     top_k_indices = torch.topk(similarities, k=min(num_matches, len(db_names)), dim=0).indices
#     top_k_db_names = [db_names[i] for i in top_k_indices.cpu().numpy()]
#     logger.info(f"Top-{num_matches} retrieved images: {top_k_db_names[:3]}...")

#     # 5. 加载查询图像列表
#     query_names = parse_image_lists(query_list_path)
#     logger.info(f"Found {len(query_names)} query images.")

#     # 6. 写入 pair 文件，每行 query 和一个 db 匹配图像（共 query × topk 行）
#     with open(output_path, 'w') as f:
#         for q_name in tqdm(query_names, desc="Writing pairs"):
#             for db_name in top_k_db_names:
#                 f.write(f"{q_name} {db_name}\n")


#     logger.info(f"Saved {len(query_names)} query pairs to {output_path}")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--clip_features_path", type=Path, required=True)
#     parser.add_argument("--text_query", type=str, required=True)
#     parser.add_argument("--num_matches", type=int, default=20)
#     parser.add_argument("--output_path", type=Path, required=True)
#     parser.add_argument("--query_list_path", type=Path, required=True)
#     args = parser.parse_args()

#     main(
#         args.clip_features_path,
#         args.text_query,
#         args.num_matches,
#         args.output_path,
#         args.query_list_path,
#     )














# =======================================================================
@torch.no_grad()
def main(
    clip_features_path: Path,
    text_query_or_csv_path: Path,
    num_loc: int,
    loc_pairs: Path,
    model_instance=None,
    processor_instance=None,
    query_list_path: Path = None,
    positive_pairs_csv: Path = None,
):
    t_total_start = time.time()
    logger.info("开始 CLIP 检索流程...")

    # --- 1. 加载 CLIP 模型 ---
    if model_instance and processor_instance:
        model = model_instance
        processor = processor_instance
    else:
        logger.warning("未提供预加载的 CLIP 模型，将从头加载。")
        model, _, processor = open_clip.create_model_and_transforms(
            MODEL_ID.split("/")[-1], pretrained=MODEL_ID.split("/")[0], device=DEVICE
        )

    # --- 2. 读取和解析文本查询 ---
    is_single_prompt = isinstance(text_query_or_csv_path, str) and not str(text_query_or_csv_path).endswith('.csv')
    if is_single_prompt:
        text_queries = [text_query_or_csv_path]
        query_names = parse_image_lists(query_list_path)
    else:
        df = pd.read_csv(text_query_or_csv_path)
        text_queries = df['description'].tolist()
        query_names = df['image_name'].tolist()
    logger.info(f"已加载 {len(query_names)} 个查询。")

    # --- 3. 编码所有文本查询 ---
    logger.info("正在编码文本查询...")
    if isinstance(processor, open_clip.tokenizer.SimpleTokenizer):
        # OpenCLIP 的分词器
        inputs = open_clip.tokenizer.tokenize(text_queries).to(DEVICE)
        text_features = model.encode_text(inputs)
    else:
        # Hugging Face 的 CLIPProcessor
        inputs = processor(text=text_queries, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        text_features = model.get_text_features(**inputs)

    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
    logger.info("文本查询编码完成。")

    # --- 4. 加载数据库图像特征 ---
    db_names, db_features = load_all_features(clip_features_path)
    db_features_tensor = torch.from_numpy(db_features).to(DEVICE)
    logger.info(f"已加载 {len(db_names)} 个数据库图像特征。")

    # --- 5. 计算相似度并进行 Top-K 检索 ---
    similarities = db_features_tensor @ text_features.T
    top_k_clip_scores, top_k_db_indices = torch.topk(similarities, k=min(num_loc, len(db_names)), dim=0)

    # --- 6. 写入配对文件 ---
    with open(loc_pairs, "w") as f:
        for i, query_name in enumerate(tqdm(query_names, desc="Writing pairs")):
            retrieved_indices = top_k_db_indices.T[i].cpu().numpy()
            retrieved_scores = top_k_clip_scores.T[i].cpu().numpy()
            retrieved_names = [db_names[j] for j in retrieved_indices]
            line_parts = [query_name]
            for name, score in zip(retrieved_names, retrieved_scores):
                line_parts.extend([name, f"{score:.6f}"])
            f.write(" ".join(line_parts) + "\n")

    logger.info(f"成功将 Top-{num_loc} 配对保存到 {loc_pairs}")
    logger.info(f"总耗时: {time.time() - t_total_start:.2f} 秒")

    # --- 核心新增：直接在此处计算并打印召回率 ---
    if positive_pairs_csv and positive_pairs_csv.exists():
        logger.info(f"正在使用真值文件 {positive_pairs_csv} 计算召回率...")
        
        # 1. 定义一个简单的本地解析器来读取真值对
        def parse_local_pairs(path: Path) -> List[tuple]:
            pairs = []
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            p1, p2 = line.strip().split(',')
                            pairs.append((p1, p2))
                        except ValueError:
                            logger.warning(f"无法解析行: '{line.strip()}'，已跳过。")
            return pairs

        # 2. 加载真值对并构建 positive_map
        try:
            gt_pairs = parse_local_pairs(positive_pairs_csv)
            positive_map = defaultdict(set)
            for q, p in gt_pairs:
                q_basename = Path(q).name
                if 'allcams/' in p:
                    p_cleaned = p.split('allcams/')[1]
                else:
                    p_cleaned = p
                positive_map[q_basename].add(p_cleaned)
        except Exception as e:
            logger.error(f"解析真值对文件 {positive_pairs_csv} 失败: {e}。跳过召回率计算。")
            return

        # 3. 调用 calculate_recall_at_k 函数
        recall_results = calculate_recall_at_k(
            query_names=query_names,
            db_names=db_names,
            top_k_indices=top_k_db_indices.T.cpu().numpy(),
            k_values=[1, 5, 10, 20],
            positive_map=positive_map
        )
        
        # 4. 打印结果
        logger.info("--- Retrieval Recall Stats ---")
        for k, recall in recall_results.items():
            logger.info(f"Recall@{k}: {recall:.2f}%")
        logger.info("------------------------------")
    # --- 新增结束 ---



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve image pairs based on CLIP text-image similarity. "
                    "Supports a single prompt for all queries or a CSV for per-query prompts."
    )
    parser.add_argument(
        "--clip_features_path", type=Path, required=True,
        help="Path to the HDF5 file containing CLIP features of the database images."
    )
    parser.add_argument(
        "--text_input", type=str, required=True,
        help="Can be either: 1) A single text prompt string (e.g., 'a photo of a street'). "
             "2) The path to a CSV file containing 'image_name' and 'description' columns."
    )
    parser.add_argument(
        "--num_loc", type=int, default=20,
        help="Number of top database images to retrieve for each query."
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path to the output file where the pairs will be written."
    )
    parser.add_argument(
        "--query_list_path", type=Path,
        help="Path to a text file listing query images. This is REQUIRED only when using a single text prompt."
    )
    parser.add_argument(
        "--positive_pairs_csv", type=Path, 
        help="用于扩展召回率计算的正样本对CSV文件。")
    
    args = parser.parse_args()

    # 智能地判断 text_input 是字符串还是文件路径
    text_input_path = Path(args.text_input)
    if text_input_path.is_file() and args.text_input.endswith('.csv'):
        input_arg = text_input_path
    else:
        # 如果不是一个存在的CSV文件，就当作是字符串 prompt
        input_arg = args.text_input

    main(
        clip_features_path=args.clip_features_path,
        text_query_or_csv_path=input_arg,
        num_loc=args.num_loc,
        loc_pairs=args.output,
        query_list_path=args.query_list_path,
    )










# @torch.no_grad()
# def main(
#     clip_features_path: Path,
#     text_query_or_csv_path: Path,
#     num_loc: int,
#     loc_pairs: Path,
#     reference_sfm_path: Path,
#     fusion_strategy: str = 'mean', # 新增参数：融合策略
#     query_list_path: Path = None,
#     model_id: str = MODEL_ID,
# ):
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.info(f"Using device: {DEVICE}")
#     logger.info(f"Generating pairs from CLIP text retrieval with '{fusion_strategy}' fusion...")

#     # 1. 加载模型
#     model = CLIPModel.from_pretrained(model_id).to(DEVICE)
#     processor = CLIPProcessor.from_pretrained(model_id)

#     # 2. 读取文本描述并拆分
#     is_single_prompt = isinstance(text_query_or_csv_path, str) and not str(text_query_or_csv_path).endswith('.csv')
    
#     if is_single_prompt:
#         # 单一prompt模式不支持句子拆分融合
#         raise NotImplementedError("Sentence fusion is only supported for CSV input.")
#     else:
#         logger.info(f"Reading prompts from CSV file: {text_query_or_csv_path}")
#         df = pd.read_csv(text_query_or_csv_path)
        
#         query_names = df['image_name'].tolist()
        
#         # --- 核心修改：拆分描述为多个句子 ---
#         # 我们假设句子在'description'列中用'|'分隔
#         # all_sentences: 所有查询的所有句子列表
#         # query_to_indices: 映射每个查询到它在all_sentences中的句子索引范围
#         all_sentences = []
#         query_to_indices = {}
#         start_idx = 0
#         for i, row in df.iterrows():
#             query_name = row['image_name']
#             sentences = [s.strip() for s in row['description'].split('.') if s.strip()]
#             if not sentences:
#                 sentences = ["a photo"] # 如果为空则提供一个默认值
#             all_sentences.extend(sentences)
#             end_idx = start_idx + len(sentences)
#             query_to_indices[query_name] = (start_idx, end_idx)
#             start_idx = end_idx
#         # --- 拆分结束 ---

#     # 3. 对所有拆分后的句子进行编码
#     logger.info(f"Encoding {len(all_sentences)} total sentences from {len(query_names)} queries...")
#     inputs = processor(text=all_sentences, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
#     text_features = model.get_text_features(**inputs)
#     text_features /= text_features.norm(dim=-1, keepdim=True)


#     # 4. 加载并过滤数据库特征 (这部分逻辑不变)
#     db_names, db_features = load_all_features(clip_features_path)
#     logger.info(f"Filtering database features against SfM model at {reference_sfm_path}")
#     sfm_images = read_images_binary(reference_sfm_path / 'images.bin')
#     sfm_image_names = {img.name for img in sfm_images.values()}
#     valid_indices = [i for i, name in enumerate(db_names) if name in sfm_image_names]
#     db_names = [db_names[i] for i in valid_indices]
#     db_features = db_features[valid_indices]
#     logger.info(f"Kept {len(db_names)} features that are in the SfM model.")
#     db_features_tensor = torch.from_numpy(db_features).to(DEVICE)

#     # 5. 计算总相似度矩阵并进行融合
#     # all_similarities 的形状是 [num_db_images, num_total_sentences]
#     all_similarities = db_features_tensor @ text_features.T
    
#     # 为每个查询图像计算融合后的相似度分数
#     fused_similarities = torch.zeros((len(db_names), len(query_names)), device=DEVICE)

#     for i, q_name in enumerate(tqdm(query_names, desc="Fusing similarities")):
#         start, end = query_to_indices[q_name]
#         query_sims = all_similarities[:, start:end] # 取出该查询对应的所有句子的相似度
        
#         if fusion_strategy == 'max':
#             fused_sim, _ = torch.max(query_sims, dim=1)
#         elif fusion_strategy == 'mean':
#             fused_sim = torch.mean(query_sims, dim=1)
#         else:
#             raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
#         fused_similarities[:, i] = fused_sim

#     # 6. 基于融合后的相似度进行Top-K检索
#     # top_k_db_indices 的形状是 [num_loc, num_queries]
#     top_k_db_indices = torch.topk(fused_similarities, k=min(num_loc, len(db_names)), dim=0).indices
#     logger.info(f"Computed top-{num_loc} pairs for {len(query_names)} queries.")



#     # --- 可视化代码：使用 t-SNE 可视化特征空间 (适配多句子融合) ---
#     from sklearn.manifold import TSNE
#     import matplotlib.pyplot as plt
#     import matplotlib.cm as cm # 导入颜色映射模块

#     logger.info("Generating t-SNE visualization for the feature space...")

#     # 1. 将所有特征合并到一个数组中 (数据库图像特征 + 所有句子特征)
#     all_features = np.vstack([db_features, text_features.cpu().numpy()])
    
#     # 2. 创建标签来区分不同类型的点
#     #    - 0: 数据库图像
#     #    - 1, 2, 3, ...: 分别对应第1, 第2, 第3, ...个查询图像的句子
#     labels = np.zeros(len(all_features), dtype=int)
#     # 数据库图像的标签为0，已经初始化好了

#     # 为每个查询图像的句子设置唯一的标签 (从1开始)
#     for i, q_name in enumerate(query_names):
#         start, end = query_to_indices[q_name]
#         # 标签的索引要加上数据库特征的数量
#         labels[len(db_names) + start : len(db_names) + end] = i + 1

#     # 3. 初始化并运行 t-SNE
#     tsne = TSNE(n_components=2, perplexity=min(30, len(all_features)-1), max_iter=300, random_state=42)
#     features_2d = tsne.fit_transform(all_features)

#     # 4. 绘制散点图
#     plt.figure(figsize=(14, 12))
    
#     # 绘制数据库图像特征 (灰色点，作为背景)
#     db_mask = labels == 0
#     plt.scatter(features_2d[db_mask, 0], features_2d[db_mask, 1], c='gray', alpha=0.3, s=15, label='Database Images')
    
#     # 为每个查询图像生成一个独特的颜色
#     colors = cm.rainbow(np.linspace(0, 1, len(query_names)))

#     # 逐个查询图像绘制其所有句子的特征点
#     for i, q_name in enumerate(query_names):
#         query_label = i + 1
#         query_mask = labels == query_label
#         if np.any(query_mask): # 确保该查询至少有一个句子
#             plt.scatter(
#                 features_2d[query_mask, 0], 
#                 features_2d[query_mask, 1], 
#                 color=colors[i], 
#                 marker='*', 
#                 s=120, 
#                 label=f'Query: {q_name[:15]}...' if i < 5 else None # 只为前5个查询显示图例
#             )

#     plt.title("t-SNE Visualization of Feature Space (Multi-Sentence Fusion)")
#     plt.xlabel("t-SNE Dimension 1")
#     plt.ylabel("t-SNE Dimension 2")
    
#     # 创建一个更简洁的图例
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label='Database Images', markerfacecolor='gray', markersize=10),
#         Line2D([0], [0], marker='*', color='w', label='Query Sentences', markerfacecolor='red', markersize=15)
#     ]
#     plt.legend(handles=legend_elements, loc='best')
    
#     # 5. 保存图像
#     tsne_path = loc_pairs.parent / "tsne_visualization_fused.png"
#     plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     logger.info(f"Saved t-SNE visualization to {tsne_path}")
#     # --- 可视化代码结束 ---


#     # 7. 写入配对文件 (这部分逻辑不变)
#     with open(loc_pairs, "w") as f:
#         for i, query_name in enumerate(tqdm(query_names, desc="Writing pairs")):
#             retrieved_indices = top_k_db_indices.T[i].cpu().numpy()
#             retrieved_names = [db_names[j] for j in retrieved_indices]
#             for r_name in retrieved_names:
#                 f.write(f"{query_name} {r_name}\n")

#     logger.info(f"Saved top-{num_loc} pairs to {loc_pairs}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Retrieve image pairs based on CLIP text-image similarity. "
#                     "Supports multi-sentence fusion from a CSV file."
#     )
#     parser.add_argument("--clip_features_path", type=Path, required=True)
#     parser.add_argument("--text_input", type=Path, required=True, help="Path to a CSV file containing 'image_name' and 'description' columns.")
#     parser.add_argument("--num_loc", type=int, default=20)
#     parser.add_argument("--output", type=Path, required=True)
#     parser.add_argument("--reference_sfm_path", type=Path, required=True, help="Path to the reference SfM model directory.")
#     parser.add_argument("--fusion_strategy", type=str, default='max', choices=['max', 'mean'], help="Strategy to fuse similarities from multiple sentences.")
#     # query_list_path is no longer needed for CSV mode
    
#     args = parser.parse_args()

#     main(
#         clip_features_path=args.clip_features_path,
#         text_query_or_csv_path=args.text_input,
#         num_loc=args.num_loc,
#         loc_pairs=args.output,
#         reference_sfm_path=args.reference_sfm_path,
#         fusion_strategy=args.fusion_strategy,
#     )






























