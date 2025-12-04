
# from sentence_transformers import SentenceTransformer, util
# import argparse
# import random
# from pathlib import Path
# import pandas as pd
# from PIL import Image
# from typing import List, Optional, Dict, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import open_clip
# from tqdm import tqdm
# import loralib as lora
# import spacy
# from functools import lru_cache
# import h5py

# # ============== Utils & LoRA (与之前版本相同，此处省略) ==============
# def set_seed(seed: int = 42):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def _replace_linear_with_lora_linear(mod: nn.Linear, r=8, alpha=16, dropout=0.0):
#     new = lora.Linear(mod.in_features, mod.out_features, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=(mod.bias is not None))
#     new = new.to(device=mod.weight.device, dtype=mod.weight.dtype)
#     with torch.no_grad():
#         new.weight.copy_(mod.weight.data)
#         if mod.bias is not None:
#             new.bias.copy_(mod.bias.data)
#     return new

# def inject_lora_openclip(model, target='all', last_n_blocks=12, r=8, alpha=16, dropout=0.0):
#     def _inject_block(blk):
#         blk.attn.out_proj = _replace_linear_with_lora_linear(blk.attn.out_proj, r, alpha, dropout)
#         blk.mlp.c_fc = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
#         blk.mlp.c_proj = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)
#     if target == 'all' or target == 'vision':
#         for blk in model.visual.transformer.resblocks[-last_n_blocks:]: _inject_block(blk)
#     if target == 'all' or target == 'text':
#         for blk in model.transformer.resblocks[-last_n_blocks:]: _inject_block(blk)
#     lora.mark_only_lora_as_trainable(model, bias='lora_only')
#     model.logit_scale.requires_grad_(True)
#     return model

# # ============== 新的数据集，加载预计算的区域特征 ==============
# class MaskFeatureDataset(Dataset):
#     def __init__(self, csv_path: Path, preprocess, image_root_dir: Path, mask_features_path: Path):
#         self.df = pd.read_csv(csv_path)
#         self.preprocess = preprocess
#         self.image_root_dir = image_root_dir
        
#         # 加载 HDF5 文件
#         self.mask_features_file = h5py.File(mask_features_path, 'r')
#         self.mask_features = self.mask_features_file['features']
#         # --- 核心修改：加载区域标签 ---
#         if 'region_labels' not in self.mask_features_file:
#             raise ValueError(f"HDF5 文件 {mask_features_path} 中缺少 'region_labels' 数据集。请先运行新版的 extract_mask_features.py。")
#         self.region_labels = self.mask_features_file['region_labels']
#         # --- 修改结束 ---
#         # 创建图像名到 HDF5 索引的映射
#         h5_names = [n.decode() for n in self.mask_features_file['image_names'][:]]
#         self.name_to_idx = {name: i for i, name in enumerate(h5_names)}

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, i):
#         row = self.df.iloc[i]
#         image_name, text = row['image_name'], row['description']
        
#         # 查找预计算特征的索引
#         h5_idx = self.name_to_idx.get(image_name)
#         if h5_idx is None:
#             return None # 如果找不到特征，则跳过此样本

#         try:
#             img = Image.open(self.image_root_dir / image_name).convert("RGB")
#             processed_img = self.preprocess(img)
#             mask_feats = torch.from_numpy(self.mask_features[h5_idx])
#             # --- 核心修改：读取并解码区域标签 ---
#             # 假设标签是以 b'string' 形式存储的
#             labels = [label.decode() for label in self.region_labels[h5_idx]]
#             # --- 修改结束 ---
#             return processed_img, text, mask_feats, labels
#         except Exception:
#             return None

# def mask_collate_fn(batch):
#     batch = [b for b in batch if b is not None]
#     if not batch:
#         return None, None, None, None
#     # --- 核心修改：处理新的标签数据 ---
#     imgs, texts, mask_feats, region_labels = zip(*batch)
#     return torch.stack(imgs), list(texts), torch.stack(mask_feats), list(region_labels)
#     # --- 修改结束 ---

# # ============== 新的模块：投影头与槽位-区域对齐损失 ==============
# class ProjectionHead(nn.Module):
#     """将 MaskFormer 特征投影到 CLIP 特征空间。"""
#     def __init__(self, input_dim, output_dim, hidden_dim=None):
#         super().__init__()
#         hidden_dim = hidden_dim or output_dim
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.gelu = nn.GELU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x):
#         return self.fc2(self.gelu(self.fc1(x)))

# class SlotRegionAlignmentLoss(nn.Module):
#     """槽位-区域对齐损失。"""
#     def __init__(self, temperature=0.1):
#         super().__init__()
#         self.tau = temperature

#     def forward(self, slot_vecs: List[torch.Tensor], region_feats: torch.Tensor):
#         batch_size = region_feats.size(0)
#         total_loss = 0.0
#         num_slots = 0

#         for b in range(batch_size):
#             if not slot_vecs[b]: continue
            
#             regions_b = F.normalize(region_feats[b], dim=-1) # [Num_Regions, Dim]

#             for e_j in slot_vecs[b]:
#                 e_j = F.normalize(e_j.to(regions_b.device), dim=-1)
                
#                 # 计算相似度
#                 sim = (regions_b @ e_j) / self.tau # [Num_Regions]
                
#                 # 正样本：最相似的那个区域
#                 pos_sim = sim.max().unsqueeze(0)

#                 # 负样本：图内其他区域 + 图外所有区域
#                 neg_sims = [sim[sim < sim.max()]]
#                 for nb in range(batch_size):
#                     if nb == b: continue
#                     regions_nb = F.normalize(region_feats[nb], dim=-1)
#                     neg_sims.append((regions_nb @ e_j) / self.tau)
                
#                 all_neg_sim = torch.cat(neg_sims)
                
#                 logits = torch.cat([pos_sim, all_neg_sim])
#                 labels = torch.zeros(1, dtype=torch.long, device=logits.device)
#                 total_loss += F.cross_entropy(logits.unsqueeze(0), labels)
#                 num_slots += 1
        
#         return total_loss / num_slots if num_slots > 0 else torch.tensor(0.0, device=region_feats.device)


# class LocatabilityScorer:
#     def __init__(self, device='cuda'):
#         self.device = device
#         print("Loading Sentence Transformer model for locatability scoring...")
#         self.st_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

#     def __call__(self, texts: List[str], mask_feats: torch.Tensor, region_labels: List[List[str]], loc_threshold=0.4):
#         batch_size = len(texts)
#         loc_scores = torch.zeros(batch_size, device=self.device)

#         text_embeddings = self.st_model.encode(
#             texts, convert_to_tensor=True, device=self.device, show_progress_bar=False
#         )

#         print("\n--- Locatability Scoring ---") # 添加一个清晰的批次分隔符

#         for i in range(batch_size):
#             unique_labels = sorted(list(set(label for label in region_labels[i] if label))) # 过滤掉空字符串
#             if not unique_labels: 
#                 # 即使没有标签，也打印文本以供分析
#                 print(f"  [Score: 0.000] (No regions) Text: {texts[i]}")
#                 continue

#             class_embeddings = self.st_model.encode(
#                 unique_labels, convert_to_tensor=True, device=self.device, show_progress_bar=False
#             )
            
#             sim_matrix_M = util.cos_sim(text_embeddings[i], class_embeddings)
#             w_loc = F.relu(sim_matrix_M)

#             num_regions = (mask_feats[i].abs().sum(dim=1) > 0).sum().float()
#             I_seg_proxy = num_regions / 20.0 # 将分母从200改为20，放大分数差异

#             score = I_seg_proxy * w_loc.mean()
#             loc_scores[i] = score
            
#             # --- 核心修改：打印详细的调试信息 ---
#             print(f"  [Score: {score:.3f}] Img Labels: {unique_labels} | Text: {texts[i]}")
#             # --- 修改结束 ---
            
#         is_locatable_mask = loc_scores > loc_threshold
#         # --- 核心修改：暂时禁用过滤，以便观察所有分数和损失 ---
#         print("--------------------------\n")
#         is_locatable_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
#         # --- 修改结束 ---
#         return is_locatable_mask
    
# # (SlotExtractor 和 Evaluation 函数与之前版本相同，此处省略)
# class SlotExtractor:
#     def __init__(self):
#         try:
#             self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
#         except IOError:
#             exit("spaCy model not found. Run: python3 -m spacy download en_core_web_sm")
#         # --- 核心修改：移除固定的 whitelist ---
#         # self.whitelist = { ... } # 不再需要

#     # --- 核心修改：修改内部处理函数，接收动态标签 ---
#     # 注意：由于现在函数的输出依赖于两个参数，lru_cache 可能会导致错误，因此移除它。
#     def _process_text(self, text: str, image_labels: set):
#         """根据图像中实际存在的标签来提取槽位。"""
#         if not image_labels:
#             return []
        
#         # 提取名词短语，当且仅当其词根存在于当前图像的标签集合中时，才保留
#         slots = [
#             chunk.text for chunk in self.nlp(text.lower()).noun_chunks 
#             if any(token.lemma_ in image_labels for token in chunk)
#         ]
#         return slots

#     # --- 核心修改：修改主调用接口 ---
#     def __call__(self, texts: List[str], region_labels: List[List[str]]):
#         """
#         为批次中的每个文本，使用其对应的图像标签来提取槽位。
        
#         Args:
#             texts: 一批次的文本描述。
#             region_labels: 一批次的区域标签列表，与 texts 一一对应。
#         """
#         # 为每张图的标签列表创建一个集合，用于快速查找
#         image_label_sets = [set(labels) for labels in region_labels]
        
#         # 为每个图文对调用 _process_text
#         return [
#             self._process_text(text, labels_set) 
#             for text, labels_set in zip(texts, image_label_sets)
#         ]

# @torch.no_grad()
# def evaluate(model, val_loader, tokenizer, device):
#     model.eval()
#     all_I, all_T = [], []
#     # --- 核心修复：接收 4 个值而不是 3 个 ---
#     for imgs, texts, _, _ in tqdm(val_loader, desc="Evaluating"):
#     # --- 修复结束 ---
#         if imgs is None: continue
#         imgs = imgs.to(device)
#         toks = tokenizer(texts).to(device)
#         img_feats = F.normalize(model.encode_image(imgs), dim=-1)
#         txt_feats = F.normalize(model.encode_text(toks), dim=-1)
#         all_I.append(img_feats)
#         all_T.append(txt_feats)
#     if not all_I: return None
#     I, T = torch.cat(all_I, dim=0), torch.cat(all_T, dim=0)
#     sim = T @ I.t()
#     gt = torch.arange(len(sim)).to(device)
#     topk = torch.topk(sim, k=20, dim=1).indices
#     r1 = (topk[:, 0] == gt).float().mean().item()
#     r5 = (topk[:, :5] == gt.view(-1, 1)).any(dim=1).float().mean().item()
#     r10 = (topk[:, :10] == gt.view(-1, 1)).any(dim=1).float().mean().item()
#     r20 = (topk[:, :20] == gt.view(-1, 1)).any(dim=1).float().mean().item()
#     return {'R@1': r1, 'R@5': r5, 'R@10': r10, 'R@20': r20}        

# # ============== Main (Mask-Align 版) ==============
# def main():
#     ap = argparse.ArgumentParser(description="Fine-tune CLIP with Mask-Region Alignment.")
#     # --- 路径参数 ---
#     ap.add_argument('--train-csv', type=Path, required=True)
#     ap.add_argument('--val-csv', type=Path, required=True)
#     ap.add_argument('--train-image-root-dir', type=Path, required=True)
#     ap.add_argument('--val-image-root-dir', type=Path, required=True)
#     ap.add_argument('--train-mask-features', type=Path, required=True, help="HDF5 file with precomputed train region features.")
#     ap.add_argument('--val-mask-features', type=Path, required=True, help="HDF5 file with precomputed val region features.")
#     ap.add_argument('--out-dir', type=Path, required=True)
    
#     # --- 模型与训练参数 ---
#     ap.add_argument('--model-name', type=str, default='ViT-L-14')
#     ap.add_argument('--pretrained', type=str, default='datacomp_xl_s13b_b90k')
#     ap.add_argument('--epochs', type=int, default=5)
#     ap.add_argument('--batch-size', type=int, default=16) # 显存占用增加，建议减小
#     ap.add_argument('--lr', type=float, default=5e-5)
#     ap.add_argument('--num-workers', type=int, default=8)
#     ap.add_argument('--use-amp', action='store_true')
#     ap.add_argument('--lambda-slot', type=float, default=1.0, help="Weight for slot-region alignment loss.")
    
#     # (LoRA 参数与之前相同)
#     ap.add_argument('--lora-target', type=str, default='all', choices=['all', 'text', 'vision'])
#     ap.add_argument('--lora-r', type=int, default=8)
#     ap.add_argument('--lora-alpha', type=int, default=16)
#     # --- 核心修复：添加缺失的 LoRA 参数 ---
#     ap.add_argument('--lora-dropout', type=float, default=0.0)
#     ap.add_argument('--lora-last-n', type=int, default=4, help="Apply LoRA to the last N blocks of vision/text towers.")
#     # --- 修复结束 ---

#     args = ap.parse_args()
#     set_seed(42)
#     args.out_dir.mkdir(parents=True, exist_ok=True)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # --- 步骤 1: 设置模型、Tokenizer 和新模块 ---
#     model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
#     tokenizer = open_clip.get_tokenizer(args.model_name)
#     # --- 核心修复：使用所有 LoRA 参数来注入模型 ---
#     model = inject_lora_openclip(
#         model, 
#         target=args.lora_target, 
#         r=args.lora_r, 
#         alpha=args.lora_alpha,
#         dropout=args.lora_dropout,
#         last_n_blocks=args.lora_last_n
#     )
#     # --- 修复结束 ---
#     model.to(device)

#     # 投影头：输入维度需与 extract_mask_features.py 中的 feature_dim 匹配 (Swin-L 是 1024)
#     # 输出维度需与 CLIP text_projection 匹配 (ViT-L 是 768)
#     projection_head = ProjectionHead(input_dim=256, output_dim=768).to(device)
    
#     slot_extractor = SlotExtractor()
#     slot_region_loss_fn = SlotRegionAlignmentLoss().to(device)

#     # --- 核心修改：实例化 LocatabilityScorer ---
#     locatability_scorer = LocatabilityScorer(device=device)
#     # --- 修改结束 ---

#     # --- 步骤 2: 设置数据加载器和优化器 ---
#     train_ds = MaskFeatureDataset(args.train_csv, preprocess, args.train_image_root_dir, args.train_mask_features)
#     val_ds = MaskFeatureDataset(args.val_csv, preprocess, args.val_image_root_dir, args.val_mask_features)
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=mask_collate_fn)
#     val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=mask_collate_fn)

#     trainable_params = list(p for p in model.parameters() if p.requires_grad) + list(projection_head.parameters())
#     optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
#     scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

#     # --- 步骤 3: 开始训练 ---
#     print("Starting training with Mask-Region alignment...")
#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         projection_head.train()
        
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
#         # --- 核心修改：接收新的 region_labels 数据 ---
#         for imgs, texts, mask_feats, region_labels in pbar:
#             if imgs is None: continue
#             # --- 修改结束 ---
#             imgs, mask_feats = imgs.to(device), mask_feats.to(device)
#             toks = tokenizer(texts).to(device)
#             optimizer.zero_grad()

#             with torch.cuda.amp.autocast(enabled=args.use_amp):
#                 # 1. 全局 InfoNCE 损失 (照旧)
#                 img_global = F.normalize(model.encode_image(imgs), dim=-1)
#                 txt_global = F.normalize(model.encode_text(toks), dim=-1)
#                 sim = torch.exp(model.logit_scale) * (img_global @ txt_global.t())
#                 labels = torch.arange(sim.size(0), device=device)
#                 loss_global = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))

#                 # 2. 槽位-区域对齐损失
#                 loss_slot = torch.tensor(0.0, device=device)
                
#                 # --- 核心修复：重构槽位损失的计算流程 ---
#                 # 1. 计算哪些样本是“可定位的”
#                 locatable_mask = locatability_scorer(texts, mask_feats, region_labels)
#                 num_locatable = locatable_mask.sum()

#                 # 2. 只为“可定位的”样本计算 loss_slot
#                 if num_locatable > 0:
#                     # a. 筛选出可定位样本的特征
#                     loc_mask_feats = mask_feats[locatable_mask]
                    
#                     # b. 为可定位的样本提取槽位
#                     loc_texts = [texts[i] for i, is_loc in enumerate(locatable_mask) if is_loc]
#                     loc_region_labels = [region_labels[i] for i, is_loc in enumerate(locatable_mask) if is_loc]
#                     loc_slots = slot_extractor(loc_texts, loc_region_labels)

#                     # c. 将所有可定位样本的槽位文本展平
#                     flat_loc_slots = [s for b_slots in loc_slots for s in b_slots]

#                     if flat_loc_slots: # 确保提取到了槽位
#                         # d. 编码槽位文本
#                         slot_toks = tokenizer(flat_loc_slots).to(device)
#                         slot_embeds = F.normalize(model.encode_text(slot_toks), dim=-1)
                        
#                         # e. 将展平的槽位向量重新分配到每个样本
#                         slot_vecs_batch, ptr = [], 0
#                         for b_slots in loc_slots:
#                             num_s = len(b_slots)
#                             slot_vecs_batch.append(list(slot_embeds[ptr : ptr + num_s]))
#                             ptr += num_s
                        
#                         # f. 投影区域特征并计算损失
#                         projected_regions = projection_head(loc_mask_feats)
#                         loss_slot = slot_region_loss_fn(slot_vecs_batch, projected_regions)
#                 # --- 修复结束 ---
#                 # 3. 合并总损失
#                 loss = loss_global + args.lambda_slot * loss_slot
            
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             pbar.set_postfix(L_global=f"{loss_global.item():.3f}", L_slot=f"{loss_slot.item():.3f}")

#         # --- 评估与保存 ---
#         metrics = evaluate(model, val_loader, tokenizer, device)
#         if metrics:
#             msg = " | ".join([f"{k}={v*100:.2f}%" for k,v in metrics.items()])
#             print(f"-> [Val Epoch {epoch}] {msg}")

#         # --- 核心修复：在保存检查点时添加元数据 ---
#         lora_state = {n: p.detach().cpu() for n, p in model.named_parameters() if 'lora_' in n or 'logit_scale' in n}
#         proj_head_state = {k: v.cpu() for k, v in projection_head.state_dict().items()}
        
#         checkpoint = {
#             'lora_state': lora_state, 
#             'projection_head_state': proj_head_state,
#             'model_name': args.model_name,
#             'pretrained': args.pretrained,
#             'lora_r': args.lora_r,
#             'lora_alpha': args.lora_alpha,
#             # --- 核心修复：保存所有 LoRA 超参数 ---
#             'lora_dropout': args.lora_dropout,
#             'lora_last_n': args.lora_last_n,
#             # --- 修复结束 ---
#             'mode': 'lora' 
#         }

#         torch.save(checkpoint, args.out_dir / f'maskclip_lora_epoch_{epoch}.pt')
#         print(f"Checkpoint for epoch {epoch} saved.\n")

# if __name__ == "__main__":
#     main()



from hloc import logger
import argparse
import random
from pathlib import Path
import pandas as pd
from PIL import Image
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open_clip
from tqdm import tqdm
import loralib as lora

# --- 新增：导入 transformers 库 ---
from transformers import CLIPProcessor, CLIPModel
# --- 结束新增 ---
from transformers import CLIPTokenizer

def set_seed(seed: int = 42):
    """设置随机种子以确保可复现性。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============== Dataset ==============


class CsvDataset(Dataset):
    """
    一个数据集，用于加载图文对。支持从多个CSV文件中加载长、短、难负样本描述。
    """
    def __init__(self, csv_path: Path, preprocess, image_root_dir: Path, 
                 short_csv_path: Optional[Path] = None, 
                 hard_negative_csv_path: Optional[Path] = None,
                 training_text_mode: str = 'long', 
                 for_hf=False):
        df = pd.read_csv(csv_path)
        assert 'image_name' in df.columns and 'description' in df.columns, \
            f"主CSV文件 {csv_path} 必须包含 'image_name' 和 'description' 列。"

        self.use_short_desc = False
        if short_csv_path:
            logger.info(f"正在加载短描述文件: {short_csv_path}")
            short_df = pd.read_csv(short_csv_path)
            assert 'image_name' in short_df.columns and 'description' in short_df.columns, \
                f"短描述CSV文件 {short_csv_path} 必须包含 'image_name' 和 'description' 列。"
            
            # 重命名列以避免冲突
            short_df = short_df.rename(columns={'description': 'short_description'})
            
            # 基于 image_name 合并两个 DataFrame
            df = pd.merge(df, short_df[['image_name', 'short_description']], on='image_name', how='inner')
            logger.info(f"成功合并 {len(df)} 个图文对进行混合训练。")
            self.use_short_desc = True
        # --- 新增结束 ---
        # --- 核心修改：当没有短描述时，在 hard_negative 模式下复用长描述 ---
        elif training_text_mode == 'hard_negative':
            logger.info("在 'hard_negative' 模式下未提供 short_csv，将使用主 'description' 列作为 L_hard 的正样本。")
            # 创建一个 'short_description' 列，其内容与 'description' 列相同
            df['short_description'] = df['description']
            self.use_short_desc = True
        # --- 修改结束 ---
        # --- 新增：加载并合并难负样本 ---
        self.use_hard_negatives = False
        if hard_negative_csv_path:
            logger.info(f"正在加载难负样本文件: {hard_negative_csv_path}")
            neg_df = pd.read_csv(hard_negative_csv_path)
            assert 'image_name' in neg_df.columns and 'description' in neg_df.columns, \
                f"难负样本CSV文件 {hard_negative_csv_path} 必须包含 'image_name' 和 'description' 列。"
            
            neg_df = neg_df.rename(columns={'description': 'hard_negative_description'})
            
            # 合并难负样本
            df = pd.merge(df, neg_df[['image_name', 'hard_negative_description']], on='image_name', how='inner')
            logger.info(f"成功合并 {len(df)} 个难负样本进行训练。")
            self.use_hard_negatives = True
        # --- 新增结束 ---


        self.paths = [image_root_dir.joinpath(p) for p in df['image_name']]
        self.long_texts = df['description'].tolist()
        if self.use_short_desc:
            self.short_texts = df['short_description'].tolist()
        if self.use_hard_negatives:
            self.hard_negative_texts = df['hard_negative_description'].tolist()
        
        # --- 核心修复：在 hard_negative 模式下，确保 short_texts 存在 ---
        # 这种模式下，主 CSV 文件被视为正样本（即短描述）
        if training_text_mode == 'hard_negative' and not self.use_short_desc:
            logger.info("在 'hard_negative' 模式下，将主 'description' 列用作正样本（短文本）。")
            self.short_texts = self.long_texts
            self.use_short_desc = True # 设置标志位以通过后续检查
        # --- 修复结束 ---
        
        self.preprocess = preprocess
        self.for_hf = for_hf
        # --- 新增：保存训练模式 ---
        self.mode = training_text_mode
        if self.mode == 'short' and not self.use_short_desc:
            raise ValueError("--training-text-mode 'short' 需要提供 --train-short-csv 文件。")
        if self.mode == 'hard_negative' and not self.use_hard_negatives:
            raise ValueError("--training-text-mode 'hard_negative' 需要提供 --train-hard-negative-csv 文件。")


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        text_to_return = None
        # --- 核心修改：调整 __getitem__ 的返回结构 ---
        if self.mode == 'hard_negative':
            # 对于第二阶段，我们需要所有三种文本
            long_text = self.long_texts[i]
            positive_text = self.short_texts[i]
            negative_text = self.hard_negative_texts[i]
            # 将它们打包成一个元组
            text_to_return = (long_text, positive_text, negative_text)
        elif self.mode == 'mixed':
            text_to_return = (self.long_texts[i], self.short_texts[i])
        elif self.mode == 'short':
            text_to_return = self.short_texts[i]
        else: # 'long' 模式或默认情况
            text_to_return = self.long_texts[i]
        # --- 修改结束 ---

        try:
            img = Image.open(self.paths[i]).convert("RGB")
            processed_img = img if self.for_hf else self.preprocess(img)
            return processed_img, text_to_return, i
        except Exception as e:
            return None, text_to_return, i


def collate_fn(batch):
    """
    自定义的 collate 函数，用于过滤掉加载失败的图像，并处理多种文本组合。
    """
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, ([], [], []), None # 返回一个能安全解包的结构
    
    imgs, texts, idxs = zip(*batch)
    
    # --- 核心修改：处理新的三元组文本格式 ---
    if isinstance(texts[0], tuple):
        num_parts = len(texts[0])
        if num_parts == 3: # hard_negative 模式: (long, positive, negative)
            longs, positives, negatives = zip(*texts)
            texts_to_return = (list(longs), list(positives), list(negatives))
        elif num_parts == 2: # mixed 模式: (long, short)
            list1, list2 = zip(*texts)
            texts_to_return = (list(list1), list(list2), []) # 添加一个空的第三元素以保持结构一致
        else:
            raise ValueError(f"未知的文本元组格式，包含 {num_parts} 个元素。")
    else:
        # 单一文本模式 ('long' 或 'short')
        texts_list = list(texts)
        texts_to_return = (texts_list, [], []) # 同样保持三元组结构
    # --- 修改结束 ---
    
    try:
        imgs_stacked = torch.stack(imgs, dim=0)
        return imgs_stacked, texts_to_return, torch.tensor(idxs, dtype=torch.long)
    except TypeError: 
        return list(imgs), texts_to_return, torch.tensor(idxs, dtype=torch.long)


# # ============== LoRA Injection (Advanced) ==============



# ============== LoRA Injection (Corrected QKV Injection) ==============

def _replace_linear_with_lora_linear(mod: nn.Linear, r=8, alpha=16, dropout=0.0):
    """用 loralib.Linear 替换一个 nn.Linear 层，并复制权重。"""
    new = lora.Linear(
        mod.in_features, mod.out_features, r=r, lora_alpha=alpha, 
        lora_dropout=dropout, bias=(mod.bias is not None)
    )
    new = new.to(device=mod.weight.device, dtype=mod.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(mod.weight.data)
        if mod.bias is not None:
            new.bias.copy_(mod.bias.data)
    return new

class LoRAInjectedQKVAttention(nn.Module):
    """
    一个自定义的注意力模块，正确地将 LoRA 注入到 Q, K, V 投影中，
    同时保持初始基线不变。
    """
    def __init__(self, original_attn: nn.MultiheadAttention, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original_attn = original_attn
        
        d_model = original_attn.embed_dim
        
        # 关键：我们为 Q, K, V 分别创建独立的 LoRA 线性层。
        # 这些层在初始化时，其 LoRA 增量为零。
        self.q_proj = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=True)
        self.k_proj = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=True)
        self.v_proj = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=True)
        
        # 从原始的 in_proj_weight 中复制权重到我们新的独立层
        with torch.no_grad():
            # in_proj_weight 的形状是 (3 * d_model, d_model)，按顺序是 W_q, W_k, W_v
            q_w, k_w, v_w = original_attn.in_proj_weight.chunk(3)
            self.q_proj.weight.copy_(q_w)
            self.k_proj.weight.copy_(k_w)
            self.v_proj.weight.copy_(v_w)
            
            # 同样处理偏置
            if original_attn.in_proj_bias is not None:
                q_b, k_b, v_b = original_attn.in_proj_bias.chunk(3)
                self.q_proj.bias.copy_(q_b)
                self.k_proj.bias.copy_(k_b)
                self.v_proj.bias.copy_(v_b)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, **kwargs):
        
        # --- 核心修复：使用 F.scaled_dot_product_attention ---
        
        # 1. 使用我们新的、可训练的 LoRA 层来计算 Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. 将 Q, K, V 变形以适应多头注意力的格式
        # (B, L, E) -> (B, L, H, D) -> (B, H, L, D)
        # B=批大小, L=序列长度, E=嵌入维度, H=头数, D=每个头的维度
        B, L, E = q.shape
        H = self.original_attn.num_heads
        D = E // H
        
        q = q.view(B, L, H, D).transpose(1, 2)
        k = k.view(B, L, H, D).transpose(1, 2)
        v = v.view(B, L, H, D).transpose(1, 2)

        # 3. 调用 PyTorch 内置的高效点积注意力实现
        # 这个函数会自动处理缩放、掩码和 softmax
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.original_attn.dropout if self.original_attn.training else 0.0
        )

        # 4. 将输出变形回原始格式
        # (B, H, L, D) -> (B, L, H, D) -> (B, L, E)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, E)

        # 5. 应用最终的输出投影（这个 out_proj 已经在外部被替换为 LoRA 层）
        attn_output = self.original_attn.out_proj(attn_output)
        
        return attn_output, None
        # --- 修复结束 ---


def inject_lora_openclip(model, target='all', last_n_blocks=12, r=8, alpha=16, dropout=0.0):
    """
    向 OpenCLIP 模型的 Transformer 块中注入 LoRA 层 (高级版，支持 QKV)。
    """
    
    def _inject_block(blk):
        # 1. 注入 Q, K, V (使用我们新的、正确的实现)
        blk.attn = LoRAInjectedQKVAttention(blk.attn, r, alpha, dropout)
        
        # 2. 注入 out_proj 和 MLP 层 (保持不变)
        # 注意：现在 out_proj 是新模块 `blk.attn` 的一个属性
        blk.attn.original_attn.out_proj = _replace_linear_with_lora_linear(blk.attn.original_attn.out_proj, r, alpha, dropout)
        blk.mlp.c_fc = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
        blk.mlp.c_proj = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)

    # 根据 target 参数选择性地注入 LoRA
    if target == 'all' or target == 'vision':
        print("Injecting LoRA into Vision Transformer (QKV, out_proj & MLP)...")
        for blk in model.visual.transformer.resblocks[-last_n_blocks:]:
            _inject_block(blk)
    
    if target == 'all' or target == 'text':
        print("Injecting LoRA into Text Transformer (QKV, out_proj & MLP)...")
        for blk in model.transformer.resblocks[-last_n_blocks:]:
            _inject_block(blk)
    
    lora.mark_only_lora_as_trainable(model, bias='lora_only')
    model.logit_scale.requires_grad_(True)
    return model






# ============== Evaluation ==============
@torch.no_grad()
def recall_at_k(text_feats: torch.Tensor, image_feats: torch.Tensor, k_list: List[int]):
    sim = text_feats @ image_feats.t()
    N = sim.size(0)
    topk = torch.topk(sim, k=max(k_list), dim=1).indices
    out = {}
    gt = torch.arange(N, device=sim.device).view(-1, 1)
    for k in k_list:
        hit = (topk[:, :k] == gt).any(dim=1)
        recall_val = hit.float().mean().item()
        out[f'R@{k}'] = recall_val
    return out

# @torch.no_grad()
# def evaluate(model, val_loader, tokenizer, device, args):
#     # ...
#     all_I, all_T = [], []
    
#     # --- 核心修复：正确解包三元组文本 ---
#     for i, (imgs, (long_texts, _, _), _) in enumerate(tqdm(val_loader, desc="Evaluating (open_clip)")):
#         # 1. 检查批次是否为空
#         if imgs is None:
#             if i == 0:
#                 logger.warning("评估时遇到空批次，可能是验证集数据加载失败。请检查 val_csv 和 val_image_root_dir。")
#             continue

#         # 2. 检查文本是否为空
#         if not long_texts:
#             continue
#     # --- 修复结束 ---

#         imgs = imgs.to(device)
        
#         # 确保 tokenizer 接收的是一个字符串列表
#         toks = tokenizer(long_texts).to(device)

#         img_feats = F.normalize(model.encode_image(imgs), dim=-1)
#         txt_feats = F.normalize(model.encode_text(toks), dim=-1)
        
#         all_I.append(img_feats)
#         all_T.append(txt_feats)
    
#     if not all_I: return None
#     I, T = torch.cat(all_I, dim=0), torch.cat(all_T, dim=0)
#     return recall_at_k(T, I, args.eval_k)


def evaluate(model, val_loader, tokenizer, device, args):
    model.eval()
    all_I, all_T = [], []
    for imgs, texts, _ in tqdm(val_loader, desc="Evaluating (open_clip)"):
        if imgs is None: continue
        imgs = imgs.to(device, non_blocking=True)

        # 展平文本列表
        if isinstance(texts[0], (list, tuple)):
            texts = [text for sublist in texts for text in sublist]

        toks = tokenizer(texts).to(device)
        zi = F.normalize(model.encode_image(imgs), dim=-1)
        zt = F.normalize(model.encode_text(toks), dim=-1)
        all_I.append(zi)
        all_T.append(zt)
    if not all_I: return None
    I, T = torch.cat(all_I, dim=0), torch.cat(all_T, dim=0)
    return recall_at_k(T, I, args.eval_k)


# --- 新增：Hugging Face 模型的评估函数 ---
@torch.no_grad()
def evaluate_hf_baseline(hf_model_id, val_ds, device, args):
    print(f"\nLoading Hugging Face model for baseline: {hf_model_id}")
    hf_model = CLIPModel.from_pretrained(hf_model_id).to(device)
    hf_processor = CLIPProcessor.from_pretrained(hf_model_id)
    
    # 为 HF 模型创建一个专用的 DataLoader
    hf_val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    hf_model.eval()
    all_I, all_T = [], []
    # --- 核心修复：正确解包三元组文本 ---
    for imgs, (long_texts, _, _), _ in tqdm(hf_val_loader, desc="Evaluating (Hugging Face)"):
        if imgs is None: continue
        
        # 只将长描述列表传递给 processor
        inputs = hf_processor(text=long_texts, images=imgs, return_tensors="pt", padding=True, truncation=True)
    # --- 修复结束 ---
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        img_feats = hf_model.get_image_features(pixel_values=inputs['pixel_values'])
        txt_feats = hf_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        zi = F.normalize(img_feats, dim=-1)
        zt = F.normalize(txt_feats, dim=-1)
        all_I.append(zi)
        all_T.append(zt)
    
    if not all_I: return None
    I, T = torch.cat(all_I, dim=0), torch.cat(all_T, dim=0)
    return recall_at_k(T, I, args.eval_k)
# --- 结束新增 ---

# ============== Main ==============
def main():
    ap = argparse.ArgumentParser(description="Fine-tune CLIP with LoRA for image-text retrieval.")
    ap.add_argument('--train-csv', type=Path, required=True)
    ap.add_argument('--val-csv', type=Path, required=True)
    ap.add_argument('--train-image-root-dir', type=Path, required=True)
    ap.add_argument('--val-image-root-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--model-name', type=str, default='ViT-L-14')
    ap.add_argument('--pretrained', type=str, default='datacomp_xl_s13b_b90k')
    # --- 核心修复：将 'hard_negative' 添加到允许的选项中 ---
    ap.add_argument('--training-text-mode', type=str, default='long', choices=['mixed', 'long', 'short', 'hard_negative'],
                    help="指定训练时使用的文本: 'mixed', 'long', 'short', 或 'hard_negative'。")
    # --- 修复结束 ---

    # # --- 核心修改：将 pretrained 参数的默认值指向本地文件 ---
    # ap.add_argument('--pretrained', type=str, default='./datacomp_xl_s13b_b90k_openclip.pt', help="Path to local open_clip checkpoint or online alias.")
    # # --- 修复结束 ---

    # --- 新增：用于指定 HF 模型的 ID ---
    ap.add_argument('--hf-model-id', type=str, default='laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', help="用于基线比较的 Hugging Face 模型 ID。")
    # --- 结束新增 ---
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--use-amp', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lora-last-n', type=int, default=12)
    ap.add_argument('--lora-r', type=int, default=8) # 8 
    ap.add_argument('--lora-alpha', type=int, default=16)
    ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--eval-k', nargs='+', type=int, default=[1, 5, 10, 20])
    # --- 核心修改：添加一个新的命令行参数来控制 LoRA 注入目标 ---
    ap.add_argument('--lora-target', type=str, default='all', choices=['all', 'text', 'vision'], 
                    help="指定 LoRA 注入的目标: 'all', 'text', 或 'vision'。")
    # --- 修改结束 ---
    # --- 新增：用于接收短描述和难负样本文件的参数 ---
    ap.add_argument('--train-short-csv', type=Path, default=None, help="可选的、包含简短描述的训练CSV文件。")
    ap.add_argument('--train-hard-negative-csv', type=Path, default=None, help="可选的、包含难负样本描述的训练CSV文件。")
    # --- 新增：为 L_hard 添加权重参数 ---
    ap.add_argument('--lambda-hard', type=float, default=0.5, help="Weight for hard negative loss in hard_negative mode.")
    
    args = ap.parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 步骤 1: 评估 Hugging Face 基线 ---
    val_ds_for_hf = CsvDataset(csv_path=args.val_csv, preprocess=None, image_root_dir=args.val_image_root_dir, for_hf=True)
    hf_metrics = evaluate_hf_baseline(args.hf_model_id, val_ds_for_hf, device, args)
    if hf_metrics:
        msg = " | ".join([f"{k}={v*100:.2f}%" for k,v in hf_metrics.items()])
        print(f"-> [Hugging Face Baseline] {msg}\n")

    # --- 步骤 2: 设置 open_clip 训练流程 ---
    print("Loading open_clip model and preprocessor...")
    
    # --- 核心修复：区分基础模型加载和 LoRA 权重加载 ---
    # 1. 确定基础模型的预训练权重
    # 如果 --pretrained 指向一个 .pt 文件，我们假设它是 LoRA 权重，
    # 那么基础模型就使用默认的 'datacomp_xl_s13b_b90k'。
    # 否则，--pretrained 本身就是基础模型的名字。
    base_pretrained_name = 'datacomp_xl_s13b_b90k' # <-- 默认基础模型
    lora_checkpoint_path = None
    if args.pretrained and args.pretrained.endswith('.pt'):
        lora_checkpoint_path = args.pretrained
        print(f"LoRA checkpoint specified: {lora_checkpoint_path}. Using '{base_pretrained_name}' as base model.")
    else:
        # 如果用户提供了非 .pt 的预训练名，则使用它
        base_pretrained_name = args.pretrained
        print(f"Base model specified: {base_pretrained_name}. No LoRA checkpoint to load.")

    # 2. 加载基础模型
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=base_pretrained_name, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)



    print("Injecting LoRA layers...")
    # 3. 注入 LoRA 层
    model = inject_lora_openclip(
        model, target=args.lora_target, last_n_blocks=args.lora_last_n, r=args.lora_r, 
        alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    model.to(device)

    # 4. 如果指定了 LoRA 检查点，现在加载它
    if lora_checkpoint_path:
        print(f"Loading LoRA weights from {lora_checkpoint_path}...")
        checkpoint = torch.load(lora_checkpoint_path, map_location=device)
        if 'lora_state' in checkpoint:
            # 加载 LoRA 权重，strict=False 允许只加载存在的键
            model.load_state_dict(checkpoint['lora_state'], strict=False)
            print("Successfully loaded LoRA state.")
        else:
            # 为了兼容旧的检查点格式
            model.load_state_dict(checkpoint, strict=False)
            print("Successfully loaded legacy LoRA checkpoint.")
    # --- 修复结束 ---
    
    print("Loading datasets for open_clip training...")
    # --- 核心修复：在实例化训练集时传入所有相关的 CSV 路径 ---
    train_ds = CsvDataset(
        csv_path=args.train_csv, 
        preprocess=preprocess, 
        image_root_dir=args.train_image_root_dir,
        short_csv_path=args.train_short_csv,
        hard_negative_csv_path=args.train_hard_negative_csv,
        training_text_mode=args.training_text_mode
    )
    
    # --- 核心修复：创建验证集时不传入任何训练集相关的文件 ---
    # 验证集只需要它自己的 val.csv 文件即可。
    val_ds = CsvDataset(
        csv_path=args.val_csv, 
        preprocess=preprocess, 
        image_root_dir=args.val_image_root_dir,
        training_text_mode='long' # 评估时始终使用长描述
    )
    # --- 修复结束 ---

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)


    # --- 步骤 3: 评估 open_clip 基线 ---
    print("\nEvaluating open_clip baseline performance on the validation set...")
    baseline_metrics = evaluate(model, val_loader, tokenizer, device, args)
    if baseline_metrics:
        msg = " | ".join([f"{k}={v*100:.2f}%" for k,v in baseline_metrics.items()])
        print(f"-> [OpenCLIP Baseline] {msg}\n")

    # --- 步骤 3.5: 评估 Hugging Face 基线 ---
    if args.hf_model_id:
        print(f"\nLoading Hugging Face model for baseline: {args.hf_model_id}")
        # --- 核心修复：为 HF 评估创建一个独立的、干净的 Dataset 实例 ---
        val_ds_for_hf = CsvDataset(
            csv_path=args.val_csv,
            preprocess=None,  # Preprocessing is done inside evaluate_hf_baseline
            image_root_dir=args.val_image_root_dir,
            training_text_mode='long', # 强制使用 'long' 模式
            for_hf=True
        )
        # --- 修复结束 ---
        hf_metrics = evaluate_hf_baseline(args.hf_model_id, val_ds_for_hf, device, args)
        if hf_metrics:
            msg = " | ".join([f"{k}={v*100:.2f}%" for k,v in hf_metrics.items()])
            print(f"-> [HuggingFace Baseline] {msg}\n")

    # --- 步骤 4: 开始训练 ---
    print(f"Starting training in '{args.training_text_mode}' mode...")
    # 获取 tokenizer 的上下文长度，用于对比
    context_length = model.context_length

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, step = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, (imgs, texts, _) in enumerate(pbar):
            if imgs is None: continue
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                img_feats = F.normalize(model.encode_image(imgs), dim=-1)
                
                # --- 核心修复：实现 L_global + β * L_hard 的组合损失 ---
                if args.training_text_mode == 'hard_negative':
                    long_texts, pos_texts, neg_texts = texts
                    batch_size = img_feats.size(0)
                    labels = torch.arange(batch_size, device=device)

                    # --- 1. 计算 L_global (基于完整长描述) ---
                    long_toks = tokenizer(long_texts).to(device)
                    long_txt_feats = F.normalize(model.encode_text(long_toks), dim=-1)

                    sim_global = torch.exp(model.logit_scale) * (img_feats @ long_txt_feats.t())
                    loss_global = 0.5 * (F.cross_entropy(sim_global, labels) + F.cross_entropy(sim_global.t(), labels))

                    # --- 2. 计算 L_hard (基于短文本+难负样本) ---
                    # 编码正负文本
                    hard_texts = pos_texts + neg_texts
                    hard_toks = tokenizer(hard_texts).to(device)
                    hard_txt_feats = F.normalize(model.encode_text(hard_toks), dim=-1)
                    
                    # 计算 Image-to-Text 相似度矩阵: [N_img, 2N_text]
                    sim_hard = torch.exp(model.logit_scale) * (img_feats @ hard_txt_feats.t())
                    
                    # L_hard 仅计算 Image -> Text 方向，以聚焦于区分难负样本
                    loss_hard = F.cross_entropy(sim_hard, labels)

                    # --- 3. 组合总损失 ---
                    loss = loss_global + args.lambda_hard * loss_hard
                    
                    pbar.set_postfix(L_global=f"{loss_global.item():.3f}", L_hard=f"{loss_hard.item():.3f}")

                elif args.training_text_mode == 'mixed':
                    long_texts, short_texts = texts
                    
                    # --- 新增：在每个 epoch 的第一个 batch 打印截断信息 ---
                    if i == 0:
                        print(f"\n--- Epoch {epoch} Tokenizer Truncation Check (Context Length: {context_length}) ---")
                        # 对第一个长文本进行分词，但不移动到 device
                        sample_long_toks = tokenizer(long_texts[0])
                        # .sum() > 0 可以计算出非 PAD 部分的有效 token 数量
                        num_tokens = (sample_long_toks > 0).sum().item()
                        print(f"  [Long Text Sample] Original: \"{long_texts[0][:200]}...\"")
                        print(f"  >>> Tokenized to {num_tokens} tokens.")
                        if num_tokens >= context_length:
                            print("      WARNING: Text was likely truncated.")
                        
                        sample_short_toks = tokenizer(short_texts[0])
                        num_tokens_short = (sample_short_toks > 0).sum().item()
                        print(f"  [Short Text Sample] Original: \"{short_texts[0]}\"")
                        print(f"  >>> Tokenized to {num_tokens_short} tokens.")
                        print("--------------------------------------------------")
                    # --- 新增结束 ---

                    # 1. 将长、短两种文本合并，并一次性编码
                    all_texts = long_texts + short_texts
                    all_toks = tokenizer(all_texts).to(device)
                    all_txt_feats = F.normalize(model.encode_text(all_toks), dim=-1)
                    
                    batch_size = img_feats.size(0)
                    long_txt_feats = all_txt_feats[:batch_size]
                    short_txt_feats = all_txt_feats[batch_size:]
                    
                    labels = torch.arange(batch_size, device=device)

                    # 2. 计算 Image-LongText 损失 (主任务)
                    sim_i_l = torch.exp(model.logit_scale) * (img_feats @ long_txt_feats.t())
                    loss_img_long = 0.5 * (F.cross_entropy(sim_i_l, labels) + F.cross_entropy(sim_i_l.t(), labels))

                    # 3. 计算 Image-ShortText 损失 (辅助任务)
                    sim_i_s = torch.exp(model.logit_scale) * (img_feats @ short_txt_feats.t())
                    loss_img_short = 0.5 * (F.cross_entropy(sim_i_s, labels) + F.cross_entropy(sim_i_s.t(), labels))

                    # --- 核心修改：移除文本对齐损失，直接加权合并 ---
                    loss = loss_img_long + args.lambda_short * loss_img_short
                    pbar.set_postfix(L_long=f"{loss_img_long.item():.3f}", L_short=f"{loss_img_short.item():.3f}")
                    # --- 修改结束 ---

                else: # 'long' 或 'short' 模式 (保持不变)
                    text_list, _ = texts 
                    toks = tokenizer(text_list).to(device)
                    txt_feats = F.normalize(model.encode_text(toks), dim=-1)
                    sim = torch.exp(model.logit_scale) * (img_feats @ txt_feats.t())
                    labels = torch.arange(sim.size(0), device=device)
                    loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                # --- 修改结束 ---
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            step += 1
            pbar.set_postfix(loss=f"{(total_loss/step):.4f}")


        metrics = evaluate(model, val_loader, tokenizer, device, args)
        if metrics:
            msg = " | ".join([f"{k}={v*100:.2f}%" for k,v in metrics.items()])
            print(f"-> [Val Epoch {epoch}] {msg}")

        lora_state = {n: p.detach().cpu() for n, p in model.named_parameters() if 'lora_' in n or 'logit_scale' in n}
        
        # --- 核心修复：保存正确的元数据 ---
        # 无论 --pretrained 参数是什么，我们都保存真正的基础模型名称
        checkpoint = {
            'mode': 'lora', 'model_name': args.model_name, 'pretrained': base_pretrained_name,
            'lora_last_n': args.lora_last_n, 'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha, 'lora_dropout': args.lora_dropout,
            'lora_state': lora_state
        }
        # --- 修复结束 ---
        torch.save(checkpoint, args.out_dir / f'lora_epoch_{epoch}.pt')
        print(f"LoRA checkpoint for epoch {epoch} saved to {args.out_dir / f'lora_epoch_{epoch}.pt'}\n")

if __name__ == "__main__":
    main()



    # python3 train_localclip.py \
    # --train-csv datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/00_train_qwen3.csv \
    # --val-csv datasets/SensorKit/clipdata/astri_412_down03/val_qwen3.csv \
    # --train-image-root-dir datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/ \
    # --val-image-root-dir datasets/SensorKit/clipdata/astri_412_down03/ \
    # --train-mask-features datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/train_mask_features.h5 \
    # --val-mask-features datasets/SensorKit/clipdata/astri_412_down03/val_mask_features.h5 \
    # --out-dir datasets/SensorKit/clipdata/finetune_output/00_qwen3_clipdata_mask_prepare_sparse03_remove/ \
    # --epochs 100 \
    # --batch-size 16 \
    # --lr 5e-5 \
    # --num-workers 8 \
    # --use-amp \
    # --lora-target text


