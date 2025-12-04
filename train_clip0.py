# import argparse
# import random
# from pathlib import Path
# import pandas as pd
# from PIL import Image
# from typing import List

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import open_clip
# from tqdm import tqdm
# import loralib as lora

# # ============== Utils ==============

# def set_seed(seed: int = 42):
#     """设置随机种子以确保可复现性。"""
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# # ============== Dataset ==============

# class CsvDataset(Dataset):
#     """
#     一个简单的数据集，用于加载一一对应的图文对。
#     CSV 文件必须包含 'filepath' 和 'title' 两列。
#     """
#     def __init__(self, csv_path: Path, preprocess, image_root_dir: Path):
#         df = pd.read_csv(csv_path)
#         assert 'filepath' in df.columns and 'title' in df.columns, \
#             f"CSV 文件 {csv_path} 必须包含 'filepath' 和 'title' 列。"

#         self.paths = [image_root_dir.joinpath(p) for p in df['filepath']]
#         self.texts = df['title'].tolist()
#         self.preprocess = preprocess

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, i):
#         """返回图像、文本和其索引。"""
#         try:
#             img = self.preprocess(Image.open(self.paths[i]).convert("RGB"))
#             return img, self.texts[i], i
#         except Exception as e:
#             # print(f"警告：加载图像失败 {self.paths[i]}: {e}")
#             return None, self.texts[i], i

# def collate_fn(batch):
#     """自定义的 collate 函数，用于过滤掉加载失败的图像。"""
#     batch = [b for b in batch if b[0] is not None]
#     if not batch:
#         return None, None, None
    
#     imgs, texts, idxs = zip(*batch)
#     return torch.stack(imgs, dim=0), list(texts), torch.tensor(idxs, dtype=torch.long)

# # ============== LoRA Injection ==============

# def _replace_linear_with_lora_linear(mod: nn.Linear, r=8, alpha=16, dropout=0.0):
#     new = lora.Linear(
#         mod.in_features, mod.out_features, r=r, lora_alpha=alpha, 
#         lora_dropout=dropout, bias=(mod.bias is not None)
#     )
#     new = new.to(device=mod.weight.device, dtype=mod.weight.dtype)
#     with torch.no_grad():
#         new.weight.copy_(mod.weight.data)
#         if mod.bias is not None:
#             new.bias.copy_(mod.bias.data)
#     return new

# def inject_lora_openclip(model, last_n_blocks=12, r=8, alpha=16, dropout=0.0):  
#     """向 OpenCLIP 模型的 Transformer 块中注入 LoRA 层。"""
#     for blk in model.visual.transformer.resblocks[-last_n_blocks:]:
#         blk.attn.out_proj = _replace_linear_with_lora_linear(blk.attn.out_proj, r, alpha, dropout)
#         blk.mlp.c_fc      = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
#         blk.mlp.c_proj    = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)
    
#     for blk in model.transformer.resblocks[-last_n_blocks:]:
#         blk.attn.out_proj = _replace_linear_with_lora_linear(blk.attn.out_proj, r, alpha, dropout)
#         blk.mlp.c_fc      = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
#         blk.mlp.c_proj    = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)
    
#     lora.mark_only_lora_as_trainable(model, bias='lora_only')
#     model.logit_scale.requires_grad_(True)
#     return model

# # ============== Recall@K Evaluation ==============

# @torch.no_grad()
# def recall_at_k(text_feats: torch.Tensor, image_feats: torch.Tensor, k_list: List[int]):
#     """
#     计算一一对应的图文检索召回率 (T->I)。
#     """
#     sim = text_feats @ image_feats.t()
#     N = sim.size(0)
#     topk = torch.topk(sim, k=max(k_list), dim=1).indices

#     out = {}
#     gt = torch.arange(N, device=sim.device).view(-1, 1)

#     for k in k_list:
#         hit = (topk[:, :k] == gt).any(dim=1)
#         recall_val = hit.float().mean().item()
#         out[f'R@{k}'] = recall_val
#     return out

# # ============== Main ==============
# def main():
#     ap = argparse.ArgumentParser(description="Fine-tune CLIP with LoRA for image-text retrieval.")
    
#     ap.add_argument('--train-csv', type=Path, required=True, help="训练集 CSV 文件。")
#     ap.add_argument('--val-csv', type=Path, required=True, help="验证集 CSV 文件。")
#     ap.add_argument('--train-image-root-dir', type=Path, required=True, help="训练集图像根目录。")
#     ap.add_argument('--val-image-root-dir', type=Path, required=True, help="验证集图像根目录。")

#     ap.add_argument('--out-dir',   type=Path, required=True)
#     ap.add_argument('--model-name', type=str, default='ViT-L-14')
#     ap.add_argument('--pretrained', type=str, default='datacomp_xl_s13b_b90k')
    
#     ap.add_argument('--epochs', type=int, default=5)
#     ap.add_argument('--batch-size', type=int, default=64)
#     ap.add_argument('--lr', type=float, default=1e-4)
#     ap.add_argument('--weight-decay', type=float, default=1e-3)
#     ap.add_argument('--num-workers', type=int, default=8)
#     ap.add_argument('--use-amp', action='store_true', help="使用自动混合精度训练。")
#     ap.add_argument('--seed', type=int, default=42)

#     ap.add_argument('--lora-last-n', type=int, default=12, help="为最后 N 个 Transformer 块注入 LoRA。")
#     ap.add_argument('--lora-r', type=int, default=8, help="LoRA 的秩。")
#     ap.add_argument('--lora-alpha', type=int, default=16, help="LoRA 的 alpha 缩放因子。")
#     ap.add_argument('--lora-dropout', type=float, default=0.0, help="LoRA 层的 dropout 率。")
    
#     ap.add_argument('--eval-k', nargs='+', type=int, default=[1, 5, 10, 20], help="用于评估 Recall@K 的 K 值列表。")

#     args = ap.parse_args()
#     set_seed(args.seed)
#     args.out_dir.mkdir(parents=True, exist_ok=True)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     print("Loading CLIP model and preprocessor...")
#     model, _, preprocess = open_clip.create_model_and_transforms(
#         args.model_name, pretrained=args.pretrained, device=device
#     )
#     tokenizer = open_clip.get_tokenizer(args.model_name)
    
#     print("Injecting LoRA layers...")
#     model = inject_lora_openclip(
#         model, last_n_blocks=args.lora_last_n, r=args.lora_r, 
#         alpha=args.lora_alpha, dropout=args.lora_dropout
#     )
#     model.to(device)
#     trainable_params = [p for p in model.parameters() if p.requires_grad]
    
#     print("Loading datasets...")
#     train_ds = CsvDataset(csv_path=args.train_csv, preprocess=preprocess, image_root_dir=args.train_image_root_dir)
#     val_ds = CsvDataset(csv_path=args.val_csv, preprocess=preprocess, image_root_dir=args.val_image_root_dir)

#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

#     optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
#     scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

#     print("Starting training...")
#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         total_loss, step = 0.0, 0
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
#         for imgs, texts, _ in pbar:
#             if imgs is None: continue

#             imgs = imgs.to(device)
#             toks = tokenizer(texts).to(device)

#             optimizer.zero_grad()
#             with torch.cuda.amp.autocast(enabled=args.use_amp):
#                 img_feats = F.normalize(model.encode_image(imgs), dim=-1)
#                 txt_feats = F.normalize(model.encode_text(toks), dim=-1)
#                 sim = torch.exp(model.logit_scale) * (img_feats @ txt_feats.t())

#                 labels = torch.arange(sim.size(0), device=device)
#                 loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
            
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
            
#             total_loss += loss.item()
#             step += 1
#             pbar.set_postfix(loss=f"{(total_loss/step):.4f}")

#         model.eval()
#         all_I, all_T = [], []
#         with torch.no_grad():
#             for imgs, texts, _ in tqdm(val_loader, desc="Evaluating"):
#                 if imgs is None: continue

#                 imgs = imgs.to(device, non_blocking=True)
#                 toks = tokenizer(texts).to(device)
#                 zi = F.normalize(model.encode_image(imgs), dim=-1)
#                 zt = F.normalize(model.encode_text(toks), dim=-1)
#                 all_I.append(zi)
#                 all_T.append(zt)

#         if not all_I:
#             print("Validation set could not be loaded. Skipping evaluation.")
#             continue

#         I = torch.cat(all_I, dim=0)
#         T = torch.cat(all_T, dim=0)

#         metrics = recall_at_k(T, I, args.eval_k)
#         msg = " | ".join([f"{k}={v*100:.2f}%" for k,v in metrics.items()])
#         print(f"[Val Epoch {epoch}] {msg}")

#         # --- 最终修复：保存与加载脚本完全兼容的检查点 ---
#         lora_state = {n: p.detach().cpu() for n, p in model.named_parameters() if 'lora_' in n or 'logit_scale' in n}
        
#         checkpoint = {
#             'mode': 'lora',
#             'model_name': args.model_name,
#             'pretrained': args.pretrained,
#             'lora_last_n': args.lora_last_n,
#             'lora_r': args.lora_r,
#             'lora_alpha': args.lora_alpha,
#             'lora_dropout': args.lora_dropout,
#             'lora_state': lora_state  # 权重字典的键名应为 'lora_state'
#         }
        
#         torch.save(checkpoint, args.out_dir / f'lora_epoch{epoch}.pt')
#         # --- 修复结束 ---
#         print(f"LoRA checkpoint for epoch {epoch} saved to {args.out_dir / f'lora_epoch{epoch}.pt'}")

# if __name__ == "__main__":
#     main()





# import argparse
# import random
# from pathlib import Path
# import pandas as pd
# from PIL import Image
# from typing import List, Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import open_clip
# from tqdm import tqdm
# import loralib as lora

# # ============== Utils ==============

# def set_seed(seed: int = 42):
#     """设置随机种子以确保可复现性。"""
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# # ============== Dataset ==============

# class CsvDataset(Dataset):
#     """
#     一个简单的数据集，用于加载一一对应的图文对。
#     CSV 文件必须包含 'filepath' 和 'title' 两列。
#     """
#     def __init__(self, csv_path: Path, preprocess, image_root_dir: Path):
#         df = pd.read_csv(csv_path)
#         assert 'filepath' in df.columns and 'title' in df.columns, \
#             f"CSV 文件 {csv_path} 必须包含 'filepath' 和 'title' 列。"

#         self.paths = [image_root_dir.joinpath(p) for p in df['filepath']]
#         self.texts = df['title'].tolist()
#         self.preprocess = preprocess

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, i):
#         """返回图像、文本和其索引。"""
#         try:
#             img = self.preprocess(Image.open(self.paths[i]).convert("RGB"))
#             return img, self.texts[i], i
#         except Exception as e:
#             # print(f"警告：加载图像失败 {self.paths[i]}: {e}")
#             return None, self.texts[i], i

# def collate_fn(batch):
#     """自定义的 collate 函数，用于过滤掉加载失败的图像。"""
#     batch = [b for b in batch if b[0] is not None]
#     if not batch:
#         return None, None, None
    
#     imgs, texts, idxs = zip(*batch)
#     return torch.stack(imgs, dim=0), list(texts), torch.tensor(idxs, dtype=torch.long)

# # ============== LoRA Injection (Advanced) ==============

# def _replace_linear_with_lora_linear(mod: nn.Linear, r=8, alpha=16, dropout=0.0):
#     new = lora.Linear(
#         mod.in_features, mod.out_features, r=r, lora_alpha=alpha, 
#         lora_dropout=dropout, bias=(mod.bias is not None)
#     )
#     new = new.to(device=mod.weight.device, dtype=mod.weight.dtype)
#     with torch.no_grad():
#         new.weight.copy_(mod.weight.data)
#         if mod.bias is not None:
#             new.bias.copy_(mod.bias.data)
#     return new

# class LoRAInjectedMultiheadAttention(nn.Module):
#     """
#     将 LoRA 注入到 MultiheadAttention 的 Q/K/V（输入侧增量），
#     并兼容 OpenCLIP 的 batch_first=True 张量形状。
#     """
#     def __init__(self, original_attn: nn.MultiheadAttention, r=8, alpha=16, dropout=0.0):
#         super().__init__()
#         self.original_attn = original_attn
#         self.batch_first = getattr(original_attn, "batch_first", False)  # OpenCLIP 为 True

#         d_model = original_attn.embed_dim
#         self.num_heads = original_attn.num_heads
#         self.head_dim = d_model // self.num_heads

#         # LoRA 作用在输入 -> q/k/v 的增量（不改 in_proj_weight 本体）
#         self.q_lora = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=False)
#         self.k_lora = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=False)
#         self.v_lora = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=False)

#     def _to_bhlh(self, x):
#         # x: (B, L, E) -> (B, H, L, D)
#         B, L, E = x.shape
#         return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

#     def _from_bhlh(self, x):
#         # x: (B, H, L, D) -> (B, L, E)
#         B, H, L, D = x.shape
#         return x.transpose(1, 2).contiguous().view(B, L, H * D)

#     def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
#                 attn_mask: Optional[torch.Tensor] = None, **kwargs):

#         # ---- 统一成 (B, L, E) ----
#         if not self.batch_first:
#             # 输入是 (L, B, E) 的情况（以防万一）
#             query = query.transpose(0, 1)
#             key   = key.transpose(0, 1)
#             value = value.transpose(0, 1)

#         # 原始 QKV（与 nn.MultiheadAttention 相同的 in-proj）
#         qkv = F.linear(query, self.original_attn.in_proj_weight, self.original_attn.in_proj_bias)
#         q, k, v = qkv.chunk(3, dim=-1)

#         # LoRA 增量：对输入分别做线性映射并加到 q/k/v 上
#         q = q + self.q_lora(query)
#         k = k + self.k_lora(key)
#         v = v + self.v_lora(value)

#         # (B, L, E) -> (B, H, L, D)
#         q = self._to_bhlh(q)
#         k = self._to_bhlh(k)
#         v = self._to_bhlh(v)

#         # ---- 注意力掩码 ----
#         # OpenCLIP 文本侧通常是 (L, L) 的加性掩码（上三角为 -inf），直接广播即可。
#         # 如果你自己传的是 bool/broadcast 维度，保持原 dtype/shape 就好，不要随便转换。
#         # PyTorch 会把 (L, L) 广播成 (B, H, L, L)。
#         attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

#         # (B, H, L, D) -> (B, L, E)
#         attn_out = self._from_bhlh(attn_out)

#         # 输出全连接（已被你外层替换为 LoRA 版 Linear）
#         attn_out = self.original_attn.out_proj(attn_out)

#         # 还原成调用方期望的布局
#         if not self.batch_first:
#             attn_out = attn_out.transpose(0, 1)

#         return attn_out, None


# def inject_lora_openclip(model, last_n_blocks=12, r=8, alpha=16, dropout=0.0):
#     """向 OpenCLIP 模型的 Transformer 块中注入 LoRA 层 (高级版)。"""
    
#     def _inject_block(blk):
#         # 注入 Q, K, V
#         blk.attn = LoRAInjectedMultiheadAttention(blk.attn, r, alpha, dropout)
        
#         # 注入 out_proj 和 MLP 层
#         blk.attn.original_attn.out_proj = _replace_linear_with_lora_linear(blk.attn.original_attn.out_proj, r, alpha, dropout)
#         blk.mlp.c_fc = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
#         blk.mlp.c_proj = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)

#     # Vision Transformer
#     for blk in model.visual.transformer.resblocks[-last_n_blocks:]:
#         _inject_block(blk)
    
#     # Text Transformer
#     for blk in model.transformer.resblocks[-last_n_blocks:]:
#         _inject_block(blk)
    
#     lora.mark_only_lora_as_trainable(model, bias='lora_only')
#     model.logit_scale.requires_grad_(True)
#     return model

# # ============== Recall@K Evaluation ==============
# @torch.no_grad()
# def recall_at_k(text_feats: torch.Tensor, image_feats: torch.Tensor, k_list: List[int]):
#     sim = text_feats @ image_feats.t()
#     N = sim.size(0)
#     topk = torch.topk(sim, k=max(k_list), dim=1).indices
#     out = {}
#     gt = torch.arange(N, device=sim.device).view(-1, 1)
#     for k in k_list:
#         hit = (topk[:, :k] == gt).any(dim=1)
#         recall_val = hit.float().mean().item()
#         out[f'R@{k}'] = recall_val
#     return out

# # ============== Main ==============
# def main():
#     ap = argparse.ArgumentParser(description="Fine-tune CLIP with LoRA for image-text retrieval.")
    
#     ap.add_argument('--train-csv', type=Path, required=True, help="训练集 CSV 文件。")
#     ap.add_argument('--val-csv', type=Path, required=True, help="验证集 CSV 文件。")
#     ap.add_argument('--train-image-root-dir', type=Path, required=True, help="训练集图像根目录。")
#     ap.add_argument('--val-image-root-dir', type=Path, required=True, help="验证集图像根目录。")

#     ap.add_argument('--out-dir',   type=Path, required=True)
#     ap.add_argument('--model-name', type=str, default='ViT-L-14')
#     ap.add_argument('--pretrained', type=str, default='datacomp_xl_s13b_b90k')
    
#     ap.add_argument('--epochs', type=int, default=5)
#     ap.add_argument('--batch-size', type=int, default=64)
#     ap.add_argument('--lr', type=float, default=1e-4)
#     ap.add_argument('--weight-decay', type=float, default=1e-3)
#     ap.add_argument('--num-workers', type=int, default=8)
#     ap.add_argument('--use-amp', action='store_true', help="使用自动混合精度训练。")
#     ap.add_argument('--seed', type=int, default=42)

#     ap.add_argument('--lora-last-n', type=int, default=12, help="为最后 N 个 Transformer 块注入 LoRA。")
#     ap.add_argument('--lora-r', type=int, default=32, help="LoRA 的秩。") # 8
#     ap.add_argument('--lora-alpha', type=int, default=16, help="LoRA 的 alpha 缩放因子。")
#     ap.add_argument('--lora-dropout', type=float, default=0.0, help="LoRA 层的 dropout 率。")
    
#     ap.add_argument('--eval-k', nargs='+', type=int, default=[1, 5, 10, 20], help="用于评估 Recall@K 的 K 值列表。")

#     args = ap.parse_args()
#     set_seed(args.seed)
#     args.out_dir.mkdir(parents=True, exist_ok=True)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     print("Loading CLIP model and preprocessor...")
#     model, _, preprocess = open_clip.create_model_and_transforms(
#         args.model_name, pretrained=args.pretrained, device=device
#     )
#     tokenizer = open_clip.get_tokenizer(args.model_name)
    
#     print("Injecting LoRA layers...")
#     model = inject_lora_openclip(
#         model, last_n_blocks=args.lora_last_n, r=args.lora_r, 
#         alpha=args.lora_alpha, dropout=args.lora_dropout
#     )
#     model.to(device)
#     trainable_params = [p for p in model.parameters() if p.requires_grad]
    
#     print("Loading datasets...")
#     train_ds = CsvDataset(csv_path=args.train_csv, preprocess=preprocess, image_root_dir=args.train_image_root_dir)
#     val_ds = CsvDataset(csv_path=args.val_csv, preprocess=preprocess, image_root_dir=args.val_image_root_dir)

#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

#     optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
#     scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

#     print("Starting training...")
#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         total_loss, step = 0.0, 0
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
#         for imgs, texts, _ in pbar:
#             if imgs is None: continue

#             imgs = imgs.to(device)
#             toks = tokenizer(texts).to(device)

#             optimizer.zero_grad()
#             with torch.cuda.amp.autocast(enabled=args.use_amp):
#                 img_feats = F.normalize(model.encode_image(imgs), dim=-1)
#                 txt_feats = F.normalize(model.encode_text(toks), dim=-1)
#                 sim = torch.exp(model.logit_scale) * (img_feats @ txt_feats.t())

#                 labels = torch.arange(sim.size(0), device=device)
#                 loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
            
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
            
#             total_loss += loss.item()
#             step += 1
#             pbar.set_postfix(loss=f"{(total_loss/step):.4f}")

#         model.eval()
#         all_I, all_T = [], []
#         with torch.no_grad():
#             for imgs, texts, _ in tqdm(val_loader, desc="Evaluating"):
#                 if imgs is None: continue

#                 imgs = imgs.to(device, non_blocking=True)
#                 toks = tokenizer(texts).to(device)
#                 zi = F.normalize(model.encode_image(imgs), dim=-1)
#                 zt = F.normalize(model.encode_text(toks), dim=-1)
#                 all_I.append(zi)
#                 all_T.append(zt)

#         if not all_I:
#             print("Validation set could not be loaded. Skipping evaluation.")
#             continue

#         I = torch.cat(all_I, dim=0)
#         T = torch.cat(all_T, dim=0)

#         metrics = recall_at_k(T, I, args.eval_k)
#         msg = " | ".join([f"{k}={v*100:.2f}%" for k,v in metrics.items()])
#         print(f"[Val Epoch {epoch}] {msg}")

#         lora_state = {n: p.detach().cpu() for n, p in model.named_parameters() if 'lora_' in n or 'logit_scale' in n}
        
#         checkpoint = {
#             'mode': 'lora',
#             'model_name': args.model_name,
#             'pretrained': args.pretrained,
#             'lora_last_n': args.lora_last_n,
#             'lora_r': args.lora_r,
#             'lora_alpha': args.lora_alpha,
#             'lora_dropout': args.lora_dropout,
#             'lora_state': lora_state
#         }
        
#         torch.save(checkpoint, args.out_dir / f'lora_epoch_{epoch}.pt')
#         print(f"LoRA checkpoint for epoch {epoch} saved to {args.out_dir / f'lora_epoch_{epoch}.pt'}")

# if __name__ == "__main__":
#     main()







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

# ============== Utils ==============

def set_seed(seed: int = 42):
    """设置随机种子以确保可复现性。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============== Dataset ==============

class CsvDataset(Dataset):
    """
    一个简单的数据集，用于加载一一对应的图文对。
    CSV 文件必须包含 'filepath' 和 'title' 两列。
    image_name, description
    """
    def __init__(self, csv_path: Path, preprocess, image_root_dir: Path, for_hf=False):
        df = pd.read_csv(csv_path)
        assert 'image_name' in df.columns and 'description' in df.columns, \
            f"CSV 文件 {csv_path} 必须包含 'filepath' 和 'title' 列。"

        self.paths = [image_root_dir.joinpath(p) for p in df['image_name']]
        self.texts = df['description'].tolist()
        self.preprocess = preprocess
        self.for_hf = for_hf # 标志位，用于区分是否为 HF 模型准备数据

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        """返回图像、文本和其索引。"""
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            # HF 模型需要原始 PIL 图像，而 open_clip 需要预处理后的张量
            processed_img = img if self.for_hf else self.preprocess(img)
            return processed_img, self.texts[i], i
        except Exception as e:
            # print(f"警告：加载图像失败 {self.paths[i]}: {e}")
            return None, self.texts[i], i

def collate_fn(batch):
    """自定义的 collate 函数，用于过滤掉加载失败的图像。"""
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None, None
    
    imgs, texts, idxs = zip(*batch)
    # 根据图像类型决定是否堆叠
    try:
        imgs_stacked = torch.stack(imgs, dim=0)
        return imgs_stacked, list(texts), torch.tensor(idxs, dtype=torch.long)
    except TypeError: # 如果是 PIL.Image 列表，直接返回列表
        return list(imgs), list(texts), torch.tensor(idxs, dtype=torch.long)


# # ============== LoRA Injection (Advanced) ==============

# def _replace_linear_with_lora_linear(mod: nn.Linear, r=8, alpha=16, dropout=0.0):
#     new = lora.Linear(
#         mod.in_features, mod.out_features, r=r, lora_alpha=alpha, 
#         lora_dropout=dropout, bias=(mod.bias is not None)
#     )
#     new = new.to(device=mod.weight.device, dtype=mod.weight.dtype)
#     with torch.no_grad():
#         new.weight.copy_(mod.weight.data)
#         if mod.bias is not None:
#             new.bias.copy_(mod.bias.data)
#     return new

# class LoRAInjectedMultiheadAttention(nn.Module):
#     def __init__(self, original_attn: nn.MultiheadAttention, r=8, alpha=16, dropout=0.0):
#         super().__init__()
#         self.original_attn = original_attn
#         self.batch_first = getattr(original_attn, "batch_first", False)

#         d_model = original_attn.embed_dim
#         self.num_heads = original_attn.num_heads
#         self.head_dim = d_model // self.num_heads

#         self.q_lora = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=False)
#         self.k_lora = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=False)
#         self.v_lora = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=False)

#     def _to_bhlh(self, x):
#         B, L, E = x.shape
#         return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

#     def _from_bhlh(self, x):
#         B, H, L, D = x.shape
#         return x.transpose(1, 2).contiguous().view(B, L, H * D)

#     def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
#                 attn_mask: Optional[torch.Tensor] = None, **kwargs):
#         if not self.batch_first:
#             query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

#         qkv = F.linear(query, self.original_attn.in_proj_weight, self.original_attn.in_proj_bias)
#         q, k, v = qkv.chunk(3, dim=-1)

#         q = q + self.q_lora(query)
#         k = k + self.k_lora(key)
#         v = v + self.v_lora(value)

#         q, k, v = self._to_bhlh(q), self._to_bhlh(k), self._to_bhlh(v)
#         attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
#         attn_out = self._from_bhlh(attn_out)
#         attn_out = self.original_attn.out_proj(attn_out)

#         if not self.batch_first:
#             attn_out = attn_out.transpose(0, 1)
#         return attn_out, None

# def inject_lora_openclip(model, last_n_blocks=12, r=8, alpha=16, dropout=0.0):
#     def _inject_block(blk):
#         blk.attn = LoRAInjectedMultiheadAttention(blk.attn, r, alpha, dropout)
#         blk.attn.original_attn.out_proj = _replace_linear_with_lora_linear(blk.attn.original_attn.out_proj, r, alpha, dropout)
#         blk.mlp.c_fc = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
#         blk.mlp.c_proj = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)

#     for blk in model.visual.transformer.resblocks[-last_n_blocks:]:
#         _inject_block(blk)
#     for blk in model.transformer.resblocks[-last_n_blocks:]:
#         _inject_block(blk)
    
#     lora.mark_only_lora_as_trainable(model, bias='lora_only')
#     model.logit_scale.requires_grad_(True)
#     return model



# # def inject_lora_openclip(model, target='all', last_n_blocks=12, r=8, alpha=16, dropout=0.0):
# #     """
# #     向 OpenCLIP 模型的 Transformer 块中注入 LoRA 层。
# #     :param target: 'all', 'text', 或 'vision'，指定注入 LoRA 的目标。
# #     """
    
# #     def _inject_block(blk):
# #         # 注入 Q, K, V
# #         blk.attn = LoRAInjectedMultiheadAttention(blk.attn, r, alpha, dropout)
        
# #         # 注入 out_proj 和 MLP 层
# #         blk.attn.original_attn.out_proj = _replace_linear_with_lora_linear(blk.attn.original_attn.out_proj, r, alpha, dropout)
# #         blk.mlp.c_fc = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
# #         blk.mlp.c_proj = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)

# #     # --- 核心修改：根据 target 参数选择性地注入 LoRA ---
# #     if target == 'all' or target == 'vision':
# #         print("Injecting LoRA into Vision Transformer...")
# #         for blk in model.visual.transformer.resblocks[-last_n_blocks:]:
# #             _inject_block(blk)
    
# #     if target == 'all' or target == 'text':
# #         print("Injecting LoRA into Text Transformer...")
# #         for blk in model.transformer.resblocks[-last_n_blocks:]:
# #             _inject_block(blk)
# #     # --- 修改结束 ---
    
# #     lora.mark_only_lora_as_trainable(model, bias='lora_only')
# #     model.logit_scale.requires_grad_(True)
# #     return model






# # ============== LoRA Injection (Simplified: No QKV Injection) ==============

# def _replace_linear_with_lora_linear(mod: nn.Linear, r=8, alpha=16, dropout=0.0):
#     """用 loralib.Linear 替换一个 nn.Linear 层，并复制权重。"""
#     new = lora.Linear(
#         mod.in_features, mod.out_features, r=r, lora_alpha=alpha, 
#         lora_dropout=dropout, bias=(mod.bias is not None)
#     )
#     new = new.to(device=mod.weight.device, dtype=mod.weight.dtype)
#     with torch.no_grad():
#         new.weight.copy_(mod.weight.data)
#         if mod.bias is not None:
#             new.bias.copy_(mod.bias.data)
#     return new

# def inject_lora_openclip(model, target='all', last_n_blocks=12, r=8, alpha=16, dropout=0.0):
#     """
#     向 OpenCLIP 模型的 Transformer 块中注入 LoRA 层。
#     此版本不注入 QKV，只注入 out_proj 和 MLP 层。
#     """
    
#     def _inject_block(blk):
#         # --- 核心修改：直接在原始 attn 模块上修改 out_proj ---
#         # 注意：不再替换整个 blk.attn 模块
#         blk.attn.out_proj = _replace_linear_with_lora_linear(blk.attn.out_proj, r, alpha, dropout)
        
#         # MLP 层的注入保持不变
#         blk.mlp.c_fc = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
#         blk.mlp.c_proj = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)

#     # 根据 target 参数选择性地注入 LoRA
#     if target == 'all' or target == 'vision':
#         print("Injecting LoRA into Vision Transformer (out_proj & MLP only)...")
#         for blk in model.visual.transformer.resblocks[-last_n_blocks:]:
#             _inject_block(blk)
    
#     if target == 'all' or target == 'text':
#         print("Injecting LoRA into Text Transformer (out_proj & MLP only)...")
#         for blk in model.transformer.resblocks[-last_n_blocks:]:
#             _inject_block(blk)
    
#     lora.mark_only_lora_as_trainable(model, bias='lora_only')
#     model.logit_scale.requires_grad_(True)
#     return model






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

@torch.no_grad()
def evaluate(model, val_loader, tokenizer, device, args):
    model.eval()
    all_I, all_T = [], []
    for imgs, texts, _ in tqdm(val_loader, desc="Evaluating (open_clip)"):
        if imgs is None: continue
        imgs = imgs.to(device, non_blocking=True)
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
    for imgs, texts, _ in tqdm(hf_val_loader, desc="Evaluating (Hugging Face)"):
        if imgs is None: continue
        inputs = hf_processor(text=texts, images=imgs, return_tensors="pt", padding=True, truncation=True)
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
    # print(open_clip.list_pretrained())  # 查看支持的权重别名

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)


    print("Injecting LoRA layers...")

    # model = inject_lora_openclip(
    #     model, last_n_blocks=args.lora_last_n, r=args.lora_r, 
    #     alpha=args.lora_alpha, dropout=args.lora_dropout
    # )

    # --- 核心修改：将新的参数传递给注入函数 ---
    model = inject_lora_openclip(
        model, target=args.lora_target, last_n_blocks=args.lora_last_n, r=args.lora_r, 
        alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    # --- 修改结束 ---

    model.to(device)
    
    print("Loading datasets for open_clip training...")
    train_ds = CsvDataset(csv_path=args.train_csv, preprocess=preprocess, image_root_dir=args.train_image_root_dir)
    val_ds = CsvDataset(csv_path=args.val_csv, preprocess=preprocess, image_root_dir=args.val_image_root_dir)

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

    # --- 步骤 4: 开始训练 ---
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, step = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, texts, _ in pbar:
            if imgs is None: continue
            imgs = imgs.to(device)
            toks = tokenizer(texts).to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                img_feats = F.normalize(model.encode_image(imgs), dim=-1)
                txt_feats = F.normalize(model.encode_text(toks), dim=-1)
                sim = torch.exp(model.logit_scale) * (img_feats @ txt_feats.t())
                labels = torch.arange(sim.size(0), device=device)
                loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
            
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
        checkpoint = {
            'mode': 'lora', 'model_name': args.model_name, 'pretrained': args.pretrained,
            'lora_last_n': args.lora_last_n, 'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha, 'lora_dropout': args.lora_dropout,
            'lora_state': lora_state
        }
        torch.save(checkpoint, args.out_dir / f'lora_epoch_{epoch}.pt')
        print(f"LoRA checkpoint for epoch {epoch} saved to {args.out_dir / f'lora_epoch_{epoch}.pt'}\n")

if __name__ == "__main__":
    main()
