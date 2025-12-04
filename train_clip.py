# import argparse, random
# from pathlib import Path
# import pandas as pd
# from PIL import Image
# from typing import List, Dict, Iterable

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, Sampler
# from transformers import CLIPModel, CLIPProcessor, CLIPConfig
# from huggingface_hub import hf_hub_download # <-- 1. 修改导入语句



# import open_clip
# from tqdm import tqdm

# # ============== Utils ==============

# def set_seed(seed: int = 42):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# # ============== Dataset ==============

# class CsvPairs(Dataset):
#     """
#     CSV 列必须包含: filepath, title
#     可选: group_id
#     - --has-group 时使用 group_id
#     - --multi-text 时对 filepath factorize 为组（同图多描述）
#     - 否则 groups=None（单正样本）
#     """
#     def __init__(self, csv_path: Path, preprocess, has_group: bool, multi_text: bool):
#         df = pd.read_csv(csv_path)
#         assert 'filepath' in df.columns and 'title' in df.columns, \
#             "CSV must contain 'filepath' and 'title' columns."
#         df['filepath'] = df['filepath'].apply(lambda p: str(Path(p).expanduser().resolve()))
#         self.paths = [Path(p) for p in df['filepath']]
#         self.texts = [str(t) for t in df['title']]

#         if has_group and 'group_id' in df.columns:
#             self.groups = df['group_id'].astype('int64').tolist()
#         elif multi_text:
#             # 关键：同一路径视为同一组，实现“同图多描述”的多正样本
#             self.groups = pd.factorize(df['filepath'])[0].astype('int64').tolist()
#         else:
#             self.groups = None  # 单正样本

#         self.preprocess = preprocess

#     def __len__(self): return len(self.paths)

#     def __getitem__(self, i):
#         img = self.preprocess(Image.open(self.paths[i]).convert("RGB"))
#         if self.groups is None:
#             return img, self.texts[i], i  # 用样本索引当组（单样本）
#         return img, self.texts[i], self.groups[i]

# def collate_fn(batch):
#     imgs, texts, gids = zip(*batch)
#     return torch.stack(imgs, dim=0), list(texts), torch.tensor(gids, dtype=torch.long)

# # ============== Optional: Grouped Batch Sampler ==============

# class GroupedBatchSampler(Sampler[List[int]]):
#     """
#     让同组样本尽量在同一批内。
#     简化实现：先按 group 排序，再按 batch_size 切块；每个 epoch 打乱组顺序。
#     """
#     def __init__(self, groups: List[int], batch_size: int, drop_last: bool):
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         # build group->indices
#         buckets: Dict[int, List[int]] = {}
#         for idx, g in enumerate(groups):
#             buckets.setdefault(int(g), []).append(idx)
#         self.group_ids = list(buckets.keys())
#         self.buckets = buckets
#         self._order = []

#     def __iter__(self) -> Iterable[List[int]]:
#         random.shuffle(self.group_ids)
#         self._order.clear()
#         cur = []
#         for gid in self.group_ids:
#             idxs = self.buckets[gid]
#             random.shuffle(idxs)
#             for i in idxs:
#                 cur.append(i)
#                 if len(cur) == self.batch_size:
#                     self._order.append(cur)
#                     cur = []
#         if (not self.drop_last) and len(cur) > 0:
#             self._order.append(cur)
#         return iter(self._order)

#     def __len__(self) -> int:
#         total = sum(len(v) for v in self.buckets.values())
#         return total // self.batch_size + (0 if self.drop_last else (1 if total % self.batch_size else 0))

# # ============== Linear Probe Projections ==============

# class Projection(nn.Module):
#     def __init__(self, dim: int, hidden: int = 0):
#         super().__init__()
#         if hidden and hidden > 0:
#             self.proj = nn.Sequential(
#                 nn.Linear(dim, hidden), nn.ReLU(inplace=True),
#                 nn.Linear(hidden, dim)
#             )
#         else:
#             self.proj = nn.Linear(dim, dim)

#     def forward(self, x):
#         x = self.proj(x)
#         return F.normalize(x, dim=-1)

# # ============== LoRA Injection (loralib) ==============

# def try_import_lora():
#     try:
#         import loralib as lora
#         return lora
#     except ImportError:
#         raise ImportError("Please install loralib: pip install loralib")

# def _replace_linear_with_lora_linear(mod: nn.Linear, r=8, alpha=16, dropout=0.0):
#     lora = try_import_lora()
#     new = lora.Linear(
#         mod.in_features, mod.out_features,
#         r=r, lora_alpha=alpha, lora_dropout=dropout,
#         bias=(mod.bias is not None)
#     )
#     with torch.no_grad():
#         new.weight.copy_(mod.weight)
#         if mod.bias is not None:
#             new.bias.copy_(mod.bias)
#     return new

# def inject_lora_openclip(model, last_n_blocks=4, r=8, alpha=16, dropout=0.0):
#     # Vision tower
#     v_blocks = model.visual.transformer.resblocks
#     for blk in v_blocks[-last_n_blocks:]:
#         blk.attn.out_proj = _replace_linear_with_lora_linear(blk.attn.out_proj, r, alpha, dropout)
#         blk.mlp.c_fc      = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
#         blk.mlp.c_proj    = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)
#     # Text tower
#     t_blocks = model.transformer.resblocks
#     for blk in t_blocks[-last_n_blocks:]:
#         blk.attn.out_proj = _replace_linear_with_lora_linear(blk.attn.out_proj, r, alpha, dropout)
#         blk.mlp.c_fc      = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
#         blk.mlp.c_proj    = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)
#     # Freeze base params
#     for n, p in model.named_parameters():
#         if 'lora_' not in n and n != 'logit_scale':
#             p.requires_grad = False
#     return model

# def lora_parameters(model):
#     return [p for n, p in model.named_parameters() if p.requires_grad]

# # ============== Multi-Positive InfoNCE ==============

# def mp_infonce(sim: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
#     """
#     sim: [B, B] 已含温度缩放后的相似度
#     groups: [B] 同组为正样本
#     """
#     G = groups.view(-1, 1)
#     pos_mask = (G == G.t())   # [B, B]
#     big_neg = torch.finfo(sim.dtype).min / 2

#     sim_pos = sim.masked_fill(~pos_mask, big_neg)  # 仅保留正样本
#     # i->t
#     lse_pos = torch.logsumexp(sim_pos, dim=1)
#     lse_all = torch.logsumexp(sim, dim=1)
#     loss_i2t = -(lse_pos - lse_all).mean()
#     # t->i
#     lse_pos_t = torch.logsumexp(sim_pos.t(), dim=1)
#     lse_all_t = torch.logsumexp(sim.t(), dim=1)
#     loss_t2i = -(lse_pos_t - lse_all_t).mean()

#     return 0.5 * (loss_i2t + loss_t2i)

# # ============== Recall@K (T->I) ==============

# @torch.no_grad()
# def recall_at_k(text_feats: torch.Tensor, image_feats: torch.Tensor, k_list: List[int], groups: torch.Tensor = None):
#     sim = text_feats @ image_feats.t()  # [N, N]
#     N = sim.size(0)
#     topk = torch.topk(sim, k=max(k_list), dim=1).indices  # [N, Kmax]
#     out = {}
#     if groups is None:
#         gt = torch.arange(N, device=sim.device)
#         for k in k_list:
#             hit = (topk[:, :k] == gt.view(-1, 1)).any(dim=1).float().mean().item()
#             out[f'R@{k}'] = hit
#     else:
#         g = groups.view(-1, 1)
#         pos_mask = (g == g.t())
#         for k in k_list:
#             idx = topk[:, :k]
#             hits = []
#             for i in range(N):
#                 hits.append((pos_mask[i][idx[i]]).any().float())
#             out[f'R@{k}'] = torch.stack(hits).mean().item()
#     return out

# # ============== Main ==============

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--train-csv', type=Path, required=True)
#     ap.add_argument('--val-csv',   type=Path, required=True)
#     ap.add_argument('--out-dir',   type=Path, required=True)
#     ap.add_argument('--model-name', type=str, default='ViT-L-14')
#     ap.add_argument('--pretrained', type=str, default='hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
#     ap.add_argument('--context-length', type=int, default=77)
#     ap.add_argument('--mode', choices=['linear','lora'], required=True)

#     # train
#     ap.add_argument('--epochs', type=int, default=3)
#     ap.add_argument('--batch-size', type=int, default=128)
#     ap.add_argument('--lr', type=float, default=1e-3)
#     ap.add_argument('--weight-decay', type=float, default=1e-3)
#     ap.add_argument('--grad-accum', type=int, default=1)
#     ap.add_argument('--num-workers', type=int, default=4)
#     ap.add_argument('--use-amp', action='store_true')
#     ap.add_argument('--seed', type=int, default=42)

#     # linear head
#     ap.add_argument('--proj-hidden', type=int, default=0)

#     # lora
#     ap.add_argument('--lora-last-n', type=int, default=4)
#     ap.add_argument('--lora-r', type=int, default=8)
#     ap.add_argument('--lora-alpha', type=int, default=16)
#     ap.add_argument('--lora-dropout', type=float, default=0.0)

#     # grouping / sampling
#     ap.add_argument('--has-group', action='store_true', help="CSV 有 group_id 列")
#     ap.add_argument('--multi-text', action='store_true', help="无 group_id 时，对同一 filepath 的多描述视为同组")
#     ap.add_argument('--grouped-sampler', action='store_true', help="尽量让同组样本进入同一 batch")

#     # eval
#     ap.add_argument('--eval-k', nargs='+', type=int, default=[1,5,10])

#     args = ap.parse_args()
#     set_seed(args.seed)
#     args.out_dir.mkdir(parents=True, exist_ok=True)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # model & preprocess & tokenizer
#     # --- 核心修改：手动加载 LongCLIP 模型 ---
#     print(f"正在加载模型架构: {args.model_name}")
#     # 1. 创建一个空的、未经训练的模型骨架
#     # 修正后的代码
#     model, _, preprocess = open_clip.create_model_and_transforms(
#         args.model_name,
#         pretrained=None,
#         text_cfg=dict(context_length=args.context_length)
#     )
    
#     print(f"正在从 Hugging Face Hub 手动加载预训练权重: {args.pretrained}")
#     # 2. 从 Hugging Face Hub 下载预训练权重
#     pretrained_model_id = args.pretrained.replace('hf-hub:', '')

#     # --- 2. 确保下载逻辑正确 ---
#     # 使用 hf_hub_download 下载模型文件
#     # LongCLIP 的权重文件名通常是 'open_clip_pytorch_model.bin' 或 'pytorch_model.bin'
#     try:
#         cached_file = hf_hub_download(repo_id=pretrained_model_id, filename='open_clip_pytorch_model.bin')
#     except Exception:
#         # 如果第一个文件名失败，尝试另一个常见的文件名
#         cached_file = hf_hub_download(repo_id=pretrained_model_id, filename='pytorch_model.bin')

#     # 使用 torch.load 加载本地文件
#     checkpoint = torch.load(cached_file, map_location=device)
    
#     # 3. 将下载的权重加载到模型骨架中
#     incompatible_keys = model.load_state_dict(checkpoint, strict=False)
#     if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
#         print("加载权重时发现不匹配的键 (通常是安全的):")
#         print(f"  - 缺失的键: {incompatible_keys.missing_keys}")
#         print(f"  - 意外的键: {incompatible_keys.unexpected_keys}")
    
#     print("LongCLIP 模型权重加载成功！")
#     # --- 修改结束 ---
#     model.to(device)
#     tokenizer = open_clip.get_tokenizer(args.model_name)

#     # datasets
#     train_ds = CsvPairs(args.train_csv, preprocess, has_group=args.has_group, multi_text=args.multi_text)
#     val_ds   = CsvPairs(args.val_csv,   preprocess, has_group=args.has_group, multi_text=args.multi_text)

#     # loaders (可选分组采样器)
#     if args.grouped-sampler and (train_ds.groups is not None):
#         sampler = GroupedBatchSampler(train_ds.groups, batch_size=args.batch_size, drop_last=True)
#         train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.num_workers,
#                                   pin_memory=True, collate_fn=collate_fn)
#     else:
#         train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
#                                   num_workers=args.num_workers, pin_memory=True, drop_last=True,
#                                   collate_fn=collate_fn)

#     val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
#                             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

#     scaler = torch.cuda.amp.GradScaler(enabled=args.use-amp)

#     # ====== build optim parts ======
#     if args.mode == 'linear':
#         # 冻结 backbone
#         for p in model.parameters(): p.requires_grad = False
#         model.eval()

#         # 读出嵌入维度（open_clip 的 text_projection 通常能反映维度）
#         if hasattr(model, 'text_projection') and model.text_projection is not None:
#             dim = model.text_projection.shape[-1]
#         else:
#             dim = 768

#         img_head = Projection(dim, hidden=args.proj_hidden).to(device)
#         txt_head = Projection(dim, hidden=args.proj_hidden).to(device)

#         # 独立温度参数
#         logit_scale = nn.Parameter(model.logit_scale.data.clone())
#         model.logit_scale.requires_grad_(False)

#         params = list(img_head.parameters()) + list(txt_head.parameters()) + [logit_scale]
#         opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

#     else:
#         # 注入 LoRA，冻结基础权重
#         model = inject_lora_openclip(
#             model,
#             last_n_blocks=args.lora_last_n,
#             r=args.lora_r,
#             alpha=args.lora_alpha,
#             dropout=args.lora_dropout
#         ).to(device)
#         params = lora_parameters(model)
#         opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

#     # ====== train loop ======
#     for epoch in range(1, args.epochs+1):
#         model.train()
#         total_loss, step = 0.0, 0
#         if args.mode == 'linear':
#             img_head.train(); txt_head.train()
#         opt.zero_grad(set_to_none=True)

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
#         for imgs, texts, gids in pbar:
#             imgs = imgs.to(device, non_blocking=True)
#             toks = tokenizer(texts, context_length=args.context_length).to(device)
#             gids = gids.to(device)

#             with torch.cuda.amp.autocast(enabled=args.use-amp):
#                 if args.mode == 'linear':
#                     with torch.no_grad():
#                         zi = F.normalize(model.encode_image(imgs), dim=-1)
#                         zt = F.normalize(model.encode_text(toks), dim=-1)
#                     zi = img_head(zi)
#                     zt = txt_head(zt)
#                     sim = torch.exp(logit_scale) * (zi @ zt.t())
#                 else:
#                     zi = F.normalize(model.encode_image(imgs), dim=-1)
#                     zt = F.normalize(model.encode_text(toks), dim=-1)
#                     sim = torch.exp(model.logit_scale) * (zi @ zt.t())

#                 if (train_ds.groups is not None):
#                     loss = mp_infonce(sim, gids)
#                 else:
#                     labels = torch.arange(sim.size(0), device=device)
#                     loss = 0.5*(F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))

#             scaler.scale(loss).backward()
#             step += 1
#             if step % args.grad_accum == 0:
#                 scaler.step(opt); scaler.update()
#                 opt.zero_grad(set_to_none=True)
#             total_loss += loss.item()
#             pbar.set_postfix(loss=f"{(total_loss/step):.4f}")

#         # ====== eval ======
#         model.eval()
#         if args.mode == 'linear':
#             img_head.eval(); txt_head.eval()

#         all_I, all_T, all_G = [], [], []
#         with torch.no_grad():
#             for imgs, texts, gids in tqdm(val_loader, desc="Eval"):
#                 imgs = imgs.to(device, non_blocking=True)
#                 toks = tokenizer(texts, context_length=args.context_length).to(device)
#                 zi = F.normalize(model.encode_image(imgs), dim=-1)
#                 zt = F.normalize(model.encode_text(toks), dim=-1)
#                 if args.mode == 'linear':
#                     zi = img_head(zi); zt = txt_head(zt)
#                 all_I.append(zi); all_T.append(zt); all_G.append(gids.to(device))

#         I = F.normalize(torch.cat(all_I, dim=0), dim=-1)
#         T = F.normalize(torch.cat(all_T, dim=0), dim=-1)
#         G = torch.cat(all_G, dim=0) if (val_ds.groups is not None) else None

#         metrics = recall_at_k(T, I, args.eval_k, groups=G)
#         msg = " | ".join([f"{k}={v*100:.2f}%" for k,v in metrics.items()])
#         print(f"[Val] {msg}")

#         # ====== save ======
#         args.out_dir.mkdir(parents=True, exist_ok=True)
#         if args.mode == 'linear':
#             ckpt = {
#                 'mode': 'linear',
#                 'img_head': img_head.state_dict(),
#                 'txt_head': txt_head.state_dict(),
#                 'logit_scale': logit_scale.data,
#                 'model_name': args.model_name,
#                 'pretrained': args.pretrained,
#                 'context_length': args.context_length
#             }
#             torch.save(ckpt, args.out_dir / f'linear_epoch{epoch}.pt')
#         else:
#             # 仅保存 LoRA 可训练参数（包含 logit_scale）
#             state = {n: p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
#             ckpt = {
#                 'mode': 'lora',
#                 'lora_state': state,
#                 'model_name': args.model_name,
#                 'pretrained': args.pretrained,
#                 'context_length': args.context_length,
#                 'lora_last_n': args.lora_last_n,
#                 'lora_r': args.lora_r,
#                 'lora_alpha': args.lora_alpha,
#                 'lora_dropout': args.lora_dropout,
#             }
#             torch.save(ckpt, args.out_dir / f'lora_epoch{epoch}.pt')

# if __name__ == "__main__":
#     main()









import argparse
import random
from pathlib import Path
import pandas as pd
from PIL import Image
from typing import List, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import open_clip
from tqdm import tqdm
import loralib as lora

# ============== Utils ==============

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============== Dataset ==============

class CsvDataset(Dataset):
    """
    CSV 必须包含: filepath, group_id
    可选的 captions_csv 包含: filepath, title
    """
    def __init__(self, csv_path: Path, preprocess, has_group: bool, default_prompt: str, captions_csv: Path = None):
        # --- 关键修改：合并两个CSV文件 ---
        
        # 1. 读取包含 group_id 的主 CSV 文件
        df_groups = pd.read_csv(csv_path)
        assert 'filepath' in df_groups.columns, f"主 CSV 文件 {csv_path} 必须包含 'filepath' 列。"
        if has_group:
            assert 'group_id' in df_groups.columns, f"当 --has-group 开启时，主 CSV 文件 {csv_path} 必须包含 'group_id' 列。"

        # 2. 如果提供了描述文件，则读取并合并
        if captions_csv:
            logger.info(f"正在从 {captions_csv} 加载文本描述...")
            df_captions = pd.read_csv(captions_csv)
            assert 'filepath' in df_captions.columns and 'title' in df_captions.columns, \
                f"描述文件 {captions_csv} 必须包含 'filepath' 和 'title' 列。"
            
            # 使用 filepath 作为键，合并两个 DataFrame
            # inner join: 只保留在两个文件中都存在的图像
            df = pd.merge(df_groups, df_captions, on='filepath', how='inner')
            
            if len(df) == 0:
                raise ValueError("合并分组文件和描述文件后，没有找到任何共同的图像。请检查两个文件中的 'filepath' 是否匹配。")
            
            logger.info(f"成功合并数据，共找到 {len(df)} 条有效记录。")
            self.texts = df['title'].tolist()
        else:
            # 如果没有提供描述文件，则回退到使用 default_prompt
            logger.warning(f"未提供描述文件，将对所有图像使用 --default-prompt。")
            df = df_groups
            self.texts = [default_prompt] * len(df)

        # --- 修改结束 ---

        self.paths = [Path(p).expanduser().resolve() for p in df['filepath']]

        if has_group:
            self.groups = df['group_id'].astype('int64').tolist()
        else:
            # 如果没有分组，每个样本自成一组
            self.groups = list(range(len(df)))

        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.preprocess(Image.open(self.paths[i]).convert("RGB"))
        # group_id 现在总是存在
        group_id = self.groups[i]
        return img, self.texts[i], group_id
    


def collate_fn(batch):
    imgs, texts, gids = zip(*batch)
    return torch.stack(imgs, dim=0), list(texts), torch.tensor(gids, dtype=torch.long)

# ============== Grouped Batch Sampler ==============

class GroupedBatchSampler(Sampler[List[int]]):
    def __init__(self, groups: List[int], batch_size: int, drop_last: bool):
        self.batch_size = batch_size
        self.drop_last = drop_last
        buckets: Dict[int, List[int]] = {}
        for idx, g in enumerate(groups):
            buckets.setdefault(int(g), []).append(idx)
        self.group_ids = list(buckets.keys())
        self.buckets = buckets

    def __iter__(self) -> Iterable[List[int]]:
        order = []
        random.shuffle(self.group_ids)
        cur = []
        for gid in self.group_ids:
            idxs = self.buckets[gid]
            random.shuffle(idxs)
            for i in idxs:
                cur.append(i)
                if len(cur) == self.batch_size:
                    order.append(cur)
                    cur = []
        if (not self.drop_last) and len(cur) > 0:
            order.append(cur)
        return iter(order)

    def __len__(self) -> int:
        total = sum(len(v) for v in self.buckets.values())
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size

# ============== LoRA Injection ==============

def _replace_linear_with_lora_linear(mod: nn.Linear, r=8, alpha=16, dropout=0.0):
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

def inject_lora_openclip(model, last_n_blocks=12, r=8, alpha=16, dropout=0.0):
    for blk in model.visual.transformer.resblocks[-last_n_blocks:]:
        blk.attn.out_proj = _replace_linear_with_lora_linear(blk.attn.out_proj, r, alpha, dropout)
        blk.mlp.c_fc      = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
        blk.mlp.c_proj    = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)
    
    for blk in model.transformer.resblocks[-last_n_blocks:]:
        blk.attn.out_proj = _replace_linear_with_lora_linear(blk.attn.out_proj, r, alpha, dropout)
        blk.mlp.c_fc      = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
        blk.mlp.c_proj    = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)
    
    lora.mark_only_lora_as_trainable(model, bias='lora_only')
    model.logit_scale.requires_grad_(True)
    return model

# ============== Multi-Positive InfoNCE ==============

def mp_infonce(sim: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    G = groups.view(-1, 1)
    pos_mask = (G == G.t())
    big_neg = torch.finfo(sim.dtype).min / 2
    sim_pos = sim.masked_fill(~pos_mask, big_neg)
    lse_pos = torch.logsumexp(sim_pos, dim=1)
    lse_all = torch.logsumexp(sim, dim=1)
    loss_i2t = -(lse_pos - lse_all).mean()
    lse_pos_t = torch.logsumexp(sim_pos.t(), dim=1)
    lse_all_t = torch.logsumexp(sim.t(), dim=1)
    loss_t2i = -(lse_pos_t - lse_all_t).mean()
    return 0.5 * (loss_i2t + loss_t2i)


# ============== Recall@K (T->I) ==============

@torch.no_grad()
def recall_at_k(text_feats: torch.Tensor, image_feats: torch.Tensor, k_list: List[int], groups: torch.Tensor):
    """
    计算多正样本下的图文检索召回率。
    """
    sim = text_feats @ image_feats.t()
    N = sim.size(0)
    topk = torch.topk(sim, k=max(k_list), dim=1).indices

    out = {}
    g = groups.view(-1, 1)
    # pos_mask[i] 是一个布尔张量，表示哪些样本与第 i 个样本同组
    pos_mask = (g == g.t()) 

    for k in k_list:
        # topk_k_indices: [N, k], 每一行是前 k 个检索结果的索引
        topk_k_indices = topk[:, :k]
        
        # 使用 advanced indexing 快速检查命中情况
        # hits[i] 会是 True 如果第 i 个文本的 Top-K 结果中至少有一个是其正样本
        hits = pos_mask[torch.arange(N).unsqueeze(1), topk_k_indices].any(dim=1)
        
        recall_val = hits.float().mean().item()
        out[f'R@{k}'] = recall_val
    return out







# ============== Main ==============
def main():
    ap = argparse.ArgumentParser(description="Fine-tune CLIP with LoRA, using separate files for groups and captions.")
    
    # --- 关键修改：为4个文件定义清晰的参数 ---
    ap.add_argument('--train-groups-csv', type=Path, required=True, help="训练集分组文件，包含 'filepath' 和 'group_id'。")
    ap.add_argument('--train-captions-csv', type=Path, default=None, help="（可选）训练集描述文件，包含 'filepath' 和 'title'。")
    ap.add_argument('--val-groups-csv', type=Path, required=True, help="验证集分组文件，包含 'filepath' 和 'group_id'。")
    ap.add_argument('--val-captions-csv', type=Path, default=None, help="（可选）验证集描述文件，包含 'filepath' 和 'title'。")
    # --- 修改结束 ---

    ap.add_argument('--out-dir',   type=Path, required=True)
    ap.add_argument('--model-name', type=str, default='ViT-L-14')
    ap.add_argument('--pretrained', type=str, default='datacomp_xl_s13b_b90k')
    
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--use-amp', action='store_true')
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--lora-last-n', type=int, default=12)
    ap.add_argument('--lora-r', type=int, default=8)
    ap.add_argument('--lora-alpha', type=int, default=16)
    
    ap.add_argument('--has-group', action='store_true', help="启用多正样本训练模式。")
    ap.add_argument('--default-prompt', type=str, default="a photo of a street scene from a car's perspective", help="当描述文件未提供时使用的默认文本。")
    
    ap.add_argument('--eval-k', nargs='+', type=int, default=[1, 5, 10, 20], help="K values for Recall@K evaluation.")

    args = ap.parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    
    model = inject_lora_openclip(
        model, last_n_blocks=args.lora_last_n, r=args.lora_r, alpha=args.lora_alpha
    )
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # --- 关键修改：使用正确的参数实例化 Dataset ---
    print("Loading training dataset...")
    train_ds = CsvDataset(
        csv_path=args.train_groups_csv, 
        preprocess=preprocess, 
        has_group=args.has_group, 
        default_prompt=args.default_prompt,
        captions_csv=args.train_captions_csv
    )
    
    print("Loading validation dataset...")
    val_ds = CsvDataset(
        csv_path=args.val_groups_csv,
        preprocess=preprocess, 
        has_group=args.has_group, 
        default_prompt=args.default_prompt,
        captions_csv=args.val_captions_csv
    )
    # --- 修改结束 ---

    if args.has_group and train_ds.groups is not None:
        print("Using GroupedBatchSampler for multi-positive training.")
        sampler = GroupedBatchSampler(train_ds.groups, batch_size=args.batch_size, drop_last=True)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    else:
        print("Using standard DataLoader with shuffling.")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, step = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, texts, gids in pbar:
            imgs, gids = imgs.to(device), gids.to(device)
            toks = tokenizer(texts).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                img_feats = F.normalize(model.encode_image(imgs), dim=-1)
                txt_feats = F.normalize(model.encode_text(toks), dim=-1)
                sim = torch.exp(model.logit_scale) * (img_feats @ txt_feats.t())

                if args.has_group and train_ds.groups is not None:
                    loss = mp_infonce(sim, gids)
                else:
                    labels = torch.arange(sim.size(0), device=device)
                    loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            step += 1
            pbar.set_postfix(loss=f"{(total_loss/step):.4f}")

        # --- 验证逻辑 ---
        model.eval()
        all_I, all_T, all_G = [], [], []
        with torch.no_grad():
            for imgs, texts, gids in tqdm(val_loader, desc="Evaluating"):
                imgs = imgs.to(device, non_blocking=True)
                toks = tokenizer(texts).to(device)
                zi = F.normalize(model.encode_image(imgs), dim=-1)
                zt = F.normalize(model.encode_text(toks), dim=-1)
                all_I.append(zi)
                all_T.append(zt)
                all_G.append(gids.to(device))

        I = torch.cat(all_I, dim=0)
        T = torch.cat(all_T, dim=0)
        G = torch.cat(all_G, dim=0) if (val_ds.groups is not None) else None

        if G is not None:
            metrics = recall_at_k(T, I, args.eval_k, groups=G)
            msg = " | ".join([f"{k}={v*100:.2f}%" for k,v in metrics.items()])
            print(f"[Val] {msg}")
        # --- 验证逻辑结束 ---

        lora_state_dict = {n: p.detach().cpu() for n, p in model.named_parameters() if 'lora_' in n or 'logit_scale' in n}
        torch.save(lora_state_dict, args.out_dir / f'lora_epoch_{epoch}.pt')
        print(f"LoRA weights saved to {args.out_dir / f'lora_epoch_{epoch}.pt'}")

if __name__ == "__main__":
    main()