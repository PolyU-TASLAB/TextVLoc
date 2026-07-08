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
from transformers import CLIPModel, CLIPProcessor

def set_seed(seed: int = 42):
    """设置随机种子以确保可复现性。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CsvDataset(Dataset):
    """
    加载图文对，只使用主 CSV 中的 description 作为训练/验证文本。
    """
    def __init__(self, csv_path: Path, preprocess, image_root_dir: Path, for_hf=False):
        df = pd.read_csv(csv_path)
        assert 'image_name' in df.columns and 'description' in df.columns, \
            f"主CSV文件 {csv_path} 必须包含 'image_name' 和 'description' 列。"

        self.paths = [image_root_dir.joinpath(p) for p in df['image_name']]
        self.texts = df['description'].tolist()

        self.preprocess = preprocess
        self.for_hf = for_hf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            processed_img = img if self.for_hf else self.preprocess(img)
            return processed_img, self.texts[i], i
        except Exception:
            return None, self.texts[i], i


def collate_fn(batch):
    """丢弃读图失败样本；文本统一为简单的 list[str]。"""
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, [], None

    imgs, texts, idxs = zip(*batch)
    
    try:
        imgs_stacked = torch.stack(imgs, dim=0)
        return imgs_stacked, list(texts), torch.tensor(idxs, dtype=torch.long)
    except TypeError: 
        return list(imgs), list(texts), torch.tensor(idxs, dtype=torch.long)

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
    """MultiheadAttention 的 Q/K/V 用独立 lora.Linear，权重从 in_proj 拷贝。"""
    def __init__(self, original_attn: nn.MultiheadAttention, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original_attn = original_attn
        d_model = original_attn.embed_dim
        self.q_proj = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=True)
        self.k_proj = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=True)
        self.v_proj = lora.Linear(d_model, d_model, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=True)
        with torch.no_grad():
            q_w, k_w, v_w = original_attn.in_proj_weight.chunk(3)
            self.q_proj.weight.copy_(q_w)
            self.k_proj.weight.copy_(k_w)
            self.v_proj.weight.copy_(v_w)
            if original_attn.in_proj_bias is not None:
                q_b, k_b, v_b = original_attn.in_proj_bias.chunk(3)
                self.q_proj.bias.copy_(q_b)
                self.k_proj.bias.copy_(k_b)
                self.v_proj.bias.copy_(v_b)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, **kwargs):
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        B, L, E = q.shape
        H, D = self.original_attn.num_heads, E // self.original_attn.num_heads
        q = q.view(B, L, H, D).transpose(1, 2)
        k = k.view(B, L, H, D).transpose(1, 2)
        v = v.view(B, L, H, D).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.original_attn.dropout if self.original_attn.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, E)
        return self.original_attn.out_proj(attn_output), None


def inject_lora_openclip(model, target='all', last_n_blocks=12, r=8, alpha=16, dropout=0.0):
    """在视觉/文本 tower 最后若干 block 注入 QKV+MLP LoRA。"""
    def _inject_block(blk):
        blk.attn = LoRAInjectedQKVAttention(blk.attn, r, alpha, dropout)
        blk.attn.original_attn.out_proj = _replace_linear_with_lora_linear(
            blk.attn.original_attn.out_proj, r, alpha, dropout
        )
        blk.mlp.c_fc = _replace_linear_with_lora_linear(blk.mlp.c_fc, r, alpha, dropout)
        blk.mlp.c_proj = _replace_linear_with_lora_linear(blk.mlp.c_proj, r, alpha, dropout)

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
def evaluate(model, val_loader, tokenizer, device, args, cross_attn_module=None):
    """验证集 R@K；若传入 cross_attn_module 则同时报告 cross 分支 recall。"""
    model.eval()
    if cross_attn_module is not None:
        cross_attn_module.eval()
    
    all_I_global, all_T = [], []
    all_I_cross = [] if cross_attn_module is not None else None
    
    for imgs, texts, _ in tqdm(val_loader, desc="Evaluating"):
        if imgs is None: 
            continue
        imgs = imgs.to(device, non_blocking=True)
        toks = tokenizer(texts).to(device)

        zi_global = F.normalize(model.encode_image(imgs), dim=-1)
        zt = F.normalize(model.encode_text(toks), dim=-1)

        all_I_global.append(zi_global)
        all_T.append(zt)

        if cross_attn_module is not None:
            vision_patches = extract_vision_patch_tokens(model, imgs)
            visual_summary, _ = cross_attn_module(zt, vision_patches)
            visual_summary_pooled = F.normalize(visual_summary.mean(dim=1), dim=-1)
            
            all_I_cross.append(visual_summary_pooled)

    if not all_I_global: 
        return None
    
    I_global = torch.cat(all_I_global, 0)
    T = torch.cat(all_T, 0)

    metrics_global = recall_at_k(T, I_global, args.eval_k)
    metrics_global = {f"{k}_global": v for k, v in metrics_global.items()}

    if all_I_cross is not None:
        I_cross = torch.cat(all_I_cross, 0)
        metrics_cross = recall_at_k(T, I_cross, args.eval_k)
        metrics_cross = {f"{k}_cross": v for k, v in metrics_cross.items()}
        metrics_global.update(metrics_cross)
    
    return metrics_global

@torch.no_grad()
def evaluate_hf_baseline(hf_model_id, val_ds, device, args):
    print(f"\nLoading Hugging Face model for baseline: {hf_model_id}")
    hf_model = CLIPModel.from_pretrained(hf_model_id).to(device)
    hf_processor = CLIPProcessor.from_pretrained(hf_model_id)

    hf_val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    hf_model.eval()
    all_I, all_T = [], []
    for imgs, texts, _ in tqdm(hf_val_loader, desc="Evaluating (Hugging Face)"):
        if imgs is None:
            continue
        inputs = hf_processor(
            text=texts, images=imgs, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        img_feats = hf_model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_feats = hf_model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        zi = F.normalize(img_feats, dim=-1)
        zt = F.normalize(txt_feats, dim=-1)
        all_I.append(zi)
        all_T.append(zt)

    if not all_I:
        return None
    I, T = torch.cat(all_I, dim=0), torch.cat(all_T, dim=0)
    return recall_at_k(T, I, args.eval_k)

def get_model_dimensions(model):
    """返回 (vision_dim, text_dim)；优先读模型属性，否则用 dummy 前向推断。"""
    vision_dim = None
    text_dim = None

    if hasattr(model.visual, 'width'):
        vision_dim = model.visual.width
    elif hasattr(model.visual, 'embed_dim'):
        vision_dim = model.visual.embed_dim

    if hasattr(model, 'text_projection') and model.text_projection is not None:
        text_dim = model.text_projection.shape[0]

    if vision_dim is None or text_dim is None:
        dummy_img = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
        dummy_text = torch.zeros(1, 77, dtype=torch.long).to(next(model.parameters()).device)
        with torch.no_grad():
            if vision_dim is None:
                vision_dim = model.visual.conv1(dummy_img).shape[1]
            if text_dim is None:
                text_dim = model.encode_text(dummy_text).shape[-1]

    return vision_dim, text_dim

class TextGuidedVisualAttention(nn.Module):
    """文本条件 cross-attn：视觉 patch 为 KV，输出多 query 视觉摘要 token。"""
    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 768,
        hidden_dim: int = 512,
        num_queries: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, text_dim)
        
    def forward(self, text_feats: torch.Tensor, vision_patches: torch.Tensor):
        B = text_feats.size(0)
        text_proj = self.text_proj(text_feats)
        vision_proj = self.vision_proj(vision_patches)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1) + text_proj.unsqueeze(1)
        attn_out, attn_weights = self.cross_attn(
            query=queries, key=vision_proj, value=vision_proj
        )
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))
        return self.output_proj(queries), attn_weights

class LightweightReranker(nn.Module):
    """文本 query + patch KV 的轻量 MHA，输出标量排序分。"""
    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 1024,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        logger.info(f"Reranker initialized: text={text_dim}, vision={vision_dim}, hidden={hidden_dim}")

    def forward(self, text_feats: torch.Tensor, vision_patches: torch.Tensor) -> torch.Tensor:
        text_proj = self.text_proj(text_feats).unsqueeze(1)
        vision_proj = self.vision_proj(vision_patches)
        attn_out, _ = self.attn(query=text_proj, key=vision_proj, value=vision_proj)
        return self.scorer(self.norm(attn_out).squeeze(1)).squeeze(-1)

class RankingLoss(nn.Module):
    """margin 或 listnet 排序损失。"""
    def __init__(self, loss_type: str = "margin", margin: float = 0.2):
        super().__init__()
        self.loss_type = loss_type
        self.margin = margin
        
        logger.info(f"Ranking Loss: type={loss_type}, margin={margin}")
    
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "margin":
            B, K = neg_scores.shape
            pos_expanded = pos_scores.unsqueeze(1).expand(-1, K)
            return F.relu(self.margin - pos_expanded + neg_scores).mean()
        if self.loss_type == "listnet":
            B, _ = neg_scores.shape
            all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
            target = torch.zeros(B, dtype=torch.long, device=all_scores.device)
            return F.cross_entropy(all_scores, target)
        raise ValueError(f"Unknown loss_type: {self.loss_type}")

def extract_vision_patch_tokens(model, images: torch.Tensor) -> torch.Tensor:
    """OpenCLIP ViT：返回 patch token 序列（不含 CLS）。"""
    x = model.visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    x = torch.cat([
        model.visual.class_embedding.to(x.dtype)
        + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        x,
    ], dim=1)
    x = x + model.visual.positional_embedding.to(x.dtype)
    x = model.visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = model.visual.transformer(x)
    x = x.permute(1, 0, 2)
    return x[:, 1:, :]

def main():
    ap = argparse.ArgumentParser(description="Fine-tune CLIP with LoRA for image-text retrieval.")
    ap.add_argument('--train-csv', type=Path, required=True)
    ap.add_argument('--val-csv', type=Path, required=True)
    ap.add_argument('--train-image-root-dir', type=Path, required=True)
    ap.add_argument('--val-image-root-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--model-name', type=str, default='ViT-L-14')
    ap.add_argument('--pretrained', type=str, default='datacomp_xl_s13b_b90k')
    ap.add_argument('--hf-model-id', type=str, default='laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', help="用于基线比较的 Hugging Face 模型 ID。")
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
    ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--eval-k', nargs='+', type=int, default=[1, 5, 10, 20])
    ap.add_argument('--lora-target', type=str, default='all', choices=['all', 'text', 'vision'],
                    help="指定 LoRA 注入的目标: 'all', 'text', 或 'vision'。")

    ap.add_argument('--use-cross-attn', action='store_true',
                    help="Enable text-guided visual attention module")
    ap.add_argument('--num-visual-queries', type=int, default=8,
                    help="Number of visual summary tokens in cross-attention")
    ap.add_argument('--lambda-cross', type=float, default=0.5,
                    help="Weight for cross-attention loss")
    ap.add_argument('--gradient-accumulation-steps', type=int, default=1,
                    help="Number of gradient accumulation steps (effective_batch_size = batch_size × this)")

    rerank_group = ap.add_argument_group('Reranker Settings')
    rerank_group.add_argument('--use-reranker', action='store_true',
                            help="Enable lightweight reranker module")
    rerank_group.add_argument('--reranker-loss-type', type=str, default='margin',
                            choices=['margin', 'listnet'],
                            help="Type of ranking loss")
    rerank_group.add_argument('--reranker-margin', type=float, default=0.2,
                            help="Margin for margin ranking loss")
    rerank_group.add_argument('--reranker-num-negatives', type=int, default=5,
                            help="Number of hard negatives for reranker training")
    rerank_group.add_argument('--lambda-rerank', type=float, default=0.3,
                            help="Weight for reranker loss")

    args = ap.parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    val_ds_for_hf = CsvDataset(csv_path=args.val_csv, preprocess=None, image_root_dir=args.val_image_root_dir, for_hf=True)
    hf_metrics = evaluate_hf_baseline(args.hf_model_id, val_ds_for_hf, device, args)
    if hf_metrics:
        msg = " | ".join([f"{k}={v*100:.2f}%" for k,v in hf_metrics.items()])
        print(f"-> [Hugging Face Baseline] {msg}\n")

    print("Loading open_clip model and preprocessor...")

    base_pretrained_name = 'datacomp_xl_s13b_b90k'
    lora_checkpoint_path = None
    if args.pretrained and args.pretrained.endswith('.pt'):
        lora_checkpoint_path = args.pretrained
        print(f"LoRA checkpoint specified: {lora_checkpoint_path}. Using '{base_pretrained_name}' as base model.")
    else:
        base_pretrained_name = args.pretrained
        print(f"Base model specified: {base_pretrained_name}. No LoRA checkpoint to load.")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=base_pretrained_name, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    vision_dim, text_dim = get_model_dimensions(model)
    logger.info(f"Detected dimensions: vision={vision_dim}, text={text_dim}")

    logger.info("Injecting LoRA layers...")
    model = inject_lora_openclip(
        model, target=args.lora_target, last_n_blocks=args.lora_last_n,
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    model.to(device)
    
    cross_attn_module = None
    if args.use_cross_attn:
        cross_attn_module = TextGuidedVisualAttention(
            text_dim=text_dim,
            vision_dim=vision_dim,
            hidden_dim=512,
            num_queries=args.num_visual_queries,
            num_heads=8,
            dropout=0.5,
        ).to(device)
        logger.info(f"Cross-Attention: text={text_dim}, vision={vision_dim}")

    reranker_module = None
    if args.use_reranker:
        reranker_module = LightweightReranker(
            text_dim=text_dim,
            vision_dim=vision_dim,
            hidden_dim=256,
            num_heads=4,
            dropout=0.1
        ).to(device)
        
        logger.info(f"Reranker: text={text_dim}, vision={vision_dim}")

    ranking_loss_fn = None
    if reranker_module is not None:
        ranking_loss_fn = RankingLoss(
            loss_type=args.reranker_loss_type,
            margin=args.reranker_margin
        ).to(device)
    
    if lora_checkpoint_path:
        print(f"Loading LoRA weights from {lora_checkpoint_path}...")
        checkpoint = torch.load(lora_checkpoint_path, map_location=device)
        if 'lora_state' in checkpoint:
            model.load_state_dict(checkpoint['lora_state'], strict=False)
            print("Successfully loaded LoRA state.")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("Successfully loaded legacy LoRA checkpoint.")

    print("Loading datasets for open_clip training...")
    train_ds = CsvDataset(
        csv_path=args.train_csv, 
        preprocess=preprocess, 
        image_root_dir=args.train_image_root_dir,
    )

    val_ds = CsvDataset(
        csv_path=args.val_csv, 
        preprocess=preprocess, 
        image_root_dir=args.val_image_root_dir,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    print("\nEvaluating open_clip baseline performance on the validation set...")
    baseline_metrics = evaluate(model, val_loader, tokenizer, device, args)
    if baseline_metrics:
        msg = " | ".join([f"{k}={v*100:.2f}%" for k,v in baseline_metrics.items()])
        print(f"-> [OpenCLIP Baseline] {msg}\n")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if cross_attn_module is not None:
        trainable_params += list(cross_attn_module.parameters())
    if reranker_module is not None:
        trainable_params += list(reranker_module.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    print(
        f"Starting training with main CSV descriptions "
        f"(cross_attn={args.use_cross_attn}, reranker={args.use_reranker}, context_length={model.context_length})."
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        if cross_attn_module is not None:
            cross_attn_module.train()
        
        total_loss, step = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for i, (imgs, texts, _) in enumerate(pbar):
            if imgs is None: 
                continue
            imgs = imgs.to(device)
            
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                img_global = F.normalize(model.encode_image(imgs), dim=-1)

                toks = tokenizer(texts).to(device)
                txt_global = F.normalize(model.encode_text(toks), dim=-1)

                sim_global = torch.exp(model.logit_scale) * (img_global @ txt_global.t())
                labels = torch.arange(sim_global.size(0), device=device)
                loss_global = 0.5 * (
                    F.cross_entropy(sim_global, labels) +
                    F.cross_entropy(sim_global.t(), labels)
                )

                loss = loss_global

                if cross_attn_module is not None:
                    vision_patches = extract_vision_patch_tokens(model, imgs)
                    visual_summary, _ = cross_attn_module(txt_global, vision_patches)
                    visual_summary_pooled = F.normalize(visual_summary.mean(dim=1), dim=-1)

                    sim_cross = torch.exp(model.logit_scale) * (visual_summary_pooled @ txt_global.t())
                    loss_cross = 0.5 * (
                        F.cross_entropy(sim_cross, labels) +
                        F.cross_entropy(sim_cross.t(), labels)
                    )

                    loss = loss + args.lambda_cross * loss_cross

                if reranker_module is not None:
                    if cross_attn_module is None:
                        vision_patches = extract_vision_patch_tokens(model, imgs)

                    pos_scores = reranker_module(txt_global, vision_patches)
                    B = imgs.size(0)
                    K = min(args.reranker_num_negatives, B - 1)
                    neg_scores_list = []
                    for b in range(B):
                        neg_indices = torch.randperm(B, device=imgs.device)
                        neg_indices = neg_indices[neg_indices != b][:K]
                        neg_patches = vision_patches[neg_indices].detach()
                        text_repeated = txt_global[b].unsqueeze(0).expand(K, -1)
                        neg_scores_list.append(reranker_module(text_repeated, neg_patches))

                    neg_scores = torch.stack(neg_scores_list)
                    loss_rerank = ranking_loss_fn(pos_scores, neg_scores)
                    loss = loss + args.lambda_rerank * loss_rerank
                
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            accum_loss += loss.item()
            
            if (i + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                total_loss += accum_loss
                step += 1

                postfix_dict = {
                    'L_global': f"{loss_global.item():.3f}",
                    'avg_loss': f"{(total_loss/step):.4f}"
                }

                if cross_attn_module is not None:
                    postfix_dict['L_cross'] = f"{loss_cross.item():.3f}"
                
                if reranker_module is not None:
                    postfix_dict['L_rerank'] = f"{loss_rerank.item():.3f}"
                
                pbar.set_postfix(**postfix_dict)
                
                accum_loss = 0.0

        if (i + 1) % args.gradient_accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        metrics = evaluate(model, val_loader, tokenizer, device, args, cross_attn_module)
        if metrics:
            msg = " | ".join([f"{k}={v*100:.2f}%" for k,v in metrics.items()])
            print(f"-> [Val Epoch {epoch}] {msg}")

        lora_state = {
            n: p.detach().cpu()
            for n, p in model.named_parameters()
            if 'lora_' in n or 'logit_scale' in n
        }
        
        checkpoint = {
            'mode': 'lora',
            'model_name': args.model_name,
            'pretrained': base_pretrained_name,
            'lora_last_n': args.lora_last_n,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'lora_state': lora_state,
        }
        
        if cross_attn_module is not None:
            checkpoint['cross_attn_state'] = cross_attn_module.state_dict()
            checkpoint['cross_attn_type'] = 'full'
            checkpoint['num_visual_queries'] = args.num_visual_queries
        
        if reranker_module is not None:
            checkpoint['reranker_state'] = reranker_module.state_dict()
            checkpoint['reranker_loss_type'] = args.reranker_loss_type
            checkpoint['reranker_margin'] = args.reranker_margin

        torch.save(checkpoint, args.out_dir / f'lora_epoch_{epoch}.pt')
        print(f"Checkpoint saved (epoch {epoch})\n")

if __name__ == "__main__":
    main()
