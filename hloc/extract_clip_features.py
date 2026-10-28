# import argparse
# from pathlib import Path
# import torch
# import h5py
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import glob
# from . import logger
# from transformers import AutoProcessor, AutoModel, SiglipProcessor, SiglipModel


# # 检查 transformers 是否安装，如果没有则提示
# try:
#     from transformers import CLIPProcessor, CLIPModel
# except ImportError:
#     raise ImportError("Please install transformers: pip install transformers")

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" 
# MODEL_ID = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
# # MODEL_ID = "zer0int/LongCLIP-GmP-ViT-L-14" 
# # MODEL_ID = "google/siglip-base-patch16-224" 


# @torch.no_grad() 
# def main(image_dir: Path, export_path: Path, model_id: str = MODEL_ID):
#     """
#     使用 CLIP 图像编码器为目录中的所有图像提取特征。
#     """
#     logger.info(f"Extracting CLIP features for images in {image_dir}...")
    
#     model = CLIPModel.from_pretrained(model_id).to(DEVICE)
#     processor = CLIPProcessor.from_pretrained(model_id)


#     # # Corrected code for extract_clip_features.py
#     # model = SiglipModel.from_pretrained(model_id).to(DEVICE)
#     # processor = SiglipProcessor.from_pretrained(model_id)


#     image_paths = glob.glob((image_dir / "**/*.jpg").as_posix(), recursive=True)
#     if not image_paths:
#         logger.error(f"No JPG images found in {image_dir}")
#         return

#     with h5py.File(export_path, 'w', libver='latest') as f:
#         for img_path_str in tqdm(image_paths, desc="Extracting CLIP features"):
#             img_path = Path(img_path_str)
#             image_name = img_path.relative_to(image_dir).as_posix()
            
#             try:
#                 image = Image.open(img_path).convert("RGB")
#                 inputs = processor(images=image, return_tensors="pt").to(DEVICE)
#                 image_features = model.get_image_features(**inputs)
                
#                 # L2 归一化特征以方便计算余弦相似度
#                 image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
#                 f[image_name] = image_features.cpu().numpy().squeeze()
#             except Exception as e:
#                 logger.warning(f"Could not process {image_name}: {e}")

#     logger.info(f"CLIP features saved to {export_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image_dir", type=Path, required=True)
#     parser.add_argument("--export_path", type=Path, required=True)
#     args = parser.parse_args()
#     main(args.image_dir, args.export_path)





import argparse
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

# LoRA library
import loralib as lora

# Local imports
from . import logger

# Transformers
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")

# OpenCLIP (will be imported conditionally when needed)
# import open_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_ID = "facebook/metaclip-l14-fullcc2.5b"


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


def load_model_and_processor(model_id: str = None, lora_ckpt_path: Path = None, device: str = "cuda"):
    """
    加载 CLIP/SigLIP/其他 VLM 模型和处理器。
    """
    # ============================================================
    # 情况 1: 使用 LoRA 微调模型（仅支持 OpenCLIP）
    # ============================================================
    if lora_ckpt_path:
        from hloc.clip_lora_checkpoint import load_openclip_lora_checkpoint

        lora_ckpt_path = Path(lora_ckpt_path)
        if not lora_ckpt_path.is_file():
            raise FileNotFoundError(lora_ckpt_path)
        logger.info(f"Loading validated LoRA model from {lora_ckpt_path}")
        model, tokenizer, preprocess, _ = load_openclip_lora_checkpoint(lora_ckpt_path, device)
        return model, tokenizer, preprocess
    
    # ============================================================
    # 情况 2: 使用预训练基线模型
    # ============================================================
    if model_id is None:
        model_id = DEFAULT_MODEL_ID
    
    logger.info(f"📦 Loading baseline pretrained model: {model_id}")
    
    model_id_lower = model_id.lower()
    
    # ============================================================
    # Case A: SigLIP（Google）
    # ============================================================
    if "siglip" in model_id_lower:
        from transformers import AutoModel, AutoProcessor
        
        model = AutoModel.from_pretrained(model_id).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        preprocess = None
        
        model.eval()
        logger.info("✅ SigLIP model loaded successfully!")
        logger.info("   Model type: Hugging Face SigLIP")
        return model, processor, preprocess
    

    
    elif "eva" in model_id.lower() or "quansun" in model_id.lower():
        import open_clip
        
        logger.info("🔧 Loading EVA-CLIP via OpenCLIP...")
        
        # ✅ 方法 1: 使用 OpenCLIP 注册的名称(推荐)
        model, _, preprocess = open_clip.create_model_and_transforms(
            'EVA02-L-14',  # ← 正确的模型名称
            pretrained='merged2b_s4b_b131k'  # ← 正确的预训练权重名称
        )
        
        # ✅ 方法 2: 手动下载权重后加载(备选)
        # model, _, preprocess = open_clip.create_model_and_transforms(
        #     'EVA02-L-14',
        #     pretrained='/path/to/downloaded/EVA02_CLIP_L_psz14_s4B.pt'
        # )
        
        model = model.to(device).eval()
        
        # ⚠️ 修正: tokenizer 也要使用 'EVA02-L-14'
        processor = open_clip.get_tokenizer('EVA02-L-14')
        
        logger.info("✅ EVA-CLIP loaded successfully!")
        logger.info(f"   Model: EVA02-L-14")
        logger.info(f"   Pretrained: merged2b_s6b_b61k")
        
        return model, processor, preprocess
    # ============================================================
    # Case B: CLIP 系列（OpenAI & LAION）
    # ============================================================
    elif any(prefix in model_id_lower for prefix in ["clip", "metaclip"]):
        from transformers import CLIPModel, CLIPProcessor
        
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        preprocess = None
        
        model.eval()
        logger.info("✅ CLIP model loaded successfully!")
        logger.info("   Model type: Hugging Face CLIP")
        return model, processor, preprocess
    
    # ============================================================
    # Case C: BLIP 系列
    # ============================================================
    elif "blip" in model_id_lower:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
        processor = BlipProcessor.from_pretrained(model_id)
        preprocess = None
        
        model.eval()
        logger.info("✅ BLIP model loaded successfully!")
        logger.info("   Model type: Hugging Face BLIP")
        return model, processor, preprocess
    
    # ============================================================
    # Case D: ViLT
    # ============================================================
    elif "vilt" in model_id_lower:
        from transformers import ViltProcessor, ViltModel
        
        model = ViltModel.from_pretrained(model_id).to(device)
        processor = ViltProcessor.from_pretrained(model_id)
        preprocess = None
        
        model.eval()
        logger.info("✅ ViLT model loaded successfully!")
        logger.info("   Model type: Hugging Face ViLT")
        return model, processor, preprocess
    
    # ============================================================
    # Case E: LongCLIP（使用 OpenCLIP）
    # ============================================================
    elif "longclip" in model_id_lower:
        import open_clip
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14',
            pretrained='datacomp_xl_s13b_b90k'
        )
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        
        model.eval()
        logger.info("✅ LongCLIP model loaded successfully!")
        logger.info("   Model type: OpenCLIP")
        return model, tokenizer, preprocess
    
    # ============================================================
    # 默认：使用 AutoModel（通用）
    # ============================================================
    else:
        from transformers import AutoModel, AutoProcessor
        
        logger.warning(f"⚠️ Unknown model type: {model_id}")
        logger.warning("   Trying AutoModel...")
        
        model = AutoModel.from_pretrained(model_id).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        preprocess = None
        
        model.eval()
        logger.info("✅ Model loaded successfully!")
        logger.info(f"   Model type: {type(model).__name__}")
        return model, processor, preprocess
    
    

def extract_features_openclip(model, preprocess, image_path, device):
    """使用 OpenCLIP 提取特征"""
    from PIL import Image
    import torch.nn.functional as F
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = F.normalize(image_features, dim=-1)
    
    return image_features.cpu().numpy().squeeze()


def extract_features_hf_clip(model, processor, image_path, device):
    """使用 Hugging Face CLIP 提取特征"""
    from PIL import Image
    import torch.nn.functional as F
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = F.normalize(image_features, dim=-1)
    
    return image_features.cpu().numpy().squeeze()


@torch.no_grad()
def main(
    image_dir: Path, 
    export_path: Path, 
    model_id: str = None,
    lora_ckpt_path: Path = None,
    batch_size: int = 32,
    device: str = "cuda"
):
    """
    使用 CLIP 图像编码器为目录中的所有图像提取特征。
    
    Args:
        image_dir: 图像目录
        export_path: 输出 H5 文件路径
        model_id: Hugging Face 模型 ID（如果不使用 LoRA）
        lora_ckpt_path: LoRA 检查点路径（如果使用微调模型）
        batch_size: 批处理大小
        device: 设备
    """
    logger.info(f"Extracting CLIP features for images in {image_dir}...")
    
    # ============================================================
    # 1. 加载模型
    # ============================================================
    model, processor, preprocess = load_model_and_processor(
        model_id=model_id,
        lora_ckpt_path=lora_ckpt_path,
        device=device
    )
    
    # 检测模型类型
    is_openclip = hasattr(model, 'encode_image')
    logger.info(f"   Model type: {'OpenCLIP' if is_openclip else 'Hugging Face CLIP'}")
    
    # ============================================================
    # 2. 获取图像路径
    # ============================================================
    image_paths = glob.glob((image_dir / "**/*.jpg").as_posix(), recursive=True)
    if not image_paths:
        logger.error(f"No JPG images found in {image_dir}")
        return
    
    logger.info(f"Found {len(image_paths)} images")
    
    # ============================================================
    # 3. 提取特征
    # ============================================================
    with h5py.File(export_path, 'w', libver='latest') as f:
        for img_path_str in tqdm(image_paths, desc="Extracting CLIP features"):
            img_path = Path(img_path_str)
            image_name = img_path.relative_to(image_dir).as_posix()
            
            try:
                # 根据模型类型选择提取方法
                if is_openclip:
                    features = extract_features_openclip(model, preprocess, img_path, device)
                else:
                    features = extract_features_hf_clip(model, processor, img_path, device)
                
                # 保存到 H5
                f[image_name] = features
            
            except Exception as e:
                logger.warning(f"Could not process {image_name}: {e}")
    
    logger.info(f"✅ CLIP features saved to {export_path}")
    logger.info(f"   Feature dimension: {features.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP features with optional LoRA fine-tuning")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing images")
    parser.add_argument("--export_path", type=Path, required=True, help="Output H5 file path")
    parser.add_argument("--model_id", type=str, default=None, 
                       help="Hugging Face model ID (ignored if using LoRA)")
    parser.add_argument("--lora_ckpt_path", type=Path, default=None,
                       help="Path to LoRA checkpoint (optional)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    main(
        image_dir=args.image_dir,
        export_path=args.export_path,
        model_id=args.model_id,
        lora_ckpt_path=args.lora_ckpt_path,
        batch_size=args.batch_size,
        device=args.device
    )










