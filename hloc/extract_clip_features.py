import argparse
from pathlib import Path
import torch
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
from . import logger
from transformers import AutoProcessor, AutoModel, SiglipProcessor, SiglipModel


# 检查 transformers 是否安装，如果没有则提示
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" 
MODEL_ID = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
# MODEL_ID = "zer0int/LongCLIP-GmP-ViT-L-14" 
# MODEL_ID = "google/siglip-base-patch16-224" 


@torch.no_grad() 
def main(image_dir: Path, export_path: Path, model_id: str = MODEL_ID):
    """
    使用 CLIP 图像编码器为目录中的所有图像提取特征。
    """
    logger.info(f"Extracting CLIP features for images in {image_dir}...")
    
    model = CLIPModel.from_pretrained(model_id).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_id)


    # # Corrected code for extract_clip_features.py
    # model = SiglipModel.from_pretrained(model_id).to(DEVICE)
    # processor = SiglipProcessor.from_pretrained(model_id)


    image_paths = glob.glob((image_dir / "**/*.jpg").as_posix(), recursive=True)
    if not image_paths:
        logger.error(f"No JPG images found in {image_dir}")
        return

    with h5py.File(export_path, 'w', libver='latest') as f:
        for img_path_str in tqdm(image_paths, desc="Extracting CLIP features"):
            img_path = Path(img_path_str)
            image_name = img_path.relative_to(image_dir).as_posix()
            
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                image_features = model.get_image_features(**inputs)
                
                # L2 归一化特征以方便计算余弦相似度
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                f[image_name] = image_features.cpu().numpy().squeeze()
            except Exception as e:
                logger.warning(f"Could not process {image_name}: {e}")

    logger.info(f"CLIP features saved to {export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--export_path", type=Path, required=True)
    args = parser.parse_args()
    main(args.image_dir, args.export_path)
