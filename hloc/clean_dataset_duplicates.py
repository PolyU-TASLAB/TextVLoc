import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import shutil

# 使用正确的相对导入
from .extractors.netvlad import NetVLAD
from .utils.image_loader import ImageLoader, collate_fn
from . import logger

@torch.no_grad()
def extract_features(image_paths, model, device):
    """使用 ImageLoader 高效地为图像列表提取NetVLAD特征。"""
    loader_conf = {'grayscale': False, 'resize': 480}
    loader = ImageLoader(image_paths, loader_conf)
    data_loader = torch.utils.data.DataLoader(
        loader, num_workers=4, batch_size=16, collate_fn=collate_fn)
    
    all_descriptors = []
    processed_paths = []
    
    for data in tqdm(data_loader, desc="  Extracting features", leave=False, ncols=80):
        if not data: continue
        images = data['image'].to(device)
        paths = data['path']
        
        # --- 关键修改：从模型输出的字典中提取特征 ---
        output_dict = model({'image': images})
        descriptors = output_dict['global_descriptor'] # NetVLAD返回一个字典
        # --- 修改结束 ---

        all_descriptors.append(descriptors.cpu().numpy())
        processed_paths.extend(paths)
    
    if not all_descriptors:
        return [], np.array([])
        
    return processed_paths, np.concatenate(all_descriptors, axis=0)

def process_and_copy_folder(source_leaf_dir, output_root_dir, model, device, args):
    """在单个叶子文件夹中通过顺序比较移除重复项，并将唯一图像复制到新位置。"""
    logger.info(f"Processing folder: {source_leaf_dir.relative_to(args.input_dir)}")
    
    relative_path = source_leaf_dir.relative_to(args.input_dir)
    output_leaf_dir = output_root_dir / relative_path
    output_leaf_dir.mkdir(parents=True, exist_ok=True)

    image_paths_on_disk = sorted(list(source_leaf_dir.glob("*.jpg")) + list(source_leaf_dir.glob("*.png")))
    if len(image_paths_on_disk) < 2:
        logger.info("  文件夹中图像少于2张，全部复制。")
        for img_path in image_paths_on_disk:
            shutil.copy(img_path, output_leaf_dir)
        return len(image_paths_on_disk), 0

    valid_image_paths, descriptors = extract_features(image_paths_on_disk, model, device)

    if len(valid_image_paths) < 2:
        logger.info("  有效图像少于2张，全部复制。")
        for img_path in valid_image_paths:
            shutil.copy(img_path, output_leaf_dir)
        return len(valid_image_paths), 0

    kept_indices = [0]
    last_kept_idx = 0

    for i in range(1, len(valid_image_paths)):
        similarity = descriptors[i] @ descriptors[last_kept_idx]
        
        if similarity <= args.duplicate_threshold:
            kept_indices.append(i)
            last_kept_idx = i
    
    for i in kept_indices:
        shutil.copy(valid_image_paths[i], output_leaf_dir)
    
    kept_count = len(kept_indices)
    removed_count = len(image_paths_on_disk) - kept_count
    logger.info(f"  完成: 保留 {kept_count} 张, 移除 {removed_count} 张重复图像。")
    return kept_count, removed_count

def main():
    parser = argparse.ArgumentParser(description="通过移除视觉重复的图像来清理训练数据集。")
    parser.add_argument("--input_dir", type=Path, required=True, help="待清理的训练数据集根目录。")
    parser.add_argument("--output_dir", type=Path, required=True, help="保存清理后数据集的新目录。")
    parser.add_argument("--duplicate_threshold", type=float, default=0.8, help="视为重复项的NetVLAD相似度阈值。")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        logger.warning(f"输出目录 {args.output_dir} 已存在且非空。可能会覆盖文件。")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Loading NetVLAD model on {device}...")
    config = {'model': {'name': 'netvlad', 'pretrained': 'vgg16_netvlad'}}
    netvlad_model = NetVLAD(config).eval().to(device)

    all_images = list(args.input_dir.rglob("*.jpg")) + list(args.input_dir.rglob("*.png"))
    if not all_images:
        logger.error(f"在 {args.input_dir} 及其子目录中未找到任何图像。")
        return
        
    images_by_dir = {}
    for img_path in all_images:
        parent_dir = img_path.parent
        if parent_dir not in images_by_dir:
            images_by_dir[parent_dir] = []
        images_by_dir[parent_dir].append(img_path)
        
    leaf_dirs = sorted(list(images_by_dir.keys()))
    
    if not leaf_dirs:
        logger.error("未找到任何包含图像的目录。")
        return

    logger.info(f"找到 {len(leaf_dirs)} 个目录进行处理。")
    
    total_kept = 0
    total_removed = 0
    for leaf_dir in tqdm(leaf_dirs, desc="Processing directories"):
        kept, removed = process_and_copy_folder(leaf_dir, args.output_dir, netvlad_model, device, args)
        total_kept += kept
        total_removed += removed

    logger.info("--- 清理完成 ---")
    logger.info(f"总共保留图像: {total_kept}")
    logger.info(f"总共移除图像: {total_removed}")
    logger.info(f"清理后的数据集已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()



# python3 -m hloc.clean_dataset_duplicates \
#         --input_dir datasets/SensorKit/clipdata/clipdata_prepare_sparse05/ \
#         --output_dir datasets/SensorKit/clipdata/clipdata_prepare_sparse03/ \
#         --duplicate_threshold 0.3


# python3 -m hloc.clean_dataset_duplicates \
#         --input_dir datasets/SensorKit/query/cam1_all/undistorted/ \
#         --output_dir datasets/SensorKit/query/cam1_all/undistorted/clipdata_sparse03/ \
#         --duplicate_threshold 0.3

# python3 -m hloc.clean_dataset_duplicates \
#         --input_dir datasets/SensorKit/2025-9-5/vehicle \
#         --output_dir datasets/SensorKit/2025-9-5/vehicle_cleaned\
#         --duplicate_threshold 0.5

# python3 -m hloc.clean_dataset_duplicates \
#         --input_dir datasets/SensorKit/clipdata/Pittsburgh250k/ \
#         --output_dir datasets/SensorKit/clipdata/Pittsburgh250k_sparse03/ \
#         --duplicate_threshold 0.3