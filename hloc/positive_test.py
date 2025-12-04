# import argparse
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import torch
# from PIL import Image
# from tqdm import tqdm

# from hloc.extractors.netvlad import NetVLAD
# from hloc import logger

# def main():
#     parser = argparse.ArgumentParser(description="Generate training triplets using NetVLAD similarity.")
#     parser.add_argument("--image_dir", type=Path, required=True, help="Directory with all training images.")
#     parser.add_argument("--output_csv", type=Path, required=True, help="Path to save the output triplet CSV file.")
#     parser.add_argument("--pos_threshold", type=float, default=0.5, help="Minimum similarity for a pair to be a positive.")
#     parser.add_argument("--neg_threshold", type=float, default=0.5, help="Maximum similarity for a pair to be a negative.")
#     parser.add_argument("--max_positives", type=int, default=5, help="Max number of positives to sample for each anchor.")
#     parser.add_argument("--max_negatives", type=int, default=5, help="Max number of hard negatives to sample for each anchor.")
#     args = parser.parse_args()

#     image_paths = list(args.image_dir.rglob("*.jpg")) + list(args.image_dir.rglob("*.png"))
#     if not image_paths:
#         raise FileNotFoundError(f"No images found in {args.image_dir}")

#     logger.info("Loading NetVLAD model...")
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # 假设NetVLAD模型已按HLOC标准放置
#     config = {
#         'model': {'name': 'netvlad', 'pretrained': 'vgg16_netvlad'},
#     }
#     netvlad_model = NetVLAD(config['model']).eval().to(device)

#     logger.info(f"Extracting NetVLAD features for {len(image_paths)} images...")
#     descriptors = []

#     with torch.no_grad():
#         for path in tqdm(image_paths, desc="Extracting features"):
#             img = Image.open(path).convert('RGB')
#             # --- 关键修复：将张量转换为 float32 类型 ---
#             img_tensor = torch.from_numpy(np.array(img) / 255.).float().permute(2, 0, 1).unsqueeze(0).to(device)
#             desc = netvlad_model({'image': img_tensor})['global_descriptor']
#             descriptors.append(desc.squeeze().cpu().numpy())
    
#     descriptors = np.array(descriptors)

    
#     logger.info("Calculating similarity matrix...")
#     # NetVLAD描述子已L2归一化，点积即余弦相似度
#     similarity_matrix = descriptors @ descriptors.T
    
#     # 忽略相似度>0.8的样本（视为重复）
#     # 为了简单起见，我们只在采样正样本时排除自身
#     np.fill_diagonal(similarity_matrix, -1) # 排除自身

#     logger.info("Generating triplets...")
#     triplets = []
#     for i in tqdm(range(len(image_paths)), desc="Generating triplets"):
#         anchor_path = image_paths[i]
#         sim_row = similarity_matrix[i]

#         # 找到正样本和负样本的索引
#         pos_indices = np.where(sim_row >= args.pos_threshold)[0]
#         neg_indices = np.where(sim_row < args.neg_threshold)[0]

#         if len(pos_indices) == 0 or len(neg_indices) == 0:
#             continue

#         # 采样
#         np.random.shuffle(pos_indices)
#         np.random.shuffle(neg_indices)
        
#         num_samples = min(len(pos_indices), len(neg_indices), args.max_positives, args.max_negatives)

#         for j in range(num_samples):
#             pos_idx = pos_indices[j]
#             neg_idx = neg_indices[j]
#             triplets.append({
#                 "anchor": str(anchor_path),
#                 "positive": str(image_paths[pos_idx]),
#                 "negative": str(image_paths[neg_idx])
#             })

#     logger.info(f"Generated {len(triplets)} triplets.")
#     df_triplets = pd.DataFrame(triplets)
#     df_triplets.to_csv(args.output_csv, index=False)
#     logger.info(f"Triplets saved to {args.output_csv}")

# if __name__ == "__main__":
#     main()



    




# import argparse
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import torch
# from PIL import Image
# from tqdm import tqdm
# import sys

# try:
#     from hloc.extractors.netvlad import NetVLAD
#     from hloc import logger
# except ImportError:
#     sys.path.append(str(Path(__file__).parent.parent.parent))
#     from hloc.extractors.netvlad import NetVLAD
#     from hloc import logger

# def extract_features(image_paths, model, device):
#     """为给定的图像路径列表提取NetVLAD特征。"""
#     descriptors = []
#     with torch.no_grad():
#         for path in tqdm(image_paths, desc="  Extracting features", leave=False):
#             try:
#                 img = Image.open(path).convert('RGB')
#                 img_tensor = torch.from_numpy(np.array(img) / 255.).float().permute(2, 0, 1).unsqueeze(0).to(device)
#                 desc = model({'image': img_tensor})['global_descriptor']
#                 descriptors.append(desc.squeeze().cpu().numpy())
#             except Exception as e:
#                 logger.warning(f"无法处理图像 {path}: {e}，已跳过。")
#                 descriptors.append(None)
    
#     valid_indices = [i for i, d in enumerate(descriptors) if d is not None]
#     if len(valid_indices) != len(image_paths):
#         logger.warning(f"成功处理了 {len(valid_indices)} / {len(image_paths)} 张图像。")
    
#     valid_paths = [image_paths[i] for i in valid_indices]
#     valid_descriptors = np.array([descriptors[i] for i in valid_indices])
    
#     return valid_paths, valid_descriptors

# def generate_groups_for_folder(folder_path, model, device, args, global_group_id_counter):
#     """
#     为CLIP训练准备数据，生成 (filepath, group_id) 对。
#     同一视觉簇（锚点+其所有正样本）的图像拥有相同的 group_id。
#     """
#     logger.info(f"Processing folder for grouping: {folder_path.relative_to(args.input_dir)}")
#     image_paths = sorted(list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")))
#     if not image_paths:
#         return [], global_group_id_counter

#     image_paths, descriptors = extract_features(image_paths, model, device)
#     if descriptors.shape[0] < 2:
#         return [], global_group_id_counter

#     similarity_matrix = descriptors @ descriptors.T
#     np.fill_diagonal(similarity_matrix, 1.0) # 自身与自身的相似度为1

#     grouped_pairs = []
#     processed_indices = set()

#     for i in tqdm(range(len(image_paths)), desc="  Grouping images", leave=False):
#         if i in processed_indices:
#             continue

#         sim_row = similarity_matrix[i]
#         pos_indices = np.where(sim_row >= args.pos_threshold)[0]
        
#         current_group_indices = set(pos_indices)
        
#         if len(current_group_indices) > 1:
#             group_id = global_group_id_counter
#             for member_idx in current_group_indices:
#                 if member_idx not in processed_indices:
#                     grouped_pairs.append({
#                         "filepath": str(image_paths[member_idx]),
#                         "group_id": group_id
#                     })
#             processed_indices.update(current_group_indices)
#             global_group_id_counter += 1

#     return grouped_pairs, global_group_id_counter

# def generate_pairs_for_folder(folder_path, model, device, args):
#     """为单个测试文件夹生成 (anchor, positive) 对，用于评估召回率。"""
#     logger.info(f"Processing test folder: {folder_path.name}")
#     image_paths = sorted(list(folder_path.rglob("*.jpg")) + list(folder_path.rglob("*.png")))
#     if not image_paths: return []

#     image_paths, descriptors = extract_features(image_paths, model, device)
#     if descriptors.shape[0] < 2: return []

#     similarity_matrix = descriptors @ descriptors.T
#     np.fill_diagonal(similarity_matrix, -1)

#     pairs = []
#     for i in tqdm(range(len(image_paths)), desc="  Generating pairs", leave=False):
#         anchor_path = image_paths[i]
#         sim_row = similarity_matrix[i]
#         pos_indices = np.where(sim_row >= args.pos_threshold)[0]
#         for pos_idx in pos_indices:
#             pairs.append({"anchor": str(anchor_path), "positive": str(image_paths[pos_idx])})
#     return pairs

# def main():
#     parser = argparse.ArgumentParser(description="智能生成训练分组或测试样本对。")
#     parser.add_argument("--input_dir", type=Path, required=True, help="根目录。")
#     parser.add_argument("--output_csv", type=Path, required=True, help="输出CSV路径。")
#     parser.add_argument("--pos_threshold", type=float, default=0.5, help="正样本的最小相似度。")
#     # 移除不再使用的参数
#     # parser.add_argument("--neg_threshold", ...)
#     # parser.add_argument("--duplicate_threshold", ...)
#     # parser.add_argument("--max_positives", ...)
#     # parser.add_argument("--max_negatives", ...)
#     args = parser.parse_args()

#     subdirs_exist = any(d.is_dir() for d in args.input_dir.iterdir())
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     logger.info(f"Loading NetVLAD model on {device}...")
#     config = {'model': {'name': 'netvlad', 'pretrained': 'vgg16_netvlad'}}
#     netvlad_model = NetVLAD(config['model']).eval().to(device)

#     if subdirs_exist:
#         logger.info("--- 训练模式启动 (生成分组ID) ---")
#         all_images = list(args.input_dir.rglob("*.jpg")) + list(args.input_dir.rglob("*.png"))
#         if not all_images:
#             logger.error(f"在 {args.input_dir} 中未找到任何图像。")
#             return
            
#         images_by_dir = {p.parent for p in all_images}
#         leaf_dirs = {d for d in images_by_dir if not any(item.is_dir() for item in d.iterdir())}
        
#         if not leaf_dirs:
#             logger.error("未找到任何包含图像的叶子目录。")
#             return

#         logger.info(f"找到 {len(leaf_dirs)} 个叶子目录进行处理。")
        
#         all_grouped_pairs = []
#         global_group_id_counter = 0
#         for leaf_dir in sorted(list(leaf_dirs)):
#             pairs, global_group_id_counter = generate_groups_for_folder(leaf_dir, netvlad_model, device, args, global_group_id_counter)
#             all_grouped_pairs.extend(pairs)
        
#         if not all_grouped_pairs:
#             logger.error("未能生成任何分组。")
#             return
            
#         logger.info(f"总共生成了 {len(all_grouped_pairs)} 条记录，分布在 {global_group_id_counter} 个组中。")
#         df_results = pd.DataFrame(all_grouped_pairs)
        
#     else:
#         logger.info("--- 测试模式启动 (生成评估对) ---")
#         all_pairs = generate_pairs_for_folder(args.input_dir, netvlad_model, device, args)
#         if not all_pairs:
#             logger.error("未能生成任何样本对。")
#             return
#         logger.info(f"总共生成了 {len(all_pairs)} 个正样本对。")
#         df_results = pd.DataFrame(all_pairs)

#     args.output_csv.parent.mkdir(parents=True, exist_ok=True)
#     df_results.to_csv(args.output_csv, index=False)
#     logger.info(f"结果已成功保存到: {args.output_csv}")

# if __name__ == "__main__":
#     main()




# python3 -m hloc.positive_test \
#         --input_dir datasets/SensorKit/clipdata/clipdata_prepare \
#         --output_csv datasets/SensorKit/clipdata/clipdata_prepare/train_triplets.csv



    # python3 -m hloc.positive_test \
    #     --input_dir datasets/SensorKit/clipdata/clipdata_prepare_cleaned \
    #     --output_csv datasets/SensorKit/clipdata/clipdata_prepare_cleaned/grouped_train_data.csv \
    #     --pos_threshold 0.5


# python3 -m hloc.positive_test \
#     --input_dir datasets/SensorKit/query/cam1_all/undistorted \ 
#     --output_csv datasets/SensorKit/query/cam1_all/positive_pairs_for_eval.csv \     
#     --pos_threshold 0.4




import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import sys

# 动态添加hloc到路径，以便在模块外也能运行
try:
    from hloc.extractors.netvlad import NetVLAD
    from hloc import logger
except ImportError:
    # 如果直接运行脚本，可能需要将hloc的父目录添加到sys.path
    hloc_path = Path(__file__).parent.parent
    if str(hloc_path) not in sys.path:
        sys.path.append(str(hloc_path))
    from hloc.extractors.netvlad import NetVLAD
    from hloc import logger

def extract_features(image_paths, model, device, desc_text="Extracting features"):
    """为给定的图像路径列表提取NetVLAD特征，并处理可能发生的错误。"""
    descriptors = []
    valid_paths = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc=f"  {desc_text}", leave=False):
            try:
                img = Image.open(path).convert('RGB')
                # NetVLAD模型内部会处理图像大小调整
                img_tensor = torch.from_numpy(np.array(img) / 255.).float().permute(2, 0, 1).unsqueeze(0).to(device)
                desc = model({'image': img_tensor})['global_descriptor']
                descriptors.append(desc.squeeze().cpu().numpy())
                valid_paths.append(path)
            except Exception as e:
                logger.warning(f"无法处理图像 {path}: {e}，已跳过。")
    
    if not valid_paths:
        return [], np.array([])
        
    return valid_paths, np.array(descriptors)

def main():
    parser = argparse.ArgumentParser(description="从一个数据库中为查询图像寻找正样本。")
    parser.add_argument("--query_dir", type=Path, required=True, help="包含查询图像的目录 (例如 cam1)。")
    parser.add_argument("--db_dir", type=Path, required=True, help="包含所有数据库图像的根目录 (例如包含 cam0-cam6)。")
    parser.add_argument("--output_csv", type=Path, required=True, help="输出CSV文件的路径。")
    parser.add_argument("--pos_threshold", type=float, default=0.4, help="定义正样本的最小相似度。")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"在 {device} 上加载 NetVLAD 模型...")
    config = {'model': {'name': 'netvlad', 'pretrained': 'vgg16_netvlad'}}
    netvlad_model = NetVLAD(config['model']).eval().to(device)

    # 1. 收集并处理查询图像
    logger.info(f"正在从查询目录 {args.query_dir} 收集图像...")
    query_image_paths = sorted(list(args.query_dir.rglob("*.jpg")) + list(args.query_dir.rglob("*.png")))
    if not query_image_paths:
        logger.error(f"在查询目录 {args.query_dir} 中未找到任何图像。")
        return
    
    query_paths, query_descriptors = extract_features(query_image_paths, netvlad_model, device, "Extracting query features")
    if len(query_paths) == 0:
        logger.error("未能成功提取任何查询图像的特征。")
        return

    # 2. 收集并处理数据库图像
    logger.info(f"正在从数据库目录 {args.db_dir} 收集图像...")
    db_image_paths = sorted(list(args.db_dir.rglob("*.jpg")) + list(args.db_dir.rglob("*.png")))
    if not db_image_paths:
        logger.error(f"在数据库目录 {args.db_dir} 中未找到任何图像。")
        return

    db_paths, db_descriptors = extract_features(db_image_paths, netvlad_model, device, "Extracting database features")
    if len(db_paths) == 0:
        logger.error("未能成功提取任何数据库图像的特征。")
        return

    # 3. 计算相似度矩阵 (Query x DB)
    logger.info("正在计算查询与数据库之间的相似度矩阵...")
    # query_descriptors: (Q, D), db_descriptors: (N, D) -> similarity_matrix: (Q, N)
    similarity_matrix = query_descriptors @ db_descriptors.T
    
    # 4. 生成正样本对
    logger.info("正在生成正样本对...")
    positive_pairs = []
    for i in tqdm(range(len(query_paths)), desc="  Generating pairs"):
        anchor_path = query_paths[i]
        sim_row = similarity_matrix[i]
        
        # 找到所有相似度超过阈值的数据库图像
        pos_indices = np.where(sim_row >= args.pos_threshold)[0]
        
        for pos_idx in pos_indices:
            positive_path = db_paths[pos_idx]
            
            # 确保查询图像不与自身匹配（以防查询目录也是数据库目录的一部分）
            if anchor_path.resolve() == positive_path.resolve():
                continue
                
            positive_pairs.append({
                "anchor": str(anchor_path.relative_to(args.query_dir.parent.parent)), # 使路径更通用
                "positive": str(positive_path.relative_to(args.db_dir.parent.parent))
            })

    if not positive_pairs:
        logger.warning("未能根据给定的阈值找到任何正样本对。")
        return

    # 5. 保存结果
    logger.info(f"总共生成了 {len(positive_pairs)} 个正样本对。")
    df_pairs = pd.DataFrame(positive_pairs)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_csv(args.output_csv, index=False)
    logger.info(f"正样本对已成功保存到: {args.output_csv}")

if __name__ == "__main__":
    main()




# python3 -m hloc.positive_test \
#     --query_dir datasets/SensorKit/query/cam1_all/undistorted \
#     --db_dir datasets/SensorKit/allcams \
#     --output_csv datasets/SensorKit/query/cam1_all/allcams_positive_pairs_for_eval.csv \
#     --pos_threshold 0.4






# import argparse
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import torch

# from . import logger
# from .utils.base_model import BaseModel
# from .utils.image_loader import ImageLoader

# # --- 模型定义 (NetVLAD) ---
# class NetVLAD(BaseModel):
#     default_conf = {
#         'model': {'name': 'netvlad', 'pretrained': 'vgg16_netvlad'},
#     }
#     required_inputs = ['image']

#     def _init(self, conf):
#         self.net = torch.hub.load(
#             'gmberton/netvlad', conf['model']['name'],
#             pretrained=conf['model']['pretrained'],
#         )

#     def _forward(self, data):
#         return self.net(data['image'])

# # --- 特征提取 ---
# @torch.no_grad()
# def extract_features(image_paths, model, device):
#     loader = ImageLoader(image_paths, {'grayscale': False, 'resize': 480})
#     loader = torch.utils.data.DataLoader(loader, num_workers=4)
    
#     all_descriptors = []
#     for data in tqdm(loader, desc="  Extracting features", leave=False):
#         images = data['image'].to(device)
#         descriptors = model({'image': images})
#         all_descriptors.append(descriptors.cpu().numpy())
    
#     return np.concatenate(all_descriptors, axis=0)

# # --- 主函数 ---
# def main():
#     parser = argparse.ArgumentParser(description="为每张图像创建一个分组，包含其所有正样本。")
#     parser.add_argument("--input_dir", type=Path, required=True, help="包含图像的根目录。")
#     parser.add_argument("--output_csv", type=Path, required=True, help="输出CSV文件的路径。")
#     parser.add_argument("--pos_threshold", type=float, default=0.6, help="定义正样本的最小相似度。")
    
#     args = parser.parse_args()

#     logger.info("--- 独立锚点分组模式启动 ---")
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     logger.info(f"Loading NetVLAD model on {device}...")
#     config = {'model': {'name': 'netvlad', 'pretrained': 'vgg16_netvlad'}}
#     netvlad_model = NetVLAD(config['model']).eval().to(device)

#     all_images = sorted(list(args.input_dir.rglob("*.jpg")) + list(args.input_dir.rglob("*.png")))
#     if not all_images:
#         logger.error(f"在 {args.input_dir} 中未找到任何图像。")
#         return

#     logger.info(f"总共找到 {len(all_images)} 张图像。")
    
#     descriptors = extract_features(all_images, netvlad_model, device)
    
#     logger.info("正在计算相似度矩阵...")
#     similarity_matrix = descriptors @ descriptors.T
    
#     all_records = []
    
#     # --- 关键修改：每个图像都作为锚点创建一个新组 ---
#     for i in tqdm(range(len(all_images)), desc="为每个锚点创建分组"):
#         anchor_path = all_images[i]
        
#         # 1. 锚点本身是组的第一个成员。组ID就是锚点的索引。
#         all_records.append({
#             "filepath": str(anchor_path),
#             "group_id": i 
#         })
        
#         # 2. 查找与该锚点相似的所有正样本
#         sim_row = similarity_matrix[i]
#         positive_indices = np.where((sim_row >= args.pos_threshold) & (np.arange(len(all_images)) != i))[0]
        
#         # 3. 将所有找到的正样本添加到同一个组中
#         for pos_idx in positive_indices:
#             positive_path = all_images[pos_idx]
#             all_records.append({
#                 "filepath": str(positive_path),
#                 "group_id": i # 使用与锚点相同的 group_id
#             })
#     # --- 修改结束 ---

#     df_results = pd.DataFrame(all_records)
    
#     num_groups = df_results['group_id'].nunique()
#     logger.info(f"总共生成了 {len(df_results)} 条记录，分布在 {num_groups} 个组中。")

#     # --- 保存结果 ---
#     df_results.to_csv(args.output_csv, index=False)
#     logger.info(f"结果已成功保存到: {args.output_csv}")

# if __name__ == "__main__":
#     main()


# # ### 新逻辑解释

# # 1.  **移除 `processed_images`**：我们完全删除了用于跟踪已处理图像的 `processed_images` 集合以及所有相关的 `if ... continue` 判断。
# # 2.  **简化循环**：现在 `for` 循环会无条件地处理 `0` 到 `N-1` 的每一张图像。
# # 3.  **直接分配 `group_id`**：对于第 `i` 次循环（即处理第 `i` 张图像作为锚点），新创建的组的 `group_id` **总是** `i`。我们不再需要 `group_id_counter`。
# # 4.  **无条件添加正样本**：找到锚点的所有正样本后，它们会被无条件地添加到当前组中，不再检查它们是否已属于其他组。

# # ### 预期结果

# # *   **日志输出**：`总共生成了 X 条记录，分布在 Y 个组中。` 这里的 `Y` 将会**严格等于**你的图像总数。`X` 会大于等于 `Y`。
# # *   **CSV 文件内容**：
# #     *   如果图像A（索引为0）和图像B（索引为1）互为正样本，你的 CSV 文件中会同时包含：
# #         *   `path/to/image_A.jpg, 0` （A是锚点）
# #         *   `path/to/image_B.jpg, 0` （B是A的正样本）
# #         *   `path/to/image_B.jpg, 1` （B是锚点）
# #         *   `path/to/image_A.jpg, 1` （A是B的正样本）
# #     *   如果图像C（索引为2）没有正样本，文件中将只包含一条记录：
# #         *   `path/to/image_C.jpg, 2`

# # 这个版本精确地实现了你“每个图像都创建一个组，允许成员重叠”的需求。# filepath: /home/wxx/HLOC_ws/hloc/positive_test.py
