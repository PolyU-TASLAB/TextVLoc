# import argparse
# import collections.abc as collections
# from pathlib import Path
# from typing import List, Optional, Union
# import time
# import h5py
# import numpy as np
# import torch

# from . import logger
# from .utils.io import list_h5_names
# from .utils.parsers import parse_image_lists
# from .utils.read_write_model import read_images_binary


# def parse_names(prefix, names, names_all):
#     if prefix is not None:
#         if not isinstance(prefix, str):
#             prefix = tuple(prefix)
#         names = [n for n in names_all if n.startswith(prefix)]
#         if len(names) == 0:
#             raise ValueError(f"Could not find any image with the prefix `{prefix}`.")
#     elif names is not None:
#         if isinstance(names, (str, Path)):
#             names = parse_image_lists(names)
#         elif isinstance(names, collections.Iterable):
#             names = list(names)
#         else:
#             raise ValueError(
#                 f"Unknown type of image list: {names}."
#                 "Provide either a list or a path to a list file."
#             )
#     else:
#         names = names_all
#     return names


# def get_descriptors(names, path, name2idx=None, key="global_descriptor"):
#     if name2idx is None:
#         with h5py.File(str(path), "r", libver="latest") as fd:
#             desc = [fd[n][key].__array__() for n in names]
#     else:
#         desc = []
#         for n in names:
#             with h5py.File(str(path[name2idx[n]]), "r", libver="latest") as fd:
#                 desc.append(fd[n][key].__array__())
#     return torch.from_numpy(np.stack(desc, 0)).float()


# def pairs_from_score_matrix(
#     scores: torch.Tensor,
#     invalid: np.array,
#     num_select: int,
#     min_score: Optional[float] = None,
# ):
#     assert scores.shape == invalid.shape
#     if isinstance(scores, np.ndarray):
#         scores = torch.from_numpy(scores)
#     invalid = torch.from_numpy(invalid).to(scores.device)
#     if min_score is not None:
#         invalid |= scores < min_score
#     scores.masked_fill_(invalid, float("-inf"))

#     # topk = torch.topk(scores, num_select, dim=1)
#     k = min(num_select, scores.shape[1])  # 防止超范围
#     topk = torch.topk(scores, k, dim=1)
    
#     indices = topk.indices.cpu().numpy()
#     valid = topk.values.isfinite().cpu().numpy()


#     pairs = []
#     for i, j in zip(*np.where(valid)):
#         pairs.append((i, indices[i, j]))
#     return pairs


# # def main(
# #     descriptors,
# #     output,
# #     num_matched,
# #     query_prefix=None,
# #     query_list=None,
# #     db_prefix=None,
# #     db_list=None,
# #     db_model=None,
# #     db_descriptors=None,
# #     db_names_precomputed: Optional[list] = None,  # <-- 1. 在这里添加新的可选参数
# #     precomputed_metadata: Optional[dict] = None,  # <-- 1. 添加新的可选参数
# #     db_descriptors_preloaded: Optional[torch.Tensor] = None,  # <-- 1. 添加新的可选参数
# # ):
# #     logger.info("Extracting image pairs from a retrieval database.")

# #     # 计时器开始
# #     start_time = time.time()
# #     # We handle multiple reference feature files.
# #     # We only assume that names are unique among them and map names to files.
# #     t0 = time.time()
# #     # --- 2. 修改预备逻辑 ---
# #     if precomputed_metadata is not None:
# #         logger.info("使用预计算的元数据 (索引和名称列表).")
# #         name2db = precomputed_metadata['name2db']
# #         db_names_h5 = precomputed_metadata['db_names_h5']
# #         query_names_h5 = precomputed_metadata['query_names_h5']
# #     else:
# #         logger.info("正在从 HDF5 文件实时解析元数据...")
# #         if db_descriptors is None:
# #             db_descriptors = descriptors
# #         if isinstance(db_descriptors, (Path, str)):
# #             db_descriptors = [db_descriptors]
# #         name2db = {n: i for i, p in enumerate(db_descriptors) for n in list_h5_names(p)}
# #         db_names_h5 = list(name2db.keys())
# #         query_names_h5 = list_h5_names(descriptors)
# #     # --- 修改结束 ---
# #     logger.info(f"预备0: {time.time() - t0:.2f} 秒")
    
# #     t_prep_start = time.time()
# #     # --- 2. 从这里开始修改数据库名称的解析逻辑 ---
# #     if db_names_precomputed is not None:
# #         logger.info(f"使用预先计算的数据库图像名称列表 (共 {len(db_names_precomputed)} 个).")
# #         db_names = db_names_precomputed
# #     elif db_model:
# #         logger.info(f"正在从 {db_model / 'images.bin'} 读取数据库图像名称...")
# #         t_read_bin = time.time()
# #         images = read_images_binary(db_model / "images.bin")
# #         logger.info(f"  - 读取和解析 images.bin 耗时: {time.time() - t_read_bin:.2f} 秒")
        
# #         t_name_extract = time.time()
# #         db_names = [i.name for i in images.values()]
# #         logger.info(f"  - 从解析结果中提取名称耗时: {time.time() - t_name_extract:.2f} 秒")
# #     else:
# #         logger.info("正在从 HDF5 名称列表或文本文件解析数据库图像名称...")
# #         db_names = parse_names(db_prefix, db_list, db_names_h5)
# #     # --- 修改结束 ---
        
# #     if len(db_names) == 0:
# #         raise ValueError("Could not find any database image.")
# #     logger.info(f"预备总耗时: {time.time() - t_prep_start:.2f} 秒")

# #     query_names = parse_names(query_prefix, query_list, query_names_h5)


# #     # --- 从这里开始修改数据库特征的加载逻辑 ---
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     if db_descriptors_preloaded is not None:
# #         logger.info("使用预加载的数据库特征张量.")
# #         db_desc = db_descriptors_preloaded
# #     else:
# #         logger.info("正在从 HDF5 文件加载数据库特征...")
# #         t0 = time.time()
# #         db_desc = get_descriptors(db_names, db_descriptors, name2db)
# #         logger.info(f"加载数据库特征耗时: {time.time() - t0:.2f} 秒")
# #     # --- 修改结束 ---


# #     # 加载查询特征
# #     t0 = time.time()
# #     query_desc = get_descriptors(query_names, descriptors)
# #     logger.info(f"加载查询特征耗时: {time.time() - t0:.2f} 秒")
# #     # 计算相似性矩阵
# #     t0 = time.time()
# #     sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))
# #     logger.info(f"计算相似性矩阵耗时: {time.time() - t0:.2f} 秒")
# #     # Avoid self-matching
# #     self = np.array(query_names)[:, None] == np.array(db_names)[None]
# #     # 生成图像对
# #     t0 = time.time()
# #     pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
# #     logger.info(f"生成图像对耗时: {time.time() - t0:.2f} 秒")
# #     pairs = [(query_names[i], db_names[j]) for i, j in pairs]

# #     logger.info(f"Found {len(pairs)} pairs.")
# #     with open(output, "w") as f:
# #         f.write("\n".join(" ".join([i, j]) for i, j in pairs))
# #     # 总耗时
# #     logger.info(f"总耗时: {time.time() - start_time:.2f} 秒")





# def main(
#     descriptors,
#     output,
#     num_matched,
#     query_prefix=None,
#     query_list=None,
#     db_prefix=None,
#     db_list=None,
#     db_model=None,
#     db_descriptors=None,
#     db_names_precomputed: Optional[list] = None,
#     precomputed_metadata: Optional[dict] = None,
#     db_descriptors_preloaded: Optional[torch.Tensor] = None,
# ):
#     logger.info("Extracting image pairs from a retrieval database.")
#     start_time = time.time()

#     # =================================================================
#     # 步骤 1: 解析元数据 (保留您的优化)
#     # 确定查询图像名 和 数据库特征文件中的图像名
#     # =================================================================
#     t0 = time.time()
#     if precomputed_metadata is not None:
#         logger.info("使用预计算的元数据 (索引和名称列表).")
#         name2db = precomputed_metadata['name2db']
#         db_names_h5 = set(precomputed_metadata['db_names_h5']) # 使用集合以提高效率
#         query_names_h5 = precomputed_metadata['query_names_h5']
#     else:
#         logger.info("正在从 HDF5 文件实时解析元数据...")
#         if db_descriptors is None: db_descriptors = descriptors
#         if isinstance(db_descriptors, (Path, str)): db_descriptors = [db_descriptors]
#         name2db = {n: i for i, p in enumerate(db_descriptors) for n in list_h5_names(p)}
#         db_names_h5 = set(name2db.keys())
#         query_names_h5 = list_h5_names(descriptors)
#     logger.info(f"预备0 (解析元数据): {time.time() - t0:.2f} 秒")

#     # =================================================================
#     # 步骤 2: 解析3D模型中的图像名 (保留您的优化)
#     # =================================================================
#     t0 = time.time()
#     if db_names_precomputed is not None:
#         logger.info(f"使用预先计算的数据库图像名称列表 (共 {len(db_names_precomputed)} 个).")
#         db_names_from_model = set(db_names_precomputed)
#     elif db_model:
#         logger.info(f"正在从 {db_model / 'images.bin'} 读取数据库图像名称...")
#         images = read_images_binary(db_model / "images.bin")
#         db_names_from_model = {i.name for i in images.values()}
#     else:
#         # 如果没有模型信息，则假定所有在特征文件中的数据库图像都有效
#         db_names_from_model = db_names_h5
#     logger.info(f"预备1 (解析3D模型名称): {time.time() - t0:.2f} 秒")

#     # =================================================================
#     # 步骤 3: 计算最终的数据库图像集 (核心逻辑：计算交集)
#     # =================================================================
#     db_names_final = sorted(list(db_names_h5.intersection(db_names_from_model)))
    
#     if not db_names_final:
#         raise ValueError("三维模型中的图像与特征文件中的图像没有交集!")
#     logger.info(f"将在 {len(db_names_final)} 张图像上进行检索 (三维模型与特征文件的交集).")

#     # =================================================================
#     # 步骤 4: 加载特征 (现在只加载交集内的特征)
#     # =================================================================
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # 加载查询特征 (这一步总是在线执行)
#     t0 = time.time()
#     query_names = parse_names(query_prefix, query_list, query_names_h5)
#     query_desc = get_descriptors(query_names, descriptors)
#     logger.info(f"加载查询特征耗时: {time.time() - t0:.2f} 秒")

#     # 加载数据库特征 (优先使用预加载的完整张量，并从中筛选子集)
#     t0 = time.time()
#     if db_descriptors_preloaded is not None and precomputed_metadata:
#         logger.info("从预加载的张量中筛选数据库特征子集...")
#         # 创建从HDF5中的名字到索引的映射
#         name_to_idx_h5 = {name: i for i, name in enumerate(precomputed_metadata['db_names_h5'])}
#         # 根据最终的交集名称列表，找到它们在完整张量中的索引
#         indices = [name_to_idx_h5[n] for n in db_names_final if n in name_to_idx_h5]
#         db_desc = db_descriptors_preloaded[indices]
#     else:
#         # 如果没有预加载张量，则回退到从文件加载
#         logger.info("从 HDF5 文件加载数据库特征...")
#         db_desc = get_descriptors(db_names_final, db_descriptors, name2db)
#     logger.info(f"加载数据库特征耗时: {time.time() - t0:.2f} 秒")

#     # =================================================================
#     # 步骤 5: 计算相似度并生成配对 (现在所有维度都已对齐)
#     # =================================================================
#     t0 = time.time()
#     sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))
#     logger.info(f"计算相似性矩阵耗时: {time.time() - t0:.2f} 秒")
    
#     # 避免自匹配 (现在 query_names 和 db_names_final 的长度与特征张量的维度完全对应)
#     self_mask = np.array(query_names)[:, None] == np.array(db_names_final)[None]
    
#     t0 = time.time()
#     pairs = pairs_from_score_matrix(sim, self_mask, num_matched, min_score=0)
#     logger.info(f"生成图像对耗时: {time.time() - t0:.2f} 秒")
    
#     pairs = [(query_names[i], db_names_final[j]) for i, j in pairs]

#     logger.info(f"Found {len(pairs)} pairs.")
#     with open(output, "w") as f:
#         f.write("\n".join(" ".join([i, j]) for i, j in pairs))
    
#     logger.info(f"总耗时: {time.time() - start_time:.2f} 秒")














# # def main(
# #     descriptors,
# #     output,
# #     num_matched,
# #     query_prefix=None,
# #     query_list=None,
# #     db_prefix=None,
# #     db_list=None,
# #     db_model=None,
# #     db_descriptors=None,
# # ):
# #     logger.info("Extracting image pairs from a retrieval database.")
# #     start_time = time.time()

# #     # 1. 确定所有可用的检索特征 (来自HDF5文件)
# #     if db_descriptors is None:
# #         db_descriptors = descriptors
# #     if isinstance(db_descriptors, (Path, str)):
# #         db_descriptors = [db_descriptors]
# #     name2db = {n: i for i, p in enumerate(db_descriptors) for n in list_h5_names(p)}
# #     db_names_h5 = set(name2db.keys())
# #     query_names_h5 = list_h5_names(descriptors)

# #     if not db_names_h5:
# #         raise ValueError("Could not find any database image in the feature files.")

# #     # 2. 确定所有在三维模型中的图像 (如果提供了模型)
# #     if db_model:
# #         logger.info(f"正在从 {db_model / 'images.bin'} 读取三维模型中的图像名称...")
# #         images = read_images_binary(db_model / "images.bin")
# #         db_names_from_model = {i.name for i in images.values()}
# #         logger.info(f"三维模型中包含 {len(db_names_from_model)} 张图像。")
# #     else:
# #         db_names_from_model = db_names_h5

# #     # 3. 计算最终的数据库图像集：必须是两者的交集
# #     db_names_final = sorted(list(db_names_from_model.intersection(db_names_h5)))
    
# #     if len(db_names_final) == 0:
# #         logger.error("错误：三维模型中的图像与特征文件中的图像没有交集！")
# #         logger.error("请确保用于构建三维模型的图像集与用于提取全局特征的图像集至少有部分重叠。")
# #         raise ValueError("No common images between the 3D model and the feature files.")
    
# #     logger.info(f"将在 {len(db_names_final)} 张图像上进行检索 (三维模型与特征文件的交集)。")

# #     # 4. 解析查询图像名称
# #     query_names = parse_names(query_prefix, query_list, query_names_h5)

# #     # 5. 加载特征 (现在只加载交集内的特征)
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     db_desc = get_descriptors(db_names_final, db_descriptors, name2db)
# #     query_desc = get_descriptors(query_names, descriptors)

# #     # 6. 计算相似度并生成配对
# #     sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))
    
# #     # 避免自匹配
# #     # 注意：这里的比较现在是在 query_names 和 db_names_final 之间进行的
# #     self = np.array(query_names)[:, None] == np.array(db_names_final)[None]
    
# #     pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
# #     pairs = [(query_names[i], db_names_final[j]) for i, j in pairs]

# #     logger.info(f"Found {len(pairs)} pairs.")
# #     with open(output, "w") as f:
# #         f.write("\n".join(" ".join([i, j]) for i, j in pairs))
    
# #     logger.info(f"总耗时: {time.time() - start_time:.2f} 秒")




# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--descriptors", type=Path, required=True)
# #     parser.add_argument("--output", type=Path, required=True)
# #     parser.add_argument("--num_matched", type=int, required=True)
# #     parser.add_argument("--query_prefix", type=str, nargs="+")
# #     parser.add_argument("--query_list", type=Path)
# #     parser.add_argument("--db_prefix", type=str, nargs="+")
# #     parser.add_argument("--db_list", type=Path)
# #     parser.add_argument("--db_model", type=Path)
# #     parser.add_argument("--db_descriptors", type=Path)
# #     args = parser.parse_args()
# #     main(**args.__dict__)






import argparse
import collections.abc as collections
from pathlib import Path
from typing import List, Optional
import time
import h5py
import numpy as np
import torch

from . import logger
from .utils.io import list_h5_names
from .utils.parsers import parse_image_lists
from .utils.read_write_model import read_images_binary


def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
        if len(names) == 0:
            raise ValueError(f"Could not find any image with the prefix `{prefix}`.")
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(f"Unknown type of image list: {names}.")
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key="global_descriptor"):
    if name2idx is None:
        with h5py.File(str(path), "r", libver="latest") as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), "r", libver="latest") as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(scores, invalid, num_select, min_score=None):
    assert scores.shape == invalid.shape
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float("-inf"))

    k = min(num_select, scores.shape[1])
    topk = torch.topk(scores, k, dim=1)
    
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    return pairs


def main(
    descriptors,
    output,
    num_matched,
    query_prefix=None,
    query_list=None,
    db_prefix=None,
    db_list=None,
    db_model=None,
    db_descriptors=None,
    db_names_precomputed=None,
    precomputed_metadata=None,
    db_descriptors_preloaded=None,
    exclude_cameras=None,
    filter_self_match=True,
):
    # ✅ 调试日志
    logger.info("="*70)
    logger.info(f"🔧 exclude_cameras = {exclude_cameras}")
    logger.info(f"🔧 filter_self_match = {filter_self_match}")
    logger.info("="*70)
    
    logger.info("Extracting image pairs from a retrieval database.")
    start_time = time.time()

    # 步骤 1: 解析元数据
    t0 = time.time()
    if precomputed_metadata is not None:
        logger.info("使用预计算的元数据 (索引和名称列表).")
        name2db = precomputed_metadata['name2db']
        db_names_h5 = set(precomputed_metadata['db_names_h5'])
        query_names_h5 = precomputed_metadata['query_names_h5']
    else:
        logger.info("正在从 HDF5 文件实时解析元数据...")
        if db_descriptors is None:
            db_descriptors = descriptors
        if isinstance(db_descriptors, (Path, str)):
            db_descriptors = [db_descriptors]
        name2db = {n: i for i, p in enumerate(db_descriptors) for n in list_h5_names(p)}
        db_names_h5 = set(name2db.keys())
        query_names_h5 = list_h5_names(descriptors)
    logger.info(f"预备0 (解析元数据): {time.time() - t0:.2f} 秒")

    # 步骤 2: 解析3D模型中的图像名
    t0 = time.time()
    if db_names_precomputed is not None:
        logger.info(f"使用预先计算的数据库图像名称列表 (共 {len(db_names_precomputed)} 个).")
        db_names_from_model = set(db_names_precomputed)
    elif db_model:
        logger.info(f"正在从 {db_model / 'images.bin'} 读取数据库图像名称...")
        images = read_images_binary(db_model / "images.bin")
        db_names_from_model = {i.name for i in images.values()}
    else:
        db_names_from_model = db_names_h5
    logger.info(f"预备1 (解析3D模型名称): {time.time() - t0:.2f} 秒")

    # 步骤 3: 计算最终的数据库图像集 (交集 + 相机过滤)
    db_names_intersection = db_names_h5.intersection(db_names_from_model)
    
    # ✅ 查看样例图像名称
    logger.info("🔍 数据库图像名称样例:")
    for i, name in enumerate(sorted(list(db_names_intersection))[:5]):
        logger.info(f"   {i+1}. {name}")
    
    # ✅ 相机过滤
    if exclude_cameras:
        logger.info(f"🔒 开始排除相机: {exclude_cameras}")
        db_names_filtered = set()
        excluded_count = 0
        excluded_per_cam = {cam: 0 for cam in exclude_cameras}
        
        for name in db_names_intersection:
            is_excluded = False
            for excluded_cam in exclude_cameras:
                # 支持多种路径格式
                if (name.startswith(f"{excluded_cam}/") or 
                    f"/{excluded_cam}/" in name):
                    
                    is_excluded = True
                    excluded_count += 1
                    excluded_per_cam[excluded_cam] += 1
                    
                    if excluded_count <= 3:
                        logger.info(f"   排除样例: {name} (相机: {excluded_cam})")
                    
                    break
            
            if not is_excluded:
                db_names_filtered.add(name)
        
        logger.info(f"✅ 排除了 {excluded_count} 张图像 (总共 {len(db_names_intersection)} 张)")
        for cam, count in excluded_per_cam.items():
            logger.info(f"   - {cam}: {count} 张")
        
        db_names_final = sorted(list(db_names_filtered))
    else:
        logger.info("⚠️ exclude_cameras=None, 未启用相机过滤")
        db_names_final = sorted(list(db_names_intersection))
    
    if not db_names_final:
        raise ValueError("数据库为空!")
    
    logger.info(f"将在 {len(db_names_final)} 张图像上进行检索 (原始: {len(db_names_from_model)}).")

    # 步骤 4: 加载特征
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    query_names = parse_names(query_prefix, query_list, query_names_h5)
    query_desc = get_descriptors(query_names, descriptors)

    if db_descriptors_preloaded is not None and precomputed_metadata:
        logger.info("从预加载的张量中筛选数据库特征子集...")
        name_to_idx_h5 = {name: i for i, name in enumerate(precomputed_metadata['db_names_h5'])}
        indices = [name_to_idx_h5[n] for n in db_names_final if n in name_to_idx_h5]
        db_desc = db_descriptors_preloaded[indices]
    else:
        db_desc = get_descriptors(db_names_final, db_descriptors, name2db)

    # 步骤 5: 计算相似度并生成配对
    sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))
    
    if filter_self_match:
        self_mask = np.zeros((len(query_names), len(db_names_final)), dtype=bool)
        db_base_names = {Path(name).name: idx for idx, name in enumerate(db_names_final)}
        
        for q_idx, q_name in enumerate(query_names):
            q_base_name = Path(q_name).name
            if q_base_name in db_base_names:
                self_mask[q_idx, db_base_names[q_base_name]] = True
        
        logger.info(f"自匹配过滤: 屏蔽了 {self_mask.sum()} 个自匹配")
    else:
        self_mask = np.array(query_names)[:, None] == np.array(db_names_final)[None]
    
    pairs = pairs_from_score_matrix(sim, self_mask, num_matched, min_score=0)
    pairs = [(query_names[i], db_names_final[j]) for i, j in pairs]

    logger.info(f"Found {len(pairs)} pairs.")
    
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))
    
    logger.info(f"总耗时: {time.time() - start_time:.2f} 秒")