
# import os
# import pandas as pd
# from pathlib import Path
# import shutil

# def filter_images_by_camera(csv_path, image_folder, output_folder, camera_id="cam2"):
#     """
#     根据 CSV 文件中的 camera_id 筛选图像，并将对应的图像复制到输出文件夹。
    
#     Args:
#         csv_path (str or Path): CSV 文件路径。
#         image_folder (str or Path): 图像所在的文件夹路径。
#         output_folder (str or Path): 筛选后的图像保存路径。
#         camera_id (str): 要筛选的 camera_id，例如 "cam2"。
#     """
#     # 加载 CSV 文件
#     df = pd.read_csv(csv_path, delimiter=",")  # 假设 CSV 文件使用制表符分隔
#     filtered_df = df[df["camera_id"] == camera_id]  # 筛选指定 camera_id 的行

#     # 获取图像名列表
#     image_names = filtered_df["image_id"].tolist()

#     # 确保输出文件夹存在
#     output_folder = Path(output_folder)
#     output_folder.mkdir(parents=True, exist_ok=True)

#     # 遍历图像文件夹，筛选并复制图像
#     image_folder = Path(image_folder)
#     for image_name in image_names:
#         image_path = image_folder / image_name
#         if image_path.exists():
#             shutil.copy(image_path, output_folder / image_name)
#             print(f"Copied: {image_path} -> {output_folder / image_name}")
#         else:
#             print(f"Warning: Image not found -> {image_path}")

#     print(f"Finished filtering images for camera_id '{camera_id}'. Saved to {output_folder}.")

# # 示例用法
# csv_path = "datasets/SensorKit/0620_gt/enu_mapping_gt.csv"  # 替换为您的 CSV 文件路径
# image_folder = "datasets/SensorKit/query/cam2_all/undistorted/"  # 替换为包含所有图像的文件夹路径
# output_folder = "datasets/SensorKit/query/cam2_mapping/undistorted/"  # 替换为保存筛选后图像的文件夹路径

# filter_images_by_camera(csv_path, image_folder, output_folder, camera_id="cam2")





import argparse
import glob
import logging
from pathlib import Path
import yaml
import cv2  # 用于读取 OpenCV 格式的 YAML 文件
import os
import time
import shutil
import torch  # <-- 1. 添加 torch 导入

from ... import (
    extract_features,
    localize_sfm,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
    reconstruction,
    pycolmap,
    extractors,
    matchers   # <-- 2. 添加 extractors 导入
)
from . import colmap_from_nvm
from ...utils.io import list_h5_names # <-- 1. 增加 import
from ...pairs_from_retrieval import get_descriptors # <-- 1. 增加 import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# def generate_query_list(dataset, image_dir, path):
#     # 指定 YAML 文件路径
#     intrinsics_filename = "config/intrinsics/camera-intrinsics_cam2.yaml"
#     fs = cv2.FileStorage(str(dataset / intrinsics_filename), cv2.FILE_STORAGE_READ)
#     if not fs.isOpened():
#         raise FileNotFoundError(f"Could not open {intrinsics_filename}")

#     # 读取 cameraMatrix
#     camera_matrix = fs.getNode("cameraMatrix").mat()
#     fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
#     cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

#     # 读取 distCoeffs（只取前 4 项）
#     dist_coeffs = fs.getNode("distCoeffs").mat()
#     k1, k2, p1, p2 = dist_coeffs[0, 0], dist_coeffs[0, 1], dist_coeffs[0, 2], dist_coeffs[0, 3]

#     # 图像宽高
#     w = int(fs.getNode("imageWidth").real())
#     h = int(fs.getNode("imageHeight").real())
#     fs.release()

#     # 相机参数格式: OPENCV width height fx fy cx cy k1 k2 p1 p2
#     camera_params = ["OPENCV", w, h, fx, fy, cx, cy, k1, k2, p1, p2]
#     camera_params = [str(p) for p in camera_params]

#     # 获取查询图像名（仅文件名，不含路径）
#     queries = glob.glob((image_dir / "*.jpg").as_posix(), recursive=True)
#     queries = [Path(q).name for q in sorted(queries)]

#     # 每行输出：图像名 + 相机内参
#     out = [[q] + camera_params for q in queries]

#     # 写入输出路径
#     with open(path, "w") as f:
#         f.write("\n".join(map(" ".join, out)))


# def generate_query_list(dataset, image_dir, path):
#     # 移除从 YAML 文件读取的逻辑，使用您提供的硬编码参数
#     # 格式: MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, k1, k2, p1, p2
#     camera_params = [
#         "OPENCV",
#         2200,
#         1740,
#         1209.8508228781561,
#         1194.7606852205797,
#         1102.1907455875778,
#         834.41813183496947,
#         0.017942628756264879,
#         -0.024676148631733136,
#         -0.0010799660950984074,
#         0.00010967916505589733
#     ]
#     camera_params = [str(p) for p in camera_params]

#     # 获取查询图像名（仅文件名，不含路径）
#     queries = glob.glob((image_dir / "*.jpg").as_posix(), recursive=True)
#     queries = [Path(q).name for q in sorted(queries)]

#     # 每行输出：图像名 + 相机内参
#     out = [[q] + camera_params for q in queries]

#     # 写入输出路径
#     with open(path, "w") as f:
#         f.write("\n".join(map(" ".join, out)))

#     logger.info(f"Generated query list with hardcoded intrinsics at {path}")





# # --- 1. 将相机参数定义为全局常量 ---
# # 格式: MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, k1, k2, p1, p2
# QUERY_CAMERA_PARAMS_LIST = [
#     "PINHOLE", 1920, 1200,
#     1086.160899, 1090.242963, 940.067502, 586.740077,
#     -0.109203, 0.063536, -0.003427, -0.000629
# ]
# # # --- 修改结束 ---


# def generate_query_list(dataset, image_dir, path):
#     # --- 使用全局常量 ---
#     camera_params = [str(p) for p in QUERY_CAMERA_PARAMS_LIST]
#     # --- 修改结束 ---

#     # 获取查询图像名（仅文件名，不含路径）
#     queries = glob.glob((image_dir / "*.jpg").as_posix(), recursive=True)
#     queries = [Path(q).name for q in sorted(queries)]

#     # 每行输出：图像名 + 相机内参
#     out = [[q] + camera_params for q in queries]

#     # 写入输出路径
#     with open(path, "w") as f:
#         f.write("\n".join(map(" ".join, out)))

#     logger.info(f"Generated query list with NEW hardcoded intrinsics at {path}")


# def generate_query_list(dataset, image_dir, path):
#     # --- 使用您新提供的相机内参进行硬编码 ---
#     # 格式: MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, k1, k2, p1, p2
#     camera_params = [
#         "PINHOLE",
#         1920,  # image_width
#         1200,  # image_height
#         1086.160899,  # fx
#         1090.242963,  # fy
#         940.067502,   # cx
#         586.740077,   # cy
#         -0.109203,    # k1
#         0.063536,     # k2
#         -0.003427,    # p1
#         -0.000629     # p2
#     ]
#     camera_params = [str(p) for p in camera_params]
#     # --- 修改结束 ---

#     # 获取查询图像名（仅文件名，不含路径）
#     queries = glob.glob((image_dir / "*.jpg").as_posix(), recursive=True)
#     queries = [Path(q).name for q in sorted(queries)]

#     # 每行输出：图像名 + 相机内参
#     out = [[q] + camera_params for q in queries]

#     # 写入输出路径
#     with open(path, "w") as f:
#         f.write("\n".join(map(" ".join, out)))

#     logger.info(f"Generated query list with NEW hardcoded intrinsics at {path}")



# # Fix——Opencv
# def generate_query_list(dataset, image_dir, path): 
#     # 移除从 YAML 文件读取的逻辑，使用您提供的硬编码参数
#     # 格式: MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, k1, k2, p1, p2
#     camera_params = [
#         "OPENCV",
#         2200,
#         1740,
#         1210.4155551559695,
#         1200.3912375066598, ########## fix
#         1102.1907455875778, 
#         834.41813183496947,
#         0.017882415329839009,
#         -0.024744979638090473,
#         -0.00097154769540342529,
#         0.00020895912883848669
#     ]
#     camera_params = [str(p) for p in camera_params]

#     # 获取查询图像名（仅文件名，不含路径）
#     queries = glob.glob((image_dir / "*.jpg").as_posix(), recursive=True)
#     queries = [Path(q).name for q in sorted(queries)]

#     # 每行输出：图像名 + 相机内参
#     out = [[q] + camera_params for q in queries]

#     # 写入输出路径
#     with open(path, "w") as f:
#         f.write("\n".join(map(" ".join, out)))

#     logger.info(f"Generated query list with hardcoded intrinsics at {path}")


def generate_query_list(dataset, image_dir, path):
    """
    为查询图像生成列表，并使用硬编码的相机内参。
    """
    # --- 关键修改：根据新的 YAML 文件更新相机内参 ---
    # 格式: MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, k1, k2, p1, p2
    camera_params = [
        "OPENCV",
        640,   # image_width
        480,   # image_height
        488.9530499488205,    # fx
        491.25772001574586,    # fy
        324.30120395988746,    # cx
        251.72243328800445,    # cy
        0.14001741143085564,   # k1
        -0.315039515960146,    # k2
        0.009553769248475456,  # p1
        0.0017408824411489351  # p2
    ]
    # --- 修改结束 ---
    
    camera_params = [str(p) for p in camera_params]

    # 获取查询图像名（仅文件名，不含路径）
    queries = glob.glob((image_dir / "*.jpg").as_posix(), recursive=True) + \
              glob.glob((image_dir / "*.png").as_posix(), recursive=True)
    queries = [Path(q).name for q in sorted(queries)]

    # 每行输出：图像名 + 相机内参
    out = [[q] + camera_params for q in queries]

    # 写入输出路径
    with open(path, "w") as f:
        f.write("\n".join(map(" ".join, out)))

    logger.info(f"Generated query list with new hardcoded intrinsics at {path}")


def run(args):
    print("Starting SensorKit pipeline...")
    logger.info("Starting pipeline...")
    
    # 设置路径
    dataset = args.dataset
    # dataset_test = args.dataset_test
    # mapping_dir = dataset / "IMAGES"
    # mapping_dir = dataset/ "mapping"  
    mapping_dir = dataset/ "allcams"  
    # query_dir = dataset / "query/cam1_all/undistorted/" 
    # query_dir = dataset / "query/412bag_1s/" 
    # query_dir = dataset / "query/test_t/undistorted/" 
    # query_dir = dataset / "query/clip_one/"  
    query_dir = dataset / "query/25-9-5-phone/"  
    # query_dir = dataset / "query/cam2_mapping/undistorted/"
    # query_dir = dataset / "query/cam2_all/undistorted/" 
    # query_dir = dataset_test / "mapping/jpg/" 
    # sift_sfm = dataset / "colmap_seque_prior/sparse_aligned/1/only_scale"
    # sift_sfm = dataset / "new_colmap_spatial_intrinsics/sparse_aligned_ecef_4"
    # sift_sfm = dataset / "new_colmap_exhaustive_fixing_intrinsics/sparse/0"
    # sift_sfm = dataset / "vocab_tree_0630_OPENCV/sparse/1/0.05_BA_ENU"
    sift_sfm = dataset / "allcams_spatial_colmap_0630/sparse/0/0.025_BA_ENU"
    # sift_sfm = dataset / "colmap_seque/sparse_aligned_enu"
    logger.info(f"Dataset path: {dataset}")
    logger.info(f"Mapping images path: {mapping_dir}")
    logger.info(f"Query images path: {query_dir}")

    outputs = args.outputs  # 输出目录
    outputs.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {outputs}")


    query_list = outputs / "queries_with_intrinsics.txt"
    # outputsba  = outputs / "cam1"  # SfM 输出目录
    outputsba  = outputs / "25-9-5-phone"  # SfM 输出目录
    # outputsba  = outputs / "412_one"  # SfM 输出目录
    # outputsba  = outputs / "412bag_1s"  # SfM 输出目录
    # reference_sfm = outputs / "sfm_superpoint+superglue/0.5_BA_CAUCHY_ENU_0626" 
    # reference_sfm = outputs / "sfm_superpointmax+lightglue"
    reference_sfm_path = outputs / "sfm_superpointmax+lightglue"
    # reference_sfm = outputs / "sfm_superpointmax+lightglue" 
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    loc_pairs = outputsba / f"pairs-query-netvlad{args.num_loc}.txt"
    results = outputsba / f"SensorKit_hloc_superpoint+lightglue_netvlad{args.num_loc}.txt"

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]  # "superpoint_max"慢 "superpoint_aachen"
    matcher_conf = match_features.confs["superpoint+lightglue"] # superpoint+lightglue superglue-fast快



    # --- 2. 在这里预加载所有在线阶段需要的模型 ---
    logger.info("Pre-loading models for the online stage...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载全局特征模型 (NetVLAD)
    GlobalModel = extract_features.dynamic_load(extractors, retrieval_conf["model"]["name"])
    global_model = GlobalModel(retrieval_conf["model"]).eval().to(device)

    # 新增：加载局部特征匹配模型
    MatcherModel = extract_features.dynamic_load(matchers, matcher_conf['model']['name'])
    local_matcher_model = MatcherModel(matcher_conf['model']).eval().to(device)
    
    
    logger.info("...Models pre-loaded and ready.")
    # --- 预加载结束 ---

    # --- 3. 在这里提前构建特征文件的元数据索引 ---
    logger.info("Pre-computing feature file metadata...")
    db_global_features_path = outputs / (retrieval_conf['output'] + '.h5')

    # 假设数据库特征只在一个文件中
    db_descriptors_list = [db_global_features_path]
    
    precomputed_db_metadata = {
        'name2db': {n: i for i, p in enumerate(db_descriptors_list) for n in list_h5_names(p)},
        'db_names_h5': list_h5_names(db_global_features_path),
    }
    logger.info("...Metadata pre-computed.")
    # --- 预计算结束 ---
    

    # --- 2. 在这里提前加载数据库的全局特征 ---
    logger.info("Pre-loading database global descriptors from HDF5...")
    t_load_db_desc = time.time()
    preloaded_db_desc = get_descriptors(
        precomputed_db_metadata['db_names_h5'], 
        db_descriptors_list, 
        precomputed_db_metadata['name2db']
    )
    logger.info(f"...Database descriptors loaded in {time.time() - t_load_db_desc:.2f}s.")
    # --- 预加载结束 ---

    # # --- 清理上一次的在线定位结果 ---
    # if outputsba.exists():
    #     logger.warning(f"Cleaning up previous online localization results in {outputsba}...")
    #     shutil.rmtree(outputsba)
    # outputsba.mkdir(exist_ok=True, parents=True)
    # # --- 清理结束 ---


    # 在这里提前加载 3D 模型 ---
    logger.info(f"Loading the 3D reference model from {reference_sfm_path}...") # <--- 使用新路径变量
    reference_sfm = pycolmap.Reconstruction(reference_sfm_path) # <--- 加载模型
    logger.info("...3D model loaded.")

    logger.info("Extracting database image names from the pre-loaded model...")
    db_names_from_model = [img.name for img in reference_sfm.images.values()]
    logger.info(f"...Found {len(db_names_from_model)} image names.")
    # --- 提取结束 ---


    # # ### OFFLINE

    # # # 生成基于共视性生成的图像对文件，用于SfM重建
    # # # 格式: 每行两个图像ID，表示它们共享视野
    # # # 可视化: 可使用networkx库将其可视化为图形网络
    
    # pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)

    # # 提取特征
    # features_db = extract_features.main(feature_conf, mapping_dir, outputs, as_half=True)

    # # 匹配特征
    # sfm_matches = match_features.main(
    #     matcher_conf, sfm_pairs, feature_conf["output"], outputs
    # )

    # # 三角化重建3d参考模型
    # triangulation.main(
    #     reference_sfm,  # 输出的参考稀疏模型
    #     sift_sfm,       # 现有的SIFT稀疏模型
    #     mapping_dir,    # 输入图像目录
    #     sfm_pairs,      # 图像对文件
    #     features_db,       # 提取的特征
    #     sfm_matches,    # 特征匹配结果
    # )

    # global_descriptors_db = extract_features.main(retrieval_conf, mapping_dir, outputs)


    ## ONLINE

    # --- 2. 在在线定位流程开始前，记录开始时间 ---
    logger.info("Starting ONLINE localization process...")
    start_time_online = time.time()
    timings = {}  # 用于存储每个步骤的耗时
    # ---
    t0 = time.time()
    # 生成查询图像列表
    generate_query_list(dataset, query_dir, query_list)
    timings['generate_query_list'] = time.time() - t0
 
    # # 步骤 1: 提取全局描述符
    t0 = time.time()
    # # 提取全局描述符提取特征 
    query_global_feats_path = outputsba / "query-global-feats.h5"
    global_descriptors_query = extract_features.main(
        retrieval_conf, query_dir, feature_path=query_global_feats_path, as_half=True, model_instance=global_model,
    )
    timings['global_feature_extraction'] = time.time() - t0

    # --- 在线阶段：为 pairs_from_retrieval 准备完整的元数据 ---
    # 将离线预计算的数据库元数据与在线生成的查询元数据合并
    full_metadata = {
        **precomputed_db_metadata, # 合并数据库元数据
        'query_names_h5': list_h5_names(query_global_feats_path) # 在线读取查询元数据
    }

    # --- 修改结束 ---
    # 步骤 2: 基于全局检索，生成定位图像对
    t0 = time.time()
    # 基于全局检索，生成定位图像对
    # db_global_features = outputs / (retrieval_conf['output'] + '.h5')
    pairs_from_retrieval.main(
        descriptors=global_descriptors_query,
        output=loc_pairs,
        num_matched=args.num_loc,
        query_list=query_list, 
        # db_descriptors=db_global_features, 
        db_descriptors=[db_global_features_path], 
        # db_model=reference_sfm_path,  # <--- 在这里传递路径，而不是加载的对象
        db_names_precomputed=db_names_from_model, # <-- 2. 传入预先提取的名称列表
        precomputed_metadata=full_metadata, # <-- 3. 传入预计算的元数据
        db_descriptors_preloaded=preloaded_db_desc, # <-- 4. 传入预加载的特征张量
    )
    timings['retrieval'] = time.time() - t0

    # 步骤 3: 提取局部特征
    t0 = time.time()
    query_local_feats_path = outputsba / "query-local-feats.h5"
    features = extract_features.main(
        feature_conf, query_dir, feature_path=query_local_feats_path, as_half=True
    )
    timings['local_feature_extraction'] = time.time() - t0
    # 步骤 4: 局部特征匹配
    t0 = time.time()
    db_local_features = outputs / (feature_conf['output'] + '.h5')
    loc_matches_path = outputsba / (matcher_conf['output'] + '.h5')
    loc_matches = match_features.main(
        conf=matcher_conf, 
        pairs=loc_pairs, 
        features=query_local_feats_path,
        features_ref=db_local_features,
        matches=loc_matches_path,  # <-- 为 matches 参数提供完整的 Path 对象
        model_instance=local_matcher_model, # <-- 4. 传入预加载的模型实例
    )
    timings['feature_matching'] = time.time() - t0

    # 步骤 5: 定位
    t0 = time.time()
    localize_sfm.main(
        reference_sfm,   # 使用生成的参考模型  reference_sfm, 
        query_list,
        loc_pairs,
        features, 
        loc_matches,
        results,
    )
    timings['pose_estimation'] = time.time() - t0




# # 步骤 6: (可选) 基于 PnP 内点生成并保存匹配可视化图像
#     t0 = time.time()
#     logger.info("Generating localization visualizations based on PnP inliers...")
#     try:
#         import pickle
#         import h5py
#         import numpy as np
#         from collections import Counter
#         from matplotlib import pyplot as plt
#         from matplotlib.patches import ConnectionPatch
#         from pprint import pprint # <--- 1. 导入 pprint
        
#         logs_path = results.parent / (results.name + "_logs.pkl")
#         if logs_path.exists():
#             with open(logs_path, 'rb') as f:
#                 logs = pickle.load(f)

#             if not logs.get('loc'):
#                 logger.warning("Logs file is empty, no localization data to visualize.")
#                 return

#             vis_dir = outputsba / "pnp_visualizations"
#             vis_dir.mkdir(exist_ok=True, parents=True)
#             logger.info(f"Saving visualizations to {vis_dir}")

#             # 创建从数据库图像ID到图像名的映射
#             db_id_to_name = {img.image_id: img.name for img in reference_sfm.images.values()}
            
#             # 使用 with 语句确保 h5 文件被正确关闭
#             with h5py.File(features, 'r') as feats_h5, \
#                  h5py.File(loc_matches, 'r') as matches_h5:

#                 db_local_features_path = outputs / (feature_conf['output'] + '.h5')
#                 with h5py.File(db_local_features_path, 'r') as db_feats_h5:

#                     pbar = logger
#                     for i, query_name in enumerate(logs['loc'].keys()):
#                         log_entry = logs['loc'][query_name]
#                         pnp_ret = log_entry.get('PnP_ret', {})

#                         # 检查定位是否成功以及内点掩码是否存在
#                         if not pnp_ret or 'inlier_mask' not in pnp_ret:
#                             pbar.warning(f"Skipping visualization for {query_name}: No PnP inliers found.")
#                             continue

#                         inlier_mask = pnp_ret['inlier_mask']
                        
#                         # 获取参与PnP的查询关键点索引和3D点ID
#                         pnp_kp_query_indices = np.array(log_entry['keypoint_index_to_db'][0])
#                         pnp_3d_ids = np.array(log_entry['points3D_ids'])

#                         # 筛选出作为内点的3D点ID
#                         inlier_3d_ids = pnp_3d_ids[inlier_mask]
#                         if len(inlier_3d_ids) == 0:
#                             pbar.warning(f"Skipping visualization for {query_name}: Zero inliers.")
#                             continue

#                         # 追溯3D内点，找到贡献最大的数据库图像
#                         db_ids_of_inliers = []


#                         # --- 1. 添加调试代码：打印参考模型中所有 3D 点的 ID ---
#                         # 注意：如果模型很大，这会打印很多信息。可以只打印前几个作为样本。
#                         all_ref_3d_ids = set(reference_sfm.points3D.keys())
#                         print(f"--- DEBUG: {query_name} ---")
#                         print(f"Total PnP inlier 3D IDs to check: {len(inlier_3d_ids)}")
#                         print(f"Sample of PnP inlier 3D IDs: {list(inlier_3d_ids)[:10]}")
#                         print(f"Total 3D points in reference_sfm model: {len(all_ref_3d_ids)}")
#                         print(f"Sample of 3D IDs in reference_sfm: {list(all_ref_3d_ids)[:10]}")
#                         # --- 调试代码结束 ---


#                         for p3d_id in inlier_3d_ids:
#                             # 修正：将 numpy.int64 强制转换为标准的 python int
#                             p3d_id_int = int(p3d_id)
#                             if p3d_id_int in reference_sfm.points3D:
#                                 point3d = reference_sfm.points3D[p3d_id_int]
#                                 db_ids_of_inliers.extend([obs.image_id for obs in point3d.track.elements])
                     

#                         if not db_ids_of_inliers:
#                             # --- 2. 添加调试代码：在失败时打印具体信息 ---
#                             print(f"!!! FAILURE TRACEBACK for {query_name} !!!")
#                             print("None of the PnP inlier IDs were found in the reference model.")
#                             # 检查是否有任何交集
#                             pnp_ids_set = set(inlier_3d_ids)
#                             intersection = pnp_ids_set.intersection(all_ref_3d_ids)
#                             print(f"Intersection size between PnP IDs and model IDs: {len(intersection)}")
#                             if len(intersection) > 0:
#                                 print(f"Found intersecting IDs, but something is still wrong. Sample: {list(intersection)[:5]}")
#                             # --- 调试代码结束 ---
#                             pbar.warning(f"Skipping visualization for {query_name}: Could not trace back inlier 3D points.")
#                             continue


#                         most_common_db_id = Counter(db_ids_of_inliers).most_common(1)[0][0]
#                         best_ref_name = db_id_to_name.get(most_common_db_id)
                        
#                         if not best_ref_name:
#                             pbar.warning(f"Skipping {query_name}: Could not map best DB ID {most_common_db_id} to a name.")
#                             continue

#                         # 加载查询图像和最佳参考图像
#                         query_image = cv2.imread(str(query_dir / query_name))
#                         ref_image = cv2.imread(str(mapping_dir / best_ref_name))
#                         if query_image is None or ref_image is None:
#                             pbar.warning(f"Skipping {query_name}: Could not read image file for query or ref '{best_ref_name}'.")
#                             continue

#                         # --- 修正：加载关键点以适应嵌套的 HDF5 结构 ---
#                         kp_query = feats_h5[query_name]['keypoints'][()]
                        
#                         # 解析 best_ref_name 以进行嵌套访问
#                         cam_id_ref, ref_name_short_ref = best_ref_name.split('/', 1)
#                         kp_ref = db_feats_h5[cam_id_ref][ref_name_short_ref]['keypoints'][()]
#                         # --- 修正结束 ---

#                         # --- 修正：加载原始匹配以适应嵌套的 HDF5 结构 ---
#                         matches_indices = None
#                         try:
#                             # best_ref_name 的格式是 "camX/image_name.jpg"
#                             # 我们需要从中解析出 camera_id 和短图像名
#                             cam_id, ref_name_short = best_ref_name.split('/', 1)

#                             # 检查嵌套路径是否存在
#                             if (query_name in matches_h5 and
#                                 cam_id in matches_h5[query_name] and
#                                 ref_name_short in matches_h5[query_name][cam_id] and
#                                 'matches0' in matches_h5[query_name][cam_id][ref_name_short]):
                                
#                                 # 按嵌套路径访问 'matches0'
#                                 matches_indices = matches_h5[query_name][cam_id][ref_name_short]['matches0'][()]
                            
#                         except (ValueError, KeyError) as e:
#                             # 捕获可能的 split 错误或 key 不存在的错误
#                             pbar.warning(f"Error accessing nested match for {query_name} vs {best_ref_name}: {e}")
                            
#                         # 如果最终没有找到任何匹配，则跳过
#                         if matches_indices is None:
#                             pbar.warning(f"Skipping {query_name}: No raw matches found for pair vs {best_ref_name} in loc_matches.h5.")
#                             continue
#                         # --- 修正结束 ---

#                         # --- 关键修改：采用 Demo 逻辑进行几何验证 ---
#                         # 确保 matches_indices 是二维的，以防只有一个匹配
#                         if matches_indices.ndim == 1:
#                             matches_indices = matches_indices.reshape(1, -1)

#                         # --- 修正：直接使用已知的全局相机模型 ---
#                         query_camera = pycolmap.Camera(
#                             model=QUERY_CAMERA_PARAMS_LIST[0],
#                             width=int(QUERY_CAMERA_PARAMS_LIST[1]),
#                             height=int(QUERY_CAMERA_PARAMS_LIST[2]),
#                             params=np.array(QUERY_CAMERA_PARAMS_LIST[3:], dtype=np.float64)
#                         )
                        
#                         # --- 备用方案：使用 OpenCV 手动实现几何验证 ---
#                         # 1. 从原始匹配中提取出对应的关键点坐标
#                         m_kp_query = kp_query[matches_indices[:, 0]]
#                         m_kp_ref = kp_ref[matches_indices[:, 1]]

#                         # 2. 获取相机内参矩阵 K
#                         K = query_camera.calibration_matrix()

#                         # 3. 使用 OpenCV 的 RANSAC 计算本质矩阵，并得到内点掩码
#                         #    threshold=1.0 是一个常用的像素误差阈值，可以调整
#                         E, inlier_mask_cv2 = cv2.findEssentialMat(
#                             m_kp_query, m_kp_ref, K,
#                             method=cv2.RANSAC,
#                             prob=0.999,
#                             threshold=1.0
#                         )
                        
#                         # 4. 将 OpenCV 的掩码 (1/0) 转换为 NumPy 的布尔掩码 (True/False)
#                         inlier_mask = inlier_mask_cv2.ravel().astype(bool)
#                         # --- 备用方案结束 ---
                        
#                         # 从返回结果中获取内点掩码和内点匹配
#                         inlier_matches = matches_indices[inlier_mask]

#                         if len(inlier_matches) == 0:
#                             pbar.warning(f"Skipping {query_name}: Geometric verification found 0 inliers for vs {best_ref_name}.")
#                             continue
#                         # --- 修改结束 ---


#                         # --- 绘图 ---
#                         fig, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=100)
#                         ax[0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
#                         ax[0].set_title(f"Query: {query_name}\n({pnp_ret.get('num_inliers', 0)} PnP inliers)", fontsize=12)
#                         ax[0].axis('off')
#                         ax[1].imshow(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
#                         ax[1].set_title(f"Best Reference: {best_ref_name}", fontsize=12)
#                         ax[1].axis('off')

#                         # 获取匹配的关键点坐标
#                         kp_query_matched = kp_query[inlier_matches[:, 0]]
#                         kp_ref_matched = kp_ref[inlier_matches[:, 1]]

#                         # 画出连接线
#                         for j in range(len(kp_query_matched)):
#                             line = ConnectionPatch(
#                                 xyA=kp_query_matched[j], coordsA=ax[0].transData,
#                                 xyB=kp_ref_matched[j], coordsB=ax[1].transData,
#                                 color=np.random.rand(3), linewidth=0.8, alpha=0.8
#                             )
#                             fig.add_artist(line)
                        
#                         vis_path = vis_dir / f"{query_name.replace('/', '_')}_vs_{best_ref_name.replace('/', '_')}.png"
#                         plt.savefig(vis_path, bbox_inches='tight', pad_inches=0)
#                         plt.close(fig)

#         else:
#             logger.warning(f"Could not find logs file for visualization: {logs_path}")

#     except Exception as e:
#         logger.error(f"An error occurred during PnP visualization: {e}", exc_info=True)
#     timings['pnp_visualization'] = time.time() - t0






    # --- 3. 在在线定位流程结束后，计算并打印耗时 ---
    end_time_online = time.time()
    total_online_time = end_time_online - start_time_online
    
    # 读取查询图像数量
    with open(query_list, 'r') as f:
        num_queries = len(f.readlines())

    if num_queries > 0:
        avg_time_per_frame = total_online_time / num_queries
        fps = 1.0 / avg_time_per_frame
        logger.info("--- Localization Performance Stats ---")
        
        # 打印每个步骤的耗时
        for step, duration in timings.items():
            avg_step_time_ms = (duration / num_queries) * 1000
            logger.info(f"  - Avg {step}: {avg_step_time_ms:.2f} ms / frame")

        logger.info(f"Total online time for {num_queries} frames: {total_online_time:.2f} seconds.")
        logger.info(f"Average total time per frame: {avg_time_per_frame*1000:.2f} ms.")
        logger.info(f"Estimated FPS: {fps:.2f} Hz.")
        logger.info("------------------------------------")
    # ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        help="Specify a single step to run (e.g., 'localization'). Default: run all steps.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/SensorKit",
        help="Path to the dataset, default: %(default)s",
    )

    # parser.add_argument(
    #     "--dataset_test",
    #     type=Path,
    #     default="datasets/SensorKit_test",
    #     help="Path to the dataset, default: %(default)s",
    # )

    parser.add_argument(
        "--outputs",
        type=Path,
        # default="outputs/SensorKit_hloc/test",
        # default="outputs/SensorKit_hloc/scale",
        # default="outputs/SensorKit_hloc/spatial_colmap_0630_BA",
        default="outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=50, # Number of image pairs for SfM, default: %(default)s", 50
        help="Number of image pairs for SfM, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=20, # Number of image pairs for loc, default: %(default)s", 20
        help="Number of image pairs for loc, default: %(default)s",
    )
    args = parser.parse_args()
    run(args)





# import pickle
# from pathlib import Path

# def analyze_pnp_results(log_path: Path):
#     """
#     加载 HLOC 日志文件，并打印出每个查询图像最底层的 PnP 求解结果。
#     """
#     if not log_path.exists():
#         print(f"错误: 日志文件未找到 -> {log_path}")
#         return

#     print(f"--- 正在分析日志文件: {log_path.name} ---")
#     with open(log_path, 'rb') as f:
#         logs = pickle.load(f)

#     loc_data = logs.get('loc', {})
#     if not loc_data:
#         print("日志中未找到定位 ('loc') 数据。")
#         return

#     for qname, log_entry in loc_data.items():
#         print(f"\n--- 查询图像: {qname} ---")
        
#         # 检查是否使用了共视聚类
#         if log_entry.get('covisibility_clustering', False):
#             print("  模式: 使用了共视聚类")
#             log_clusters = log_entry.get('log_clusters', [])
#             best_cluster_idx = log_entry.get('best_cluster')
            
#             if not log_clusters:
#                 print("  警告: 未找到聚类日志。")
#                 continue

#             for i, cluster_log in enumerate(log_clusters):
#                 pnp_result = cluster_log.get('PnP_ret')
#                 is_best = " (最佳选择)" if i == best_cluster_idx else ""
#                 print(f"  [聚类 {i}{is_best}] PnP 结果: {pnp_result}")
        
#         else:
#             print("  模式: 未使用共视聚类")
#             pnp_result = log_entry.get('PnP_ret')
#             print(f"  PnP 结果: {pnp_result}")


# log_file = Path("outputs/SensorKit_hloc/test_t_aachen+lightglue/412_one/SensorKit_hloc_superpoint+lightglue_netvlad20.txt_logs.pkl")
# analyze_pnp_results(log_file)


























import numpy as np
from scipy.spatial.transform import Rotation as R
import re

def convert_quat_pose_to_outputs(input_txt, output_txt_with_name, output_tum_txt):
    with open(input_txt, 'r') as fin, \
         open(output_txt_with_name, 'w') as fout1, \
         open(output_tum_txt, 'w') as fout2:

        for line in fin:
            parts = line.strip().split()
            if len(parts) != 8:
                continue  # 跳过无效行
            name, qw, qx, qy, qz, tx, ty, tz = parts
            qw, qx, qy, qz = map(float, [qw, qx, qy, qz])
            tx, ty, tz = map(float, [tx, ty, tz])

            # 从相机到世界的旋转
            rot_c2w = R.from_quat([qx, qy, qz, qw]).inv()
            quat_c2w = rot_c2w.as_quat()  # [qx, qy, qz, qw]
            R_c2w = rot_c2w.as_matrix()

            t = np.array([tx, ty, tz])
            cam_center = -R_c2w @ t  # 世界坐标下的相机中心

            # 输出文件1：name + 四元数 + 相机位置
            fout1.write(f"{name} {quat_c2w[0]:.6f} {quat_c2w[1]:.6f} {quat_c2w[2]:.6f} {quat_c2w[3]:.6f} "
                        f"{cam_center[0]:.6f} {cam_center[1]:.6f} {cam_center[2]:.6f}\n")

            # 尝试从 name 中提取 timestamp（假设是数字）
            match = re.search(r'(\d+)', name)
            if match:
                timestamp = match.group(1)
            else:
                timestamp = "0"

            # 输出文件2：TUM 格式 timestamp tx ty tz qx qy qz qw
            fout2.write(f"{timestamp} {cam_center[0]:.6f} {cam_center[1]:.6f} {cam_center[2]:.6f} "
                        f"{quat_c2w[0]:.6f} {quat_c2w[1]:.6f} {quat_c2w[2]:.6f} {quat_c2w[3]:.6f}\n")

    print(f"Saved:\n  → {output_txt_with_name}\n  → {output_tum_txt}")

convert_quat_pose_to_outputs(
    "outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/25-9-5-phone/SensorKit_hloc_superpoint+lightglue_netvlad20.txt",
    "outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/25-9-5-phone/SensorKit_query_centers.txt",
    "outputs/evo_tum/enu_mapping_query/cam0_phone_SensorKit_query_centers_tum.txt"
)


# ########### ns TUM s
# import numpy as np

# def convert_timestamp_ns_to_s(input_path, output_path):
#     """
#     Reads a TUM-like file, converts the first column (timestamp) from nanoseconds to seconds,
#     and saves the modified data to a new file.
#     """
#     output_lines = []
#     with open(input_path, 'r') as f_in:
#         for line in f_in:
#             if line.startswith("#"): # Preserve comment lines
#                 output_lines.append(line)
#                 continue
            
#             parts = line.strip().split()
#             if len(parts) == 8: # Expecting timestamp tx ty tz qx qy qz qw
#                 try:
#                     timestamp_ns = float(parts[0])
#                     timestamp_s = timestamp_ns / 1e9  # Convert nanoseconds to seconds
                    
#                     # Reconstruct the line with the new timestamp
#                     # and format all numbers to .6f
#                     new_line_parts = [f"{timestamp_s:.6f}"] + [f"{float(p):.6f}" for p in parts[1:]]
#                     output_lines.append(" ".join(new_line_parts) + "\n")
#                 except ValueError:
#                     print(f"Warning: Could not parse numbers in line, skipping: {line.strip()}")
#                     output_lines.append(line) # Append original line if parsing fails
#             else:
#                 # If the line doesn't have 8 parts, append it as is (e.g., empty lines or malformed)
#                 output_lines.append(line)

#     with open(output_path, "w") as f_out:
#         for out_line in output_lines:
#             f_out.write(out_line)
    
#     print(f"Converted timestamps and saved to {output_path}")


# input_file_path = "outputs/evo_tum/enu_mapping_query/cam1_spatial_colmap_0630_BA_SensorKit_query_centers_tum.txt"
# output_file_path = "outputs/evo_tum/enu_mapping_query/s_cam1_spatial_colmap_0630_BA_SensorKit_query_centers_tum.txt" # New output file name


# import os
# if os.path.exists(input_file_path):
#     convert_timestamp_ns_to_s(input_file_path, output_file_path)
# else:
#     print(f"Input file not found: {input_file_path}")
    


# import numpy as np
# from bisect import bisect_left

# def load_tum_file(file_path):
#     """
#     Load TUM trajectory into a sorted list: [[timestamp, tx, ty, tz, qx, qy, qz, qw]]
#     """
#     data = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             if line.strip() and not line.startswith("#"):
#                 parts = line.strip().split()
#                 if len(parts) == 8:
#                     timestamp = float(parts[0])
#                     values = list(map(float, parts[1:]))
#                     data.append([timestamp] + values)
#     data.sort()  # ensure sorted by timestamp
#     return data

# def find_nearest_index(timestamps, target_time):
#     """
#     Find the index of the timestamp in 'timestamps' that is closest to 'target_time'
#     """
#     idx = bisect_left(timestamps, target_time)
#     if idx == 0:
#         return 0
#     if idx == len(timestamps):
#         return len(timestamps) - 1
#     before = timestamps[idx - 1]
#     after = timestamps[idx]
#     return idx - 1 if abs(before - target_time) < abs(after - target_time) else idx

# def match_and_filter_nearest(est_data, gt_data, threshold=10.0):
#     """
#     Match est to gt using nearest timestamp and filter based on 3D distance
#     """
#     gt_timestamps = [row[0] for row in gt_data]
#     gt_dict = {row[0]: row[1:4] for row in gt_data}

#     filtered = []
#     for est_row in est_data:
#         est_time = est_row[0]
#         est_pos = np.array(est_row[1:4])

#         nearest_idx = find_nearest_index(gt_timestamps, est_time)
#         gt_time = gt_timestamps[nearest_idx]
#         gt_pos = np.array(gt_dict[gt_time])

#         dist = np.linalg.norm(est_pos - gt_pos)
#         if dist <= threshold:
#             filtered.append(est_row)
#         else:
#             print(f"Frame at {est_time:.6f} rejected (distance = {dist:.3f}m to nearest gt @ {gt_time:.6f})")

#     return np.array(filtered)

# def save_tum_trajectory(file_path, data):
#     with open(file_path, 'w') as f:
#         for row in data:
#             f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*row))

# if __name__ == "__main__":
#     est_file = "outputs/evo_tum/enu_mapping_query/s_cam1_spatial_colmap_0630_BA_SensorKit_query_centers_tum.txt"
#     gt_file = "outputs/evo_tum/enu_mapping_query/enu_all_query_gt.tum"
#     output_file = "outputs/evo_tum/enu_mapping_query/s_cam1_spatial_colmap_0630_BA_SensorKit_query_centers_tum_filtered.txt"
#     threshold = 10.0  # meters # 10.0

#     print("Loading estimated trajectory...")
#     est_data = load_tum_file(est_file)

#     print("Loading ground truth trajectory...")
#     gt_data = load_tum_file(gt_file)

#     print("Filtering using nearest time and distance threshold...")
#     filtered_data = match_and_filter_nearest(est_data, gt_data, threshold)

#     print(f"Saving filtered trajectory to {output_file}...")
#     save_tum_trajectory(output_file, filtered_data)

#     print(f"Done. Original: {len(est_data)} frames, Filtered: {len(filtered_data)} frames")



