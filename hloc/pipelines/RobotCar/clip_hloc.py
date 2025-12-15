import pandas as pd 
from groundingdino.util.inference import load_model as load_grounding_dino_model, predict as predict_grounding_dino
import groundingdino.datasets.transforms as T
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq # <-- 核心修改：导入新类
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer 
from transformers import CLIPModel as HfCLIPModel, CLIPProcessor, CLIPTokenizer
import argparse
import glob
import logging
from pathlib import Path
import yaml
import cv2  # 用于读取 OpenCV 格式的 YAML 文件
import os
import time
import h5py
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel # <-- 1. 确保导入
import open_clip
import numpy as np
from bisect import bisect_left
# with h5py.File("outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/clip_features.h5", "r") as f:
#     for key in f.keys():
#         print(f"Key: {key} ({type(key)})")
import csv 
import itertools 

from ... import (
    extract_features,
    localize_sfm,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,  # // 不再使用
    triangulation,
    reconstruction,
    extract_clip_features,      # <-- 1. 导入新的 CLIP 特征提取器
    pairs_from_clip_retrieval, 
    generate_captions, # <-- 2. 导入新的 CLIP 检索器
    matchers,
    pycolmap,
)
from . import colmap_from_nvm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import yaml  # 用于解析 YAML 文件
import glob
from pathlib import Path


# --- 新增：将辅助函数移至此处 ---
def load_tum_file(file_path):
    """
    Load TUM trajectory into a sorted list: [[timestamp, tx, ty, tz, qx, qy, qz, qw]]
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split()
                if len(parts) == 8:
                    timestamp = float(parts[0])
                    values = list(map(float, parts[1:]))
                    data.append([timestamp] + values)
    data.sort()  # ensure sorted by timestamp
    return data

# --- 核心新增：从查询CSV加载真值数据的函数 ---
def load_query_gt_from_csv(csv_path):
    """
    从包含查询图像真值的CSV文件加载数据，并转换为TUM格式的列表。
    CSV格式: camera_id,timestamp_ns,...,X_enu,Y_enu,Z_enu,quat_x,quat_y,quat_z,quat_w
    返回: [[timestamp_s, tx, ty, tz, qx, qy, qz, qw], ...]
    """
    gt_data = []
    logger.info(f"Loading query ground truth from CSV: {csv_path}")
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 将纳秒时间戳转换为秒
                timestamp_s = float(row['timestamp_ns']) / 1e9
                tx, ty, tz = float(row['X_enu']), float(row['Y_enu']), float(row['Z_enu'])
                qx, qy, qz, qw = float(row['quat_x']), float(row['quat_y']), float(row['quat_z']), float(row['quat_w'])
                # 注意TUM格式的四元数顺序是 qx, qy, qz, qw
                gt_data.append([timestamp_s, tx, ty, tz, qx, qy, qz, qw])
        gt_data.sort() # 按时间戳排序
        logger.info(f"Loaded {len(gt_data)} query ground truth poses.")
        return gt_data
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Failed to load or parse query ground truth CSV: {e}")
        return None
# --- 新增结束 ---

def find_nearest_index(timestamps, target_time):
    """
    Find the index of the timestamp in 'timestamps' that is closest to 'target_time'
    """
    idx = bisect_left(timestamps, target_time)
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    before = timestamps[idx - 1]
    after = timestamps[idx]
    return idx - 1 if abs(before - target_time) < abs(after - target_time) else idx
# --- 新增结束 ---

def convert_retrieval_to_pairs(retrieval_path, pairs_path):
    """
    将检索文件 (query db1 score1 db2 score2 ...) 转换为
    简单的图像对文件 (query db1 \n query db2 \n ...)。
    """
    pairs = []
    with open(retrieval_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            query = parts[0]
            db_images = parts[1::2]  # 每隔一个取一个，得到所有数据库图像名
            for db in db_images:
                pairs.append((query, db))
    
    with open(pairs_path, 'w') as f:
        f.write('\n'.join(' '.join(p) for p in pairs))
    logger.info(f"Converted {len(pairs)} pairs from {retrieval_path} to {pairs_path}")



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
    # 格式: MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, k1, k2, p1, p2
    camera_params = [
        "OPENCV",
        2200,
        1740,
        1210.4155551559695,
        1200.3912375066598,
        1102.1907455875778, 
        834.41813183496947,
        0.017882415329839009,
        -0.024744979638090473,
        -0.00097154769540342529,
        0.00020895912883848669
    ]
    camera_params_str = [str(p) for p in camera_params]

    logger.info(f"正在从目录 {image_dir} 中递归搜索查询图像...")
    
    # 使用 rglob 进行递归搜索，以找到所有子目录中的 .jpg 文件
    all_query_paths = sorted(list(image_dir.rglob("*.jpg")))
    
    if not all_query_paths:
        logger.warning(f"在 {image_dir} 中没有找到任何 .jpg 查询图像。请检查路径。")
        # 创建一个空文件以避免后续步骤失败
        open(path, 'w').close()
        return

    # 提取相对于 image_dir 的相对路径，例如 'cam0/12345.jpg'
    # 这对于确保与 hloc 其他部分的名称一致性至关重要
    queries_relative = [p.relative_to(image_dir).as_posix() for p in all_query_paths]

    # 每行输出：相对图像路径 + 相机内参
    out = [[q] + camera_params_str for q in queries_relative]

    # 写入输出路径
    with open(path, "w") as f:
        f.write("\n".join(map(" ".join, out)))

    logger.info(f"为 {len(out)} 张图像生成了查询列表，并保存于 {path}")




def localize_from_retrieved_poses(
    loc_pairs_path, db_poses_gt_paths, results_path,
    # ... (其他参数保持不变) ...
    **kwargs):
    
    """
    纯文本定位：直接使用检索结果中分数最高的候选位姿。
    此版本不使用断点续传，每次都会覆盖结果文件。
    """
    logger.info(f"Performing text-only localization from: {loc_pairs_path}")
    logger.warning("Breakpoint resume is disabled. The results file will be overwritten.")

    # 1. 读取检索结果 (逻辑不变)
    retrieval = {}
    with open(loc_pairs_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qname = parts[0]
            db_names = parts[1::2]
            scores = np.array([float(s) for s in parts[2::2]], dtype=np.float64)
            retrieval[qname] = (db_names, scores)

    # 2. 加载数据库图像的真值位姿 (逻辑不变)
    db_name_to_pose = {}
    logger.info(f"Loading database poses from {len(db_poses_gt_paths)} ground truth CSV file(s)...")
    for db_path in db_poses_gt_paths:
        try:
            with open(db_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 假设数据库图像名格式为 'camera_id/image_id'
                    full_image_name = f"{row['camera_id']}/{row['image_id']}"
                    tvec = np.array([row['X_enu'], row['Y_enu'], row['Z_enu']], dtype=np.float64)
                    qvec = np.array([row['quat_w'], row['quat_x'], row['quat_y'], row['quat_z']], dtype=np.float64)
                    db_name_to_pose[full_image_name] = (tvec, qvec)
        except Exception as e:
            logger.error(f"Failed to load poses from {db_path}: {e}")
            continue
    logger.info(f"Loaded {len(db_name_to_pose)} database poses.")

    # --- 核心修改：移除断点续传，使用 'w' 模式 ---
    # 3. 以写入模式打开结果文件，并遍历所有查询
    with open(results_path, 'w') as results_file_handle:
        for qname, (dbs, scores) in tqdm(retrieval.items(), desc="Text-only localization (Top-1)"):
            
            final_tvec, final_qvec = None, None

            # 直接选择 CLIP 全局分数最高的
            if dbs:
                best_db_name = dbs[0]
                if best_db_name in db_name_to_pose:
                    final_tvec, final_qvec = db_name_to_pose[best_db_name]

            if final_tvec is not None and final_qvec is not None:
                q_str = ' '.join(map(str, final_qvec))
                t_str = ' '.join(map(str, final_tvec))
                results_file_handle.write(f'{qname} {q_str} {t_str}\n')
    
    logger.info(f"Processing finished. All results saved to {results_path}")
    # --- 修改结束 ---


    # # 3. 遍历每个查询图像，进行地理聚类
    # localized = []
    # for qname, (dbs, scores) in retrieval.items():
    #     candidates = []
    #     for name, score in zip(dbs, scores):
    #         if name in db_name_to_pose:
    #             tvec, qvec = db_name_to_pose[name]
    #             candidates.append({'name': name, 'tvec': tvec, 'qvec': qvec, 'score': score})

    #     if not candidates:
    #         logger.warning(f"No valid candidates for {qname}.")
    #         continue

    #     # --- 地理聚类 ---
    #     logger.info(f"Performing geographic clustering for {len(candidates)} candidates of {qname}...")
    #     tvecs = np.array([c['tvec'] for c in candidates])
    #     from sklearn.cluster import DBSCAN

    #     # 使用 DBSCAN 进行聚类
    #     clustering = DBSCAN(eps=10.0, min_samples=2).fit(tvecs)  # eps 是聚类半径，单位为米
    #     labels = clustering.labels_

    #     # 找到最大簇
    #     unique_labels, counts = np.unique(labels, return_counts=True)
    #     if len(unique_labels) == 0 or (len(unique_labels) == 1 and unique_labels[0] == -1):
    #         logger.warning(f"No valid geographic clusters found for {qname}. Falling back to best CLIP score.")
    #         best_candidate = max(candidates, key=lambda c: c['score'])
    #         final_tvec, final_qvec = best_candidate['tvec'], best_candidate['qvec']
    #     else:
    #         # 找到最大簇的标签
    #         largest_cluster_label = unique_labels[np.argmax(counts)]
    #         largest_cluster = [candidates[i] for i in range(len(candidates)) if labels[i] == largest_cluster_label]

    #         # 计算最大簇的平均位置
    #         cluster_tvecs = np.array([c['tvec'] for c in largest_cluster])
    #         final_tvec = np.mean(cluster_tvecs, axis=0)

    #         # 从最大簇中选择分数最高的候选作为姿态来源
    #         best_in_cluster = max(largest_cluster, key=lambda c: c['score'])
    #         final_qvec = best_in_cluster['qvec']

    #         logger.info(f"Pose for {qname} is fused from the largest geographic cluster ({len(largest_cluster)} members).")

    #     # 保存最终结果
    #     localized.append((qname, final_qvec, final_tvec))

    # # 4. 写入结果文件
    # with open(results_path, 'w') as f:
    #     for name, q, t in localized:
    #         q_str = ' '.join(map(str, q))
    #         t_str = ' '.join(map(str, t))
    #         f.write(f'{name} {q_str} {t_str}\n')

    # logger.info(f"Saved {len(localized)} localized poses to {results_path}")




def run(args):
    print("Starting SensorKit pipeline...")
    logger.info("Starting pipeline...")
    
    # 设置路径
    dataset = args.dataset
    # dataset_test = args.dataset_test
    # mapping_dir = dataset / "IMAGES"
    # mapping_dir = dataset/ "mapping"  
    mapping_dir = dataset/ "allcams"  
    # mapping_dir = dataset / "allcams_downsampled_10"  # 使用下采样后的数据集
    # query_dir = dataset / "allcams_downsampled_10/cam1/" 
    # mapping_dir = dataset / "mapping_allcams_downsampled_10_10"  # 使用下采样后的数据集
    # query_dir = dataset / "query_allcams_downsampled_10_10" 
    query_dir = dataset / "query/cam1_all/0_undistorted" ##################
    # query_dir = dataset / "query/cam1_all/half" ##################  
    # query_dir = dataset / "clipdata/astri_412_down03" 
    # query_dir = dataset / "query/clip_one/"  
    # query_dir = dataset / "query/cam1_vlm_test/" 
    sift_sfm = dataset / "allcams_spatial_colmap_0630/sparse/0/0.025_BA_ENU"
    logger.info(f"Dataset path: {dataset}")
    logger.info(f"Mapping images path: {mapping_dir}")
    logger.info(f"Query images path: {query_dir}")

    outputs = args.outputs  # 输出目录
    outputs.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {outputs}")
    
    # --- 核心修复：在函数开头定义 sfm_dir ---
    sfm_dir = outputs 
    # --- 修复结束 ---

    # --- 核心修改：在这里也加入 LoRA 后缀逻辑 ---
    if args.lora_ckpt_path and args.lora_ckpt_path != 'default':
        lora_suffix = Path(args.lora_ckpt_path).stem
    else:
        lora_suffix = "no_lora"
    # --- 修改结束 ---
    # --- 核心修改：在定义 results 和 loc_pairs 文件名时使用 lora_suffix ---
    loc_pairs = outputs / f"pairs-query-clip{args.num_loc}_{lora_suffix}.txt"
    results = outputs / f"SensorKit_hloc_superpoint+superglue_clip{args.num_loc}_{args.caption_model}_{lora_suffix}.txt"
    # --- 修改结束 ---

    query_list = outputs / "queries_with_intrinsics.txt"
    # reference_sfm = outputs / "sfm_superpointmax+lightglue"
    # outputsba  = outputs / "one_text"  # SfM 输出目录
    reference_sfm_path = outputs / "sfm_superpointmax+lightglue" # 预先存在的 SfM 模型路径
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"

    # loc_pairs = outputs / f"downsampled10_10_pairs-query-netvlad{args.num_loc}.txt"
    # results = outputs / f"downsampled10_10_SensorKit_hloc_superpoint+superglue_netvlad{args.num_loc}.txt"
    # loc_pairs = outputs / f"downsampled10_10_pairs-query-clip{args.num_loc}.txt"
    # results = outputs / f"SensorKit_hloc_superpoint+superglue_clip{args.num_loc}_{args.caption_model}_downsampled10_10.txt"
    # loc_pairs = outputs / f"pairs-query-clip{args.num_loc}.txt"
    
    # results = outputs / f"SensorKit_hloc_superpoint+superglue_clip{args.num_loc}_{args.caption_model}_cam1.txt"
    
    # results = outputsba / f"SensorKit_hloc_superpoint+superglue_clip{args.num_loc}_{args.caption_model}_one.txt"
    # results = outputs / f"SensorKit_hloc_superpoint+superglue_clip{args.num_loc}_{args.caption_model}_cam1_0805.txt"

    # retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]  # "superpoint_max"慢 "superpoint_aachen"
    matcher_conf = match_features.confs["superpoint+lightglue"] # superpoint+lightglue superglue-fast快
 
    # clip_features_path = outputs / "clip_features_one.h5"
    # clip_features_path = outputs / "downsampled_clip_features_cam1_.h5"
    # clip_features_path = outputs / "clip_features_cam1_0.h5"  # large,只有cam4 (false 叉)
    # clip_features_path = outputs / "clip_features_cam1_.h5"  # large
    # clip_features_path = outputs / "clip_features_cam1_long.h5" # long
    clip_features_path = outputs / "clip_features_cam1_datacomp.h5" # large-datacomp
    # clip_features_path = outputs / "clip_features_one_long.h5"
    # clip_features_path = outputs / "sigclip_features_cam1_sigclip.h5"

    # # --- 新增：离线提取数据库的 CLIP 特征 ---
    # logger.info("Extracting CLIP features for the database...")
    # extract_clip_features.main(mapping_dir, clip_features_path)


    # ### OFFLINE
    logger.info("--- Starting OFFLINE pre-processing ---")

    # --- 核心修改：模型预加载逻辑 ---
    t_offline_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 核心修改：添加详细的调试日志 ---
    if args.lora_ckpt_path:
        # 将用户提供的路径转换为绝对路径
        absolute_lora_path = args.lora_ckpt_path.resolve()
        logger.info(f"提供的 LoRA 检查点路径: {args.lora_ckpt_path}")
        logger.info(f"解析后的绝对路径: {absolute_lora_path}")
        logger.info(f"检查路径是否存在... {absolute_lora_path.exists()}")

        if absolute_lora_path.exists():
            # 如果路径存在，则加载微调模型
            logger.info("路径有效，正在加载微调模型...")
            clip_model, clip_tokenizer, _ = pairs_from_clip_retrieval.load_finetuned_model(
                absolute_lora_path, device
            )
            clip_processor = clip_tokenizer 
        else:
            # 如果路径不存在，打印错误并退出或回退
            logger.error("错误：LoRA 检查点文件未找到！请检查路径。")
            logger.warning(f"回退到基础 CLIP 模型: {args.clip_model_id}")
            clip_model = CLIPModel.from_pretrained(args.clip_model_id).to(device)
            clip_processor = CLIPProcessor.from_pretrained(args.clip_model_id)
            # --- 修复结束 ---
    else:
        # 如果根本没有提供 lora_ckpt_path 参数
        logger.info(f"未提供 LoRA 检查点。使用基础 CLIP 模型: {args.clip_model_id}")
        clip_model = CLIPModel.from_pretrained(args.clip_model_id).to(device)
        clip_processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    # --- 调试修改结束 ---

    # --- 预加载修改结束 ---
    
    # # --- 2. 新增：预加载 CLIP 模型 ---
    # t_offline_start = time.time()
    # logger.info("Pre-loading CLIP model for retrieval...")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model = CLIPModel.from_pretrained(args.clip_model_id).to(device)
    # clip_processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    # logger.info(f"CLIP model pre-loaded in {time.time() - t_offline_start:.2f}s.")
    # # --- 预加载结束 ---


    # 新增：加载局部特征匹配模型
    MatcherModel = extract_features.dynamic_load(matchers, matcher_conf['model']['name'])
    local_matcher_model = MatcherModel(matcher_conf['model']).eval().to(device)
    
    
    logger.info("...Models pre-loaded and ready.")
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



    # # 生成基于共视性生成的图像对文件，用于SfM重建
    # # 格式: 每行两个图像ID，表示它们共享视野
    # # 可视化: 可使用networkx库将其可视化为图形网络
    
    # pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)

    # # 提取特征
    # features_db = extract_features.main(feature_conf, mapping_dir, outputs, as_half=True)

    # #  提取特征
    # features = extract_features.main(feature_conf, mapping_dir, outputs, as_half=True)

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
    #     features,       # 提取的特征
    #     sfm_matches,    # 特征匹配结果
    # )

    # # reconstruction.main(
    # #     reference_sfm, 
    # #     mapping_dir,    
    # #     sfm_pairs, 
    # #     features, 
    # #     sfm_matches,
    # # )


    ### ONLINE
    # --- 2. 在在线定位流程开始前，记录开始时间 ---
    logger.info("Starting ONLINE localization process...")
    start_time_online = time.time()
    timings = {}  # 用于存储每个步骤的耗时
    # ---
    t0 = time.time()
    # 生成查询图像列表
    generate_query_list(dataset, query_dir, query_list)
    timings['generate_query_list'] = time.time() - t0
 

    # # 步骤 2: 基于clip进行文本检索，生成定位图像对
    # t0 = time.time()

    # # --- 定义你的文本查询 ---
    # TEXT_PROMPT = "I’m at a roundabout in front of a glass building with the Huawei logo. Huawei headquarters is on my right side. The road curves to the left." # <--- 在这里输入你的文本描述
    # # 1740554624827797760.jpg
    # pairs_from_clip_retrieval.main(
    #     clip_features_path,
    #     TEXT_PROMPT,
    #     args.num_loc,
    #     loc_pairs,
    #     query_list_path=query_list,
    # )
    # timings['clip_retrieval'] = time.time() - t0
    

    # 修改后:
    caption_model_name = args.caption_model
    logger.info(f"Using caption model for retrieval: {caption_model_name}")
    # ---

    # 根据模型名称动态生成 CSV 文件名 (这部分逻辑保持不变)
    # prompts_filename = f"downsampled10_query_prompts_{caption_model_name}_cam1_clip20.csv"
    # prompts_filename = f"query_prompts_{caption_model_name}_sigclip20_one.csv"
    # prompts_filename = f"query_prompts_{caption_model_name}_sigclip10.csv" ### 长文本
    # prompts_filename = f"query_prompts_llava.csv" ### llava文本
    prompts_filename = f"qwen3_query_prompts_{caption_model_name}_77clip10.csv" ### 短文本
    # prompts_filename = f"half_qwen3_query_prompts_qwen-vl_77clip10.csv" 
    # prompts_csv_path = dataset / "query/cam1_all/undistorted/clipdata_sparse03/val_qwen2.csv"
    # prompts_csv_path = dataset / "clipdata/astri_412_down03/val_qwen3.csv"
    # prompts_csv_path = dataset / "query/cam1_all/0_undistorted/val_qwen2.csv"
    # prompts_filename = f"query_prompts_{caption_model_name}_77clip10_remove.csv"
    # prompts_filename = f"query_prompts_{caption_model_name}_77clip10_new0.csv"
    # prompts_filename = f"downsampled10_10_query_prompts_qwen-vl_clip10.csv"
    prompts_csv_path = outputs / prompts_filename
    # prompts_csv_path = dataset / prompts_filename

    # ... (caption_args 的定义保持不变)
    caption_args = argparse.Namespace(

        query_dir=query_dir,
        output_csv=prompts_csv_path,
        model=caption_model_name, # 这里也使用来自命令行的模型名称
        device=None,
        image_list=None  # <-- 核心修复：添加缺失的属性
    )

    if not prompts_csv_path.exists():
        logger.info(f"Prompt file {prompts_csv_path} not found. Generating with local model: {caption_args.model}...")
        
        t_gen = time.time()
        generate_captions.main(caption_args)
        timings['caption_generation'] = time.time() - t_gen
    else:
        logger.info(f"Found existing prompt file: {prompts_csv_path}")

    # ... (为配对文件命名的逻辑保持不变)
    loc_pairs_filename = f"pairs-query-clip{args.num_loc}-{caption_model_name}.txt"
    # loc_pairs_filename = f"pairs-query-clip{args.num_loc}-{caption_model_name}_1.txt"
    # loc_pairs = outputsba / loc_pairs_filename
    loc_pairs = outputs / loc_pairs_filename



    # --- 3. 调用 pairs_from_clip_retrieval.main ---
    t0 = time.time() # 重新计时
    pairs_from_clip_retrieval.main(
        clip_features_path=clip_features_path,
        text_query_or_csv_path=prompts_csv_path,
        num_loc=args.num_loc,
        loc_pairs=loc_pairs,
        # --- 仅传递 CLIP 检索相关参数 ---
        model_instance=clip_model,
        processor_instance=clip_processor,
        # --- 修复结束 ---
        query_list_path=query_list,
        # --- 核心修复：将召回率计算所需的真值文件路径传递下去 ---
        positive_pairs_csv=args.positive_pairs_csv,
    )
    timings['clip_retrieval'] = time.time() - t0

    

    # # 步骤 1: 提取全局描述符
    # t0 = time.time()
    # # 提取全局描述符提取特征 
    # global_descriptors = extract_features.main(retrieval_conf, mapping_dir, outputs)
    # global_descriptors = extract_features.main(retrieval_conf, query_dir, outputs)
    # timings['global_feature_extraction'] = time.time() - t0

    # # 步骤 2: 基于全局检索，生成定位图像对
    # t0 = time.time()
    # # 基于全局检索，生成定位图像对
    # pairs_from_retrieval.main(
    #     global_descriptors,
    #     loc_pairs,
    #     args.num_loc,
    #     query_list=query_list,  
    #     db_model=reference_sfm,
    # )
    # timings['retrieval'] = time.time() - t0




    # # 步骤 1: 为数据库和查询图像分别提取全局描述符
    # t0 = time.time()
    
    # # 为数据库图像创建独立的特征配置和文件
    # db_retrieval_conf = retrieval_conf.copy()
    # db_retrieval_conf['output'] = 'db_' + retrieval_conf['output']
    # logger.info(f"为数据库图像提取全局特征 -> {db_retrieval_conf['output']}")
    # db_global_descriptors = extract_features.main(db_retrieval_conf, mapping_dir, outputs)

    # # 为查询图像创建独立的特征配置和文件
    # query_retrieval_conf = retrieval_conf.copy()
    # query_retrieval_conf['output'] = 'query_' + retrieval_conf['output']
    # logger.info(f"为查询图像提取全局特征 -> {query_retrieval_conf['output']}")
    # query_global_descriptors = extract_features.main(query_retrieval_conf, query_dir, outputs)
    
    # timings['global_feature_extraction'] = time.time() - t0

    # # 步骤 2: 基于分离的特征库进行全局检索
    # t0 = time.time()
    # logger.info("正在从分离的数据库中检索图像对...")
    # pairs_from_retrieval.main(
    #     query_global_descriptors,  # <-- 使用查询特征文件作为检索源
    #     loc_pairs,
    #     args.num_loc,
    #     db_descriptors=db_global_descriptors, # <-- 明确指定数据库特征文件
    #     query_list=query_list,  
    #     db_model=reference_sfm,
    # )
    # timings['retrieval'] = time.time() - t0

    


    
    # 步骤 3: 提取局部特征
    t0 = time.time()
    # query_local_feats_path = outputsba / "query-local-feats.h5"
    query_local_feats_path = outputs / "query-local-feats.h5"
    features = extract_features.main(
        feature_conf, query_dir, feature_path=query_local_feats_path, as_half=True
    )
    timings['local_feature_extraction'] = time.time() - t0


    # # 步骤 4: 局部特征匹配
    # t0 = time.time()
    # db_local_features = outputs / (feature_conf['output'] + '.h5')
    # loc_matches_path = outputs / (matcher_conf['output'] + '.h5')
    # # loc_matches_path = outputsba / (matcher_conf['output'] + '.h5')
    # loc_matches = match_features.main(
    #     conf=matcher_conf, 
    #     pairs=loc_pairs, 
    #     features=query_local_feats_path,
    #     features_ref=db_local_features,
    #     matches=loc_matches_path,  # <-- 为 matches 参数提供完整的 Path 对象
    #     model_instance=local_matcher_model, # <-- 4. 传入预加载的模型实例
    # )
    # timings['feature_matching'] = time.time() - t0

    # # 步骤 5: 定位
    # t0 = time.time()
    # localize_sfm.main(
    #     reference_sfm,   # 使用生成的参考模型  reference_sfm, 
    #     query_list,
    #     loc_pairs,
    #     features, 
    #     loc_matches,
    #     results,
    # )
    # timings['pose_estimation'] = time.time() - t0




# --- 核心修改：根据 --text_only_loc 参数选择定位流程 ---
    if args.text_only_loc:
        # 如果是纯文本定位，调用修改后的函数并跳过后续步骤
        t0 = time.time()

        # --- 核心修改：定义包含所有数据库位姿的CSV文件列表 ---
        db_poses_gt_paths = [
            dataset / "0620_gt/enu_all_mapping_gt.csv",
            dataset / "0620_gt/enu_all_query_gt.csv"
        ]
        
        for path in db_poses_gt_paths:
            if not path.exists():
                logger.error(f"Required database poses file not found: {path}")
                return # 找不到任何一个文件都直接退出
        # --- 修改结束 ---


        # --- 核心修复：在这里定义 gt_data 变量 ---
        query_gt_csv_path = dataset / "0620_gt/enu_all_query_gt.csv"
        gt_data = load_query_gt_from_csv(query_gt_csv_path)
        # --- 修复结束 ---


        # --- 定义对比文件的输出路径 (这部分逻辑保持不变) ---
        comparison_filename = f"onlytext_pose_comparison_clip1.txt"
        comparison_file_path = outputs / comparison_filename
        # ---

    #     # --- 核心修改：调用函数时，传递文件列表 ---
    #     localize_from_retrieved_poses(
    #         loc_pairs, 
    #         db_poses_gt_paths, # <-- 使用包含两个路径的列表
    #         results, 
    #         args.num_loc,
    #         gt_data=gt_data,
    #         comparison_file_path=comparison_file_path
    #     )
    #     timings['pose_estimation'] = time.time() - t0
    # else:

    # --- 核心新增：定义用于聚类分析的新日志文件 ---
        cluster_analysis_filename = f"cluster_analysis_{lora_suffix}.csv"
        cluster_analysis_file_path = outputs / cluster_analysis_filename
        # --- 新增结束 ---
        db_local_features = outputs / (feature_conf['output'] + '.h5')
        # --- 核心修改：调用函数时，传递几何验证所需的组件 ---
        # --- 核心修改：调用新的 localize_from_retrieved_poses ---
        localize_from_retrieved_poses(
            loc_pairs_path=loc_pairs, 
            db_poses_gt_paths=db_poses_gt_paths,
            results_path=results,
            query_prompts_csv=prompts_csv_path,
            image_root_dir=mapping_dir,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            # --- 保持原有参数 ---
            gt_data=gt_data,
            comparison_file_path=comparison_file_path,
            cluster_analysis_file=cluster_analysis_file_path,
        )
        timings['pose_estimation'] = time.time() - t0
    
        # localize_from_retrieved_poses(
        #     loc_pairs_path=loc_pairs, 
        #     db_poses_gt_paths=db_poses_gt_paths,
        #     results_path=results, 
        #     num_loc=args.num_loc,
        #     gt_data=gt_data,
        #     comparison_file_path=comparison_file_path,
        #     cluster_analysis_file=cluster_analysis_file_path
        # )
        # timings['pose_estimation'] = time.time() - t0
    else:

        # 否则，执行标准的几何定位流程

        # --- 新增步骤：将检索文件转换为图像对文件 ---
        t0 = time.time()
        logger.info("Converting retrieval file to pairs file for matching...")
        pairs_for_matching = outputs / f"pairs-for-matching-{caption_model_name}.txt"
        
        # --- 核心修复：调用我们自己的转换函数，而不是 hloc 的 ---
        convert_retrieval_to_pairs(loc_pairs, pairs_for_matching)
        # --- 修复结束 ---

        timings['convert_retrieval_to_pairs'] = time.time() - t0
        # --- 新增步骤结束 ---
        
        # 步骤 4: 局部特征匹配
        t0 = time.time()
        db_local_features = outputs / (feature_conf['output'] + '.h5')
        loc_matches_path = outputs / (matcher_conf['output'] + '.h5')
        loc_matches = match_features.main(
            conf=matcher_conf, 
            # pairs=loc_pairs, 
            pairs=pairs_for_matching,  # <-- 使用转换后的图像对文件
            features=query_local_feats_path,
            features_ref=db_local_features,
            matches=loc_matches_path,
            model_instance=local_matcher_model,
        )
        timings['feature_matching'] = time.time() - t0

        # 步骤 5: 定位
        t0 = time.time()
        localize_sfm.main(
            reference_sfm,
            query_list,
            # loc_pairs,
            pairs_for_matching,
            features, 
            loc_matches,
            results,
        )
        timings['pose_estimation'] = time.time() - t0
    # --- 修改结束 ---




    # # 步骤 3: 提取局部特征
    # t0 = time.time()
    # # 提取特征
    # features = extract_features.main(feature_conf, query_dir, outputs, as_half=True)
    # timings['local_feature_extraction'] = time.time() - t0
    # # 步骤 4: 局部特征匹配
    # t0 = time.time()
    # loc_matches = match_features.main(
    #     matcher_conf, loc_pairs, feature_conf["output"], outputs
    # )
    # timings['feature_matching'] = time.time() - t0


    # # 步骤 5: 定位
    # t0 = time.time()
    # localize_sfm.main(
    #     reference_sfm,   # 使用生成的参考模型  reference_sfm, 
    #     query_list,
    #     loc_pairs,
    #     features,
    #     loc_matches,
    #     results,
    # )
    # timings['pose_estimation'] = time.time() - t0


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
        # default="outputs/SensorKit_hloc/test_t_aachen+lightglue",
        default="outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025",
        # default="outputs/SensorKit_hloc/allcams_spatial_colmap_0630_one_0.025",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=10, # Number of image pairs for SfM, default: %(default)s", 50
        help="Number of image pairs for SfM, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=20, # Number of image pairs for loc, default: %(default)s", 20
        help="Number of image pairs for loc, default: %(default)s",
    )

    # --- 关键修改 2：在这里添加新的命令行参数 ---
    parser.add_argument(
        "--caption_model", 
        type=str, 
        default="qwen-vl", 
        choices=['blip', 'llava', 'qwen-vl'],
        help="用于生成描述以进行检索的模型。"
    )
    # ---

    parser.add_argument(
        "--clip_model_id",
        type=str,
        # default="zer0int/LongCLIP-GmP-ViT-L-14", # <-- 4. 将模型ID参数化
        # default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 
        default="laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        help="The Hugging Face model ID for the CLIP model.",
    )

    # --- 新增 LoRA 检查点参数 ---
    parser.add_argument(
        "--lora_ckpt_path",
        type=Path,
        default=None,
        help="Path to the fine-tuned LoRA checkpoint (.pt file). If provided, this will be used for retrieval.",
    )
    # ---

    # --- 在文件末尾添加新的命令行索引正样本文件参数 ---
    parser.add_argument(
        "--positive_pairs_csv",
        type=Path,
        default=None,
        help="可选的CSV文件路径，其中包含用于扩展召回率计算的 (anchor, positive) 对。",
    )
    # ---

    # --- 新增 --text_only_loc 参数 ---
    parser.add_argument(
        '--text_only_loc', 
        action='store_true', 
        help='Perform text-only localization by using the poses of retrieved DB images.'
    )
    # ---

    args = parser.parse_args()
    run(args)












# import numpy as np
# from scipy.spatial.transform import Rotation as R
# import re

# def convert_quat_pose_to_outputs(input_txt, output_txt_with_name, output_tum_txt):
#     with open(input_txt, 'r') as fin, \
#          open(output_txt_with_name, 'w') as fout1, \
#          open(output_tum_txt, 'w') as fout2:

#         for line in fin:
#             parts = line.strip().split()
#             if len(parts) != 8:
#                 continue  # 跳过无效行
#             name, qw, qx, qy, qz, tx, ty, tz = parts
#             qw, qx, qy, qz = map(float, [qw, qx, qy, qz])
#             tx, ty, tz = map(float, [tx, ty, tz])

#             # 从相机到世界的旋转
#             rot_c2w = R.from_quat([qx, qy, qz, qw]).inv()
#             quat_c2w = rot_c2w.as_quat()  # [qx, qy, qz, qw]
#             R_c2w = rot_c2w.as_matrix()

#             t = np.array([tx, ty, tz])
#             cam_center = -R_c2w @ t  # 世界坐标下的相机中心

#             # 输出文件1：name + 四元数 + 相机位置
#             fout1.write(f"{name} {quat_c2w[0]:.6f} {quat_c2w[1]:.6f} {quat_c2w[2]:.6f} {quat_c2w[3]:.6f} "
#                         f"{cam_center[0]:.6f} {cam_center[1]:.6f} {cam_center[2]:.6f}\n")

#             # 尝试从 name 中提取 timestamp（假设是数字）
#             match = re.search(r'(\d+)', name)
#             if match:
#                 timestamp = match.group(1)
#             else:
#                 timestamp = "0"

#             # 输出文件2：TUM 格式 timestamp tx ty tz qx qy qz qw
#             fout2.write(f"{timestamp} {cam_center[0]:.6f} {cam_center[1]:.6f} {cam_center[2]:.6f} "
#                         f"{quat_c2w[0]:.6f} {quat_c2w[1]:.6f} {quat_c2w[2]:.6f} {quat_c2w[3]:.6f}\n")

#     print(f"Saved:\n  → {output_txt_with_name}\n  → {output_tum_txt}")

# convert_quat_pose_to_outputs(
#     # "outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/SensorKit_hloc_superpoint+superglue_clip20_qwen-vl_cam1.txt",
#     "outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/SensorKit_hloc_superpoint+superglue_clip20_qwen-vl_lora_epoch_36.txt",
#     "outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/SensorKit_query_centers_clip_qwen-vl_cam1.txt",
#     "outputs/evo_tum/enu_mapping_query/clip_allcams_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_qwen-vl_cam1.txt"
# )


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


# input_file_path = "outputs/evo_tum/enu_mapping_query/clip_allcams_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_qwen-vl_cam1.txt"
# output_file_path = "outputs/evo_tum/enu_mapping_query/s_clip_allcams_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_qwen-vl_cam1.txt" # New output file name


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

# def match_and_filter_nearest(est_data, gt_data, threshold=10.0): # 10.0
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
#     est_file = "outputs/evo_tum/enu_mapping_query/s_clip_allcams_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_qwen-vl_cam1.txt"
#     gt_file = "outputs/evo_tum/enu_mapping_query/enu_all_query_gt.tum"
#     output_file = "outputs/evo_tum/enu_mapping_query/s_clip_allcams_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_qwen-vl_cam1_tum_filtered.txt"
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










# ===================================================================
# ==================== 后处理脚本（纯文本） ====================
# ===================================================================

import numpy as np
from bisect import bisect_left
import re
import os

def convert_hloc_to_tum(hloc_results_path, output_tum_path):
    """
    将 hloc 的输出文件 (name qw qx qy qz tx ty tz) 转换为 TUM 格式 (timestamp tx ty tz qx qy qz qw)。
    """
    tum_lines = []
    logger.info(f"Converting HLOC results '{hloc_results_path}' to TUM format '{output_tum_path}'...")
    with open(hloc_results_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            
            name, qw, qx, qy, qz, tx, ty, tz = parts
            
            # 从文件名中提取纳秒时间戳并转换为秒
            timestamp_ns = float(Path(name).stem)
            timestamp_s = timestamp_ns / 1e9
            
            # 按照 TUM 格式重新排列：timestamp tx ty tz qx qy qz qw
            tum_line = f"{timestamp_s:.6f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}"
            tum_lines.append(tum_line)
            
    with open(output_tum_path, 'w') as f:
        f.write('\n'.join(tum_lines))
    logger.info(f"Successfully converted {len(tum_lines)} poses.")

def load_tum_file(file_path):
    """
    加载 TUM 轨迹文件。
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split()
                if len(parts) == 8:
                    data.append([float(p) for p in parts])
    data.sort()
    return data

def find_nearest_index(timestamps, target_time):
    """
    在时间戳列表中查找最接近目标时间的索引。
    """
    idx = bisect_left(timestamps, target_time)
    if idx == 0: return 0
    if idx == len(timestamps): return len(timestamps) - 1
    before = timestamps[idx - 1]
    after = timestamps[idx]
    return idx - 1 if abs(before - target_time) < abs(after - target_time) else idx

def match_and_filter_nearest(est_data, gt_data, threshold=10.0):
    """
    使用最近的时间戳匹配估计轨迹和真值轨迹，并根据3D距离进行过滤。
    """
    gt_timestamps = [row[0] for row in gt_data]
    gt_dict = {row[0]: np.array(row[1:4]) for row in gt_data}

    filtered = []
    for est_row in est_data:
        est_time = est_row[0]
        est_pos = np.array(est_row[1:4])

        nearest_idx = find_nearest_index(gt_timestamps, est_time)
        gt_time = gt_timestamps[nearest_idx]
        gt_pos = gt_dict[gt_time]

        dist = np.linalg.norm(est_pos - gt_pos)
        if dist <= threshold:
            filtered.append(est_row)
        else:
            logger.warning(f"Frame at {est_time:.6f} rejected (distance = {dist:.3f}m to nearest gt @ {gt_time:.6f})")

    return filtered

def save_tum_trajectory(file_path, data):
    """
    保存 TUM 轨迹数据。
    """
    with open(file_path, 'w') as f:
        for row in data:
            f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*row))

# # --- 主后处理逻辑 ---
# if __name__ == "__main__":
#     # 1. 定义输入和输出文件路径
#     # 这是 hloc 流程生成的、包含 ENU 坐标的原始结果文件
#     hloc_results_file = "outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/SensorKit_hloc_superpoint+superglue_clip1_qwen-vl_cam1.txt"
    
#     # 转换后的 TUM 格式文件
#     tum_file_unfiltered = "outputs/evo_tum/enu_mapping_query/clip_qwen-vl_cam1_unfiltered.tum"
    
#     # 真值文件
#     gt_file = "outputs/evo_tum/enu_mapping_query/enu_all_query_gt.tum"
    
#     # 最终过滤后的 TUM 文件
#     tum_file_filtered = "outputs/evo_tum/enu_mapping_query/clip_qwen-vl_cam1_filtered.tum"
    
#     # 过滤阈值
#     distance_threshold = 10.0


# --- 主后处理逻辑 ---
if __name__ == "__main__":
    # --- 新增代码：从命令行解析 LoRA 模型路径 ---
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_ckpt_path', type=str, default='default', help='Path to the lora checkpoint')
    args, unknown = parser.parse_known_args()

    # 从 LoRA 路径生成一个唯一的文件名后缀
    if args.lora_ckpt_path != 'default':
        lora_suffix = Path(args.lora_ckpt_path).stem  # e.g., "lora_epoch_6"
    else:
        lora_suffix = "no_lora"
    # --- 新增代码结束 ---

    # 1. 定义输入和输出文件路径 (使用动态后缀)
    # 这是 hloc 流程生成的、包含 ENU 坐标的原始结果文件
    hloc_results_file = f"outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/SensorKit_hloc_superpoint+superglue_clip1_qwen-vl_{lora_suffix}.txt"
    
    # 转换后的 TUM 格式文件
    tum_file_unfiltered = f"outputs/evo_tum/enu_mapping_query/clip_qwen-vl_cam1_{lora_suffix}_unfiltered.tum"
    
    # 真值文件
    gt_file = "outputs/evo_tum/enu_mapping_query/enu_all_query_gt.tum"
    
    # 最终过滤后的 TUM 文件
    tum_file_filtered = f"outputs/evo_tum/enu_mapping_query/clip_qwen-vl_cam1_{lora_suffix}_filtered.tum"
    
    # 过滤阈值
    distance_threshold = 10.0

    # 2. 执行后处理流程
    if os.path.exists(hloc_results_file):
        # 步骤 1: 将 HLOC 结果转换为 TUM 格式
        convert_hloc_to_tum(hloc_results_file, tum_file_unfiltered)

        # 步骤 2: 加载数据并进行过滤
        logger.info("Loading trajectories for filtering...")
        est_data = load_tum_file(tum_file_unfiltered)
        gt_data = load_tum_file(gt_file)

        logger.info(f"Filtering with distance threshold: {distance_threshold}m...")
        filtered_data = match_and_filter_nearest(est_data, gt_data, distance_threshold)

        # 步骤 3: 保存过滤后的轨迹
        logger.info(f"Saving filtered trajectory to {tum_file_filtered}...")
        save_tum_trajectory(tum_file_filtered, filtered_data)

        logger.info(f"Post-processing done. Original: {len(est_data)} frames, Filtered: {len(filtered_data)} frames.")
    else:
        logger.error(f"Input file for post-processing not found: {hloc_results_file}")

























# import numpy as np
# from scipy.spatial.transform import Rotation as R
# import re

# def convert_quat_pose_to_outputs(input_txt, output_txt_with_name, output_tum_txt):
#     with open(input_txt, 'r') as fin, \
#          open(output_txt_with_name, 'w') as fout1, \
#          open(output_tum_txt, 'w') as fout2:

#         for line in fin:
#             parts = line.strip().split()
#             if len(parts) != 8:
#                 continue  # 跳过无效行
#             name, qw, qx, qy, qz, tx, ty, tz = parts
#             qw, qx, qy, qz = map(float, [qw, qx, qy, qz])
#             tx, ty, tz = map(float, [tx, ty, tz])

#             # 从相机到世界的旋转
#             rot_c2w = R.from_quat([qx, qy, qz, qw]).inv()
#             quat_c2w = rot_c2w.as_quat()  # [qx, qy, qz, qw]
#             R_c2w = rot_c2w.as_matrix()

#             t = np.array([tx, ty, tz])
#             cam_center = -R_c2w @ t  # 世界坐标下的相机中心

#             # 输出文件1：name + 四元数 + 相机位置
#             fout1.write(f"{name} {quat_c2w[0]:.6f} {quat_c2w[1]:.6f} {quat_c2w[2]:.6f} {quat_c2w[3]:.6f} "
#                         f"{cam_center[0]:.6f} {cam_center[1]:.6f} {cam_center[2]:.6f}\n")

#             # 尝试从 name 中提取 timestamp（假设是数字）
#             match = re.search(r'(\d+)', name)
#             if match:
#                 timestamp = match.group(1)
#             else:
#                 timestamp = "0"

#             # 输出文件2：TUM 格式 timestamp tx ty tz qx qy qz qw
#             fout2.write(f"{timestamp} {cam_center[0]:.6f} {cam_center[1]:.6f} {cam_center[2]:.6f} "
#                         f"{quat_c2w[0]:.6f} {quat_c2w[1]:.6f} {quat_c2w[2]:.6f} {quat_c2w[3]:.6f}\n")

#     print(f"Saved:\n  → {output_txt_with_name}\n  → {output_tum_txt}")

# convert_quat_pose_to_outputs(
#     "outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/downsampled10_10_SensorKit_hloc_superpoint+superglue_netvlad10.txt",
#     "outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/downsampled10_10_SensorKit_query_centers_netvlad10_cam1_0805.txt",
#     "outputs/evo_tum/enu_mapping_query/downsampled10_10_allcams_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_netvlad10_cam1_0805_tum.txt"
# )


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


# input_file_path = "outputs/evo_tum/enu_mapping_query/downsampled10_10_allcams_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_netvlad10_cam1_0805_tum.txt"
# output_file_path = "outputs/evo_tum/enu_mapping_query/s_downsampled10_10_allcams_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_netvlad10_cam1_0805_tum.txt" # New output file name


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
#     est_file = "outputs/evo_tum/enu_mapping_query/s_downsampled10_10_allcams_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_netvlad10_cam1_0805_tum.txt"
#     gt_file = "outputs/evo_tum/enu_mapping_query/enu_all_query_gt.tum"
#     output_file = "outputs/evo_tum/enu_mapping_query/s_downsampled10_10_allcams_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_netvlad10_cam1_0805_tum_filtered.txt"
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


