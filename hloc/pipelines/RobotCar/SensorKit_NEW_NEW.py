
import argparse
import glob
import logging
from pathlib import Path
import yaml
import cv2  # 用于读取 OpenCV 格式的 YAML 文件
import os
import time


from ... import (
    extract_features,
    localize_sfm,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
    reconstruction,
)
from . import colmap_from_nvm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def generate_query_list(dataset, image_dir, path): 
    # 移除从 YAML 文件读取的逻辑，使用您提供的硬编码参数
    # 格式: MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, k1, k2, p1, p2
    camera_params = [
        "OPENCV",
        2200,
        1740,
        1210.4155551559695,
        1200.3912375066598, ########## fix
        1102.1907455875778, 
        834.41813183496947,
        0.017882415329839009,
        -0.024744979638090473,
        -0.00097154769540342529,
        0.00020895912883848669
    ]
    camera_params = [str(p) for p in camera_params]

    # 获取查询图像名（仅文件名，不含路径）
    queries = glob.glob((image_dir / "*.jpg").as_posix(), recursive=True)
    queries = [Path(q).name for q in sorted(queries)]

    # 每行输出：图像名 + 相机内参
    out = [[q] + camera_params for q in queries]

    # 写入输出路径
    with open(path, "w") as f:
        f.write("\n".join(map(" ".join, out)))

    logger.info(f"Generated query list with hardcoded intrinsics at {path}")


# def generate_query_list(dataset, image_dir, path):
#     # 移除从 YAML 文件读取的逻辑，使用您提供的硬编码参数
#     # 格式: MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, k1, k2, p1, p2
#     camera_params = [
#         "OPENCV",
#         2200,
#         1740,
#         1209.8508228781561,
#         1194.7606852205797,
#         1102.1907455875778,   0.05
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



# def generate_query_list(dataset, image_dir, path): 
#     # 移除从 YAML 文件读取的逻辑，使用您提供的硬编码参数
#     # 格式: MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, k1, k2, p1, p2
#     camera_params = [
#         "OPENCV", 
#         2200,
#         1740,
#         1208.5590572281003,
#         1159.3881884168909, ########## 0.025
#         1102.1907455875778, 
#         834.41813183496947,
#         0.018496193941085062,
#         -0.024476519342250945,
#         -0.0015408571181731994,
#         0.00019147601151739993
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




# def generate_query_list(dataset, image_dir, path): 
#     # 移除从 YAML 文件读取的逻辑，使用您提供的硬编码参数
#     # 格式: MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, k1, k2, p1, p2
#     camera_params = [
#         "OPENCV", 
#         2200,
#         1740,
#         1206.8018214667377,
#         1101.5254976125998, ########## 0.01
#         1102.1907455875778, 
#         834.41813183496947,
#         0.019894466973868499,
#         -0.024634816949375509,
#         -0.0022547922079267867,
#         0.0004954222136537526
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



def run(args):
    print("Starting SensorKit pipeline...")
    logger.info("Starting pipeline...")
    
    # 设置路径
    dataset = args.dataset
    # dataset_test = args.dataset_test
    # mapping_dir = dataset / "IMAGES"
    # mapping_dir = dataset/ "mapping"  
    mapping_dir = dataset/ "allcams"  
    # mapping_dir = dataset / "allcams_downsampled_5"  # 使用下采样后的数据集
    # query_dir = dataset / "allcams_downsampled_5/cam1/" 
    query_dir = dataset / "query/cam1_all/undistorted/" 
    # query_dir = dataset / "query/test_t/undistorted/" 
    # query_dir = dataset / "query/cam2_mapping/undistorted/"
    # query_dir = dataset / "query/cam2_all/undistorted/" 
    # query_dir = dataset_test / "mapping/jpg/" 
    # sift_sfm = dataset / "colmap_seque_prior/sparse_aligned/1/only_scale"
    # sift_sfm = dataset / "new_colmap_spatial_intrinsics/sparse_aligned_ecef_4"
    # sift_sfm = dataset / "new_colmap_exhaustive_fixing_intrinsics/sparse/0"
    sift_sfm = dataset / "allcams_spatial_colmap_0630/sparse/0/0.025_BA_ENU"
    # sift_sfm = dataset / "allcams_spatial_colmap_0630/sparse/0/0.01_BA_ENU"
    # sift_sfm = dataset / "colmap_seque/sparse_aligned_enu"
    logger.info(f"Dataset path: {dataset}")
    logger.info(f"Mapping images path: {mapping_dir}")
    logger.info(f"Query images path: {query_dir}")

    outputs = args.outputs  # 输出目录
    outputs.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {outputs}")

    query_list = outputs / "queries_with_intrinsics.txt"
    # outputsba  = outputs / "BA_Results/0.05_BA_CAUCHY_ENU"  # SfM 输出目录
    # reference_sfm = outputs / "sfm_superpoint+superglue/0.5_BA_CAUCHY_ENU_0626" 
    reference_sfm = outputs / "sfm_superpointmax+lightglue"
    # reference_sfm = outputs / "sfm_superpointmax+lightglue" 
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    loc_pairs = outputs / f"pairs-query-netvlad{args.num_loc}.txt"
    results = outputs / f"SensorKit_hloc_superpoint+superglue_netvlad{args.num_loc}.txt"

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]  # "superpoint_max"慢 "superpoint_aachen"
    matcher_conf = match_features.confs["superpoint+lightglue"] # superpoint+lightglue superglue-fast快

 

    ### OFFLINE

    # 生成基于共视性生成的图像对文件，用于SfM重建
    # 格式: 每行两个图像ID，表示它们共享视野
    # 可视化: 可使用networkx库将其可视化为图形网络
    
    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)

    # 提取特征
    features = extract_features.main(feature_conf, mapping_dir, outputs, as_half=True)

    # 匹配特征
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    # 三角化重建3d参考模型
    triangulation.main(
        reference_sfm,  # 输出的参考稀疏模型
        sift_sfm,       # 现有的SIFT稀疏模型
        mapping_dir,    # 输入图像目录
        sfm_pairs,      # 图像对文件
        features,       # 提取的特征
        sfm_matches,    # 特征匹配结果
    )

    # reconstruction.main(
    #     reference_sfm, 
    #     mapping_dir,    
    #     sfm_pairs, 
    #     features, 
    #     sfm_matches,
    # )


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
 
    # 步骤 1: 提取全局描述符
    t0 = time.time()
    # 提取全局描述符提取特征 
    global_descriptors = extract_features.main(retrieval_conf, mapping_dir, outputs)
    global_descriptors = extract_features.main(retrieval_conf, query_dir, outputs)
    timings['global_feature_extraction'] = time.time() - t0

    # 步骤 2: 基于全局检索，生成定位图像对
    t0 = time.time()
    # 基于全局检索，生成定位图像对
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_list=query_list,  
        db_model=reference_sfm,
    )
    timings['retrieval'] = time.time() - t0


    # 步骤 3: 提取局部特征
    t0 = time.time()
    # 提取特征
    features = extract_features.main(feature_conf, query_dir, outputs, as_half=True)
    timings['local_feature_extraction'] = time.time() - t0
    # 步骤 4: 局部特征匹配
    t0 = time.time()
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
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
        default="outputs/SensorKit_hloc/downsample_mapping_spatial_colmap_0630_CAM1_0.025",
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
    "outputs/SensorKit_hloc/downsample_mapping_spatial_colmap_0630_CAM1_0.025/SensorKit_hloc_superpoint+superglue_netvlad20.txt",
    "outputs/SensorKit_hloc/downsample_mapping_spatial_colmap_0630_CAM1_0.025/SensorKit_query_centers.txt",
    "outputs/evo_tum/enu_mapping_query/downsample_mapping_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_tum.txt"
)


########### ns TUM s
import numpy as np

def convert_timestamp_ns_to_s(input_path, output_path):
    """
    Reads a TUM-like file, converts the first column (timestamp) from nanoseconds to seconds,
    and saves the modified data to a new file.
    """
    output_lines = []
    with open(input_path, 'r') as f_in:
        for line in f_in:
            if line.startswith("#"): # Preserve comment lines
                output_lines.append(line)
                continue
            
            parts = line.strip().split()
            if len(parts) == 8: # Expecting timestamp tx ty tz qx qy qz qw
                try:
                    timestamp_ns = float(parts[0])
                    timestamp_s = timestamp_ns / 1e9  # Convert nanoseconds to seconds
                    
                    # Reconstruct the line with the new timestamp
                    # and format all numbers to .6f
                    new_line_parts = [f"{timestamp_s:.6f}"] + [f"{float(p):.6f}" for p in parts[1:]]
                    output_lines.append(" ".join(new_line_parts) + "\n")
                except ValueError:
                    print(f"Warning: Could not parse numbers in line, skipping: {line.strip()}")
                    output_lines.append(line) # Append original line if parsing fails
            else:
                # If the line doesn't have 8 parts, append it as is (e.g., empty lines or malformed)
                output_lines.append(line)

    with open(output_path, "w") as f_out:
        for out_line in output_lines:
            f_out.write(out_line)
    
    print(f"Converted timestamps and saved to {output_path}")


input_file_path = "outputs/evo_tum/enu_mapping_query/downsample_mapping_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_tum.txt"
output_file_path = "outputs/evo_tum/enu_mapping_query/s_downsample_mapping_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_tum.txt" # New output file name


import os
if os.path.exists(input_file_path):
    convert_timestamp_ns_to_s(input_file_path, output_file_path)
else:
    print(f"Input file not found: {input_file_path}")
    


import numpy as np
from bisect import bisect_left

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

def match_and_filter_nearest(est_data, gt_data, threshold=10.0):
    """
    Match est to gt using nearest timestamp and filter based on 3D distance
    """
    gt_timestamps = [row[0] for row in gt_data]
    gt_dict = {row[0]: row[1:4] for row in gt_data}

    filtered = []
    for est_row in est_data:
        est_time = est_row[0]
        est_pos = np.array(est_row[1:4])

        nearest_idx = find_nearest_index(gt_timestamps, est_time)
        gt_time = gt_timestamps[nearest_idx]
        gt_pos = np.array(gt_dict[gt_time])

        dist = np.linalg.norm(est_pos - gt_pos)
        if dist <= threshold:
            filtered.append(est_row)
        else:
            print(f"Frame at {est_time:.6f} rejected (distance = {dist:.3f}m to nearest gt @ {gt_time:.6f})")

    return np.array(filtered)

def save_tum_trajectory(file_path, data):
    with open(file_path, 'w') as f:
        for row in data:
            f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*row))

if __name__ == "__main__":
    est_file = "outputs/evo_tum/enu_mapping_query/s_downsample_mapping_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_tum.txt"
    gt_file = "outputs/evo_tum/enu_mapping_query/enu_all_query_gt.tum"
    output_file = "outputs/evo_tum/enu_mapping_query/s_downsample_mapping_spatial_colmap_0630_CAM1_0.025_SensorKit_query_centers_tum_filtered.txt"
    threshold = 10.0  # meters # 10.0

    print("Loading estimated trajectory...")
    est_data = load_tum_file(est_file)

    print("Loading ground truth trajectory...")
    gt_data = load_tum_file(gt_file)

    print("Filtering using nearest time and distance threshold...")
    filtered_data = match_and_filter_nearest(est_data, gt_data, threshold)

    print(f"Saving filtered trajectory to {output_file}...")
    save_tum_trajectory(output_file, filtered_data)

    print(f"Done. Original: {len(est_data)} frames, Filtered: {len(filtered_data)} frames")



