
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


import yaml  # 用于解析 YAML 文件
import glob
from pathlib import Path


# def generate_query_list_from_model(reference_sfm, image_dir, path):
#     """
#     从参考模型中提取相机内参，并生成查询图像列表。
    
#     Args:
#         reference_sfm (pycolmap.Reconstruction): 参考模型。
#         image_dir (Path): 查询图像所在的文件夹。
#         path (Path): 输出查询图像列表的路径。
#     """
#     # 提取参考模型中的相机内参
#     cameras = {}
#     for camera_id, camera in reference_sfm.cameras.items():
#         # 强制将相机模型设置为 SIMPLE_RADIAL
#         params = [
#             "SIMPLE_RADIAL",  # 强制使用 SIMPLE_RADIAL 模型
#             camera.width,
#             camera.height,
#             *camera.params[:4],  # 只保留前 4 个参数（焦距、光心坐标、畸变系数）
#         ]
#         cameras[camera_id] = [str(p) for p in params]

#     # 获取查询图像路径
#     queries = glob.glob((image_dir / "*.jpg").as_posix(), recursive=True)
#     queries = [Path(q).name for q in sorted(queries)]  # 只保留文件名

#     # 假设所有查询图像使用相同的相机
#     default_camera_id = list(cameras.keys())[0]
#     out = [[q] + cameras[default_camera_id] for q in queries]

#     # 写入输出文件
#     with open(path, "w") as f:
#         f.write("\n".join(map(" ".join, out)))



def generate_query_list(dataset, image_dir, path):
    # 只处理 cam1 的内参文件
    intrinsics_filename = "config/intrinsics/camera-intrinsics_cam1.yaml"  # cam1 的 YAML 文件路径
    cameras = {}

    # 使用 cv2.FileStorage 读取 cam1 的 YAML 文件
    fs = cv2.FileStorage(str(dataset / intrinsics_filename), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open {intrinsics_filename}")

    # 从 cameraMatrix 中提取 fx, fy, cx, cy
    camera_matrix = fs.getNode("cameraMatrix").mat()
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # 获取图像宽高  
    w = int(fs.getNode("imageWidth").real())
    h = int(fs.getNode("imageHeight").real())

    # 构造相机参数
    # params = ["SIMPLE_RADIAL", w, h, fx, cx, cy, 0.0]
    # params = ["SIMPLE_RADIAL", 2200, 1740, 1228.4566560292494, 1100, 870, -0.048229458708200092]
    params = ["SIMPLE_RADIAL", 2200, 1740, 1242.5232843372257, 1100, 870, -0.02761191606812281]
    # params = ["PINHOLE", w, h, fx, fy, cx, cy]
    cameras["cam1"] = [str(p) for p in params]  # 只存储 cam1 的参数


    fs.release()

    # 获取 cam1 文件夹下的查询图像路径
    # queries = glob.glob((image_dir / "cam1_all/**/*.jpg").as_posix(), recursive=True)
    queries = glob.glob((image_dir / "*.jpg").as_posix(), recursive=True)
    queries = [Path(q).name for q in sorted(queries)]  # 只保留文件名

    # 为 cam1 的查询图像添加相机内参
    out = [[q] + cameras["cam1"] for q in queries]

    # 写入输出文件
    with open(path, "w") as f:
        f.write("\n".join(map(" ".join, out)))


# def generate_query_list(dataset, image_dir, path):
#     # 只处理 cam1 的内参文件
#     intrinsics_filename = "config/intrinsics/camera-intrinsics_cam2.yaml"  # cam1 的 YAML 文件路径
#     cameras = {}

#     # 使用 cv2.FileStorage 读取 cam1 的 YAML 文件
#     fs = cv2.FileStorage(str(dataset / intrinsics_filename), cv2.FILE_STORAGE_READ)
#     if not fs.isOpened():
#         raise FileNotFoundError(f"Could not open {intrinsics_filename}")

#     # 从 cameraMatrix 中提取 fx, fy, cx, cy
#     camera_matrix = fs.getNode("cameraMatrix").mat()
#     fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
#     cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

#     # 获取图像宽高
#     w = int(fs.getNode("imageWidth").real())
#     h = int(fs.getNode("imageHeight").real())

#     # 构造相机参数
#     # params = ["SIMPLE_RADIAL", w, h, fx, cx, cy, 0.0]
#     params = ["SIMPLE_RADIAL", 2200, 1740, 1228.0692203952146, 1100, 870, -0.015858411386629093]

#     # params = ["PINHOLE", w, h, fx, fy, cx, cy]
#     cameras["cam2"] = [str(p) for p in params]  # 只存储 cam1 的参数


#     fs.release()

#     # 获取 cam1 文件夹下的查询图像路径
#     # queries = glob.glob((image_dir / "cam1_all/**/*.jpg").as_posix(), recursive=True)
#     queries = glob.glob((image_dir / "*.jpg").as_posix(), recursive=True)
#     queries = [Path(q).name for q in sorted(queries)]  # 只保留文件名

#     # 为 cam1 的查询图像添加相机内参
#     out = [[q] + cameras["cam2"] for q in queries]

#     # 写入输出文件
#     with open(path, "w") as f:
#         f.write("\n".join(map(" ".join, out)))




# import cv2
# import glob
# from pathlib import Path

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




def run(args):
    print("Starting SensorKit pipeline...")
    logger.info("Starting pipeline...")
    
    # 设置路径
    dataset = args.dataset
    # dataset_test = args.dataset_test
    mapping_dir = dataset / "IMAGES"
    # mapping_dir = dataset/ "mapping"  
    query_dir = dataset / "query/cam1_all/undistorted/" 
    # query_dir = dataset / "query/cam1_all/downsample/" 
    # query_dir = dataset / "query/cam2_mapping/undistorted/" 
    # query_dir = dataset_test / "mapping/jpg/" 
    # sift_sfm = dataset / "colmap_seque_prior/sparse_aligned/1/only_scale"
    # sift_sfm = dataset / "new_colmap_spatial_intrinsics/sparse_aligned_ecef_4"
    # sift_sfm = dataset / "new_colmap_exhaustive_fixing_intrinsics/sparse/0"
    # sift_sfm = dataset / "colmap_seque/sparse/1/BA_ENU"
    # sift_sfm = dataset / "colmap_seque/sparse_aligned_enu"
    logger.info(f"Dataset path: {dataset}")
    logger.info(f"Mapping images path: {mapping_dir}")
    logger.info(f"Query images path: {query_dir}")

    outputs = args.outputs  # 输出目录
    outputs.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {outputs}")

    query_list = outputs / "queries_with_intrinsics.txt"
    outputsba  = outputs / "BA_Results/0.05_BA_CAUCHY_ENU"  # SfM 输出目录
    # reference_sfm = outputs / "sfm_superpoint+superglue/0.5_BA_CAUCHY_ENU_0626" 
    reference_sfm = outputs / "sfm_superpointmax+lightglue"
    # reference_sfm = outputs / "sfm_superpointmax+lightglue" 
    # sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    loc_pairs = outputsba / f"pairs-query-netvlad{args.num_loc}.txt"
    results = outputsba / f"SensorKit_hloc_superpoint+superglue_netvlad{args.num_loc}.txt"

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_max"]  # "superpoint_max" "superpoint_aachen"
    matcher_conf = match_features.confs["superpoint+lightglue"]  


 

    # ### OFFLINE

    # # 生成基于共视性生成的图像对文件，用于SfM重建
    # # 格式: 每行两个图像ID，表示它们共享视野
    # # 可视化: 可使用networkx库将其可视化为图形网络
    
    # pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)

    # 提取特征
    features = extract_features.main(feature_conf, mapping_dir, outputsba, as_half=True)

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

    # 生成查询图像列表
    generate_query_list(dataset, query_dir, query_list)
 

    # 提取全局描述符提取特征 
    global_descriptors = extract_features.main(retrieval_conf, mapping_dir, outputsba)
    global_descriptors = extract_features.main(retrieval_conf, query_dir, outputsba)

    # 基于全局检索，生成定位图像对
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_list=query_list,  
        db_model=reference_sfm,
    )
    # 提取特征
    features = extract_features.main(feature_conf, query_dir, outputsba, as_half=True)
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputsba
    )

    # 定位

    localize_sfm.main(
        reference_sfm,   # 使用生成的参考模型  reference_sfm, 
        query_list,
        loc_pairs,
        features,
        loc_matches,
        results,
    )


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
        default="outputs/SensorKit_hloc/ENU_BA_sparse_seque",
        # default="outputs/SensorKit_hloc/exhaustive_fixing_intrinsics",
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
    "outputs/SensorKit_hloc/ENU_BA_sparse_seque/BA_Results/0.05_BA_CAUCHY_ENU/SensorKit_hloc_superpoint+superglue_netvlad20.txt",
    "outputs/SensorKit_hloc/ENU_BA_sparse_seque/BA_Results/0.05_BA_CAUCHY_ENU/SensorKit_query_centers.txt",
    "outputs/evo_tum/enu_mapping_query/0.05_BA_CAUCHY_sparse_seque_SensorKit_query_centers_tum.txt"
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


input_file_path = "outputs/evo_tum/enu_mapping_query/0.05_BA_CAUCHY_sparse_seque_SensorKit_query_centers_tum.txt"
output_file_path = "outputs/evo_tum/enu_mapping_query/s_0.05_BA_CAUCHY_sparse_seque_SensorKit_query_centers_tum.txt" # New output file name

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
    est_file = "outputs/evo_tum/enu_mapping_query/s_0.05_BA_CAUCHY_sparse_seque_SensorKit_query_centers_tum.txt"
    gt_file = "outputs/evo_tum/enu_mapping_query/enu_all_query_gt.tum"
    output_file = "outputs/evo_tum/enu_mapping_query/s_0.05_BA_CAUCHY_sparse_seque_SensorKit_query_centers_tum_filtered.txt"
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













# from pathlib import Path
# from PIL import Image

# def convert_png_to_jpg(input_dir, output_dir):
#     """
#     将指定文件夹下的所有 PNG 图片转换为 JPG 格式，并保存到输出文件夹。
    
#     Args:
#         input_dir (str or Path): 输入文件夹路径，包含 PNG 图片。
#         output_dir (str or Path): 输出文件夹路径，用于保存转换后的 JPG 图片。
#     """
#     input_dir = Path(input_dir)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出文件夹存在

#     for png_file in input_dir.glob("*.png"):  # 遍历所有 PNG 文件
#         with Image.open(png_file) as img:
#             # 转换为 RGB 模式（JPG 不支持透明通道）
#             rgb_img = img.convert("RGB")
#             # 保存为 JPG 格式
#             jpg_file = output_dir / f"{png_file.stem}.jpg"  # 使用相同文件名，扩展名改为 .jpg
#             rgb_img.save(jpg_file, "JPEG")
#             print(f"Converted: {png_file} -> {jpg_file}")

# # 示例用法
# input_folder = "datasets/SensorKit_test/mapping"  # 替换为包含 PNG 图片的文件夹路径
# output_folder = "datasets/SensorKit_test/mapping/jpg"   # 替换为保存 JPG 图片的文件夹路径
# convert_png_to_jpg(input_folder, output_folder)











# import csv # 导入 CSV 模块
# import os
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# from hloc.utils.read_write_model import read_model, write_model, Image
# import matplotlib.pyplot as plt # 已导入，但在此代码段中未使用
# from mpl_toolkits.mplot3d import Axes3D # 已导入，但在此代码段中未使用

# # ========== 配置 (针对模型对齐部分) ==========
# # 这些路径是针对模型对齐脚本的，请确保它们是正确的
# IMAGES_BIN_PATH_ALIGN = "outputs/SensorKit_hloc/only_scale/sfm_superpoint+superglue"  # 用于对齐的原始 images.bin
# GPS_PRIOR_PATH_ALIGN = "datasets/SensorKit/mapping_camera_ecef_poses_with_quaternion.csv" # ECEF 先验位姿
# # GPS_PRIOR_PATH_QUERY = "datasets/SensorKit/query_camera_ecef_poses_with_quaternion.csv" # 查询图像的 ECEF 先验位姿
# OUTPUT_DIR_ALIGN = "outputs/SensorKit_hloc/only_scale/sfm_superpoint+superglue/points_scaled_only"  # 对齐模型的输出
# TRANSFORM_PARAMS_DIR = OUTPUT_DIR_ALIGN # 相似变换参数将保存在此目录中

# # outputs/SensorKit_hloc/only_scale/sfm_superpoint+superglue

# os.makedirs(OUTPUT_DIR_ALIGN, exist_ok=True)

# # ========== 辅助函数 (针对模型对齐部分) ==========
# def qvec2rotmat_align(qvec): # 重命名以避免与 convert_quat_pose_to_xyz 中的潜在冲突
#     return R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

# def rotmat2qvec_align(Rmat): # 重命名
#     quat = R.from_matrix(Rmat).as_quat()
#     return np.array([quat[3], quat[0], quat[1], quat[2]])

# def compute_camera_center_align(qvec, tvec): # 重命名
#     Rmat = qvec2rotmat_align(qvec)
#     center = -Rmat.T @ tvec
#     return center

# def estimate_similarity_transform_align(source, target): # 重命名
#     assert source.shape == target.shape
#     mu_src = np.mean(source, axis=0)
#     mu_tgt = np.mean(target, axis=0)
#     src_centered = source - mu_src
#     tgt_centered = target - mu_tgt
#     cov_matrix = src_centered.T @ tgt_centered / source.shape[0]
#     U, S, Vt = np.linalg.svd(cov_matrix)
#     R_est = Vt.T @ U.T
#     if np.linalg.det(R_est) < 0:
#         Vt[-1, :] *= -1
#         R_est = Vt.T @ U.T
#     var_src = np.var(src_centered, axis=0).sum()
#     scale = np.sum(S) / var_src # 如果要将比例固定为1，则 scale = 1.0
#     t_est = mu_tgt - scale * R_est @ mu_src
#     return scale, R_est, t_est

# def read_prior_gps_align(path): # 重命名
#     gps = {}
#     with open(path, 'r', newline='') as f:
#         reader = csv.DictReader(f, delimiter=',')
#         for row in reader:
#             image_full_path = row['image_path']
#             name = os.path.basename(image_full_path)
#             try:
#                 qw = float(row['quat_w_c2e'])
#                 qx = float(row['quat_x_c2e'])
#                 qy = float(row['quat_y_c2e'])
#                 qz = float(row['quat_z_c2e'])
#                 tx = float(row['pos_x_ecef'])
#                 ty = float(row['pos_y_ecef'])
#                 tz = float(row['pos_z_ecef'])
#                 gps[name] = (qw, qx, qy, qz, tx, ty, tz)
#             except ValueError as e:
#                 print(f"警告 (对齐): 由于数据转换错误，跳过行: {row} - {e}")
#                 continue
#             except KeyError as e:
#                 print(f"警告 (对齐): 由于缺少键，跳过行: {e} 在行 {row} 中")
#                 continue
#     return gps

# # ========== 主程序 (模型对齐部分) ==========
# def align_colmap_model_and_save_params():
#     print("正在运行模型对齐脚本...")
#     print("📥 (对齐) 正在读取数据...")
#     cameras, images, points3D = read_model(IMAGES_BIN_PATH_ALIGN)
#     gps_priors = read_prior_gps_align(GPS_PRIOR_PATH_ALIGN)

#     image_names = {image.name: image_id for image_id, image in images.items()}
#     common_names = sorted(set(image_names.keys()) & set(gps_priors.keys()))

#     if not common_names:
#         print("❌ (对齐) 错误: 在 COLMAP 模型和 GPS 先验文件中未找到共同图像。")
#         # ... (错误消息详情) ...
#         return False # 返回 False 表示失败

#     print(f"✅ (对齐) 找到 {len(common_names)} 张共同图像用于对齐。")

#     sparse_centers = []
#     gps_centers_for_align = [] # 变量重命名
#     for name in common_names:
#         image_id = image_names[name]
#         image = images[image_id]
#         sparse_centers.append(compute_camera_center_align(image.qvec, image.tvec))
#         _, _, _, _, tx, ty, tz = gps_priors[name]
#         gps_centers_for_align.append(np.array([tx, ty, tz]))

#     sparse_centers_np = np.array(sparse_centers) # 变量重命名
#     gps_centers_np = np.array(gps_centers_for_align) # 变量重命名

#     if sparse_centers_np.shape[0] < 2 or gps_centers_np.shape[0] < 2:
#         print(f"❌ (对齐) 错误: 用于估计相似变换的匹配点数量不足 ({sparse_centers_np.shape[0]}个)。至少需要2个点。")
#         return False

#     print("🔍 (对齐) 正在估计相似变换 Sim(3)...")
#     scale, R_est, t_est = estimate_similarity_transform_align(sparse_centers_np, gps_centers_np)
#     print(f"✨ (对齐) 估计得到: scale={scale:.6f}")

#     # 保存变换参数
#     os.makedirs(TRANSFORM_PARAMS_DIR, exist_ok=True)
#     np.savetxt(os.path.join(TRANSFORM_PARAMS_DIR, "R_est_transform.txt"), R_est)
#     np.savetxt(os.path.join(TRANSFORM_PARAMS_DIR, "scale_transform.txt"), np.array([scale]))
#     np.savetxt(os.path.join(TRANSFORM_PARAMS_DIR, "t_est_transform.txt"), t_est)
#     print(f"💾 (对齐) 相似变换参数已保存到 {TRANSFORM_PARAMS_DIR}")

#     # 仅将缩放应用于 3D 点
#     print("🔧 (对齐) 正在将缩放应用于 3D 点...")
#     new_points = {}
#     for pid, point in points3D.items():
#         xyz_scaled = scale * point.xyz # 不应用 R_est 和 t_est
#         new_points[pid] = point._replace(xyz=xyz_scaled)

#     # 不更改图像位姿 (使用原始位姿)
#     # new_images 字典将按原样使用原始图像对象。
#     # 或者，您可以在调用 write_model 时直接传递原始 'images'。
#     # 为清楚起见，此处创建了 new_images，但其内容与原始图像相同。
#     print("ℹ️ (对齐) 相机位姿未更改。将使用原始位姿。")
#     new_images_aligned = {}
#     for img_id, img_data in images.items():
#         new_images_aligned[img_id] = img_data # 复制原始图像数据

#     # 保存新模型 (包含缩放点和原始位姿)
#     print("🔧 (对齐) 正在将新模型保存到目录...")
#     write_model(cameras, new_images_aligned, new_points, OUTPUT_DIR_ALIGN, ext=".bin")
#     print(f"🎯 (对齐) 完成！新的稀疏模型已保存到 {OUTPUT_DIR_ALIGN}")
#     return True # 返回 True 表示成功

# # ========== HLoc 输出转换函数 ==========

# # R_est, scale, t_est 变换参数文件的预期路径
# # 此路径应与 align_colmap_model_and_save_params 中为 TRANSFORM_PARAMS_DIR 设置的路径匹配。
# R_EST_FILE_PATH_CONVERT = os.path.join(TRANSFORM_PARAMS_DIR, "R_est_transform.txt")
# SCALE_FILE_PATH_CONVERT = os.path.join(TRANSFORM_PARAMS_DIR, "scale_transform.txt")
# T_EST_FILE_PATH_CONVERT = os.path.join(TRANSFORM_PARAMS_DIR, "t_est_transform.txt")

# loaded_R_est_transform = None
# loaded_scale_transform = None
# loaded_t_est_transform = None

# def load_transform_parameters():
#     global loaded_R_est_transform, loaded_scale_transform, loaded_t_est_transform
#     if os.path.exists(R_EST_FILE_PATH_CONVERT):
#         try:
#             loaded_R_est_transform = np.loadtxt(R_EST_FILE_PATH_CONVERT)
#             print(f"✅ (转换) 已从 {R_EST_FILE_PATH_CONVERT} 成功加载 R_est 变换矩阵。")
#         except Exception as e:
#             print(f"⚠️ (转换) 警告: 无法从 {R_EST_FILE_PATH_CONVERT} 加载 R_est 变换矩阵: {e}。")
#     else:
#         print(f"⚠️ (转换) 警告: 在 {R_EST_FILE_PATH_CONVERT} 未找到 R_est 变换矩阵文件。")

#     if os.path.exists(SCALE_FILE_PATH_CONVERT):
#         try:
#             loaded_scale_transform = np.loadtxt(SCALE_FILE_PATH_CONVERT).item()
#             print(f"✅ (转换) 已从 {SCALE_FILE_PATH_CONVERT} 成功加载缩放变换因子: {loaded_scale_transform:.6f}。")
#         except Exception as e:
#             print(f"⚠️ (转换) 警告: 无法从 {SCALE_FILE_PATH_CONVERT} 加载缩放变换因子: {e}。")
#     else:
#         print(f"⚠️ (转换) 警告: 在 {SCALE_FILE_PATH_CONVERT} 未找到缩放变换因子文件。")

#     if os.path.exists(T_EST_FILE_PATH_CONVERT):
#         try:
#             loaded_t_est_transform = np.loadtxt(T_EST_FILE_PATH_CONVERT)
#             print(f"✅ (转换) 已从 {T_EST_FILE_PATH_CONVERT} 成功加载 t_est 变换向量。")
#         except Exception as e:
#             print(f"⚠️ (转换) 警告: 无法从 {T_EST_FILE_PATH_CONVERT} 加载 t_est 变换向量: {e}。")
#     else:
#         print(f"⚠️ (转换) 警告: 在 {T_EST_FILE_PATH_CONVERT} 未找到 t_est 变换向量文件。")

# def rotmat2qvec_convert(Rmat): # 将在 convert_quat_pose_to_xyz 中使用的 rotmat2qvec
#     quat_xyzw = R.from_matrix(Rmat).as_quat()
#     return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

# def convert_quat_pose_to_xyz(input_txt, output_poses_file):
#     """
#     将 HLoc 输出位姿转换为新坐标系，并以 "image_name qw qx qy qz pos_x pos_y pos_z" 格式保存。
#     应用保存的 Sim(3) 参数 (scale, R_est, t_est)。
#     """
#     if loaded_R_est_transform is None or loaded_scale_transform is None or loaded_t_est_transform is None:
#         print("❌ (转换) 错误: 相似变换参数未完全加载。无法继续转换。")
#         print("   请先运行 align_colmap_model_and_save_params() 或检查参数文件路径。")
#         return

#     with open(input_txt, 'r') as fin, open(output_poses_file, 'w') as fout:
#         fout.write("# image_name qw qx qy qz pos_x pos_y pos_z\n")
#         for line in fin:
#             parts = line.strip().split()
#             if len(parts) != 8 or line.startswith("#"):
#                 continue
            
#             name, qw_hloc, qx_hloc, qy_hloc, qz_hloc, tx_hloc, ty_hloc, tz_hloc = parts
#             qw_hloc, qx_hloc, qy_hloc, qz_hloc = map(float, [qw_hloc, qx_hloc, qy_hloc, qz_hloc])
#             tx_hloc, ty_hloc, tz_hloc = map(float, [tx_hloc, ty_hloc, tz_hloc])

#             R_hloc_orig = R.from_quat([qx_hloc, qy_hloc, qz_hloc, qw_hloc]).as_matrix()
#             t_hloc_vec = np.array([tx_hloc, ty_hloc, tz_hloc])
#             center_hloc_orig = -R_hloc_orig.T @ t_hloc_vec

#             # 应用相似变换
#             # 1. 变换相机中心: C_new = scale * (R_align @ C_hloc) + t_align
#             center_final = loaded_scale_transform * (loaded_R_est_transform @ center_hloc_orig) + loaded_t_est_transform
            
#             # 2. 变换相机旋转: R_new = R_align @ R_hloc_orig
#             R_final = loaded_R_est_transform @ R_hloc_orig
            
#             q_final_colmap = rotmat2qvec_convert(R_final)
#             q_final_w, q_final_x, q_final_y, q_final_z = q_final_colmap
            
#             fout.write(
#                 f"{name} {q_final_w:.8f} {q_final_x:.8f} {q_final_y:.8f} {q_final_z:.8f} "
#                 f"{center_final[0]:.3f} {center_final[1]:.3f} {center_final[2]:.3f}\n"
#             )
            
#     print(f"✅ (转换) 转换后的相机位姿已保存到 {output_poses_file}")

# # --- 脚本执行流程 ---
# if __name__ == "__main__": # 此部分可以在 hloc 管道的运行函数内部或作为单独的脚本运行。
#     # 步骤 1: 对齐 COLMAP 模型并保存变换参数
#     alignment_successful = align_colmap_model_and_save_params()

#     if alignment_successful:
#         # 步骤 2: 加载保存的变换参数
#         load_transform_parameters()

#         # 步骤 3: 转换 HLoc 输出位姿 (示例调用)
#         # 需要替换为实际的 HLoc 输出文件路径和所需的输出文件路径。
#         hloc_output_file = "outputs/SensorKit_hloc/only_scale/SensorKit_hloc_superpoint+superglue_netvlad20.txt"
#         transformed_poses_output_file = "outputs/SensorKit_hloc/only_scale/SensorKit_query_poses_fully_transformed.txt"
        
#         if os.path.exists(hloc_output_file):
#             convert_quat_pose_to_xyz(
#                 hloc_output_file,
#                 transformed_poses_output_file
#             )
#         else:
#             print(f"❌ (转换) 错误: 未找到 HLoc 输出文件 {hloc_output_file}。")
#     else:
#         print("❌ 模型对齐失败，跳过 HLoc 位姿转换。")





# ... (SensorKit.py 的其余 hloc 管道代码，例如 run(args) 函数和 argparse 部分) ...
# 应在 hloc 的 localize_sfm.main(...) 调用之后调用 convert_quat_pose_to_xyz。
# 例如，在 run(args) 函数内部:
#
#    localize_sfm.main(...)
#
#    # 在 localize_sfm 之后执行转换
#    print("HLoc 定位完成。现在尝试进行位姿转换...")
#    load_transform_parameters() # 假设 localize_sfm 不会更改变换参数
#    hloc_results_file = results # 与 localize_sfm.main 的 results 参数相同
#    transformed_output_file = outputs / f"SensorKit_hloc_superpoint+superglue_netvlad{args.num_loc}_transformed.txt"
#    if os.path.exists(hloc_results_file):
#        convert_quat_pose_to_xyz(hloc_results_file, transformed_output_file)
#    else:
#        logger.error(f"未找到 HLoc 结果文件 {hloc_results_file}，跳过转换。")
#
# `align_colmap_model_and_save_params()` 应在 hloc 管道运行之前运行一次。
# 或者，它可以集成到 hloc 管道中，但通常是一个单独的准备步骤。





# import numpy as np
# from scipy.spatial.transform import Rotation as R

# def convert_quat_pose_to_xyz(input_txt, output_xyz):
#     with open(input_txt, 'r') as fin, open(output_xyz, 'w') as fout:
#         for line in fin:
#             parts = line.strip().split()
#             if len(parts) != 8:
#                 continue  # 跳过无效行
#             name, qw, qx, qy, qz, tx, ty, tz = parts
#             qw, qx, qy, qz = map(float, [qw, qx, qy, qz])
#             tx, ty, tz = map(float, [tx, ty, tz])

#             # 创建旋转矩阵
#             rot = R.from_quat([qx, qy, qz, qw])
#             R_mat = rot.as_matrix()

#             t = np.array([tx, ty, tz])
#             cam_center = -R_mat.T @ t  # 世界坐标下的相机中心

#             # 输出为 CloudCompare 可读的格式
#             fout.write(f"{cam_center[0]:.6f} {cam_center[1]:.6f} {cam_center[2]:.6f}\n")
#     print(f"Camera centers saved to {output_xyz}")


# convert_quat_pose_to_xyz(
#     "outputs/SensorKit_hloc/spatial_sparse_aligned_ecef/SensorKit_hloc_superpoint+superglue_netvlad20.txt",
#     "outputs/SensorKit_hloc/spatial_sparse_aligned_ecef/SensorKit_query_centers.xyz"
# )




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
#     "outputs/SensorKit_hloc/BA_Results/0.5_BA_CAUCHY_ENU_0626_CAM2/SensorKit_hloc_superpoint+superglue_netvlad20.txt",
#     "outputs/SensorKit_hloc/BA_Results/0.5_BA_CAUCHY_ENU_0626_CAM2/SensorKit_query_centers.txt",
#     "outputs/evo_tum/enu_mapping_query/0.5_BA_CAUCHY_ENU_0626_CAM2_SensorKit_query_centers_tum.txt"
# )



# import numpy as np
# from scipy.spatial.transform import Rotation as R
# import os # 用于检查文件是否存在

# # 在 convert_quat_pose_to_xyz 函数定义之前或脚本的早期部分加载 R_est

# # R_est 矩阵文件的期望路径。
# # 此路径应与 colmap_ours.py 中 OUTPUT_DIR 下保存 R_est_transform.txt 的路径一致。
# # 假设 colmap_ours.py 的 OUTPUT_DIR 是 "datasets/SensorKit/colmap_seque_prior/sparse_aligned/1/1_only_scale"
# R_EST_FILE_PATH = "datasets/SensorKit/colmap_seque_prior/sparse_aligned/1/1_only_scale/R_est_transform.txt"
# loaded_R_est_transform = None

# def convert_quat_pose_to_xyz(input_txt, output_xyz, R_transform_matrix=None):
#     with open(input_txt, 'r') as fin, open(output_xyz, 'w') as fout:
#         for line in fin:
#             parts = line.strip().split()
#             if len(parts) != 8:
#                 continue  # 跳过无效行
#             name, qw, qx, qy, qz, tx, ty, tz = parts
#             qw, qx, qy, qz = map(float, [qw, qx, qy, qz])
#             tx, ty, tz = map(float, [tx, ty, tz])

#             # 创建旋转矩阵
#             rot = R.from_quat([qx, qy, qz, qw]) # HLoc 输出的四元数通常是 [qx, qy, qz, qw] for scipy
#             R_mat = rot.as_matrix()

#             t_vec = np.array([tx, ty, tz])
#             cam_center = -R_mat.T @ t_vec  # 在HLoc参考模型坐标系下的相机中心

#             if R_transform_matrix is not None:
#                 # 应用从 colmap_ours.py 加载的 R_est 变换
#                 # 请务必理解此操作的含义（见上面的重要注意事项）
#                 # 此操作假设 cam_center 需要通过 R_transform_matrix 旋转以达到最终坐标系
#                 cam_center = R_transform_matrix @ cam_center
#             else:
#                 print("ℹ️ 未提供 R_transform_matrix，相机中心未进行额外旋转。")
  
#             fout.write(f"{cam_center[0]:.6f} {cam_center[1]:.6f} {cam_center[2]:.6f}\n")
#     print(f"Camera centers saved to {output_xyz}")


# convert_quat_pose_to_xyz(
#     "outputs/SensorKit_hloc/only_scale/SensorKit_hloc_superpoint+superglue_netvlad20.txt",
#     "outputs/SensorKit_hloc/only_scale/SensorKit_query_centers.xyz",
#     R_transform_matrix=loaded_R_est_transform # 将加载的 R_est 传递给函数
# )
