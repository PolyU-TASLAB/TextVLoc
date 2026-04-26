
import argparse
import csv
import glob
import logging
import os
import re
import shutil
import time
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R_scipy

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
from ...pairs_from_retrieval import get_descriptors
from ...utils.io import get_keypoints, get_matches, list_h5_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_mapping_dir_against_sfm(mapping_dir: Path, sfm_dir: Path, max_examples: int = 10):
    """Fail fast if the image folder does not match the reference SfM model."""
    reconstruction = pycolmap.Reconstruction(sfm_dir)
    sfm_image_names = sorted(img.name for img in reconstruction.images.values())
    missing = [name for name in sfm_image_names if not (mapping_dir / name).exists()]
    if not missing:
        return

    examples = "\n".join(f"  - {name}" for name in missing[:max_examples])
    raise FileNotFoundError(
        "The reference SfM model and mapping_dir do not refer to the same image set.\n"
        f"mapping_dir: {mapping_dir}\n"
        f"sfm_dir: {sfm_dir}\n"
        f"Missing {len(missing)} images from mapping_dir; examples:\n{examples}\n"
        "Use the same image root that was used to build the SfM model, or rebuild the SfM "
        "model from the images currently under mapping_dir."
    )



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



# Fix——Opencv
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
#     """
#     为查询图像生成列表，并使用硬编码的相机内参。
    
#     内参来源: cam0 内参（Kalibr 格式转换为 COLMAP OPENCV 格式）
#     - camera_model: pinhole
#     - distortion_model: radtan
#     - resolution: [800, 600]
#     """
#     import glob
#     from pathlib import Path
    
#     # # ✅ 核心修改：使用新的 cam0 内参,目前最好
#     # camera_params = [
#     #     "OPENCV",  # camera_model
#     #     800,   # image_width
#     #     600,   # image_height
#     #     545.4207913448012,   # fx (projection_parameters.fx)
#     #     525.0972476625767,    # fy (projection_parameters.fy)
#     #     398.68865832032634,   # cx (projection_parameters.cx)
#     #     297.07348437586717,  # cy (projection_parameters.cy)
#     #     0.04366182775900429,  # k1 (distortion_parameters.k1)
#     #     -0.08196607013832706, # k2 (distortion_parameters.k2)
#     #     -0.0022363745939624194, # p1 (distortion_parameters.p1)
#     #     -0.0012539861294505836  # p2 (distortion_parameters.p2)
#     # ]


#     # # ✅ 核心修改：使用新的 cam0 内参
#     # camera_params = [
#     #     "OPENCV",  # camera_model
#     #     800,   # image_width
#     #     600,   # image_height
#     #     540.4207913448012,   # fx (projection_parameters.fx)
#     #     521.0972476625767,    # fy (projection_parameters.fy)
#     #     419.68865832032634,   # cx (projection_parameters.cx)
#     #     297.07348437586717,  # cy (projection_parameters.cy)
#     #     0.04366182775900429,  # k1 (distortion_parameters.k1)
#     #     -0.08196607013832706, # k2 (distortion_parameters.k2)
#     #     -0.0022363745939624194, # p1 (distortion_parameters.p1)
#     #     -0.0012539861294505836  # p2 (distortion_parameters.p2)
#     # ]


#     # ✅ 核心修改：使用新的 cam0 内参
#     camera_params = [
#         "OPENCV",  # camera_model
#         800,   # image_width
#         600,   # image_height
#         508.40123289156384,   # fx (projection_parameters.fx)
#         507.7491856816255,    # fy (projection_parameters.fy)
#         393.2454921892426,   # cx (projection_parameters.cx)
#         283.13640678005714,   # cy (projection_parameters.cy)
#         0.03026578931460552,  # k1 (distortion_parameters.k1)
#         -0.06536301126399395, # k2 (distortion_parameters.k2)
#         -0.00312274462694366, # p1 (distortion_parameters.p1)
#         -0.0023545716621999196  # p2 (distortion_parameters.p2)
#     ]


#     # camera_params = [
#     #     "OPENCV",        # ← 关键修正!
#     #     800,
#     #     600,
#     #     554.4207913448012,
#     #     550.0972476625767,
#     #     398.68865832032634,
#     #     297.07348437586717,
#     #     0.04366182775900429,
#     #     -0.08196607013832706,
#     #     -0.0022363745939624194,
#     #     -0.0012539861294505836
#     # ]


#     camera_params = [str(p) for p in camera_params]

#     # ✅ 查找图像（支持 .jpg 和 .png）
#     queries = glob.glob(str(image_dir / "*.jpg")) + \
#               glob.glob(str(image_dir / "*.png"))
#     queries = [Path(q).name for q in sorted(queries)]

#     # ✅ 验证
#     logger.info(f"Found {len(queries)} query images in {image_dir}")
#     if len(queries) == 0:
#         raise FileNotFoundError(f"No images found in {image_dir}!")
    
#     logger.info(f"Sample query images: {queries[:5]}")

#     # ✅ 生成输出
#     out = [[q] + camera_params for q in queries]

#     # ✅ 写入文件
#     with open(path, "w") as f:
#         f.write("\n".join(map(" ".join, out)))

#     logger.info(f"Generated query list with cam0 intrinsics at {path}")





def is_cam1_query_dir(query_dir: Path) -> bool:
    return any(p.lower() == "cam1" for p in query_dir.parts)


def default_cam1_gt_csv(dataset: Path) -> Path:
    return dataset / "1218_1148_CAM_images" / "enu_all_query_gt.csv"


def read_top1_retrieval_pairs(pairs_path: Path) -> dict:
    """pairs 文件中按 C 序遍历时，每个 query 的首个邻居即 NetVLAD top-1。"""
    top1 = {}
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            q, ref = parts[0], parts[1]
            if q not in top1:
                top1[q] = ref
    return top1


def enu_all_query_gt_csv_to_tum(csv_path: Path, out_tum: Path) -> int:
    """enu_all_query_gt.csv -> TUM，ENU 位置 + 四元数 (qx,qy,qz,qw)。"""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = (r.get("image_id") or "").strip()
            if not name:
                continue
            ts = float(r["timestamp_ns"]) / 1e9
            x, y, z = float(r["X_enu"]), float(r["Y_enu"]), float(r["Z_enu"])
            qx, qy, qz, qw = (
                float(r["quat_x"]),
                float(r["quat_y"]),
                float(r["quat_z"]),
                float(r["quat_w"]),
            )
            rows.append((ts, x, y, z, qx, qy, qz, qw))
    rows.sort(key=lambda x: x[0])
    out_tum.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tum, "w", encoding="utf-8") as fo:
        fo.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, x, y, z, qx, qy, qz, qw in rows:
            fo.write(
                f"{ts:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
    return len(rows)


def convert_quat_pose_to_outputs(input_txt: Path, output_txt_with_name: Path, output_tum_txt: Path):
    with open(input_txt, "r", encoding="utf-8") as fin, open(
        output_txt_with_name, "w", encoding="utf-8"
    ) as fout1, open(output_tum_txt, "w", encoding="utf-8") as fout2:
        for line in fin:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            name, qw, qx, qy, qz, tx, ty, tz = parts
            qw, qx, qy, qz = map(float, [qw, qx, qy, qz])
            tx, ty, tz = map(float, [tx, ty, tz])
            rot_c2w = R_scipy.from_quat([qx, qy, qz, qw]).inv()
            quat_c2w = rot_c2w.as_quat()
            R_c2w = rot_c2w.as_matrix()
            t = np.array([tx, ty, tz])
            cam_center = -R_c2w @ t
            fout1.write(
                f"{name} {quat_c2w[0]:.6f} {quat_c2w[1]:.6f} {quat_c2w[2]:.6f} {quat_c2w[3]:.6f} "
                f"{cam_center[0]:.6f} {cam_center[1]:.6f} {cam_center[2]:.6f}\n"
            )
            match = re.search(r"(\d+)", name)
            timestamp = match.group(1) if match else "0"
            fout2.write(
                f"{timestamp} {cam_center[0]:.6f} {cam_center[1]:.6f} {cam_center[2]:.6f} "
                f"{quat_c2w[0]:.6f} {quat_c2w[1]:.6f} {quat_c2w[2]:.6f} {quat_c2w[3]:.6f}\n"
            )
    logger.info("Saved query centers:\n  %s\n  %s", output_txt_with_name, output_tum_txt)


def convert_timestamp_ns_to_s(input_path: Path, output_path: Path):
    output_lines = []
    with open(input_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if line.startswith("#"):
                output_lines.append(line)
                continue
            parts = line.strip().split()
            if len(parts) == 8:
                try:
                    timestamp_ns = float(parts[0])
                    timestamp_s = timestamp_ns / 1e9
                    new_line_parts = [f"{timestamp_s:.6f}"] + [
                        f"{float(p):.6f}" for p in parts[1:]
                    ]
                    output_lines.append(" ".join(new_line_parts) + "\n")
                except ValueError:
                    logger.warning("Could not parse line, skipping: %s", line.strip())
                    output_lines.append(line)
            else:
                output_lines.append(line)
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.writelines(output_lines)
    logger.info("Converted timestamps to seconds: %s", output_path)


def load_tum_file(file_path: Path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split()
                if len(parts) == 8:
                    timestamp = float(parts[0])
                    values = list(map(float, parts[1:]))
                    data.append([timestamp] + values)
    data.sort(key=lambda r: r[0])
    return data


def find_nearest_index(timestamps, target_time):
    idx = bisect_left(timestamps, target_time)
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    before = timestamps[idx - 1]
    after = timestamps[idx]
    return idx - 1 if abs(before - target_time) < abs(after - target_time) else idx


def match_and_filter_nearest(est_data, gt_data, threshold: float):
    gt_timestamps = [row[0] for row in gt_data]
    gt_dict = {row[0]: row[1:4] for row in gt_data}
    filtered = []
    for est_row in est_data:
        est_time = est_row[0]
        est_pos = np.array(est_row[1:4])
        nearest_idx = find_nearest_index(gt_timestamps, est_time)
        gt_time = gt_timestamps[nearest_idx]
        gt_pos = np.array(gt_dict[gt_time])
        dist = float(np.linalg.norm(est_pos - gt_pos))
        if dist <= threshold:
            filtered.append(est_row)
        else:
            logger.info(
                "Frame at %.6f rejected (dist=%.3f m to nearest gt @ %.6f)",
                est_time,
                dist,
                gt_time,
            )
    return np.array(filtered)


def save_tum_trajectory(file_path: Path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(
                "{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*row)
            )


def translation_errors_vs_gt(est_data, gt_data):
    # 不能用 `if not est_data`：numpy 多维数组会触发 truth value ambiguous
    if est_data is None or gt_data is None:
        return np.array([])
    if len(est_data) == 0 or len(gt_data) == 0:
        return np.array([])
    gt_timestamps = [row[0] for row in gt_data]
    errs = []
    for est_row in est_data:
        i = find_nearest_index(gt_timestamps, est_row[0])
        gt_pos = np.array(gt_data[i][1:4])
        est_pos = np.array(est_row[1:4])
        errs.append(float(np.linalg.norm(est_pos - gt_pos)))
    return np.array(errs, dtype=np.float64)


def _gt_unit_tangents_xy(gt_data: list) -> tuple[np.ndarray, np.ndarray]:
    """GT 轨迹在 E-N (tx,ty) 平面上的单位切向 t 与左法向 n（横向有符号：沿 n 为正）。"""
    n = len(gt_data)
    gt_xy = np.array([[row[1], row[2]] for row in gt_data], dtype=np.float64)
    tangents = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        if n == 1:
            d = np.array([1.0, 0.0])
        elif i == 0:
            d = gt_xy[1] - gt_xy[0]
        elif i == n - 1:
            d = gt_xy[-1] - gt_xy[-2]
        else:
            d = gt_xy[i + 1] - gt_xy[i - 1]
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            d = np.array([1.0, 0.0])
            norm = 1.0
        tangents[i] = d / norm
    left_normal = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return tangents, left_normal


def planar_long_lat_errors_vs_gt(est_data, gt_data):
    """
    相对最近时间 GT 点的位置误差在水平面 (E-N) 的模，以及相对 GT 路径切向的分解：
    - horizontal: sqrt(dx^2 + dy^2)，不含高程
    - longitudinal: 沿 GT 轨迹切向的分量（沿迹，有符号）
    - lateral: 沿左法向的分量（横向/偏航向侧向，有符号）
    """
    if est_data is None or gt_data is None:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )
    if len(est_data) == 0 or len(gt_data) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )
    gt_timestamps = [row[0] for row in gt_data]
    tangents, left_n = _gt_unit_tangents_xy(gt_data)
    gt_xy = np.array([[row[1], row[2]] for row in gt_data], dtype=np.float64)

    horiz, lon, lat = [], [], []
    for est_row in est_data:
        i = find_nearest_index(gt_timestamps, est_row[0])
        e = np.array([est_row[1], est_row[2]], dtype=np.float64) - gt_xy[i]
        horiz.append(float(np.linalg.norm(e)))
        lon.append(float(np.dot(e, tangents[i])))
        lat.append(float(np.dot(e, left_n[i])))
    return (
        np.array(horiz, dtype=np.float64),
        np.array(lon, dtype=np.float64),
        np.array(lat, dtype=np.float64),
    )


def _log_planar_error_stats(phase: str, horiz: np.ndarray, lon: np.ndarray, lat: np.ndarray):
    if horiz.size == 0:
        return
    abs_lon = np.abs(lon)
    abs_lat = np.abs(lat)
    logger.info(
        "%s horizontal (E-N) error (m): mean=%.3f median=%.3f max=%.3f rmse=%.3f n=%d",
        phase,
        float(np.mean(horiz)),
        float(np.median(horiz)),
        float(np.max(horiz)),
        float(np.sqrt(np.mean(horiz**2))),
        len(horiz),
    )
    logger.info(
        "%s lateral |cross-track| (m): mean=%.3f median=%.3f max=%.3f rmse_signed=%.3f n=%d",
        phase,
        float(np.mean(abs_lat)),
        float(np.median(abs_lat)),
        float(np.max(abs_lat)),
        float(np.sqrt(np.mean(lat**2))),
        len(lat),
    )
    logger.info(
        "%s longitudinal |along-track| (m): mean=%.3f median=%.3f max=%.3f rmse_signed=%.3f n=%d",
        phase,
        float(np.mean(abs_lon)),
        float(np.median(abs_lon)),
        float(np.max(abs_lon)),
        float(np.sqrt(np.mean(lon**2))),
        len(lon),
    )


def _planar_metrics_dict(
    horiz: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    translation_3d: np.ndarray | None = None,
) -> dict:
    """结构化指标，供 metrics.txt、条形图与文本图共用。"""
    abs_lon = np.abs(lon)
    abs_lat = np.abs(lat)
    m: dict = {}
    if translation_3d is not None and translation_3d.size > 0:
        e = translation_3d.astype(np.float64)
        m["3d"] = {
            "mean": float(np.mean(e)),
            "median": float(np.median(e)),
            "max": float(np.max(e)),
            "rmse": float(np.sqrt(np.mean(e**2))),
            "n": int(e.size),
        }
    m["horiz"] = {
        "mean": float(np.mean(horiz)),
        "median": float(np.median(horiz)),
        "max": float(np.max(horiz)),
        "rmse": float(np.sqrt(np.mean(horiz**2))),
    }
    m["lat"] = {
        "mean_abs": float(np.mean(abs_lat)),
        "median_abs": float(np.median(abs_lat)),
        "max_abs": float(np.max(abs_lat)),
        "rmse_signed": float(np.sqrt(np.mean(lat**2))),
    }
    m["lon"] = {
        "mean_abs": float(np.mean(abs_lon)),
        "median_abs": float(np.median(abs_lon)),
        "max_abs": float(np.max(abs_lon)),
        "rmse_signed": float(np.sqrt(np.mean(lon**2))),
    }
    return m


def _planar_metrics_text(
    horiz: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    translation_3d: np.ndarray | None = None,
    md: dict | None = None,
) -> str:
    """与日志一致的误差摘要（metrics.txt / 文本图）。"""
    if md is None:
        md = _planar_metrics_dict(horiz, lon, lat, translation_3d=translation_3d)
    lines = ["Metrics (vs nearest GT time)", ""]

    if "3d" in md:
        g = md["3d"]
        lines.append("3D translation (m)")
        lines.append(
            f"  mean={g['mean']:.3f}  med={g['median']:.3f}  "
            f"max={g['max']:.3f}  rmse={g['rmse']:.3f}  n={g['n']}"
        )
        lines.append("")

    h = md["horiz"]
    lines.append("Horizontal |e| E–N (m)")
    lines.append(
        f"  mean={h['mean']:.3f}  med={h['median']:.3f}  max={h['max']:.3f}  rmse={h['rmse']:.3f}"
    )
    lines.append("")
    la = md["lat"]
    lines.append("Lateral |cross-track| (m)")
    lines.append(
        f"  mean|.|={la['mean_abs']:.3f}  med|.|={la['median_abs']:.3f}  "
        f"max|.|={la['max_abs']:.3f}  rmse(sgn)={la['rmse_signed']:.3f}"
    )
    lines.append("")
    lo = md["lon"]
    lines.append("Longitudinal |along-track| (m)")
    lines.append(
        f"  mean|.|={lo['mean_abs']:.3f}  med|.|={lo['median_abs']:.3f}  "
        f"max|.|={lo['max_abs']:.3f}  rmse(sgn)={lo['rmse_signed']:.3f}"
    )
    return "\n".join(lines)


def _save_planar_metrics_figures(
    out_dir: Path,
    title_line: str,
    n_frames: int,
    t0: float,
    stats_txt: str,
    md: dict,
    plt,
):
    """metrics 文本排版图 + mean/median/rmse 对比条形图。"""
    out_dir = Path(out_dir)

    fig, ax = plt.subplots(figsize=(10.0, 7.5), dpi=120)
    ax.axis("off")
    header = f"{title_line}\nn_frames={n_frames}  abs_t0={t0:.9f}\n"
    full_txt = header + "\n" + stats_txt
    ax.text(
        0.03,
        0.98,
        full_txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.9, edgecolor="0.45"),
    )
    p_txt = out_dir / "metrics_text_panel.png"
    fig.savefig(p_txt, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    logger.info("Saved: %s", p_txt)

    labels = []
    means = []
    medians = []
    rmses = []
    if "3d" in md:
        labels.append("3D translation")
        means.append(md["3d"]["mean"])
        medians.append(md["3d"]["median"])
        rmses.append(md["3d"]["rmse"])
    labels.extend(["Horizontal E–N", "Lateral |.|", "Longitudinal |.|"])
    h = md["horiz"]
    means.append(h["mean"])
    medians.append(h["median"])
    rmses.append(h["rmse"])
    la = md["lat"]
    means.append(la["mean_abs"])
    medians.append(la["median_abs"])
    rmses.append(la["rmse_signed"])
    lo = md["lon"]
    means.append(lo["mean_abs"])
    medians.append(lo["median_abs"])
    rmses.append(lo["rmse_signed"])

    x = np.arange(len(labels), dtype=np.float64)
    w = 0.24
    fig2, ax2 = plt.subplots(figsize=(9.5, 4.8), dpi=120)
    ax2.bar(x - w, means, width=w, label="mean", color="#1f77b4", alpha=0.88)
    ax2.bar(x, medians, width=w, label="median", color="#ff7f0e", alpha=0.88)
    ax2.bar(x + w, rmses, width=w, label="rmse", color="#2ca02c", alpha=0.88)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=14, ha="right")
    ax2.set_ylabel("Error (m)")
    ax2.set_title("Summary: mean / median / rmse (lateral & longitudinal use mean|.| / rmse signed)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.35)
    ax2.axhline(0.0, color="0.4", lw=0.5)
    fig2.suptitle(title_line, fontsize=10, y=1.02)
    fig2.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    p_bar = out_dir / "metrics_comparison_bars.png"
    fig2.savefig(p_bar, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig2)
    logger.info("Saved: %s", p_bar)


def save_planar_error_outputs(
    times: np.ndarray,
    horiz: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    eval_root: Path,
    stem: str,
    tag: str,
    title_line: str,
    translation_3d: np.ndarray | None = None,
):
    """
    轨迹平面误差：metrics.txt；指标可视化 metrics_text_panel.png、metrics_comparison_bars.png；
    曲线/散点各一张 PNG。输出目录：eval_root / f"{stem}_{tag}" /
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if horiz.size == 0 or len(times) != len(horiz):
        return
    eval_root = Path(eval_root)
    out_dir = eval_root / f"{stem}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_rel = times.astype(np.float64) - float(times[0])
    t0 = float(times[0])
    md = _planar_metrics_dict(horiz, lon, lat, translation_3d=translation_3d)
    stats_txt = _planar_metrics_text(horiz, lon, lat, translation_3d, md=md)

    metrics_path = out_dir / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(title_line + "\n")
        f.write(f"n_frames={len(horiz)}  abs_t0={t0:.9f}\n\n")
        f.write(stats_txt)
        f.write("\n")
    logger.info("Saved planar metrics: %s", metrics_path)
    _save_planar_metrics_figures(out_dir, title_line, len(horiz), t0, stats_txt, md, plt)

    def _save_ts(fig_path: Path, y, ylabel: str, plot_title: str, color: str, zero_line: bool):
        fig, ax = plt.subplots(1, 1, figsize=(9.0, 3.8), dpi=120)
        if zero_line:
            ax.axhline(0.0, color="0.35", lw=0.6)
        ax.plot(t_rel, y, color=color, lw=0.9, alpha=0.92)
        ax.set_xlabel(f"Time from first frame (s); abs t0 = {t0:.6f}")
        ax.set_ylabel(ylabel)
        ax.set_title(plot_title)
        ax.grid(True, alpha=0.35)
        fig.suptitle(title_line, fontsize=10, y=1.02)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
        fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        logger.info("Saved: %s", fig_path)

    _save_ts(
        out_dir / "horizontal_vs_time.png",
        horiz,
        "Horizontal |e| (m)",
        "E–N plane error magnitude",
        "#1f77b4",
        zero_line=False,
    )
    _save_ts(
        out_dir / "lateral_vs_time.png",
        lat,
        "Lateral (m)",
        "Cross-track (+ = left of GT forward)",
        "#ff7f0e",
        zero_line=True,
    )
    _save_ts(
        out_dir / "longitudinal_vs_time.png",
        lon,
        "Longitudinal (m)",
        "Along-track (+ = with GT tangent)",
        "#2ca02c",
        zero_line=True,
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.2), dpi=120)
    sc = ax.scatter(lon, lat, c=t_rel, s=12, cmap="viridis", alpha=0.78, edgecolors="none")
    ax.axhline(0.0, color="0.35", lw=0.5)
    ax.axvline(0.0, color="0.35", lw=0.5)
    ax.set_xlabel("Longitudinal error (m)")
    ax.set_ylabel("Lateral error (m)")
    ax.set_title("Longitudinal vs lateral (color = time from start)")
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, ax=ax, label="Time from start (s)", shrink=0.82)
    fig.suptitle(title_line, fontsize=10, y=1.02)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    p_sc = out_dir / "lateral_vs_longitudinal_scatter.png"
    fig.savefig(p_sc, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    logger.info("Saved: %s", p_sc)
    logger.info("Planar error plots directory: %s", out_dir)


def run_cam1_traj_eval(results_txt: Path, outputs: Path, gt_csv: Path, filter_dist_m: float):
    stem = results_txt.stem
    centers_named = outputs / f"{stem}_query_centers_named.txt"
    tum_ns = outputs / f"{stem}_query_centers_tum_ns.txt"
    tum_s = outputs / f"{stem}_query_centers_tum_s.txt"
    gt_tum = outputs / "enu_all_query_gt_cam1.tum"
    filtered_path = outputs / f"{stem}_query_centers_tum_s_filtered.txt"

    convert_quat_pose_to_outputs(results_txt, centers_named, tum_ns)
    convert_timestamp_ns_to_s(tum_ns, tum_s)
    n_gt = enu_all_query_gt_csv_to_tum(gt_csv, gt_tum)
    logger.info("Wrote %d GT poses to %s", n_gt, gt_tum)

    est_data = load_tum_file(tum_s)
    gt_data = load_tum_file(gt_tum)
    if len(est_data) == 0 or len(gt_data) == 0:
        logger.warning("Trajectory eval skipped: empty est or GT.")
        return
    errs_all = translation_errors_vs_gt(est_data, gt_data)
    logger.info(
        "Pre-filter translation error vs enu_all_query_gt (m): mean=%.3f median=%.3f max=%.3f n=%d",
        float(np.mean(errs_all)),
        float(np.median(errs_all)),
        float(np.max(errs_all)),
        len(errs_all),
    )
    h_all, lon_all, lat_all = planar_long_lat_errors_vs_gt(est_data, gt_data)
    _log_planar_error_stats("Pre-filter", h_all, lon_all, lat_all)
    eval_fig_dir = outputs / "traj_eval_plots"
    t_all = np.array([row[0] for row in est_data], dtype=np.float64)
    save_planar_error_outputs(
        t_all,
        h_all,
        lon_all,
        lat_all,
        eval_fig_dir,
        stem,
        "pre_filter",
        f"{stem}: pre-filter vs enu_all_query_gt (cam1)",
        translation_3d=errs_all,
    )

    filtered = match_and_filter_nearest(est_data, gt_data, filter_dist_m)
    save_tum_trajectory(filtered_path, filtered)
    errs_f = translation_errors_vs_gt(filtered, gt_data)
    if len(errs_f):
        logger.info(
            "Post-filter (%.1fm) translation error (m): mean=%.3f median=%.3f rmse=%.3f n=%d -> %s",
            filter_dist_m,
            float(np.mean(errs_f)),
            float(np.median(errs_f)),
            float(np.sqrt(np.mean(errs_f**2))),
            len(errs_f),
            filtered_path,
        )
        h_f, lon_f, lat_f = planar_long_lat_errors_vs_gt(filtered, gt_data)
        _log_planar_error_stats(f"Post-filter ({filter_dist_m:.1f}m)", h_f, lon_f, lat_f)
        t_f = np.asarray(filtered, dtype=np.float64)[:, 0]
        save_planar_error_outputs(
            t_f,
            h_f,
            lon_f,
            lat_f,
            eval_fig_dir,
            stem,
            "post_filter",
            f"{stem}: post-filter ({filter_dist_m:.1f}m) vs enu_all_query_gt (cam1)",
            translation_3d=errs_f,
        )
    logger.info(
        "Trajectory eval done. Original %d frames, filtered %d.",
        len(est_data),
        len(filtered),
    )


def visualize_top1_retrieval_matches(
    top1_refs: dict,
    query_dir: Path,
    mapping_dir: Path,
    matches_h5: Path,
    query_feats_h5: Path,
    db_feats_h5: Path,
    out_dir: Path,
    max_images: int,
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import ConnectionPatch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "top1_retrieval_match_stats.csv"
    n_ok = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["query", "ref", "n_matches", "mean_score"])
        for qi, (qname, ref_name) in enumerate(top1_refs.items()):
            if qi >= max_images:
                break
            try:
                kpq = get_keypoints(query_feats_h5, qname)
                kpr = get_keypoints(db_feats_h5, ref_name)
                m, scores = get_matches(matches_h5, qname, ref_name)
            except Exception as e:
                logger.warning("Top1 match IO failed %s vs %s: %s", qname, ref_name, e)
                w.writerow([qname, ref_name, -1, ""])
                continue
            n_m = len(m)
            mean_sc = float(np.mean(scores)) if len(scores) else 0.0
            w.writerow([qname, ref_name, n_m, f"{mean_sc:.6f}"])
            qimg = cv2.imread(str(query_dir / qname))
            rimg = cv2.imread(str(mapping_dir / ref_name))
            if qimg is None or rimg is None:
                logger.warning("Missing image for viz: %s or %s", qname, ref_name)
                continue
            fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
            ax[0].imshow(cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB))
            ax[0].set_title(f"Query {qname}\n{n_m} matches (top-1 ref)")
            ax[0].axis("off")
            ax[1].imshow(cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB))
            ax[1].set_title(f"Ref {ref_name}")
            ax[1].axis("off")
            show = min(80, n_m)
            sel = np.linspace(0, n_m - 1, show, dtype=int) if n_m else []
            for j in sel:
                i0, i1 = int(m[j, 0]), int(m[j, 1])
                line = ConnectionPatch(
                    xyA=kpq[i0],
                    coordsA=ax[0].transData,
                    xyB=kpr[i1],
                    coordsB=ax[1].transData,
                    color="lime",
                    linewidth=0.6,
                    alpha=0.7,
                )
                fig.add_artist(line)
            safe = f"{qname.replace('/', '_')}__{ref_name.replace('/', '_')}.png"
            plt.savefig(out_dir / safe, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            n_ok += 1
    logger.info(
        "Top1 retrieval match viz: %d figures under %s; stats %s",
        n_ok,
        out_dir,
        csv_path,
    )


def run(args):
    print("Starting SensorKit pipeline...")
    logger.info("Starting pipeline...")

    if getattr(args, "eval_only", False):
        dataset = args.dataset
        outputs = args.outputs
        outputs.mkdir(exist_ok=True, parents=True)
        results = (
            args.results
            if getattr(args, "results", None) is not None
            else outputs / f"SensorKit_hloc_superpoint+lightglue_netvlad{args.num_loc}.txt"
        )
        if not results.is_file():
            raise FileNotFoundError(f"--eval-only: localization results not found: {results}")
        gt_csv = Path(args.gt_csv) if args.gt_csv else default_cam1_gt_csv(dataset)
        if not gt_csv.is_file():
            raise FileNotFoundError(f"--eval-only: GT CSV not found: {gt_csv}")
        run_cam1_traj_eval(results, outputs, gt_csv, args.cam1_filter_dist_m)
        logger.info("Eval-only finished.")
        return

    # 设置路径
    dataset = args.dataset
    # dataset_test = args.dataset_test
    # mapping_dir = dataset / "IMAGES"
    mapping_dir = dataset/ "1218_1148_CAM_images"  
    # mapping_dir = dataset/ "1218CAM"  
    # query_dir = dataset / "query/cam1_all/0_undistorted" 
    # query_dir = dataset / "query/412bag_1s/" 
    # query_dir = dataset / "query/test_t/undistorted/" 
    # query_dir = dataset / "query/clip_one/"  
    query_dir = dataset / "query/1218c3outCarHeng_jpg/"  
    # query_dir = dataset / "cam1/"  
    # query_dir = dataset / "query/cam2_mapping/undistorted/"
    # query_dir = dataset / "query/cam2_all/undistorted/" 
    # query_dir = dataset_test / "mapping/jpg/" 
    # sift_sfm = dataset / "colmap_seque_prior/sparse_aligned/1/only_scale"
    # sift_sfm = dataset / "new_colmap_spatial_intrinsics/sparse_aligned_ecef_4"
    # sift_sfm = dataset / "new_colmap_exhaustive_fixing_intrinsics/sparse/0"
    # sift_sfm = dataset / "vocab_tree_0630_OPENCV/sparse/1/0.05_BA_ENU"
    # sift_sfm = dataset / "allcams_spatial_colmap_0630/sparse/0/0.025_BA_ENU"
    sift_sfm = dataset / "allcams_spatial_colmap_1218_1148_CAM_images/sparse/2/0.025_BA_ENU_fix_intrinsic"
    # sift_sfm = dataset / "colmap_seque/sparse_aligned_enu"
    logger.info(f"Dataset path: {dataset}")
    logger.info(f"Mapping images path: {mapping_dir}")
    logger.info(f"Query images path: {query_dir}")

    outputs = args.outputs  # 输出目录
    outputs.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {outputs}")
    outputsba = outputs / "online_tmp"


    query_list = outputs / "queries_with_intrinsics.txt"
    # outputsba  = outputs / "cam1"  # SfM 输出目录
    # outputsba  = outputs / "1218c3outCarHeng"  # SfM 输出目录
    # outputsba  = outputs / "412_one"  # SfM 输出目录
    # outputsba  = outputs / "412bag_1s"  # SfM 输出目录
    # reference_sfm = outputs / "sfm_superpoint+superglue/0.5_BA_CAUCHY_ENU_0626" 
    # reference_sfm = outputs / "sfm_superpointmax+lightglue"
    reference_sfm_path = outputs / "sfm_superpointmax+lightglue"
    reference_sfm = outputs / "sfm_superpointmax+lightglue" 
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    loc_pairs = outputs / f"pairs-query-netvlad{args.num_loc}.txt"
    results = outputs / f"SensorKit_hloc_superpoint+lightglue_netvlad{args.num_loc}.txt"

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
    
    
    # logger.info("...Models pre-loaded and ready.")
    # # --- 预加载结束 ---

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

    # --- 清理上一次的在线定位结果 ---
    if outputsba.exists():
        logger.warning(f"Cleaning up previous online localization results in {outputsba}...")
        shutil.rmtree(outputsba)
    outputsba.mkdir(exist_ok=True, parents=True)
    # --- 清理结束 ---


    # 在这里提前加载 3D 模型 ---
    logger.info(f"Loading the 3D reference model from {reference_sfm}...") # <--- 使用新路径变量
    reference_sfm = pycolmap.Reconstruction(reference_sfm) # <--- 加载模型
    logger.info("...3D model loaded.")

    logger.info("Extracting database image names from the pre-loaded model...")
    db_names_from_model = [img.name for img in reference_sfm.images.values()]
    logger.info(f"...Found {len(db_names_from_model)} image names.")
    # --- 提取结束 ---


    # # ### OFFLINE

    # validate_mapping_dir_against_sfm(mapping_dir, sift_sfm)

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
    #     reference_sfm_path,  # 输出的参考稀疏模型
    #     sift_sfm,       # 现有的SIFT稀疏模型
    #     mapping_dir,    # 输入图像目录
    #     sfm_pairs,      # 图像对文件
    #     features_db,       # 提取的特征
    #     sfm_matches,    # 特征匹配结果
    # )

    # global_descriptors_db = extract_features.main(retrieval_conf, mapping_dir, outputs)




    # # --- 2. 在这里预加载所有在线阶段需要的模型 ---
    # logger.info("Pre-loading models for the online stage...")
    # device = "cuda" if torch.cuda.is_available() else "cpu" 
    
    # # 加载全局特征模型 (NetVLAD)
    # GlobalModel = extract_features.dynamic_load(extractors, retrieval_conf["model"]["name"])
    # global_model = GlobalModel(retrieval_conf["model"]).eval().to(device)

    # # 新增：加载局部特征匹配模型
    # MatcherModel = extract_features.dynamic_load(matchers, matcher_conf['model']['name'])
    # local_matcher_model = MatcherModel(matcher_conf['model']).eval().to(device)
    
    
    # # logger.info("...Models pre-loaded and ready.")
    # # # --- 预加载结束 ---

    # # --- 3. 在这里提前构建特征文件的元数据索引 ---
    # logger.info("Pre-computing feature file metadata...")
    # db_global_features_path = outputs / (retrieval_conf['output'] + '.h5')

    # # 假设数据库特征只在一个文件中
    # db_descriptors_list = [db_global_features_path]
    
    # precomputed_db_metadata = {
    #     'name2db': {n: i for i, p in enumerate(db_descriptors_list) for n in list_h5_names(p)},
    #     'db_names_h5': list_h5_names(db_global_features_path),
    # }
    # logger.info("...Metadata pre-computed.")
    # # --- 预计算结束 ---
    

    # # --- 2. 在这里提前加载数据库的全局特征 ---
    # logger.info("Pre-loading database global descriptors from HDF5...")
    # t_load_db_desc = time.time()
    # preloaded_db_desc = get_descriptors(
    #     precomputed_db_metadata['db_names_h5'], 
    #     db_descriptors_list, 
    #     precomputed_db_metadata['name2db']
    # )
    # logger.info(f"...Database descriptors loaded in {time.time() - t_load_db_desc:.2f}s.")
    # # --- 预加载结束 ---

    # # # --- 清理上一次的在线定位结果 ---
    # # if outputsba.exists():
    # #     logger.warning(f"Cleaning up previous online localization results in {outputsba}...")
    # #     shutil.rmtree(outputsba)
    # # outputsba.mkdir(exist_ok=True, parents=True)
    # # # --- 清理结束 ---


    # # 在这里提前加载 3D 模型 ---
    # logger.info(f"Loading the 3D reference model from {reference_sfm}...") # <--- 使用新路径变量
    # reference_sfm = pycolmap.Reconstruction(reference_sfm) # <--- 加载模型
    # logger.info("...3D model loaded.")

    # logger.info("Extracting database image names from the pre-loaded model...")
    # db_names_from_model = [img.name for img in reference_sfm.images.values()]
    # logger.info(f"...Found {len(db_names_from_model)} image names.")
    # # --- 提取结束 ---

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
    query_global_feats_path = outputs / "query-global-feats--.h5"
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
    db_global_features = outputs / (retrieval_conf['output'] + '.h5')
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
    query_local_feats_path = outputs / "query-local-feats.h5"
    features = extract_features.main(
        feature_conf, query_dir, feature_path=query_local_feats_path, as_half=True
    )
    timings['local_feature_extraction'] = time.time() - t0
    # 步骤 4: 局部特征匹配
    t0 = time.time()
    db_local_features = outputs / (feature_conf['output'] + '.h5')
    loc_matches_path = outputs / (matcher_conf['output'] + '.h5')
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

    if not getattr(args, "skip_post_eval", False):
        top1_refs = read_top1_retrieval_pairs(loc_pairs)
        if top1_refs and args.max_top1_vis > 0:
            visualize_top1_retrieval_matches(
                top1_refs,
                query_dir,
                mapping_dir,
                loc_matches,
                query_local_feats_path,
                db_local_features,
                outputs / "eval_top1_retrieval",
                max_images=args.max_top1_vis,
            )
        gt_csv = Path(args.gt_csv) if args.gt_csv else default_cam1_gt_csv(dataset)
        if gt_csv.is_file():
            if not is_cam1_query_dir(query_dir):
                logger.info(
                    "Query 路径未匹配 cam1 目录名，仍对 GT 做轨迹评估（请确认与 query 是否同一套真值）: %s",
                    gt_csv,
                )
            run_cam1_traj_eval(results, outputs, gt_csv, args.cam1_filter_dist_m)
        else:
            logger.warning("真值 CSV 未找到，跳过轨迹评估: %s（可用 --gt-csv 指定）", gt_csv)




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
        # default="outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025",
        default="outputs/SensorKit_hloc/allcams_spatial_colmap_1218CAM_1148",
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
        default=5, # Number of image pairs for loc, default: %(default)s", 20
        help="Number of image pairs for loc, default: %(default)s",
    )
    parser.add_argument(
        "--gt-csv",
        type=str,
        default=None,
        help="轨迹评估用真值 CSV；默认 dataset/.../20251218_phone_hkstp_enu.txt（与 query 是否 cam1 无关）",
    )
    parser.add_argument(
        "--cam1-filter-dist-m",
        type=float,
        default=10.0,
        help="cam1 轨迹与 GT 最近邻时间对齐后，按位置距离(m)过滤离群帧",
    )
    parser.add_argument(
        "--max-top1-vis",
        type=int,
        default=0,
        help="query vs NetVLAD top-1 参考图匹配可视化数量上限，0 关闭",
    )
    parser.add_argument(
        "--skip-post-eval",
        action="store_true",
        help="跳过 top1 匹配可视化与 enu_all_query_gt 轨迹评估（见 --gt-csv）",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="不跑定位：仅用已有 hloc 结果 txt 转 TUM 并对 cam1 的 enu_all_query_gt 做平移误差统计（需 --outputs 与结果文件）",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="与 --eval-only 连用：指定定位结果 txt；默认 outputs/SensorKit_hloc_superpoint+lightglue_netvlad{num_loc}.txt",
    )
    args = parser.parse_args()
    run(args)
