import argparse
import csv
import hashlib
import logging
import re
import time
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation as R_scipy
from tqdm import tqdm

from ... import (
    extract_features,
    localize_sfm,
    match_features,
    extract_clip_features,
    generate_captions,
    matchers,
    pairs_from_clip_retrieval,
    pycolmap,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- 从查询 CSV 加载真值 ---
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


def default_query_gt_csv(dataset: Path) -> Path:
    return dataset / "0620_gt" / "enu_all_query_gt.csv"


def enu_all_query_gt_csv_to_tum(csv_path: Path, out_tum: Path) -> int:
    """Convert SensorKit ENU GT CSV to TUM format: timestamp tx ty tz qx qy qz qw."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp_s = float(row["timestamp_ns"]) / 1e9
            rows.append(
                (
                    timestamp_s,
                    float(row["X_enu"]),
                    float(row["Y_enu"]),
                    float(row["Z_enu"]),
                    float(row["quat_x"]),
                    float(row["quat_y"]),
                    float(row["quat_z"]),
                    float(row["quat_w"]),
                )
            )
    rows.sort(key=lambda r: r[0])
    out_tum.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tum, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for row in rows:
            f.write(
                f"{row[0]:.9f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} "
                f"{row[4]:.6f} {row[5]:.6f} {row[6]:.6f} {row[7]:.6f}\n"
            )
    return len(rows)


def convert_hloc_result_to_query_centers(
    input_txt: Path, output_txt_with_name: Path, output_tum_txt: Path
):
    """
    hloc output is world-to-camera pose: image qw qx qy qz tx ty tz.
    Convert it to camera center in world/ENU coordinates.
    """
    with open(input_txt, "r", encoding="utf-8") as fin, open(
        output_txt_with_name, "w", encoding="utf-8"
    ) as fout_named, open(output_tum_txt, "w", encoding="utf-8") as fout_tum:
        for line in fin:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            name, qw, qx, qy, qz, tx, ty, tz = parts
            qw, qx, qy, qz = map(float, [qw, qx, qy, qz])
            tvec = np.array(list(map(float, [tx, ty, tz])), dtype=np.float64)
            rot_c2w = R_scipy.from_quat([qx, qy, qz, qw]).inv()
            quat_c2w = rot_c2w.as_quat()
            center = -rot_c2w.as_matrix() @ tvec

            fout_named.write(
                f"{name} {quat_c2w[0]:.6f} {quat_c2w[1]:.6f} "
                f"{quat_c2w[2]:.6f} {quat_c2w[3]:.6f} "
                f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n"
            )
            match = re.search(r"(\d+)", name)
            timestamp_s = (float(match.group(1)) / 1e9) if match else 0.0
            fout_tum.write(
                f"{timestamp_s:.9f} {center[0]:.6f} {center[1]:.6f} {center[2]:.6f} "
                f"{quat_c2w[0]:.6f} {quat_c2w[1]:.6f} "
                f"{quat_c2w[2]:.6f} {quat_c2w[3]:.6f}\n"
            )
    logger.info("Saved query centers:\n  %s\n  %s", output_txt_with_name, output_tum_txt)


def convert_query_center_result_to_tum(
    input_txt: Path, output_txt_with_name: Path, output_tum_txt: Path
):
    """
    Text-only localization output stores tx ty tz as the ENU camera center already.
    Convert directly to TUM without applying the HLOC world-to-camera transform.
    """
    with open(input_txt, "r", encoding="utf-8") as fin, open(
        output_txt_with_name, "w", encoding="utf-8"
    ) as fout_named, open(output_tum_txt, "w", encoding="utf-8") as fout_tum:
        for line in fin:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            name, qw, qx, qy, qz, tx, ty, tz = parts
            center = np.array(list(map(float, [tx, ty, tz])), dtype=np.float64)
            match = re.search(r"(\d+)", name)
            timestamp_s = (float(match.group(1)) / 1e9) if match else 0.0

            fout_named.write(
                f"{name} {float(qx):.6f} {float(qy):.6f} "
                f"{float(qz):.6f} {float(qw):.6f} "
                f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n"
            )
            fout_tum.write(
                f"{timestamp_s:.9f} {center[0]:.6f} {center[1]:.6f} {center[2]:.6f} "
                f"{float(qx):.6f} {float(qy):.6f} "
                f"{float(qz):.6f} {float(qw):.6f}\n"
            )
    logger.info("Saved query centers:\n  %s\n  %s", output_txt_with_name, output_tum_txt)


def load_tum_file(file_path: Path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 8:
                data.append([float(parts[0])] + list(map(float, parts[1:])))
    data.sort(key=lambda row: row[0])
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


def translation_errors_vs_gt(est_data, gt_data):
    if est_data is None or gt_data is None or len(est_data) == 0 or len(gt_data) == 0:
        return np.array([], dtype=np.float64)
    gt_timestamps = [row[0] for row in gt_data]
    errors = []
    for est_row in est_data:
        idx = find_nearest_index(gt_timestamps, est_row[0])
        gt_pos = np.array(gt_data[idx][1:4], dtype=np.float64)
        est_pos = np.array(est_row[1:4], dtype=np.float64)
        errors.append(float(np.linalg.norm(est_pos - gt_pos)))
    return np.array(errors, dtype=np.float64)


def _gt_unit_tangents_xy(gt_data):
    gt_xy = np.array([[row[1], row[2]] for row in gt_data], dtype=np.float64)
    tangents = np.zeros_like(gt_xy)
    for i in range(len(gt_xy)):
        if len(gt_xy) == 1:
            direction = np.array([1.0, 0.0])
        elif i == 0:
            direction = gt_xy[1] - gt_xy[0]
        elif i == len(gt_xy) - 1:
            direction = gt_xy[-1] - gt_xy[-2]
        else:
            direction = gt_xy[i + 1] - gt_xy[i - 1]
        norm = float(np.linalg.norm(direction))
        tangents[i] = direction / norm if norm > 1e-9 else np.array([1.0, 0.0])
    left_normal = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return tangents, left_normal


def planar_long_lat_errors_vs_gt(est_data, gt_data):
    if est_data is None or gt_data is None or len(est_data) == 0 or len(gt_data) == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty
    gt_timestamps = [row[0] for row in gt_data]
    tangents, left_normal = _gt_unit_tangents_xy(gt_data)
    gt_xy = np.array([[row[1], row[2]] for row in gt_data], dtype=np.float64)
    horiz, lon, lat = [], [], []
    for est_row in est_data:
        idx = find_nearest_index(gt_timestamps, est_row[0])
        error_xy = np.array(est_row[1:3], dtype=np.float64) - gt_xy[idx]
        horiz.append(float(np.linalg.norm(error_xy)))
        lon.append(float(np.dot(error_xy, tangents[idx])))
        lat.append(float(np.dot(error_xy, left_normal[idx])))
    return (
        np.array(horiz, dtype=np.float64),
        np.array(lon, dtype=np.float64),
        np.array(lat, dtype=np.float64),
    )


def save_tum_trajectory(file_path: Path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(
                "{:.9f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    *row
                )
            )


def match_and_filter_nearest(est_data, gt_data, threshold_m: float):
    if len(est_data) == 0 or len(gt_data) == 0:
        return np.array([], dtype=np.float64)
    gt_timestamps = [row[0] for row in gt_data]
    filtered = []
    for est_row in est_data:
        idx = find_nearest_index(gt_timestamps, est_row[0])
        gt_pos = np.array(gt_data[idx][1:4], dtype=np.float64)
        est_pos = np.array(est_row[1:4], dtype=np.float64)
        if float(np.linalg.norm(est_pos - gt_pos)) <= threshold_m:
            filtered.append(est_row)
    return np.array(filtered, dtype=np.float64)


def write_traj_eval_metrics(
    metrics_path: Path,
    title: str,
    translation_3d: np.ndarray,
    horizontal: np.ndarray,
    longitudinal: np.ndarray,
    lateral: np.ndarray,
):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        if translation_3d.size:
            f.write(
                "3D translation (m): "
                f"mean={np.mean(translation_3d):.3f} "
                f"median={np.median(translation_3d):.3f} "
                f"max={np.max(translation_3d):.3f} "
                f"rmse={np.sqrt(np.mean(translation_3d**2)):.3f} "
                f"n={len(translation_3d)}\n"
            )
        if horizontal.size:
            f.write(
                "Horizontal E-N (m): "
                f"mean={np.mean(horizontal):.3f} "
                f"median={np.median(horizontal):.3f} "
                f"max={np.max(horizontal):.3f} "
                f"rmse={np.sqrt(np.mean(horizontal**2)):.3f}\n"
            )
            f.write(
                "Lateral |cross-track| (m): "
                f"mean_abs={np.mean(np.abs(lateral)):.3f} "
                f"median_abs={np.median(np.abs(lateral)):.3f} "
                f"max_abs={np.max(np.abs(lateral)):.3f} "
                f"rmse_signed={np.sqrt(np.mean(lateral**2)):.3f}\n"
            )
            f.write(
                "Longitudinal |along-track| (m): "
                f"mean_abs={np.mean(np.abs(longitudinal)):.3f} "
                f"median_abs={np.median(np.abs(longitudinal)):.3f} "
                f"max_abs={np.max(np.abs(longitudinal)):.3f} "
                f"rmse_signed={np.sqrt(np.mean(longitudinal**2)):.3f}\n"
            )


def run_traj_eval(
    results_txt: Path,
    outputs: Path,
    gt_csv: Path,
    filter_dist_m: float,
    results_are_query_centers: bool = False,
):
    stem = results_txt.stem
    eval_dir = outputs / "traj_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    centers_named = eval_dir / f"{stem}_query_centers_named.txt"
    centers_tum = eval_dir / f"{stem}_query_centers_tum.txt"
    gt_tum = eval_dir / "enu_all_query_gt.tum"
    filtered_tum = eval_dir / f"{stem}_query_centers_tum_filtered_{filter_dist_m:.1f}m.txt"

    if results_are_query_centers:
        convert_query_center_result_to_tum(results_txt, centers_named, centers_tum)
    else:
        convert_hloc_result_to_query_centers(results_txt, centers_named, centers_tum)
    n_gt = enu_all_query_gt_csv_to_tum(gt_csv, gt_tum)
    logger.info("Wrote %d GT poses to %s", n_gt, gt_tum)

    est_data = load_tum_file(centers_tum)
    gt_data = load_tum_file(gt_tum)
    if len(est_data) == 0 or len(gt_data) == 0:
        logger.warning("Trajectory eval skipped: empty estimated trajectory or GT.")
        return

    errors = translation_errors_vs_gt(est_data, gt_data)
    horizontal, longitudinal, lateral = planar_long_lat_errors_vs_gt(est_data, gt_data)
    logger.info(
        "Trajectory error vs GT (m): mean=%.3f median=%.3f max=%.3f rmse=%.3f n=%d",
        float(np.mean(errors)),
        float(np.median(errors)),
        float(np.max(errors)),
        float(np.sqrt(np.mean(errors**2))),
        len(errors),
    )
    metrics_path = eval_dir / f"{stem}_metrics.txt"
    write_traj_eval_metrics(
        metrics_path,
        f"{stem}: localization result vs nearest GT timestamp",
        errors,
        horizontal,
        longitudinal,
        lateral,
    )

    filtered = match_and_filter_nearest(est_data, gt_data, filter_dist_m)
    save_tum_trajectory(filtered_tum, filtered)
    filtered_errors = translation_errors_vs_gt(filtered, gt_data)
    if filtered_errors.size:
        logger.info(
            "Post-filter (%.1fm) translation error (m): mean=%.3f median=%.3f rmse=%.3f n=%d",
            filter_dist_m,
            float(np.mean(filtered_errors)),
            float(np.median(filtered_errors)),
            float(np.sqrt(np.mean(filtered_errors**2))),
            len(filtered_errors),
        )
    logger.info("Trajectory eval outputs saved under %s", eval_dir)


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


def generate_query_list(_dataset, image_dir, path):
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


# ============================================================
# Region-aware reranking (no query image)
#   - Use HF CLIP vision patch tokens -> grid region pooling
#   - Score(candidate) = max(region·text) or top-m mean
#   - Combine with original retrieval score if desired
# ============================================================

def split_text_to_phrases(text, method="sentence", max_length=77):
    """
    将长文本拆分为多个短文本片段。
    
    Args:
        text: 原始文本
        method: 拆分方法
            - "sentence": 按句子拆分（正则表达式）
            - "clause": 按逗号拆分
            - "window": 滑动窗口
            - "none": 不拆分
        max_length: 每个片段的最大字符数（CLIP 限制 77）
    
    Returns:
        List[str]: 短文本片段列表
    """
    # ============================================================
    # 方法 1: 正则表达式句子分割（默认）
    # ============================================================
    if method == "sentence":
        phrases = re.split(r'[.!?。！？]', text)
        phrases = [p.strip() for p in phrases if p.strip()]
        
        final_phrases = []
        for phrase in phrases:
            if len(phrase) > max_length:
                sub_phrases = [p.strip() for p in phrase.split(',') if p.strip()]
                final_phrases.extend(sub_phrases)
            else:
                final_phrases.append(phrase)
        
        return final_phrases
    
    # ============================================================
    # 方法 2: 按逗号拆分
    # ============================================================
    elif method == "clause":
        phrases = re.split(r'[,;，；]', text)
        return [p.strip() for p in phrases if p.strip()]
    
    # ============================================================
    # 方法 3: 滑动窗口
    # ============================================================
    elif method == "window":
        words = text.split()
        window_size = 10
        step = 5
        phrases = []
        for i in range(0, len(words), step):
            phrase = ' '.join(words[i:i+window_size])
            if phrase.strip():
                phrases.append(phrase.strip())
        return phrases
    
    # ============================================================
    # 默认：不拆分
    # ============================================================
    else:
        return [text]

def split_image_to_grid_patches(img: Image.Image, grid: int):
    """
    img: PIL.Image
    grid: e.g. 3 or 4
    return: List[PIL.Image]
    """
    W, H = img.size
    patches = []

    w_step = W // grid
    h_step = H // grid

    for gy in range(grid):
        for gx in range(grid):
            left   = gx * w_step
            upper  = gy * h_step
            right  = W if gx == grid - 1 else (gx + 1) * w_step
            lower  = H if gy == grid - 1 else (gy + 1) * h_step
            patch = img.crop((left, upper, right, lower))
            patches.append(patch)

    return patches


@torch.no_grad()
def _encode_text_any(clip_model, clip_processor, texts, device):
    """
    兼容多种模型的文本编码。
    """
    # --- Case A: CLIP / SigLIP / MetaCLIP ---
    if hasattr(clip_model, "get_text_features"):
        inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        features = clip_model.get_text_features(**inputs)
    
    # --- Case B: BLIP ---
    elif hasattr(clip_model, "generate"):
        # BLIP 需要先编码文本
        inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        features = clip_model.get_text_features(**inputs) if hasattr(clip_model, "get_text_features") else \
                   clip_model.text_encoder(**inputs).last_hidden_state[:, 0, :]
    
    # --- Case C: ViLT ---
    elif hasattr(clip_model, "pooler"):
        # ViLT 需要图像输入（使用黑色占位符）
        dummy_img = Image.new('RGB', (224, 224), color='black')
        inputs = clip_processor(images=[dummy_img]*len(texts), text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = clip_model(**inputs)
        features = outputs.pooler_output
    
    # --- Case D: OpenCLIP ---
    elif hasattr(clip_model, "encode_text"):
        toks = clip_processor(texts).to(device)
        features = clip_model.encode_text(toks)
    
    else:
        raise TypeError(f"Unsupported model type: {type(clip_model)}")
    
    return F.normalize(features.float(), dim=-1)



@torch.no_grad()
def _encode_image_regions_any(
    clip_model,
    clip_processor,
    image_patches,   # List[PIL.Image] or Tensor [N,3,H,W]
    preprocess,      # open_clip 的 preprocess（非常关键）
    device,
):
    """
    返回: (N, D) normalized image embeddings
    """
    # -------- open_clip --------
    if hasattr(clip_model, "encode_image"):
        if isinstance(image_patches[0], torch.Tensor):
            imgs = torch.stack(image_patches).to(device)
        else:
            imgs = torch.stack([preprocess(im) for im in image_patches]).to(device)

        feats = clip_model.encode_image(imgs)

    # -------- HuggingFace CLIP --------
    elif hasattr(clip_model, "get_image_features"):
        inputs = clip_processor(images=image_patches, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = clip_model.get_image_features(**inputs)

    else:
        raise TypeError(f"Unknown CLIP model type: {type(clip_model)}")

    return F.normalize(feats.float(), dim=-1)


@torch.no_grad()
def _region_text_score(regions, text_emb, top_m=2, reduce="topm_mean"):
    """
    regions: [R,D] normalized
    text_emb: [D] normalized
    return scalar score
    """
    sims = regions @ text_emb  # [R]
    if reduce == "max":
        return sims.max().item()
    # default: top-m mean
    m = min(top_m, sims.numel())
    return sims.topk(m).values.mean().item()


@torch.no_grad()
def rerank_loc_pairs_region(
    loc_pairs_in,
    loc_pairs_out,
    prompts_csv,
    image_root_dir,
    clip_model,
    clip_processor,
    preprocess,
    device="cuda",
    rerank_top_k=20,
    grid=3,
    top_m=1,
    reduce="max",
    alpha=0.75,
    region_pool="mean",
    missing_ok=True,
    # ✅ 召回率计算参数
    positive_pairs_csv=None,
    query_gt_csv=None,
    db_poses_paths=None,
    distance_thresholds=[5.0, 10.0, 15.0],  # ← 多阈值
):
    """Grid-based region reranking with multi-threshold evaluation."""
    
    # ============================================================
    # ✅ 修复: 在函数开头统一导入依赖
    # ============================================================
    import pandas as pd
    import csv
    
    df = pd.read_csv(prompts_csv)
    q2text = dict(zip(df["image_name"].astype(str), df["description"].astype(str)))

    def _parse_line(line):
        parts = line.strip().split()
        q = parts[0]
        dbs = parts[1::2]
        scs = [float(x) for x in parts[2::2]]
        return q, dbs, scs

    def _format_line(q, dbs, scs):
        out = [q]
        for n, s in zip(dbs, scs):
            out += [n, f"{s:.6f}"]
        return " ".join(out)

    clip_model.eval()

    with open(loc_pairs_in, "r") as f:
        lines = f.readlines()

    new_lines = []
    
    # ============================================================
    # ✅ 修复: 初始化多阈值统计
    # ============================================================
    retrieval_stats = {k: 0 for k in [1, 5, 10, 20]}
    localization_stats_multi = {
        threshold: {k: 0 for k in [1, 5, 10, 20]}
        for threshold in distance_thresholds
    }
    total_queries = 0
    
    # ============================================================
    # ✅ 修复 1: 加载并标准化正样本对
    # ============================================================
    positive_pairs = {}
    if positive_pairs_csv and positive_pairs_csv.exists():
        logger.info(f"📂 Loading positive pairs from: {positive_pairs_csv}")
        df_pos = pd.read_csv(positive_pairs_csv)
        logger.info(f"   CSV shape: {df_pos.shape}")
        logger.info(f"   CSV columns: {df_pos.columns.tolist()}")
        
        for _, row in df_pos.iterrows():
            # ✅ 标准化 anchor (移除 '_all/0_undistorted' 部分)
            anchor = str(row['anchor'])
            # 例如: "cam1_all/0_undistorted/1740554132806470102.jpg" → "cam1/1740554132806470102.jpg"
            if 'cam1_all/0_undistorted/' in anchor:
                anchor_normalized = anchor.replace('cam1_all/0_undistorted/', 'cam1/')
            else:
                # 如果格式未知,尝试提取 camX/timestamp.jpg
                timestamp = anchor.split('/')[-1]  # 提取文件名
                cam_id = 'cam1'  # 默认假设 cam1
                anchor_normalized = f"{cam_id}/{timestamp}"
            
            # ✅ 标准化 positive (移除 'SensorKit/allcams/' 前缀)
            positive = str(row['positive'])
            # 例如: "SensorKit/allcams/cam1/1740554132806470102.jpg" → "cam1/1740554132806470102.jpg"
            if 'SensorKit/allcams/' in positive:
                positive_normalized = positive.replace('SensorKit/allcams/', '')
            else:
                # 如果格式未知,尝试提取 camX/timestamp.jpg
                parts = positive.split('/')
                if len(parts) >= 2:
                    positive_normalized = f"{parts[-2]}/{parts[-1]}"  # camX/timestamp.jpg
                else:
                    positive_normalized = positive
            
            # 存储标准化后的映射
            if anchor_normalized not in positive_pairs:
                positive_pairs[anchor_normalized] = []
            positive_pairs[anchor_normalized].append(positive_normalized)
        
        logger.info(f"✅ Loaded {len(positive_pairs)} anchor-positive pairs (normalized).")
        
        # 打印前 3 个样例 (标准化后)
        sample_keys = list(positive_pairs.keys())[:3]
        for key in sample_keys:
            logger.info(f"   Sample (normalized): {key} → {positive_pairs[key][:2]}")
    else:
        logger.warning(f"⚠️ No positive_pairs_csv provided or file not found: {positive_pairs_csv}")


    # ============================================================
    # ✅ 修复 2: 加载查询真值
    # ============================================================
    query_gt = {}
    if query_gt_csv and query_gt_csv.exists():
        logger.info(f"📂 Loading query ground truth from: {query_gt_csv}")
        with open(query_gt_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = f"{row['camera_id']}/{row['image_id']}"
                tvec = np.array([float(row['X_enu']), float(row['Y_enu']), float(row['Z_enu'])])
                query_gt[image_name] = tvec
        
        logger.info(f"✅ Loaded {len(query_gt)} query ground truth poses.")
        
        # 打印前 3 个样例
        sample_keys = list(query_gt.keys())[:3]
        for key in sample_keys:
            logger.info(f"   Sample: {key} → {query_gt[key]}")
    else:
        logger.warning(f"⚠️ No query_gt_csv provided or file not found: {query_gt_csv}")
    
    # ============================================================
    # ✅ 修复 3: 加载数据库位姿
    # ============================================================
    db_poses = {}
    if db_poses_paths:
        logger.info(f"📂 Loading database poses from {len(db_poses_paths)} files...")
        for db_path in db_poses_paths:
            if not db_path.exists():
                logger.warning(f"⚠️ Database poses file not found: {db_path}")
                continue
                
            with open(db_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    full_name = f"{row['camera_id']}/{row['image_id']}"
                    tvec = np.array([float(row['X_enu']), float(row['Y_enu']), float(row['Z_enu'])])
                    db_poses[full_name] = tvec
        
        logger.info(f"✅ Loaded {len(db_poses)} database poses.")
        
        # 打印前 3 个样例
        sample_keys = list(db_poses.keys())[:3]
        for key in sample_keys:
            logger.info(f"   Sample: {key} → {db_poses[key]}")
    else:
        logger.warning(f"⚠️ No db_poses_paths provided.")
    
    # ============================================================
    # ✅ 调试: 打印第一个查询的匹配情况
    # ============================================================
    first_debug = True
    
    # ============================================================
    # 主循环: Grid Reranking + 召回率计算
    # ============================================================
    first_debug = True

    for line in tqdm(lines, desc=f"Grid rerank (grid={grid}, topK={rerank_top_k})"):
        if not line.strip():
            continue
        q, db_names, base_scores = _parse_line(line)
        
        # ============================================================
        # ✅ 修复: 标准化查询名称
        # ============================================================
        q_normalized = q
        
        # 情况 1: 如果 q 仅包含文件名 (例如 "1740554132806470102.jpg")
        if '/' not in q:
            # 尝试通过时间戳匹配 positive_pairs
            timestamp = q.replace('.jpg', '')
            matching_keys = [k for k in positive_pairs.keys() if timestamp in k]
            
            if matching_keys:
                q_normalized = matching_keys[0]
            else:
                # 尝试在 query_gt 中查找
                matching_gt = [k for k in query_gt.keys() if timestamp in k]
                if matching_gt:
                    q_normalized = matching_gt[0]
                else:
                    # 最后回退: 默认 cam1
                    q_normalized = f"cam1/{q}"
        
        # ✅ 调试: 打印第一个查询的标准化结果
        if first_debug and (positive_pairs or query_gt):
            logger.info("="*70)
            logger.info(f"🔍 Debugging first query:")
            logger.info(f"   Original: {q}")
            logger.info(f"   Normalized: {q_normalized}")
            logger.info(f"   Sample db_names: {db_names[:3]}")
            
            if positive_pairs:
                if q_normalized in positive_pairs:
                    logger.info(f"   ✅ Found in positive_pairs: {positive_pairs[q_normalized][:2]}")
                else:
                    logger.info(f"   ❌ STILL not found in positive_pairs")
                    logger.info(f"   💡 Sample positive_pairs keys: {list(positive_pairs.keys())[:3]}")
            
            if query_gt:
                if q_normalized in query_gt:
                    logger.info(f"   ✅ Found in query_gt: {query_gt[q_normalized]}")
                else:
                    logger.info(f"   ❌ STILL not found in query_gt")
                    logger.info(f"   💡 Sample query_gt keys: {list(query_gt.keys())[:3]}")
            
            logger.info("="*70)
            first_debug = False


        text = q2text.get(q, None)
        
        if text is None:
            new_lines.append(line.strip())
            continue

        # ✅ 编码文本
        text_emb = _encode_text_any(clip_model, clip_processor, [text], device)[0]

        # ✅ Grid Reranking (仅对 top-K 重排)
        K = min(rerank_top_k, len(db_names))
        region_scores = [None] * len(db_names)

        for i in range(K):
            name = db_names[i]
            img_path = (image_root_dir / name)
            
            if not img_path.exists():
                if missing_ok:
                    region_scores[i] = -1e9
                    continue
                else:
                    raise FileNotFoundError(f"DB image not found: {img_path}")

            img = Image.open(img_path).convert("RGB")
            region_patches = split_image_to_grid_patches(img, grid)
            
            region_embs = _encode_image_regions_any(
                clip_model,
                clip_processor,
                region_patches,
                preprocess,
                device,
            )
            
            s_reg = _region_text_score(region_embs, text_emb, top_m=top_m, reduce=reduce)
            region_scores[i] = s_reg

        # ✅ 融合分数
        final_scores = base_scores[:]
        for i in range(K):
            if region_scores[i] is None:
                continue
            final_scores[i] = alpha * base_scores[i] + (1 - alpha) * region_scores[i]

        # ✅ 重新排序 top-K
        top_idx = list(range(K))
        top_idx.sort(key=lambda i: final_scores[i], reverse=True)

        db_new = [db_names[i] for i in top_idx] + db_names[K:]
        sc_new = [final_scores[i] for i in top_idx] + base_scores[K:]
        
        new_lines.append(_format_line(q, db_new, sc_new))
        
        # ============================================================
        # ✅ 计算召回率 (使用 Rerank 后的顺序)
        # ============================================================
        if positive_pairs or query_gt:
            total_queries += 1
            
            # 1. 检索召回率 (Retrieval Recall)
            if q_normalized in positive_pairs:
                gt_set = set(positive_pairs[q_normalized])
                
                for k in retrieval_stats.keys():
                    # ✅ 修复: 使用 Rerank 后的顺序 db_new
                    retrieved_set = set(db_new[:k])  # ← 关键修复!
                    if not retrieved_set.isdisjoint(gt_set):
                        retrieval_stats[k] += 1
            
            # 2. 多阈值定位召回率 (Localization Recall)
            if q_normalized in query_gt:
                query_pos = query_gt[q_normalized]
                
                for threshold in distance_thresholds:
                    for k in [1, 5, 10, 20]:
                        # ✅ 这里已经正确使用了 db_new
                        for db_name in db_new[:k]:
                            if db_name in db_poses:
                                db_pos = db_poses[db_name]
                                dist = np.linalg.norm(query_pos - db_pos)
                                if dist <= threshold:
                                    localization_stats_multi[threshold][k] += 1
                                    break  # 找到一个符合条件的就算成功，跳出循环

    # ============================================================
    # ✅ 保存 Rerank 结果
    # ============================================================
    with open(loc_pairs_out, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    # ============================================================
    # ✅ 打印多阈值统计
    # ============================================================
    if total_queries > 0:
        logger.info("="*70)
        logger.info("📊 Reranking Evaluation Results")
        logger.info("="*70)
        
        # 检索召回率
        if positive_pairs:
            logger.info("🔍 Retrieval Recall (after reranking):")
            for k in [1, 5, 10, 20]:
                recall = (retrieval_stats[k] / total_queries) * 100
                logger.info(f" R@{k:2d} = {recall:6.2f}% ({retrieval_stats[k]}/{total_queries})")
        
        # ✅ 多阈值定位召回率
        if query_gt:
            logger.info(f"\n📍 Localization Recall (multi-threshold):")
            for threshold in distance_thresholds:
                logger.info(f"\n   Threshold: {threshold}m")
                for k in [1, 5, 10, 20]:
                    count = localization_stats_multi[threshold][k]
                    recall = (count / total_queries) * 100
                    logger.info(f" R@{k:2d} = {recall:6.2f}% ({count}/{total_queries})")
        
        logger.info("="*70)

    return loc_pairs_out


@torch.no_grad()
def extract_vision_patch_tokens_openclip(model, images: torch.Tensor) -> torch.Tensor:
    """
    从 open_clip 的 ViT 视觉塔提取 patch tokens（去掉 CLS）。
    从 open_clip ViT 视觉塔提取 patch tokens（与训练脚本中 extract_vision_patch_tokens 一致）。

    images: [B, 3, H, W]
    return: [B, N, Dv]
    """
    x = model.visual.conv1(images)  # [B, Dv, gh, gw]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, N, Dv]

    x = torch.cat([
        model.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        ),
        x
    ], dim=1)  # [B, N+1, Dv]

    x = x + model.visual.positional_embedding.to(x.dtype)
    x = model.visual.ln_pre(x)
    x = x.permute(1, 0, 2)          # [N+1, B, Dv]
    x = model.visual.transformer(x)
    x = x.permute(1, 0, 2)          # [B, N+1, Dv]

    return x[:, 1:, :]              # [B, N, Dv]


@torch.no_grad()
def crossattn_score_one_image(
    clip_model,
    cross_attn_module,
    preprocess,
    pil_img: Image.Image,
    text_emb: torch.Tensor,  # [Dt] normalized
    device="cuda"
) -> float:
    """
    对一张候选 DB 图，计算 cross-attn/token-select 的文本引导“局部摘要”分数：
      1) 提取 patch tokens
      2) cross_attn_module(txt_global, vision_patches) -> 视觉摘要
      3) summary 与 text_emb 点积（带 logit_scale）
    """
    # 1) preprocess
    img_t = preprocess(pil_img).unsqueeze(0).to(device)  # [1,3,H,W]

    # 2) patch tokens（视觉塔前向一次）
    vision_patches = extract_vision_patch_tokens_openclip(clip_model, img_t)  # [1,N,Dv]

    # 3) cross-attn / token_select 输出
    txt = text_emb.unsqueeze(0)  # [1,Dt]
    out = cross_attn_module(txt, vision_patches)

    if isinstance(out, tuple):
        # full 模式：(visual_summary, attn_weights)
        visual_summary = out[0]                     # [1, num_queries, Dt]
        visual_summary_pooled = visual_summary.mean(dim=1)  # [1,Dt]
    else:
        # token_select：selected_patches
        selected = out                              # [1, top_k, Dt]
        visual_summary_pooled = selected.mean(dim=1)       # [1,Dt]

    visual_summary_pooled = F.normalize(visual_summary_pooled, dim=-1)  # [1,Dt]

    # 4) similarity（标量）
    score = torch.exp(clip_model.logit_scale) * (visual_summary_pooled @ txt.t())  # [1,1]
    return float(score.item())


@torch.no_grad()
def rerank_loc_pairs_crossattn(
    loc_pairs_in,
    loc_pairs_out,
    prompts_csv,
    image_root_dir,
    clip_model,
    clip_processor,
    preprocess,
    cross_attn_module,
    device="cuda",
    rerank_top_k=20,
    alpha=0.5,          # final = alpha*base + (1-alpha)*cross
    missing_ok=True,
):
    """
    用 Cross-Attention / Token-Select 在 topK 上做 rerank（不再 grid 切块）：
    - text 还是整句（你也可以之后再接“拆句”版本）
    - 每张候选图只跑一次视觉塔，得到 patch tokens，再过 cross_attn_module 得摘要向量
    """
    df = pd.read_csv(prompts_csv)
    q2text = dict(zip(df["image_name"].astype(str), df["description"].astype(str)))

    def _parse_line(line):
        parts = line.strip().split()
        q = parts[0]
        dbs = parts[1::2]
        scs = [float(x) for x in parts[2::2]]
        return q, dbs, scs

    def _format_line(q, dbs, scs):
        out = [q]
        for n, s in zip(dbs, scs):
            out += [n, f"{s:.6f}"]
        return " ".join(out)

    clip_model.eval()
    cross_attn_module.eval()

    with open(loc_pairs_in, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in tqdm(lines, desc=f"CrossAttn rerank (topK={rerank_top_k})"):
        if not line.strip():
            continue

        q, db_names, base_scores = _parse_line(line)
        text = q2text.get(q, None)
        if text is None:
            new_lines.append(line.strip())
            continue

        # 编码文本（与检索时一致：text tower -> normalize）
        text_emb = _encode_text_any(clip_model, clip_processor, [text], device)[0]  # [Dt], normalized

        K = min(rerank_top_k, len(db_names))
        cross_scores = [None] * len(db_names)

        for i in range(K):
            name = db_names[i]
            img_path = (image_root_dir / name)
            if not img_path.exists():
                if missing_ok:
                    cross_scores[i] = -1e9
                    continue
                raise FileNotFoundError(f"DB image not found: {img_path}")

            pil_img = Image.open(img_path).convert("RGB")
            s_cross = crossattn_score_one_image(
                clip_model=clip_model,
                cross_attn_module=cross_attn_module,
                preprocess=preprocess,
                pil_img=pil_img,
                text_emb=text_emb,
                device=device,
            )
            cross_scores[i] = s_cross

        # 融合分数，只重排 topK
        final_scores = base_scores[:]
        for i in range(K):
            if cross_scores[i] is None:
                continue
            final_scores[i] = alpha * base_scores[i] + (1 - alpha) * cross_scores[i]

        # topK 内排序，其余保持原顺序
        idx_top = list(range(K))
        idx_top.sort(key=lambda j: final_scores[j], reverse=True)

        new_dbs = [db_names[j] for j in idx_top] + db_names[K:]
        new_scs = [final_scores[j] for j in idx_top] + final_scores[K:]

        new_lines.append(_format_line(q, new_dbs, new_scs))

    with open(loc_pairs_out, "w") as f:
        for l in new_lines:
            f.write(l + "\n")



@torch.no_grad()
def extract_vision_patch_tokens_any(model, images: torch.Tensor, device="cuda", apply_projection=False) -> torch.Tensor:
    """
    从多种 CLIP 模型提取 patch tokens (去掉 CLS token)。
    
    Args:
        model: CLIP 模型实例
        images: [B, 3, H, W] 输入图像张量
        device: 设备
        apply_projection: 是否应用最终的投影层 (默认 False,用于 Reranker)
    
    Returns:
        [B, N, D] patch tokens
    """
    # ============================================================
    # Case 1: OpenCLIP
    # ============================================================
    if hasattr(model, 'visual') and hasattr(model, 'token_embedding'):
        x = model.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        
        x = torch.cat([
            model.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x
        ], dim=1)
        
        x = x + model.visual.positional_embedding.to(x.dtype)
        x = model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = model.visual.transformer(x)
        x = x.permute(1, 0, 2)
        
        patch_tokens = x[:, 1:, :]  # [B, N, D_original] (去掉 CLS)
        
        # ✅ 修复: 只在明确要求时才应用投影层
        if apply_projection and hasattr(model.visual, 'proj') and model.visual.proj is not None:
            B, N, D = patch_tokens.shape
            patch_tokens_flat = patch_tokens.reshape(B * N, D)
       
            patch_tokens_proj = patch_tokens_flat @ model.visual.proj
            patch_tokens = patch_tokens_proj.reshape(B, N, -1)
        
        return patch_tokens
    
    # ============================================================
    # Case 2: HuggingFace CLIP
    # ============================================================
    elif hasattr(model, 'vision_model') and hasattr(model, 'text_model'):
        vision_outputs = model.vision_model(
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # ✅ 使用倒数第二层的隐藏状态 (未投影)
        if apply_projection:
            patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
        else:
            # 使用原始维度 (1024 for ViT-L)
            hidden_states = vision_outputs.hidden_states
            patch_tokens = hidden_states[-2][:, 1:, :]  # 倒数第二层
        
        return patch_tokens
    
    # ============================================================
    # 其他情况: 抛出错误
    # ============================================================
    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")
    

@torch.no_grad()
def rerank_with_trained_reranker(
    loc_pairs_in,
    loc_pairs_out,
    prompts_csv,
    image_root_dir,
    clip_model,
    clip_processor,
    preprocess,
    reranker_module,  # ✅ 训练好的 Reranker
    device="cuda",
    rerank_top_k=20,
    alpha=0.5,
    missing_ok=True,
):
    """
    使用训练好的 Reranker 模块对候选图进行重排序
    
    优势:
        - 速度快 (一次提取 patch tokens,共享计算)
        - 准确度高 (专门优化排序目标)
        - 无需事后调参 (端到端学习)
    """
    df = pd.read_csv(prompts_csv)
    q2text = dict(zip(df["image_name"].astype(str), df["description"].astype(str)))

    def _parse_line(line):
        parts = line.strip().split()
        q = parts[0]
        dbs = parts[1::2]
        scs = [float(x) for x in parts[2::2]]
        return q, dbs, scs

    def _format_line(q, dbs, scs):
        out = [q]
        for n, s in zip(dbs, scs):
            out += [n, f"{s:.6f}"]
        return " ".join(out)

    clip_model.eval()
    reranker_module.eval()

    with open(loc_pairs_in, "r") as f:
        lines = f.readlines()

    new_lines = []
    
    for line in tqdm(lines, desc=f"Reranker (topK={rerank_top_k})"):
        if not line.strip():
            continue
        
        q, db_names, base_scores = _parse_line(line)
        text = q2text.get(q, None)
        
        if text is None:
            new_lines.append(line.strip())
            continue

        # 步骤 1: 编码文本
        text_emb = _encode_text_any(clip_model, clip_processor, [text], device)[0]  # [D]

        # 步骤 2: 对 top-K 候选图进行 rerank
        K = min(rerank_top_k, len(db_names))
        rerank_scores = [None] * len(db_names)

        for i in range(K):
            name = db_names[i]
            img_path = (image_root_dir / name)
            
            if not img_path.exists():
                if missing_ok:
                    rerank_scores[i] = -1e9
                    continue
                else:
                    raise FileNotFoundError(f"DB image not found: {img_path}")

            # 提取 patch tokens
            pil_img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            
            patch_tokens = extract_vision_patch_tokens_any(clip_model, img_tensor, device)  # [1, N, D]
            
            # ✅ 使用 Reranker 打分
            score = reranker_module(
                text_emb.unsqueeze(0),  # [1, D]
                patch_tokens            # [1, N, D]
            ).item()
            
            rerank_scores[i] = score

        # 步骤 3: 融合分数并重排
        final_scores = base_scores[:]
        for i in range(K):
            if rerank_scores[i] is None or rerank_scores[i] == -1e9:
                continue
            final_scores[i] = alpha * base_scores[i] + (1 - alpha) * rerank_scores[i]

        # 仅对 top-K 重新排序
        top_idx = list(range(K))
        top_idx.sort(key=lambda i: final_scores[i], reverse=True)

        db_new = [db_names[i] for i in top_idx] + db_names[K:]
        sc_new = [final_scores[i] for i in top_idx] + base_scores[K:]
        
        new_lines.append(_format_line(q, db_new, sc_new))

    # 保存结果
    with open(loc_pairs_out, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    logger.info(f"✅ Reranker reranking complete: {loc_pairs_out}")
    return loc_pairs_out


# ============================================================
# ✅ 修改: 根据模型 ID 动态生成特征文件路径
# ============================================================
def get_clip_features_path(outputs_dir, model_id, checkpoint_path=None):
    """
    根据模型 ID 生成唯一的 CLIP 特征文件路径。
    
    Args:
        outputs_dir: 输出目录
        model_id: 模型 ID (例如 "openai/clip-vit-large-patch14")
    
    Returns:
        Path: 特征文件路径
    """
    # 清理模型 ID,移除特殊字符
    safe_model_name = model_id.replace("/", "_").replace("-", "_")
    
    # 生成文件名
    if checkpoint_path is None:
        features_filename = f"clip_features_{safe_model_name}.h5"
    else:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        features_filename = (f"clip_features_{safe_model_name}_"
                             f"{checkpoint_path.stem}_{digest.hexdigest()[:12]}.h5")
    
    return outputs_dir / features_filename


def run(args):
    
    # ============================================================
    # ✅ 步骤 1: 配置根 Logger (影响所有模块)
    # ============================================================
    outputs = args.outputs
    outputs.mkdir(exist_ok=True, parents=True)
    
    model_suffix = args.clip_model_id.replace("/", "_").replace("-", "_")
    log_filename = f"{model_suffix}.log"
    log_path = outputs / log_filename
    
    # ✅ 关键修改 1: 获取根 logger 并清理所有处理器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    
    # ✅ 关键修改 2: 同时清理所有已存在的子 logger
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).setLevel(logging.INFO)
    
    # 1. 文件处理器
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '[%(asctime)s %(name)s %(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # 2. 终端处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # ✅ 关键修改 3: 强制所有子 logger 传播到根 logger
    logging.getLogger('hloc').propagate = True
    logging.getLogger('hloc.pairs_from_clip_retrieval').propagate = True
    
    # 创建本模块的 logger
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info(f"📝 Logging to: {log_path}")
    logger.info(f"📦 Model: {args.clip_model_id}")
    logger.info(f"📂 Output directory: {outputs}")
    logger.info("="*80)
    
    
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

    if getattr(args, "eval_only", False):
        eval_results = args.results if args.results is not None else results
        gt_csv = Path(args.gt_csv) if args.gt_csv else default_query_gt_csv(dataset)
        if not eval_results.is_file():
            raise FileNotFoundError(f"--eval-only: localization results not found: {eval_results}")
        if not gt_csv.is_file():
            raise FileNotFoundError(f"--eval-only: GT CSV not found: {gt_csv}")
        run_traj_eval(
            eval_results,
            outputs,
            gt_csv,
            args.filter_dist_m,
            results_are_query_centers=(
                getattr(args, "text_only_loc", False)
                or getattr(args, "results_are_query_centers", False)
            ),
        )
        logger.info("Eval-only finished.")
        return

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
    # clip_features_path = outputs / "clip_features_cam1_datacomp.h5" # large-datacomp
    # clip_features_path = outputs / "clip_features_cam1_datacomp_ft_.h5" # large-datacomp
    # clip_features_path = outputs / "clip_features_cam1_datacomp_ft_nofw.h5" # large-datacomp
    # clip_features_path = outputs / "clip_features_vit_large_patch14.h5"
    # clip_features_path = outputs / "clip_features_cam1_sigclip384.h5"
    # clip_features_path = outputs / "clip_features_cam1_eva2.h5"
    # clip_features_path = outputs / "clip_features_cam1_metaclip.h5"

    # # --- 新增：离线提取数据库的 CLIP 特征 ---
    # logger.info("Extracting CLIP features for the database...")
    # extract_clip_features.main(mapping_dir, clip_features_path)

#     # --- 修改后 ---
#     logger.info("Extracting CLIP features for the database...")
#     extract_clip_features.main(
#     image_dir=mapping_dir,
#     export_path=clip_features_path,
#     model_id=args.clip_model_id if not args.lora_ckpt_path else None,
#     lora_ckpt_path=args.lora_ckpt_path,
#     batch_size=32,
# )


    clip_features_path = get_clip_features_path(outputs, args.clip_model_id, args.lora_ckpt_path)
    logger.info(f"📂 Using CLIP features file: {clip_features_path}")
    
    # ============================================================
    # ✅ 如果特征文件不存在,自动提取
    # ============================================================
    if not clip_features_path.exists():
        logger.info(f"⚠️ Features file not found. Extracting features for model: {args.clip_model_id}")
        extract_clip_features.main(
            image_dir=mapping_dir,
            export_path=clip_features_path,
            model_id=args.clip_model_id if not args.lora_ckpt_path else None,
            lora_ckpt_path=args.lora_ckpt_path,
            batch_size=32,
        )
    else:
        logger.info(f"✅ Found existing features file: {clip_features_path}")
    
    
    # ### OFFLINE
    logger.info("--- Starting OFFLINE pre-processing ---")

    # --- 核心修改：模型预加载逻辑 ---
    t_offline_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # --- 核心修改：添加详细的调试日志 ---
    # if args.lora_ckpt_path:
    #     # 将用户提供的路径转换为绝对路径
    #     absolute_lora_path = args.lora_ckpt_path.resolve()
    #     logger.info(f"提供的 LoRA 检查点路径: {args.lora_ckpt_path}")
    #     logger.info(f"解析后的绝对路径: {absolute_lora_path}")
    #     logger.info(f"检查路径是否存在... {absolute_lora_path.exists()}")

    #     if absolute_lora_path.exists():
    #         # 如果路径存在，则加载微调模型
    #         logger.info("路径有效，正在加载微调模型...")
    #         # clip_model, clip_tokenizer, preprocess = pairs_from_clip_retrieval.load_finetuned_model(
    #         # absolute_lora_path, device
    #         # )
    #         # clip_processor = clip_tokenizer
            
    #         # ✅ 修复：接收 4 个返回值
    #         clip_model, clip_tokenizer, preprocess, cross_attn_module = pairs_from_clip_retrieval.load_finetuned_model(
    #             absolute_lora_path, device
    #         )
    #         clip_processor = clip_tokenizer
            
    #         # ✅ 检查是否成功加载
    #         if cross_attn_module is not None:
    #             print("✅ Cross-Attention module loaded successfully!")
    #             print(f"   Type: {type(cross_attn_module).__name__}")
    #             print(f"   Parameters: {sum(p.numel() for p in cross_attn_module.parameters())}")
    #         else:
    #             print("❌ Cross-Attention module is None")

    #     else:
    #         # 如果路径不存在，打印错误并退出或回退
    #         logger.error("错误：LoRA 检查点文件未找到！请检查路径。")
    #         logger.warning(f"回退到基础 CLIP 模型: {args.clip_model_id}")
    #         clip_model = CLIPModel.from_pretrained(args.clip_model_id).to(device)
    #         clip_processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    #         # --- 修复结束 ---
    # else:
    #     # 如果根本没有提供 lora_ckpt_path 参数
    #     logger.info(f"未提供 LoRA 检查点。使用基础 CLIP 模型: {args.clip_model_id}")
    #     clip_model = CLIPModel.from_pretrained(args.clip_model_id).to(device)
    #     clip_processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    # # --- 调试修改结束 ---

    # 在 run() 函数中


    # ============================================================
    # 模型加载逻辑（第 1040-1090 行附近）
    # ============================================================
    if args.lora_ckpt_path and args.lora_ckpt_path.exists():
        # ✅ 情况 1: 使用微调的 LoRA 模型（仅支持 OpenCLIP）
        logger.info(f"🔧 Loading fine-tuned LoRA model from {args.lora_ckpt_path}")
        clip_model, clip_tokenizer, preprocess, cross_attn_module = pairs_from_clip_retrieval.load_finetuned_model(
            args.lora_ckpt_path, device
        )
        clip_processor = clip_tokenizer
        lora_suffix = Path(args.lora_ckpt_path).stem


        # ===========================================================
        # ✅ 新增: 加载 Reranker 模块 (如果检查点中包含)
        reranker_module = None
        checkpoint = torch.load(args.lora_ckpt_path, map_location=device)
        
        if 'reranker_state' in checkpoint:
            logger.info("✅ Found Reranker weights in checkpoint, loading...")
            
            # 重建 Reranker 模块
            from train_localclip import LightweightReranker  # 导入训练脚本中的类
            
            text_dim = 768  # 根据您的模型调整
            vision_dim = 1024
            
            reranker_module = LightweightReranker(
                text_dim=text_dim,
                vision_dim=vision_dim,
                hidden_dim=256,
                num_heads=4,
                dropout=0.1
            ).to(device)
            
            reranker_module.load_state_dict(checkpoint['reranker_state'])
            reranker_module.eval()
            
            logger.info(f"✅ Reranker loaded successfully!")
            logger.info(f"   Parameters: {sum(p.numel() for p in reranker_module.parameters())}")
        else:
            logger.info("ℹ️ No Reranker weights found in checkpoint (skipping)")

    else:
        # ✅ 情况 2: 使用预训练基线模型
        logger.info(f"📦 Loading baseline pretrained model: {args.clip_model_id}")
        
        # 检测模型类型
        model_id_lower = args.clip_model_id.lower()
        
        # ============================================================
        # Case A: SigLIP（Google）
        # ============================================================
        if "siglip" in model_id_lower:
            from transformers import AutoModel, AutoProcessor
            
            clip_model = AutoModel.from_pretrained(args.clip_model_id).to(device)
            clip_processor = AutoProcessor.from_pretrained(args.clip_model_id)
            preprocess = None
            cross_attn_module = None
            lora_suffix = args.clip_model_id.replace("/", "_")
            
            logger.info("✅ SigLIP model loaded successfully!")
            logger.info(f"   Model type: {type(clip_model).__name__}")



        elif "eva" in model_id_lower or "quansun" in model_id_lower:
            import open_clip
            
            logger.info("🔧 Loading EVA-CLIP via OpenCLIP...")
            
            # 加载模型
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                'EVA02-L-14',
                pretrained='merged2b_s4b_b131k'
            )
            clip_model = clip_model.to(device).eval()
            clip_processor = open_clip.get_tokenizer('EVA02-L-14')
            cross_attn_module = None
            lora_suffix = "eva-clip"
            
            logger.info("✅ EVA-CLIP loaded!")
            logger.info(f"   Model: EVA02-L-14")
            logger.info(f"   Pretrained: merged2b_s4b_b131k")
        
        # ============================================================
        # Case B: CLIP 系列（OpenAI & LAION）
        # ============================================================
        elif any(prefix in model_id_lower for prefix in ["clip", "metaclip"]):
            from transformers import CLIPModel, CLIPProcessor
            
            clip_model = CLIPModel.from_pretrained(args.clip_model_id).to(device)
            clip_processor = CLIPProcessor.from_pretrained(args.clip_model_id)
            preprocess = None
            cross_attn_module = None
            lora_suffix = args.clip_model_id.replace("/", "_")
            
            logger.info("✅ CLIP model loaded successfully!")
            logger.info(f"   Model type: {type(clip_model).__name__}")
        
        # ============================================================
        # Case C: BLIP 系列
        # ============================================================
        elif "blip" in model_id_lower:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            clip_model = BlipForConditionalGeneration.from_pretrained(args.clip_model_id).to(device)
            clip_processor = BlipProcessor.from_pretrained(args.clip_model_id)
            preprocess = None
            cross_attn_module = None
            lora_suffix = args.clip_model_id.replace("/", "_")
            
            logger.info("✅ BLIP model loaded successfully!")
            logger.info(f"   Model type: {type(clip_model).__name__}")
        
        # ============================================================
        # Case D: ViLT
        # ============================================================
        elif "vilt" in model_id_lower:
            from transformers import ViltProcessor, ViltModel
            
            clip_model = ViltModel.from_pretrained(args.clip_model_id).to(device)
            clip_processor = ViltProcessor.from_pretrained(args.clip_model_id)
            preprocess = None
            cross_attn_module = None
            lora_suffix = args.clip_model_id.replace("/", "_")
            
            logger.info("✅ ViLT model loaded successfully!")
            logger.info(f"   Model type: {type(clip_model).__name__}")
        
        # ============================================================
        # Case E: LongCLIP（使用 OpenCLIP）
        # ============================================================
        elif "longclip" in model_id_lower:
            import open_clip
            
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14',
                pretrained='datacomp_xl_s13b_b90k'
            )
            clip_model = model.to(device)
            clip_processor = open_clip.get_tokenizer('ViT-L-14')
            cross_attn_module = None
            lora_suffix = "longclip"
            
            logger.info("✅ LongCLIP model loaded successfully!")
            logger.info("   Model type: OpenCLIP")
        
        # ============================================================
        # 默认：抛出错误
        # ============================================================
        else:
            raise ValueError(
                f"Unsupported model: {args.clip_model_id}\n"
                f"Supported models:\n"
                f"  - CLIP: openai/clip-*, laion/CLIP-*\n"
                f"  - SigLIP: google/siglip-*\n"
                f"  - BLIP: Salesforce/blip-*\n"
                f"  - ViLT: dandelin/vilt-*\n"
                f"  - MetaCLIP: facebook/metaclip-*"
            )

    logger.info(f"✅ Model loaded. Suffix for results: {lora_suffix}")

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
    # prompts_filename = f"nofw_qwen3_query_prompts_{caption_model_name}_77clip10.csv" ### 短文本
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



    # # --- 3. 调用 pairs_from_clip_retrieval.main ---
    # t0 = time.time() # 重新计时
    # pairs_from_clip_retrieval.main(
    #     clip_features_path=clip_features_path,
    #     text_query_or_csv_path=prompts_csv_path,
    #     num_loc=args.num_loc,
    #     loc_pairs=loc_pairs,
    #     # --- 仅传递 CLIP 检索相关参数 ---
    #     model_instance=clip_model,
    #     processor_instance=clip_processor,
    #     # --- 修复结束 ---
    #     query_list_path=query_list,
    #     # --- 核心修复：将召回率计算所需的真值文件路径传递下去 ---
    #     positive_pairs_csv=args.positive_pairs_csv,
    # )
    # timings['clip_retrieval'] = time.time() - t0

    # ✅ 启用逐帧模式
    t0 = time.time()
    pairs_from_clip_retrieval.main(
        clip_features_path=clip_features_path,
        text_query_or_csv_path=prompts_csv_path,
        num_loc=args.num_loc,
        loc_pairs=loc_pairs,
        model_instance=clip_model,
        processor_instance=clip_processor,
        query_list_path=query_list,
        positive_pairs_csv=args.positive_pairs_csv,
        per_frame_timing=True,  # ✅ 启用逐帧处理
        # ✅ 新增: 排除双目相机图像
        exclude_cameras=['cam1'],
        filter_self_match=True,
    ) 
    timings['clip_retrieval'] = time.time() - t0
    


    # --- After pairs_from_clip_retrieval.main(...) finished and loc_pairs is created ---

    # if args.enable_region_rerank:
    #     logger.info("🚀 Starting Region-aware reranking (no query image)...")

    #     loc_pairs_region = outputs / (loc_pairs.stem + f"_region_g{args.region_grid}_k{args.region_top_k}.txt")

    #     rerank_loc_pairs_region(
    #         loc_pairs_in=loc_pairs,
    #         loc_pairs_out=loc_pairs_region,
    #         prompts_csv=prompts_csv_path,
    #         image_root_dir=mapping_dir,     # 这里要指向你的 DB 图像根目录（候选图像所在）
    #         clip_model=clip_model,
    #         clip_processor=clip_processor,
    #         preprocess=preprocess, 
    #         device=device,
    #         rerank_top_k=args.region_top_k,
    #         grid=args.region_grid,
    #         top_m=args.region_top_m,
    #         reduce=args.region_reduce,
    #         alpha=args.region_alpha,
    #         region_pool=args.region_pool,
    #     )

    #     logger.info(f"✅ Region reranking done. New pairs: {loc_pairs_region}")
    #     loc_pairs = loc_pairs_region   # 关键：后续流程用新的排序文件



    if args.enable_region_rerank:
        logger.info("🚀 Starting Region-aware reranking...")

        loc_pairs_region = outputs / (loc_pairs.stem + f"_rerank_k{args.region_top_k}.txt")
        
        t0 = time.time()

        # ============================================================
        # ✅ 修改: 优先使用 Grid Reranking (基线方法)
        # ============================================================
        logger.info("✅ Using Grid-based region reranking (baseline).")
        rerank_loc_pairs_region(
            loc_pairs_in=loc_pairs,
            loc_pairs_out=loc_pairs_region,
            prompts_csv=prompts_csv_path,
            image_root_dir=mapping_dir,
            clip_model=clip_model,
            clip_processor=clip_processor,
            preprocess=preprocess,
            device=device,
            rerank_top_k=args.region_top_k,
            grid=args.region_grid,
            top_m=args.region_top_m,
            reduce=args.region_reduce,
            alpha=args.region_alpha,
            region_pool=args.region_pool,
            # ✅ 修复: 确保传递完整的路径对象
            positive_pairs_csv=args.positive_pairs_csv,  # 确保不是 None
            query_gt_csv=dataset / "0620_gt/enu_all_query_gt.csv",
            db_poses_paths=[
                dataset / "0620_gt/enu_all_mapping_gt.csv",
                dataset / "0620_gt/enu_all_query_gt.csv"
            ],
            distance_thresholds=[5.0, 10.0, 15.0],  # ← 多阈值
        )

        timings['region_rerank'] = time.time() - t0
        logger.info(f"✅ Grid reranking done in {timings['region_rerank']:.2f}s.")
        logger.info(f"   New pairs file: {loc_pairs_region}")
        
        # ✅ 更新 loc_pairs 为 rerank 后的文件
        loc_pairs = loc_pairs_region
        
        # # ============================================================
        # # 优先级 1: 使用训练好的 Reranker (最准确)
        # # ============================================================
        # if reranker_module is not None:
        #     logger.info("✅ Using trained Reranker module for reranking.")
        #     rerank_with_trained_reranker(
        #         loc_pairs_in=loc_pairs,
        #         loc_pairs_out=loc_pairs_region,
        #         prompts_csv=prompts_csv_path,
        #         image_root_dir=mapping_dir,
        #         clip_model=clip_model,
        #         clip_processor=clip_processor,
        #         preprocess=preprocess,
        #         reranker_module=reranker_module,
        #         device=device,
        #         rerank_top_k=args.region_top_k,
        #         alpha=args.region_alpha,
        #     )
        
        # # ============================================================
        # # 优先级 2: 使用 Cross-Attention (次优)
        # # ============================================================
        # elif args.region_use_cross_attn and (cross_attn_module is not None):
        #     logger.info("✅ Using Cross-Attention/Token-Select rerank.")
        #     rerank_loc_pairs_crossattn(
        #         loc_pairs_in=loc_pairs,
        #         loc_pairs_out=loc_pairs_region,
        #         prompts_csv=prompts_csv_path,
        #         image_root_dir=mapping_dir,
        #         clip_model=clip_model,
        #         clip_processor=clip_processor,
        #         preprocess=preprocess,
        #         cross_attn_module=cross_attn_module,
        #         device=device,
        #         rerank_top_k=args.region_top_k,
        #         alpha=args.region_alpha,
        #     )
        
        # # ============================================================
        # # 优先级 3: 回退到 Grid Reranking (基线)
        # # ============================================================
        # else:
        #     logger.info("⚠️ No Reranker/Cross-Attn. Fallback to grid region rerank.")
        #     rerank_loc_pairs_region(
        #         loc_pairs_in=loc_pairs,
        #         loc_pairs_out=loc_pairs_region,
        #         prompts_csv=prompts_csv_path,
        #         image_root_dir=mapping_dir,
        #         clip_model=clip_model,
        #         clip_processor=clip_processor,
        #         preprocess=preprocess,
        #         device=device,
        #         rerank_top_k=args.region_top_k,
        #         grid=args.region_grid,
        #         top_m=args.region_top_m,
        #         reduce=args.region_reduce,
        #         alpha=args.region_alpha,
        #         region_pool=args.region_pool,
        #     )

        # timings['region_rerank'] = time.time() - t0
        # logger.info(f"✅ Reranking done in {timings['region_rerank']:.2f}s. New pairs: {loc_pairs_region}")
        # loc_pairs = loc_pairs_region




    


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
    db_local_features = extract_features.main(
        feature_conf, mapping_dir, outputs, as_half=True
    )
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

        t0 = time.time()
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

    if not getattr(args, "skip_post_eval", False):
        gt_csv = Path(args.gt_csv) if args.gt_csv else default_query_gt_csv(dataset)
        if results.is_file() and gt_csv.is_file():
            run_traj_eval(
                results,
                outputs,
                gt_csv,
                args.filter_dist_m,
                results_are_query_centers=getattr(args, "text_only_loc", False),
            )
        elif not results.is_file():
            logger.warning("定位结果文件未找到，跳过轨迹评估: %s", results)
        else:
            logger.warning("真值 CSV 未找到，跳过轨迹评估: %s（可用 --gt-csv 指定）", gt_csv)




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
    rerank_region = parser.add_argument_group("Region Reranking (no query image)")
    rerank_region.add_argument("--enable_region_rerank", action="store_true")
    rerank_region.add_argument("--region_top_k", type=int, default=20, help="Only rerank top-K candidates per query.")
    rerank_region.add_argument("--region_grid", type=int, default=3, help="Grid size (e.g., 3 or 4).")
    rerank_region.add_argument("--region_top_m", type=int, default=1, help="top-m regions averaged for score.")
    rerank_region.add_argument("--region_reduce", type=str, default="max", choices=["topm_mean", "max"])
    rerank_region.add_argument("--region_alpha", type=float, default=0.75, help="final = alpha*base + (1-alpha)*region")
    rerank_region.add_argument("--region_pool", type=str, default="mean", choices=["mean", "max"])
    rerank_region.add_argument("--region_use_cross_attn", action="store_true",
                           help="Use cross-attn/token-select module for rerank (topK online).")

    
    # ✅ 新增: 召回率评估参数
    rerank_region.add_argument(
        "--evaluate_rerank_recall",
        action="store_true",
        help="Compute retrieval and localization recall after reranking."
    )

    rerank_region.add_argument(
        "--distance_threshold",
        type=float,
        default=10.0,
        help="Distance threshold (meters) for localization recall (default: 10.0)."
    )
    

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

    # parser.add_argument(
    #     "--clip_model_id",
    #     type=str,
    #     # default="zer0int/LongCLIP-GmP-ViT-L-14", # <-- 4. 将模型ID参数化
    #     # default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 
    #     default="laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    #     help="The Hugging Face model ID for the CLIP model.",
    # )

    parser.add_argument(
    "--clip_model_id",
    type=str,
    default="laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    choices=[
        # ============================================================
        # CLIP 系列 (OpenAI & LAION)
        # ============================================================
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-base-patch16",
        "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        
        # ============================================================
        # EVA2-CLIP
        # ============================================================
        "QuanSun/EVA-CLIP",
        
        # ============================================================
        # SigLIP (Google)
        # ============================================================
        "google/siglip-base-patch16-224",
        "google/siglip-large-patch16-384",
        
        # ============================================================
        # MetaCLIP (Meta)
        # ============================================================
        "facebook/metaclip-l14-fullcc2.5b",   # Large, 2.5B 数据（推荐）
        
        # ============================================================
        # LongCLIP
        # ============================================================
        "zer0int/LongCLIP-GmP-ViT-L-14",  
        
        # ============================================================
        # VLM 系列 (BLIP, ViLT, X-VLM)
        # ============================================================
        "Salesforce/blip-image-captioning-base",
        "Salesforce/blip2-opt-2.7b",
        "dandelin/vilt-b32-finetuned-coco",
    ],
    help="预训练模型 ID (Hugging Face Model Hub)。用于基线对比，不进行微调。",
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

    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=None,
        help="轨迹评估用真值 CSV；默认使用 dataset/0620_gt/enu_all_query_gt.csv。",
    )
    parser.add_argument(
        "--filter-dist-m",
        type=float,
        default=10.0,
        help="轨迹评估后按最近 GT 距离过滤离群帧的阈值，单位米。",
    )
    parser.add_argument(
        "--skip-post-eval",
        action="store_true",
        help="跳过定位结果 txt 转换和 GT 轨迹误差评估。",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="不跑定位，只处理已有定位结果 txt 并和 GT 做轨迹误差评估。",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="与 --eval-only 连用：指定已有定位结果 txt；默认使用当前参数推导出的 results 文件名。",
    )
    parser.add_argument(
        "--results-are-query-centers",
        action="store_true",
        help="与 --eval-only 连用：结果 txt 的 tx ty tz 已经是 ENU 相机中心，不再做 HLOC 位姿转换。",
    )

    # --- 新增 --text_only_loc 参数 ---
    parser.add_argument(
        '--text_only_loc', 
        action='store_true', 
        help='Perform text-only localization by using the poses of retrieved DB images.'
    )
    # ---

    args = parser.parse_args()
    run(args)
