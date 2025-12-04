import os
import subprocess
import sqlite3
import cv2
import numpy as np

# ===== 路径设置 =====
IMAGE_DIR = "datasets/SensorKit/mapping"  # 包含 cam0, cam2 等子文件夹的父目录
# 建议为新模型创建一个新的输出目录，以避免与旧结果混淆
OUTPUT_DIR = "datasets/SensorKit/vocab_tree_0630_OPENCV"

DB_PATH = os.path.join(OUTPUT_DIR, "database.db")
SPARSE_DIR = os.path.join(OUTPUT_DIR, "sparse")

# 包含相机内参 YAML 文件的目录
INTRINSICS_DIR = "datasets/SensorKit/config/intrinsics"
# 相机文件夹列表
CAMERA_FOLDERS = ["cam0", "cam2", "cam3", "cam4", "cam5", "cam6"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPARSE_DIR, exist_ok=True)

# ===== 1. 特征提取 (使用 OPENCV 模型) - 初始运行 =====
print("Step 1: Running initial feature extraction with OPENCV model...")

if os.path.exists(DB_PATH):
    print(f"Removing existing database: {DB_PATH}")
    os.remove(DB_PATH)

# 注意：这里我们将 camera_model 改为 OPENCV
# COLMAP 会为每个相机文件夹创建一个使用 OPENCV 模型的相机条目，但参数是默认的
subprocess.run([
    "colmap", "feature_extractor",
    "--database_path", DB_PATH,
    "--image_path", IMAGE_DIR,
    "--ImageReader.camera_model", "OPENCV",
    "--ImageReader.single_camera_per_folder", "1",
    "--SiftExtraction.use_gpu", "1"
], check=True)
print("Initial feature extraction complete.")

# ===== Python 脚本: 从 YAML 更新数据库中的相机内参 =====
print("\nStep 1.5: Updating camera intrinsics in the database from YAML files (OPENCV model)...")

def parse_camera_yaml_for_opencv(yaml_path):
    """
    从 OpenCV YAML 文件中解析 OPENCV 模型所需的参数。
    OPENCV 模型参数: [fx, fy, cx, cy, k1, k2, p1, p2]
    """
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f"Failed to open YAML file: {yaml_path}")

    # 提取相机矩阵和畸变系数
    camera_matrix = fs.getNode("cameraMatrix").mat()
    dist_coeffs = fs.getNode("distCoeffs").mat() # 应该是 1x4 (k1, k2, p1, p2)

    # 从相机矩阵提取内参
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # 提取畸变系数
    k1 = dist_coeffs[0, 0]
    k2 = dist_coeffs[0, 1]
    p1 = dist_coeffs[0, 2]
    p2 = dist_coeffs[0, 3]

    # 提取图像尺寸
    image_width = int(fs.getNode("imageWidth").real())
    image_height = int(fs.getNode("imageHeight").real())
    fs.release()

    # 组合成 COLMAP OPENCV 模型所需的8个参数
    params = np.array([fx, fy, cx, cy, k1, k2, p1, p2], dtype=np.float64)
    
    # COLMAP 中 OPENCV 模型的 ID 是 4
    model_id = 4
    return model_id, image_width, image_height, params

def get_camera_id_for_folder_from_db(db_path, folder_name_pattern):
    """从数据库中根据文件夹名获取 camera_id"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 图像名在数据库中通常以 "folder/image.jpg" 的形式存储
    query = "SELECT camera_id FROM images WHERE name LIKE ? LIMIT 1"
    pattern = f"{folder_name_pattern}/%"
    cursor.execute(query, (pattern,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    else:
        print(f"Warning: No image found for folder pattern '{pattern}' in the database. Cannot determine camera_id.")
        return None

# --- 开始更新数据库 ---
db_conn_intrinsics = sqlite3.connect(DB_PATH)
db_cursor_intrinsics = db_conn_intrinsics.cursor()

for cam_folder_name in CAMERA_FOLDERS:
    yaml_file_path = os.path.join(INTRINSICS_DIR, f"camera-intrinsics_{cam_folder_name}.yaml")
    if not os.path.exists(yaml_file_path):
        print(f"Warning: YAML file not found for {cam_folder_name} at {yaml_file_path}. Skipping.")
        continue

    try:
        # 使用新的解析函数
        model_id, width, height, params_array = parse_camera_yaml_for_opencv(yaml_file_path)
    except Exception as e:
        print(f"Error parsing YAML file {yaml_file_path}: {e}. Skipping.")
        continue

    camera_id_to_update = get_camera_id_for_folder_from_db(DB_PATH, cam_folder_name)
    if camera_id_to_update is not None:
        print(f"Updating intrinsics for folder '{cam_folder_name}' (DB camera_id: {camera_id_to_update})")
        params_blob = params_array.tobytes() # 将 numpy 数组转换为二进制 blob
        
        # 更新数据库中的相机条目
        update_query = """
        UPDATE cameras
        SET model = ?, width = ?, height = ?, params = ?, prior_focal_length = 0
        WHERE camera_id = ?"""
        db_cursor_intrinsics.execute(update_query, (model_id, width, height, params_blob, camera_id_to_update))
        db_conn_intrinsics.commit()
        print(f"Successfully updated intrinsics for camera_id {camera_id_to_update} for {cam_folder_name}.")
    else:
        print(f"Skipping intrinsics update for {cam_folder_name} as its camera_id could not be determined.")

db_conn_intrinsics.close()
print("Camera intrinsics update process complete.")

# # ===== 2. 特征匹配 =====
# print("\nStep 2: Running sequential matcher...")
# subprocess.run([
#     "colmap", "sequential_matcher",
#     "--database_path", DB_PATH,
#     "--SiftMatching.guided_matching", "1", # 如果图像是连续的，引导匹配很有用
#     "--SiftMatching.use_gpu", "1"
# ], check=True)
# print("Sequential matching complete.")


print("\nStep 2: Running vocab_tree_matcher...")
vocab_tree_path = "vocab_tree_flickr100K_words256K.bin" # 假设词汇树文件在工作目录下

subprocess.run([
    "colmap", "vocab_tree_matcher",
    "--database_path", DB_PATH,
    "--VocabTreeMatching.vocab_tree_path", vocab_tree_path,
    "--SiftMatching.use_gpu", "1"
], check=True)
print("Vocab tree matching complete.")


# ===== 3. 稀疏重建 (Mapper) =====
print("\nStep 3: Running mapper...")
subprocess.run([
    "colmap", "mapper",
    "--database_path", DB_PATH,
    "--image_path", IMAGE_DIR,
    "--output_path", SPARSE_DIR,
    "--Mapper.min_num_matches", "15",
    "--Mapper.init_min_num_inliers", "20",
    # # --- 添加以下参数来固定内参 ---
    "--Mapper.ba_refine_focal_length", "0",
    "--Mapper.ba_refine_principal_point", "0",
    "--Mapper.ba_refine_extra_params", "0"  # 固定所有畸变参数
], check=True)

print("\n✅ Reconstruction complete, output saved in:", SPARSE_DIR)
print("包含文件：cameras.bin, images.bin, points3D.bin")







