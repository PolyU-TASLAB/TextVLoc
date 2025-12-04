# import os
# import subprocess
# import sqlite3
# import yaml
# import numpy as np
# import cv2

# # ===== 路径设置 =====
# IMAGE_DIR = "datasets/SensorKit/mapping_0617"  # Parent directory containing cam0, cam2, etc.
# OUTPUT_DIR = "datasets/SensorKit/seque_colmap_0617"
# DB_PATH = os.path.join(OUTPUT_DIR, "database.db")
# SPARSE_DIR = os.path.join(OUTPUT_DIR, "sparse")

# # Directory containing your camera intrinsics YAML files
# INTRINSICS_DIR = "datasets/SensorKit/config/intrinsics"
# # List of camera folder names
# CAMERA_FOLDERS = ["cam0", "cam2", "cam3", "cam4", "cam5", "cam6"]


# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(SPARSE_DIR, exist_ok=True)

# # ===== 1. 特征提取（使用默认 SIFT） - Initial Run =====
# print("Step 1: Running initial feature extraction...")

# # Add these lines to remove the existing database if it exists
# if os.path.exists(DB_PATH):
#     print(f"Removing existing database: {DB_PATH}")
#     os.remove(DB_PATH)


# subprocess.run([
#     "colmap", "feature_extractor",
#     "--database_path", DB_PATH,
#     "--image_path", IMAGE_DIR,
#     "--ImageReader.camera_model", "PINHOLE", # Use PINHOLE model for all cameras
#     # "--ImageReader.single_camera_per_folder", "1", 
#     "--SiftExtraction.use_gpu", "1"
# ], check=True)


# print("Initial feature extraction complete.")

# # # ===== Python Script to Update Camera Intrinsics in DB =====
# # print("\nStep 1.5: Updating camera intrinsics in the database from YAML files...")


# # def parse_camera_yaml_for_pinhole(yaml_path):
# #     """Parses the camera intrinsics YAML for PINHOLE model parameters."""
# #     fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
# #     if not fs.isOpened():
# #         raise ValueError(f"Failed to open YAML file: {yaml_path}")

# #     # Read camera matrix
# #     camera_matrix = fs.getNode("cameraMatrix").mat()
# #     fx = camera_matrix[0, 0]
# #     fy = camera_matrix[1, 1]
# #     cx = camera_matrix[0, 2]
# #     cy = camera_matrix[1, 2]

# #     # Read image dimensions
# #     image_width = int(fs.getNode("imageWidth").real())
# #     image_height = int(fs.getNode("imageHeight").real())

# #     fs.release()

# #     # COLMAP PINHOLE model ID is 1
# #     params = np.array([fx, fy, cx, cy], dtype=np.float64)
# #     return 1, image_width, image_height, params

# # def get_camera_id_for_folder_from_db(db_path, folder_name_pattern):
  
# #     conn = sqlite3.connect(db_path)
# #     cursor = conn.cursor()
    
# #     query = "SELECT camera_id FROM images WHERE name LIKE ? LIMIT 1"
# #     # Pattern should match images within the specific camera's folder, e.g., 'cam0/%'
# #     pattern = f"{folder_name_pattern}/%" 
# #     cursor.execute(query, (pattern,))
# #     result = cursor.fetchone()
# #     conn.close()
    
# #     if result:
# #         return result[0]
# #     else:
# #         print(f"Warning: No image found for folder pattern '{pattern}' in the database. Cannot determine camera_id.")
# #         return None

# # db_conn = sqlite3.connect(DB_PATH)
# # db_cursor = db_conn.cursor()

# # for cam_folder_name in CAMERA_FOLDERS:
# #     yaml_file_path = os.path.join(INTRINSICS_DIR, f"camera-intrinsics_{cam_folder_name}.yaml")
    
# #     if not os.path.exists(yaml_file_path):
# #         print(f"Warning: YAML file not found for {cam_folder_name} at {yaml_file_path}. Skipping.")
# #         continue
        
# #     try:
# #         model_id, width, height, params_array = parse_camera_yaml_for_pinhole(yaml_file_path)
# #     except Exception as e:
# #         print(f"Error parsing YAML file {yaml_file_path}: {e}. Skipping.")
# #         continue
        
# #     # Get the camera_id that feature_extractor created for this folder
# #     # The folder_name_pattern for the LIKE query is just the camera folder name (e.g., "cam0")
# #     camera_id_to_update = get_camera_id_for_folder_from_db(DB_PATH, cam_folder_name)
    
# #     if camera_id_to_update is not None:
# #         print(f"Updating intrinsics for folder '{cam_folder_name}' (DB camera_id: {camera_id_to_update})")
# #         print(f"  Model: PINHOLE ({model_id}), Width: {width}, Height: {height}")
# #         print(f"  Params (fx, fy, cx, cy): {params_array}")
        
# #         params_blob = params_array.tobytes() # Convert NumPy array to blob for SQLite
        
# #         update_query = """
# #         UPDATE cameras 
# #         SET model = ?, width = ?, height = ?, params = ?, prior_focal_length = 0
# #         WHERE camera_id = ?
# #         """
# #         # prior_focal_length = 0 indicates that the provided params are the actual values.
# #         db_cursor.execute(update_query, (model_id, width, height, params_blob, camera_id_to_update))
# #         db_conn.commit()
# #         print(f"Successfully updated camera_id {camera_id_to_update} for {cam_folder_name}.")
# #     else:
# #         print(f"Skipping update for {cam_folder_name} as its camera_id could not be determined in the database.")

# # db_conn.close()
# # print("Camera intrinsics update process complete.")

# # ===== 2. 匹配特征（序列匹配，适合行驶序列） =====  ### exhaustive_matcher
# print("\nStep 2: Running sequential matcher...")
# subprocess.run([
#     "colmap", "sequential_matcher", # sequential_matcher; exhaustive_matcher; space matcher
#     "--database_path", DB_PATH,
#     "--SiftMatching.guided_matching", "1",  # 采用引导匹配提高精度
#     # You might need to add SiftMatching options if defaults are not desired
#     "--SiftMatching.use_gpu", "1"
# ], check=True)
# print("Sequential matching complete.")

# # ===== 3. 建立 SfM 模型 =====
# print("\nStep 3: Running mapper...")
# subprocess.run([
#     "colmap", "mapper",
#     "--database_path", DB_PATH,
#     "--image_path", IMAGE_DIR,  # 确保此路径与 feature_extractor 中的一致
#     "--output_path", SPARSE_DIR,
#     "--Mapper.min_num_matches", "15",
#     "--Mapper.init_min_num_inliers", "20",
#     # Consider if you want to fix intrinsics during BA:
#     # "--Mapper.ba_refine_focal_length", "0",
#     # "--Mapper.ba_refine_principal_point", "0",
#     # "--Mapper.ba_refine_extra_params", "0"
# ], check=True)

# print("\n✅ 模型构建完成，输出目录为：", SPARSE_DIR)
# print("包含文件：cameras.bin, images.bin, points3D.bin")



# # # ===== 4. 使用位置先验进行光束法平差 (Bundle Adjustment) =====
# # print("\nStep 4: Running Bundle Adjustment with position priors...")

# # # 定义输入和输出路径
# # # mapper 通常在 SPARSE_DIR 中创建一个名为 '0' 的子目录来存放模型
# # # 如果有多个模型，您可能需要选择正确的目录
# # input_model_path = os.path.join(SPARSE_DIR, "0") 
# # output_model_path = os.path.join(OUTPUT_DIR, "sparse-ba-prior")
# # os.makedirs(output_model_path, exist_ok=True)

# # # 检查输入模型是否存在
# # if not os.path.exists(input_model_path):
# #     print(f"Error: Input model for bundle adjustment not found at {input_model_path}. Skipping BA.")
# # else:
# #     subprocess.run([
# #         "colmap", "bundle_adjuster",
# #         "--input_path", input_model_path,
# #         "--output_path", output_model_path,
# #         "--BundleAdjustment.use_position_priors", "1",
# #         "--BundleAdjustment.position_sigma", "1.0"
# #     ], check=True)
# #     print(f"✅ 使用先验位姿的光束法平差完成。优化后的模型保存在: {output_model_path}")







import os
import subprocess
import sqlite3
import yaml # 保留导入，即使cv2处理yaml，以防其他地方使用
import numpy as np
import cv2
import csv # 新增导入 csv 模块


# ===== 路径设置 =====
# IMAGE_DIR = "datasets/SensorKit/IMAGES"
# IMAGE_DIR = "datasets/SensorKit/mapping"  # Parent directory containing cam0, cam2, etc.
# IMAGE_DIR = "datasets/SensorKit/allcams" # 修改输出目录名以反映变化
IMAGE_DIR = "datasets/SensorKit/2025-9-5/vehicle_0905" # 修改输出目录名以反映变化
# IMAGE_DIR = "datasets/SensorKit/mapping_downsampled_5"  # 使用下采样后的数据集
# OUTPUT_DIR = "datasets/SensorKit/mapping_downsampled_5_colmap_0807/" 
# OUTPUT_DIR = "datasets/SensorKit/colmap_seque" # 修改输出目录名以反映变化
# OUTPUT_DIR = "datasets/SensorKit/allcams_spatial_colmap_0630" # 修改输出目录名以反映变化
# OUTPUT_DIR = "datasets/SensorKit/spatial_colmap_0905" # 修改输出目录名以反映变化
OUTPUT_DIR = "datasets/SensorKit/seque_colmap_0905" # 修改输出目录名以反映变化



DB_PATH = os.path.join(OUTPUT_DIR, "database.db")
SPARSE_DIR = os.path.join(OUTPUT_DIR, "sparse")

# Directory containing your camera intrinsics YAML files
INTRINSICS_DIR = "datasets/SensorKit/config/intrinsics"
# ENU_POSES_CSV_PATH = "datasets/SensorKit/0620_gt/enu_all_mapping_gt.csv" 
# ENU_POSES_CSV_PATH_1 = "datasets/SensorKit/0620_gt/enu_all_query_gt.csv" 
ENU_POSES_CSV_PATH = "datasets/SensorKit/2025-9-5/vehicle_cleaned/enu_all_mapping_gt.csv" 
# ENU_POSES_CSV_PATH_1 = "datasets/SensorKit/2025-9-5/vehicle_cleaned/enu_all_query_gt.csv" 
# llh_POSES_CSV_PATH = "datasets/SensorKit/mapping_camera_llh_poses.csv"
# List of camera folder names
CAMERA_FOLDERS = ["cam0", "cam2", "cam3", "cam4", "cam5", "cam6"]
# CAMERA_FOLDERS = ["cam0", "cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]
# CAMERA_FOLDERS = ["image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPARSE_DIR, exist_ok=True)



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


# ===== Step 1.7: Update Image ENU Priors in DB from CSV (NEW) =====
print("\nStep 1.7: Updating image ENU priors in the database from CSV file...")

# def update_images_with_ecef_priors(db_path, ecef_csv_path, image_dir_colmap_root):
#     """
#     Updates the COLMAP database with ECEF priors (tx, ty, tz) from a CSV file.
#     The CSV must contain 'image_path', 'pos_x_ecef', 'pos_y_ecef', 'pos_z_ecef'.
#     'image_path' in CSV is an absolute path. We will extract a relative path
#     based on image_dir_colmap_root to match 'name' in COLMAP's images table.
#     """
#     if not os.path.exists(ecef_csv_path):
#         print(f"Error: ECEF CSV file not found at {ecef_csv_path}. Skipping ECEF prior update.")
#         return 0

#     print(f"Reading ECEF priors from: {ecef_csv_path}")
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
    
#     ecef_priors_to_update_in_db = {} # Stores {colmap_image_name: (tx, ty, tz)}
    
#     # Determine the anchor string for splitting the absolute path from CSV
#     # This should be the part of the absolute path that corresponds to IMAGE_DIR
#     # For example, if IMAGE_DIR is "datasets/SensorKit/mapping",
#     # and CSV path is "/abs/path/to/datasets/SensorKit/mapping/cam0/img.jpg",
#     # we need to find "datasets/SensorKit/mapping/" or just "mapping/" if it's unique enough.
#     # A simple approach is to use the basename of image_dir_colmap_root as part of the anchor.
#     image_dir_basename = os.path.basename(image_dir_colmap_root.rstrip(os.sep))
#     # We expect the CSV path to contain something like ".../mapping/cam0/..."
#     # So, the anchor to split *after* would be "mapping/"
#     path_anchor_to_split_after = image_dir_basename + os.sep

#     print(f"  Using '{path_anchor_to_split_after}' as anchor to extract relative paths from CSV image_path column.")

#     try:
#         with open(ecef_csv_path, 'r', newline='') as f:
#             # Assuming tab-separated based on the image
#             reader = csv.DictReader(f, delimiter=',') 
            
#             # Verify required columns exist in the CSV
#             # Adjust column names if they are different in your actual CSV header
#             required_cols = ['image_path', 'pos_x_ecef', 'pos_y_ecef', 'pos_z_ecef']
#             # required_cols = ['image_path', 'latitude', 'longitude', 'height']
#             actual_fieldnames = [name.strip() for name in reader.fieldnames] # Clean field names

#             col_mapping = {}
#             missing_cols = []
#             for req_col in required_cols:
#                 found = False
#                 for act_col in actual_fieldnames:
#                     if req_col == act_col: # Exact match first
#                         col_mapping[req_col] = act_col
#                         found = True
#                         break
#                 if not found: # Try case-insensitive or partial match if needed (simplified here)
#                     missing_cols.append(req_col)
            
#             if missing_cols:
#                 print(f"Error: ECEF CSV file '{ecef_csv_path}' is missing one or more required columns: {missing_cols}.")
#                 print(f"  Available columns in CSV: {actual_fieldnames}")
#                 conn.close()
#                 return 0
            

#             for row_idx, row in enumerate(reader):
#                 try:
#                     abs_image_path_csv = row[col_mapping['image_path']]
                    
#                     # Extract the relative path that should match COLMAP's image.name
#                     # Example: /home/wxx/.../mapping/cam0/undistorted/img.jpg
#                     # We want: cam0/img.jpg
#                     if path_anchor_to_split_after in abs_image_path_csv:
#                         # path_after_anchor will be like "cam0/undistorted/img.jpg"
#                         path_after_anchor = abs_image_path_csv.split(path_anchor_to_split_after, 1)[1]
#                         parts = path_after_anchor.split(os.sep)

#                         # path_after_anchor = abs_image_path_csv.split(path_anchor_to_split_after, 1)[1]
#                         # colmap_image_name = os.path.basename(path_after_anchor)

#                         if len(parts) >= 2:
#                             # parts[0] is "cam0", parts[-1] is "img.jpg"
#                             colmap_image_name = os.path.join(parts[0], parts[-1])
#                         else:
#                             # Fallback if the path structure is not as expected (e.g., no subfolder after anchor)
#                             print(f"Warning: Path structure for '{abs_image_path_csv}' after anchor '{path_anchor_to_split_after}' is not 'folder/file'. Using basename instead.")
#                             colmap_image_name = os.path.basename(abs_image_path_csv)
#                     else:
#                         # Fallback or error if anchor is not found
#                         print(f"Warning: Anchor '{path_anchor_to_split_after}' not found in CSV image_path '{abs_image_path_csv}' for row {row_idx+1}. Skipping this row.")
#                         print(f"  Ensure IMAGE_DIR ('{image_dir_colmap_root}') correctly corresponds to a segment in the CSV image paths.")
#                         continue
                        
#                     pos_x = float(row[col_mapping['pos_x_ecef']])
#                     pos_y = float(row[col_mapping['pos_y_ecef']])
#                     pos_z = float(row[col_mapping['pos_z_ecef']])

#                     # pos_x = float(row[col_mapping['longitude']])
#                     # pos_y = float(row[col_mapping['latitude']])
#                     # pos_z = float(row[col_mapping['height']])
                    
#                     ecef_priors_to_update_in_db[colmap_image_name] = (pos_x, pos_y, pos_z)
#                 except ValueError as ve:
#                     print(f"Warning: Skipping ECEF data for row {row_idx+1} ('{abs_image_path_csv if 'abs_image_path_csv' in locals() else 'N/A'}') due to invalid number format for ECEF coordinates: {ve}")
#                     continue
#                 except KeyError as ke:
#                     print(f"Warning: Skipping row {row_idx+1} due to missing key {ke}. Check CSV column names.")
#                     continue
#     except Exception as e:
#         print(f"Error reading ECEF CSV file {ecef_csv_path}: {e}")
#         conn.close()
#         return 0

#     if not ecef_priors_to_update_in_db:
#         print("No valid ECEF priors read and processed from CSV. Skipping ECEF prior update in DB.")
#         conn.close()
#         return 0

#     updated_count = 0
#     not_found_in_db_count = 0
    
#     try:
#         cursor.execute("SELECT prior_tx, prior_ty, prior_tz FROM images LIMIT 1;")
#     except sqlite3.OperationalError:
#         print("Columns prior_tx, prior_ty, prior_tz not found in images table. Attempting to add them.")
#         try:
#             cursor.execute("ALTER TABLE images ADD COLUMN prior_tx REAL;")
#             cursor.execute("ALTER TABLE images ADD COLUMN prior_ty REAL;")
#             cursor.execute("ALTER TABLE images ADD COLUMN prior_tz REAL;")
#             conn.commit()
#             print("Successfully added prior_tx, prior_ty, prior_tz columns to images table.")
#         except sqlite3.Error as e_alter:
#             print(f"Failed to add ECEF prior columns to images table: {e_alter}. Cannot proceed with ECEF update.")
#             conn.close()
#             return -1 # Indicate critical failure

#     for colmap_name, (tx, ty, tz) in ecef_priors_to_update_in_db.items():
#         try:
#             cursor.execute("""
#                 UPDATE images 
#                 SET prior_tx = ?, prior_ty = ?, prior_tz = ?
#                 WHERE name = ?
#             """, (tx, ty, tz, colmap_name))
            
#             if cursor.rowcount > 0:
#                 updated_count += 1
#             else:
#                 if not_found_in_db_count < 10:
#                      print(f"Warning: Image name '{colmap_name}' (derived from CSV) not found in the COLMAP database.")
#                 not_found_in_db_count +=1
#         except sqlite3.Error as e_update:
#             print(f"SQLite error when updating ECEF for '{colmap_name}': {e_update}")
    
#     if not_found_in_db_count > 10:
#         print(f"Warning: ...and {not_found_in_db_count - 10} more images from ECEF CSV were not found in the database.")

#     conn.commit()
#     conn.close()

#     print(f"ECEF prior update: {updated_count} images updated in the database.")
#     if not_found_in_db_count > 0:
#          print(f"ECEF prior update: {not_found_in_db_count} images from CSV (after path processing) were not found in the database.")
#     if updated_count == 0 and len(ecef_priors_to_update_in_db) > 0:
#         print("WARNING: No ECEF priors were actually updated in the database, though CSV data was read. Check for mismatches in image names.")
#         print(f"  Example of processed image name from CSV: {list(ecef_priors_to_update_in_db.keys())[0] if ecef_priors_to_update_in_db else 'N/A'}")
#         # You might want to query and print an example image name from the DB here for comparison
#         # Example:
#         # temp_conn = sqlite3.connect(db_path)
#         # temp_cursor = temp_conn.cursor()
#         # temp_cursor.execute("SELECT name FROM images LIMIT 1;")
#         # db_example_name = temp_cursor.fetchone()
#         # if db_example_name:
#         #     print(f"  Example image name from COLMAP DB: {db_example_name[0]}")
#         # temp_conn.close()

#     return updated_count


# update_images_with_ecef_priors(DB_PATH, ECEF_POSES_CSV_PATH, IMAGE_DIR)
# print("Image ECEF priors update process complete.")




# import os
# import csv
# import sqlite3
# import numpy as np

# def update_pose_priors_with_ecef(db_path, ecef_csv_path, image_dir_colmap_root):
#     if not os.path.exists(ecef_csv_path):
#         print(f"Error: ECEF CSV file not found at {ecef_csv_path}.")
#         return 0

#     print(f"Reading ECEF priors from: {ecef_csv_path}")
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     ecef_priors = {}  # {colmap_image_name: (tx, ty, tz)}

#     image_dir_basename = os.path.basename(image_dir_colmap_root.rstrip(os.sep))
#     path_anchor_to_split_after = image_dir_basename + os.sep
#     print(f"Using anchor '{path_anchor_to_split_after}' to extract image names.")

#     try:
#         with open(ecef_csv_path, 'r', newline='') as f:
#             reader = csv.DictReader(f)
#             required_cols = ['image_path', 'pos_x_ecef', 'pos_y_ecef', 'pos_z_ecef']
#             fieldnames = [name.strip() for name in reader.fieldnames]
#             col_mapping = {}

#             for col in required_cols:
#                 match = [f for f in fieldnames if f.lower() == col.lower()]
#                 if match:
#                     col_mapping[col] = match[0]
#                 else:
#                     print(f"Missing column '{col}' in CSV.")
#                     return 0

#             for row in reader:
#                 try:
#                     abs_path = row[col_mapping['image_path']]
#                     if path_anchor_to_split_after in abs_path:
#                         relative_path = abs_path.split(path_anchor_to_split_after, 1)[1]
#                         parts = relative_path.split(os.sep)
#                         if len(parts) >= 2:
#                             colmap_name = os.path.join(parts[0], parts[-1])
#                         else:
#                             colmap_name = os.path.basename(abs_path)
#                     else:
#                         print(f"Anchor not found in path '{abs_path}', skipping.")
#                         continue

#                     tx = float(row[col_mapping['pos_x_ecef']])
#                     ty = float(row[col_mapping['pos_y_ecef']])
#                     tz = float(row[col_mapping['pos_z_ecef']])
#                     ecef_priors[colmap_name] = (tx, ty, tz)
#                 except Exception as e:
#                     print(f"Error processing row: {e}")
#     except Exception as e:
#         print(f"Error reading CSV: {e}")
#         conn.close()
#         return 0

#     if not ecef_priors:
#         print("No valid ECEF priors parsed.")
#         conn.close()
#         return 0

#     updated_count = 0
#     not_found_count = 0

#     for name, (tx, ty, tz) in ecef_priors.items():
#         cursor.execute("SELECT image_id FROM images WHERE name = ?", (name,))
#         result = cursor.fetchone()
#         if result:
#             image_id = result[0]
#             position_blob = np.array([tx, ty, tz], dtype=np.float64).tobytes()

#             try:
#                 cursor.execute("""
#                     INSERT OR REPLACE INTO pose_priors (image_id, position, coordinate_system)
#                     VALUES (?, ?, 0)
#                 """, (image_id, position_blob))
#                 updated_count += 1
#             except sqlite3.Error as e:
#                 print(f"Failed to insert pose prior for image '{name}': {e}")
#         else:
#             if not_found_count < 10:
#                 print(f"Image '{name}' not found in COLMAP database.")
#             not_found_count += 1

#     conn.commit()
#     conn.close()

#     print(f"Updated {updated_count} pose priors.")
#     if not_found_count > 0:
#         print(f"{not_found_count} images not found in the database.")
#     return updated_count


# # 调用函数
# update_pose_priors_with_ecef(DB_PATH, ENU_POSES_CSV_PATH, IMAGE_DIR)
# print("Pose priors ECEF update process complete.")








# def update_pose_priors_with_enu(db_path, enu_csv_path):
#     """
#     从 CSV 文件更新 COLMAP 数据库中的 ENU 位置先验。
#     CSV 文件必须包含 'camera_id', 'image_id', 'X_enu', 'Y_enu', 'Z_enu' 列。
#     此函数会填充 'pose_priors' 表
#     """
#     if not os.path.exists(enu_csv_path):
#         print(f"错误: ENU CSV 文件未找到于 {enu_csv_path}.")
#         return 0

#     print(f"正在从以下位置读取 ENU 先验: {enu_csv_path}")
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     # --- 如果 'pose_priors' 表不存在，则创建它 ---
#     try:
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS pose_priors (
#                 image_id INTEGER PRIMARY KEY NOT NULL,
#                 position BLOB,
#                 coordinate_system INTEGER NOT NULL,
#                 FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
#         """)
#         conn.commit()
#         print("已确保 'pose_priors' 表存在。")
#     except sqlite3.Error as e:
#         print(f"创建 'pose_priors' 表时出错: {e}")
#         conn.close()
#         return 0

#     enu_priors = {}  # {colmap_image_name: (tx, ty, tz)}

#     try:
#         with open(enu_csv_path, 'r', newline='') as f:
#             reader = csv.DictReader(f)
            
#             # 清理列名以处理潜在的前后空格
#             reader.fieldnames = [name.strip() for name in reader.fieldnames]
            
#             required_cols = ['camera_id', 'image_id', 'X_enu', 'Y_enu', 'Z_enu']
            
#             # 检查所有必需的列是否存在
#             if not all(col in reader.fieldnames for col in required_cols):
#                 missing = [col for col in required_cols if col not in reader.fieldnames]
#                 print(f"错误: CSV 文件中缺少必需的列: {missing}")
#                 print(f"可用的列: {reader.fieldnames}")
#                 conn.close()
#                 return 0

#             for row in reader:
#                 try:
#                     cam_folder = row['camera_id'].strip()
#                     image_filename = row['image_id'].strip()
                    
#                     # 构建与 COLMAP 数据库中 'name' 列匹配的名称 (例如, "cam4/12345.jpg")
#                     colmap_name = os.path.join(cam_folder, image_filename)

#                     tx = float(row['X_enu'])
#                     ty = float(row['Y_enu'])
#                     tz = float(row['Z_enu'])
                    
#                     enu_priors[colmap_name] = (tx, ty, tz)
#                 except (ValueError, KeyError) as e:
#                     print(f"警告: 因错误跳过行: {e}。行内容: {row}")
#                     continue
#     except Exception as e:
#         print(f"读取 CSV 文件时出错: {e}")
#         conn.close()
#         return 0

#     if not enu_priors:
#         print("未从 CSV 文件中解析出有效的 ENU 先验。")
#         conn.close()
#         return 0

#     updated_count = 0
#     not_found_count = 0

#     for name, (tx, ty, tz) in enu_priors.items():
#         cursor.execute("SELECT image_id FROM images WHERE name = ?", (name,))
#         result = cursor.fetchone()
#         if result:
#             image_id = result[0]
#             # 从 numpy 数组创建位置的二进制 blob
#             position_blob = np.array([tx, ty, tz], dtype=np.float64).tobytes()

#             try:
#                 # 插入或替换此 image_id 的先验
#                 # 坐标系 0 = ECEF, 1 = ENU
#                 cursor.execute("""
#                     INSERT OR REPLACE INTO pose_priors (image_id, position, coordinate_system)
#                     VALUES (?, ?, 1)
#                 """, (image_id, position_blob))
#                 updated_count += 1
#             except sqlite3.Error as e:
#                 print(f"为图像 '{name}' 插入 ENU 位姿先验失败: {e}")
#         else:
#             if not_found_count < 10:
#                 print(f"警告: 在 COLMAP 数据库中未找到来自 CSV 的图像 '{name}'。")
#             not_found_count += 1
    
#     if not_found_count >= 10:
#         print(f"警告: ...以及另外 {not_found_count - 10} 张图像未找到。")

#     conn.commit()
#     conn.close()

#     print(f"成功更新/插入了 {updated_count} 个 ENU 位姿先验。")
#     if not_found_count > 0:
#         print(f"来自 CSV 的 {not_found_count} 张图像在数据库中未找到。")
    
#     return updated_count


# # 调用函数
# # 注意：现在函数只需要两个参数
# update_pose_priors_with_enu(DB_PATH, ENU_POSES_CSV_PATH)
# print("位姿先验 ENU 更新过程完成。")




def _read_priors_from_csv(enu_csv_path, camera_filter=None):
    """
    从单个CSV文件读取ENU先验，并可选择性地只为特定相机读取。
    :param enu_csv_path: CSV文件的路径。
    :param camera_filter: 一个包含允许的相机ID的列表/集合，如果为None则读取所有。
    :return: 一个字典 {colmap_image_name: (tx, ty, tz)}
    """
    if not os.path.exists(enu_csv_path):
        print(f"警告: ENU CSV 文件未找到于 {enu_csv_path}。跳过此文件。")
        return {}

    print(f"正在从以下位置读取 ENU 先验: {enu_csv_path}")
    enu_priors = {}
    try:
        with open(enu_csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            required_cols = ['camera_id', 'image_id', 'X_enu', 'Y_enu', 'Z_enu']
            if not all(col in reader.fieldnames for col in required_cols):
                missing = [col for col in required_cols if col not in reader.fieldnames]
                print(f"错误: CSV 文件 '{os.path.basename(enu_csv_path)}' 中缺少必需的列: {missing}")
                return {}

            for row in reader:
                cam_folder = row['camera_id'].strip()
                # 如果设置了过滤器，并且当前相机不在过滤器中，则跳过
                if camera_filter and cam_folder not in camera_filter:
                    continue
                
                try:
                    image_filename = row['image_id'].strip()
                    colmap_name = os.path.join(cam_folder, image_filename)
                    tx = float(row['X_enu'])
                    ty = float(row['Y_enu'])
                    tz = float(row['Z_enu'])
                    enu_priors[colmap_name] = (tx, ty, tz)
                except (ValueError, KeyError) as e:
                    print(f"警告: 因错误跳过行: {e}。行内容: {row}")
                    continue
    except Exception as e:
        print(f"读取 CSV 文件 '{enu_csv_path}' 时出错: {e}")
    
    return enu_priors

def update_all_pose_priors(db_path, mapping_csv, query_csv, all_camera_folders):
    """
    从多个CSV文件更新所有相机的位姿先验。
    cam1 从 query_csv 获取，其他相机从 mapping_csv 获取。
    """
    # 1. 为除 cam1 之外的所有相机读取先验
    mapping_cameras = [cam for cam in all_camera_folders if cam != 'cam1']
    all_priors = _read_priors_from_csv(mapping_csv, camera_filter=mapping_cameras)
    print(f"从 mapping CSV 读到 {len(all_priors)} 个先验。")

    # # 2. 为 cam1 读取先验并合并
    # cam1_priors = _read_priors_from_csv(query_csv, camera_filter=['cam1'])
    # print(f"从 query CSV 读到 {len(cam1_priors)} 个 'cam1' 的先验。")
    # all_priors.update(cam1_priors)

    # if not all_priors:
    #     print("未从任何 CSV 文件中解析出有效的 ENU 先验。")
    #     return 0

    # 3. 连接数据库并更新
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pose_priors (
                image_id INTEGER PRIMARY KEY NOT NULL,
                position BLOB,
                coordinate_system INTEGER NOT NULL,
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"创建 'pose_priors' 表时出错: {e}")
        conn.close()
        return 0

    updated_count = 0
    not_found_count = 0
    for name, (tx, ty, tz) in all_priors.items():
        cursor.execute("SELECT image_id FROM images WHERE name = ?", (name,))
        result = cursor.fetchone()
        if result:
            image_id = result[0]
            position_blob = np.array([tx, ty, tz], dtype=np.float64).tobytes()
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO pose_priors (image_id, position, coordinate_system)
                    VALUES (?, ?, 1)
                """, (image_id, position_blob))
                updated_count += 1
            except sqlite3.Error as e:
                print(f"为图像 '{name}' 插入 ENU 位姿先验失败: {e}")
        else:
            if not_found_count < 10:
                print(f"警告: 在 COLMAP 数据库中未找到来自 CSV 的图像 '{name}'。")
            not_found_count += 1
    
    if not_found_count >= 10:
        print(f"警告: ...以及另外 {not_found_count - 10} 张图像未找到。")

    conn.commit()
    conn.close()

    print(f"成功更新/插入了 {updated_count} 个 ENU 位姿先验。")
    if not_found_count > 0:
        print(f"来自 CSV 的 {not_found_count} 张图像在数据库中未找到。")
    
    return updated_count

# ... (脚本中其他函数如 parse_camera_yaml_for_opencv 等保持不变) ...

# ===== 在主流程中调用新的更新函数 =====
# ... (在 "Camera intrinsics update process complete." 之后) ...

print("\nStep 1.7: Updating image ENU priors in the database from CSV files...")
# 调用新的、更灵活的函数
# update_all_pose_priors(DB_PATH, ENU_POSES_CSV_PATH, ENU_POSES_CSV_PATH_1, CAMERA_FOLDERS)
update_all_pose_priors(DB_PATH, ENU_POSES_CSV_PATH, None, CAMERA_FOLDERS)
print("位姿先验 ENU 更新过程完成。")




# ===== 2. 匹配特征（序列匹配，适合行驶序列） =====  
# 基于图像的采集顺序（时间顺序）进行匹配。
# 只匹配相邻帧（或一定范围内的帧），例如视频或相机轨迹中相邻图像之间。
# 优点：计算量较小，速度快，适合连续拍摄场景。
# 缺点：当相邻视角变化较小时效果好，但如果存在闭环或远距离重复场景，容易错过匹配。

print("\nStep 2: Running sequential matcher...")
subprocess.run([
    "colmap", "sequential_matcher", # sequential_matcher; exhaustive_matcher; space matcher
    "--database_path", DB_PATH,
    "--SiftMatching.guided_matching", "1",  # 采用引导匹配提高精度
    # You might need to add SiftMatching options if defaults are not desired
    "--SiftMatching.use_gpu", "1"
], check=True)
print("Sequential matching complete.")


    
# # ===== 2. 匹配特征 (空间匹配) =====
# print("\nStep 2: Running spatial matcher...")
# subprocess.run([
#     "colmap", "spatial_matcher",
#     "--database_path", DB_PATH,
#     "--SpatialMatching.max_distance", "100", # 仅对距离小于 100 米的图像进行特征匹配
#     "--SpatialMatching.ignore_z", "0",
#     "--SpatialMatching.max_num_neighbors", "50",  # Optional: Max number of neighbors to match against，每个图像最多匹配的邻近图像数量。
#     # "--SpatialMatching.is_gps", "0", # 指定图像位置信息是否为 GPS 坐标（经纬度），1: 使用 GPS 经纬度（自动转换成米）
#     "--SiftMatching.use_gpu", "1" # For the SIFT matching part
# ], check=True)
# print("Spatial matching complete.")


# ===== 3. 建立 SfM 模型 =====
print("\nStep 3: Running mapper...")
subprocess.run([
    "colmap", "mapper",
    "--database_path", DB_PATH,
    "--image_path", IMAGE_DIR,
    "--output_path", SPARSE_DIR, # Mapper outputs to a subfolder (e.g., SPARSE_DIR/0)
    "--Mapper.min_num_matches", "15",
    "--Mapper.init_min_num_inliers", "20", # 10
    #     # --- 添加以下参数来固定内参 ---
    # "--Mapper.ba_refine_focal_length", "0",
    # "--Mapper.ba_refine_principal_point", "0",
    # "--Mapper.ba_refine_extra_params", "0"  # 固定所有畸变参数
], check=True)
print("Mapper complete.")




# print("\nStep 4: Aligning all models in SPARSE_DIR to ECEF coordinates...")

# # 遍历 sparse 目录下所有子目录（假设都是数字命名的模型目录）
# model_dirs = []
# for name in os.listdir(SPARSE_DIR):
#     path = os.path.join(SPARSE_DIR, name)
#     if os.path.isdir(path) and name.isdigit():
#         model_dirs.append((int(name), path))
# # 按数字排序，保证顺序
# model_dirs.sort()

# if not model_dirs:
#     print(f"Error: No model directories found in {SPARSE_DIR}. Skipping alignment.")
# else:
#     # 为 model_aligner 创建 geo-coordinates 文件（只需创建一次）
#     geocoords_txt_path = os.path.join(OUTPUT_DIR, "ecef_geocoords_for_aligner.txt")
#     if create_geocoords_file_for_aligner(ECEF_POSES_CSV_PATH, geocoords_txt_path, IMAGE_DIR):
#         for idx, unaligned_model_path in model_dirs:
#             aligned_model_dir = os.path.join(OUTPUT_DIR, f"sparse_aligned_ecef_{idx}")
#             os.makedirs(aligned_model_dir, exist_ok=True)

#             aligner_command = [
#                 "colmap", "model_aligner",
#                 "--input_path", unaligned_model_path,
#                 "--output_path", aligned_model_dir,
#                 "--ref_images_path", geocoords_txt_path,
#                 "--alignment_type", "ecef",
#                 "--ref_is_gps", "0",
#                 "--alignment_max_error", "1",
#             ]

#             print(f"\nRunning model_aligner for model {idx} with command: {' '.join(aligner_command)}")
#             try:
#                 subprocess.run(aligner_command, check=True)
#                 print(f"Model {idx} alignment complete.")
#                 print(f"Aligned model saved to: {aligned_model_dir}")
#             except subprocess.CalledProcessError as e:
#                 print(f"Error during model {idx} alignment: {e}")
#                 print("Proceeding with next model (if possible) or check errors.")
#             except FileNotFoundError:
#                 print("Error: colmap command not found. Make sure COLMAP is installed and in your PATH.")
#                 break
#     else:
#         print("Skipping model alignment due to issues creating the geo-coordinates file.")



# print("\n✅ 模型构建完成 (使用空间匹配和ECEF先验)，输出目录为：", SPARSE_DIR)
# print("包含文件：cameras.bin, images.bin, points3D.bin")

















# import os
# import subprocess
# import csv # 如果 create_geocoords_file_for_aligner 需要
# import re # 新增导入 re 模块
# from hloc.utils.read_write_model import read_images_binary # <--- 新增导入

# # ===== 路径设置 =====
# # 确保这些路径已在您的脚本中正确定义
# # IMAGE_DIR = "datasets/SensorKit/mapping"
# # OUTPUT_DIR = "datasets/SensorKit/new_colmap_spatial_intrinsics_aligned_only"
# # SPARSE_DIR = "datasets/SensorKit/new_colmap_spatial_intrinsics/sparse"
# # ECEF_POSES_CSV_PATH = "datasets/SensorKit/mapping_camera_ecef_poses.csv"
# # def create_geocoords_file_for_aligner(...): ... # 确保此函数已定义

# print("\nStep 4: Aligning all models in SPARSE_DIR to ECEF coordinates...")

# model_dirs = []
# if not os.path.exists(SPARSE_DIR):
#     print(f"Error: Sparse directory '{SPARSE_DIR}' does not exist. Cannot find models to align.")
# else:
#     for name in os.listdir(SPARSE_DIR):
#         path = os.path.join(SPARSE_DIR, name)
#         if os.path.isdir(path) and name.isdigit():
#             model_dirs.append((int(name), path))
#     model_dirs.sort()

# # 用于存储所有模型统计数据的文件路径
# stats_output_txt_path = os.path.join(OUTPUT_DIR, "model_alignment_stats.txt")

# if not model_dirs:
#     print(f"Error: No model directories found in {SPARSE_DIR}. Skipping alignment.")
# else:
#     geocoords_txt_path = os.path.join(OUTPUT_DIR, "ecef_geocoords_for_aligner.txt")
    
#     with open(stats_output_txt_path, 'w') as stats_file:
#         stats_file.write("# Model_Index Mean_Alignment_Error Median_Alignment_Error Num_Registered_Images\n")

#         if create_geocoords_file_for_aligner(ECEF_POSES_CSV_PATH, geocoords_txt_path, IMAGE_DIR):
#             for idx, unaligned_model_path in model_dirs:
#                 aligned_model_dir = os.path.join(OUTPUT_DIR, f"sparse_aligned_ecef_{idx}")
#                 os.makedirs(aligned_model_dir, exist_ok=True)

#                 aligner_command = [
#                     "colmap", "model_aligner",
#                     "--input_path", unaligned_model_path,
#                     "--output_path", aligned_model_dir,
#                     "--ref_images_path", geocoords_txt_path,
#                     "--alignment_type", "ecef",
#                     "--ref_is_gps", "0",
#                     "--alignment_max_error", "1",
#                 ]

#                 print(f"\nRunning model_aligner for model {idx} (from {unaligned_model_path})")
#                 print(f"  Command: {' '.join(aligner_command)}")
                
#                 mean_align_error, median_align_error = None, None
#                 num_images = None # 初始化为 None

#                 try:
#                     aligner_result = subprocess.run(aligner_command, check=True, capture_output=True, text=True, encoding='utf-8')
#                     print(f"Model {idx} alignment complete.")
#                     print(f"Aligned model saved to: {aligned_model_dir}")

#                     # 强制打印 STDOUT 和 STDERR 进行调试 (如果需要，可以取消注释)
#                     # print(f"--- DEBUG STDOUT for model {idx} ---")
#                     # print(aligner_result.stdout)
#                     # print(f"--- END DEBUG STDOUT for model {idx} ---")
#                     # print(f"--- DEBUG STDERR for model {idx} ---")
#                     # print(aligner_result.stderr)
#                     # print(f"--- END DEBUG STDERR for model {idx} ---")

#                     error_pattern = r"Alignment error:\s*([\d.]+)\s*\(mean\),\s*([\d.]+)\s*\(median\)"
#                     residuals_pattern = r"Residuals:\s*([\d.]+)\s*\(mean\),\s*([\d.]+)\s*\(median\)"
#                     error_match = None
                    
#                     combined_output = aligner_result.stdout + "\n" + aligner_result.stderr
                    
#                     for line in combined_output.splitlines():
#                         current_match = re.search(error_pattern, line)
#                         if current_match:
#                             error_match = current_match
#                             break 
#                         if not error_match: 
#                             current_match = re.search(residuals_pattern, line)
#                             if current_match:
#                                 error_match = current_match
#                                 break
                    
#                     if error_match:
#                         mean_align_error = float(error_match.group(1))
#                         median_align_error = float(error_match.group(2))
#                         print(f"  Extracted Alignment Error: Mean={mean_align_error:.6f}, Median={median_align_error:.6f}")
#                     else:
#                         print(f"  Warning: Could not parse alignment error from COLMAP output for model {idx}.")
#                         # print(f"  Combined COLMAP output (first 1000 chars) for model {idx}:\n{combined_output[:1000]}")

#                     # --- 从 images.bin 读取注册图像数量 ---
#                     images_bin_path = os.path.join(aligned_model_dir, "images.bin")
#                     if os.path.exists(images_bin_path):
#                         try:
#                             images_data = read_images_binary(images_bin_path)
#                             num_images = len(images_data)
#                             print(f"  Extracted Number of Images from images.bin: {num_images}")
#                         except Exception as e_read_bin:
#                             print(f"  Error reading {images_bin_path}: {e_read_bin}")
#                             num_images = "ErrorReadingBin" # 或保持为 None
#                     else:
#                         print(f"  Warning: {images_bin_path} not found. Cannot determine number of images.")
#                         num_images = "BinNotFound" # 或保持为 None
#                     # --- 结束读取 images.bin ---

#                 except subprocess.CalledProcessError as e:
#                     print(f"Error during model {idx} alignment: {e}")
#                     if e.stdout: print("COLMAP aligner stdout on error:\n", e.stdout)
#                     if e.stderr: print("COLMAP aligner stderr on error:\n", e.stderr)
#                 except FileNotFoundError:
#                     print("Error: colmap command not found. Make sure COLMAP is installed and in your PATH.")
#                     break 

#                 mean_err_str = f"{mean_align_error:.6f}" if mean_align_error is not None else "N/A"
#                 median_err_str = f"{median_align_error:.6f}" if median_align_error is not None else "N/A"
#                 num_img_str = str(num_images) if num_images is not None else "N/A" # num_images 现在可能包含错误字符串
                
#                 stats_file.write(f"{idx} {mean_err_str} {median_err_str} {num_img_str}\n")
#                 stats_file.flush() 
#         else:
#             print("Skipping model alignment due to issues creating the geo-coordinates file.")
#             stats_file.write("# Skipped model alignment due to issues creating the geo-coordinates file.\n")
    
#     print(f"\n📝 Model alignment statistics saved to: {stats_output_txt_path}")

# print("\n✅ Model alignment process and statistics collection complete.")














# import os
# import subprocess

# # ===== 路径设置 =====
# IMAGE_DIR = "datasets/SensorKit_test/mapping/jpg"  # 图像路径（建议使用正前摄像头提取帧）
# OUTPUT_DIR = "datasets/SensorKit_test/colmap_seque"
# DB_PATH = os.path.join(OUTPUT_DIR, "database.db")
# SPARSE_DIR = os.path.join(OUTPUT_DIR, "sparse")

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(SPARSE_DIR, exist_ok=True)

# # ===== 1. 特征提取（使用默认 SIFT） =====
# subprocess.run([
#     "colmap", "feature_extractor",
#     "--database_path", DB_PATH,
#     "--image_path", IMAGE_DIR
# ], check=True)

# # ===== 2. 匹配特征（序列匹配，适合行驶序列） =====
# subprocess.run([
#     "colmap", "sequential_matcher",
#     "--database_path", DB_PATH,
#     "--SiftMatching.guided_matching", "1",  # 采用引导匹配提高精度
# ], check=True)

# # ===== 3. 建立 SfM 模型 =====
# subprocess.run([
#     "colmap", "mapper",
#     "--database_path", DB_PATH,
#     "--image_path", IMAGE_DIR,
#     "--output_path", SPARSE_DIR,
#     "--Mapper.min_num_matches", "15",                # 控制三角化图像数目
#     "--Mapper.init_min_num_inliers", "20",
# ], check=True)

# print("\n✅ 模型构建完成，输出目录为：", SPARSE_DIR)
# print("包含文件：cameras.bin, images.bin, points3D.bin")









# import os
# import subprocess
# import shutil

# # ===== 路径设置 =====
# IMAGE_DIR = "datasets/SensorKit/IMAGES"  # 图像路径
# OUTPUT_DIR = "datasets/SensorKit/colmap_seque_prior"
# DB_PATH = os.path.join(OUTPUT_DIR, "database.db")
# SPARSE_DIR = os.path.join(OUTPUT_DIR, "sparse")
# PRIOR_POSE_PATH = "datasets/SensorKit/prior_poses_for_colmap.txt"  # 你的先验 pose 文件路径


# # 创建临时目录，仅包含前 100 张图像
# TEMP_IMAGE_DIR = "datasets/SensorKit/IMAGES_100"
# os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
# all_images = sorted(os.listdir(IMAGE_DIR))[:10]  # 获取前 10 张图像
# for img in all_images:
#     shutil.copy(os.path.join(IMAGE_DIR, img), os.path.join(TEMP_IMAGE_DIR, img))

# # 更新 IMAGE_DIR 为临时目录
# IMAGE_DIR = TEMP_IMAGE_DIR



# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(SPARSE_DIR, exist_ok=True)

# # ===== 1. 特征提取（使用默认 SIFT） =====
# subprocess.run([
#     "colmap", "feature_extractor",
#     "--database_path", DB_PATH,
#     "--image_path", IMAGE_DIR
# ], check=True)

# # ===== 2. 匹配特征（序列匹配，适合行驶序列） =====
# subprocess.run([
#     "colmap", "sequential_matcher",
#     "--database_path", DB_PATH,
#     "--SiftMatching.guided_matching", "1",  # 引导匹配
# ], check=True)


# # ===== 3. 建图 =====
# subprocess.run([
#     "colmap", "mapper",
#     "--database_path", DB_PATH,
#     "--image_path", IMAGE_DIR,
#     "--output_path", SPARSE_DIR,
#     "--Mapper.min_num_matches", "15",
#     "--Mapper.init_min_num_inliers", "20",
# ], check=True)


# print("\n✅ 基于先验位姿的模型构建完成，输出目录为：", SPARSE_DIR)
# print("包含文件：cameras.bin, images.bin, points3D.bin")










# # ===== 基于先验位姿对齐稀疏模型 =====

# import os
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# from hloc.utils.read_write_model import read_model, write_model, Image

# # ========== 配置 ==========
# IMAGES_BIN_PATH = "datasets/SensorKit/colmap_seque/sparse/1"  # 原 images.bin
# GPS_PRIOR_PATH = "datasets/SensorKit/ecef_prior_poses_for_colmap.txt"  # 图片名 四元数 位置 的先验文件
# OUTPUT_DIR = "datasets/SensorKit/colmap_seque_prior/sparse_aligned/1/1_only_scale"  # 输出文件夹

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ========== 辅助函数 ==========
# def qvec2rotmat(qvec):
#     """Convert COLMAP qvec format to rotation matrix."""
#     return R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

# def rotmat2qvec(Rmat):
#     """Convert rotation matrix to COLMAP qvec format."""
#     quat = R.from_matrix(Rmat).as_quat()  # (x, y, z, w)
#     return np.array([quat[3], quat[0], quat[1], quat[2]])  # (w, x, y, z)

# def read_prior_gps(path):
#     gps = {}
#     with open(path, 'r') as f:
#         for line in f:
#             if line.startswith('#') or len(line.strip()) == 0:
#                 continue
#             parts = line.strip().split()
#             name = parts[0]
#             qw, qx, qy, qz = map(float, parts[1:5])
#             tx, ty, tz = map(float, parts[5:8])
#             gps[name] = (qw, qx, qy, qz, tx, ty, tz)
#     return gps

# def compute_camera_center(qvec, tvec):
#     """C = -R^T * t"""
#     Rmat = qvec2rotmat(qvec)
#     center = -Rmat.T @ tvec
#     return center

# def estimate_similarity_transform(source, target):
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
#     scale = np.sum(S) / var_src
#     t_est = mu_tgt - scale * R_est @ mu_src
#     return scale, R_est, t_est

# # ========== 主程序 ==========

# print("📥 读取数据中...")
# # 读取 .bin 文件
# cameras, images, points3D = read_model(IMAGES_BIN_PATH)
# gps_priors = read_prior_gps(GPS_PRIOR_PATH)


# # 提取 images 的文件名作为键
# image_names = {image.name: image_id for image_id, image in images.items()}

# # 提取匹配的图片
# common_names = sorted(set(image_names.keys()) & set(gps_priors.keys()))
# print(f"✅ 匹配到 {len(common_names)} 张共同图片用于对齐。")


# # 提取 camera 中心 & GPS 中心
# sparse_centers = []
# gps_centers = []
# for name in common_names:
#     image_id = image_names[name]
#     image = images[image_id]
#     sparse_centers.append(compute_camera_center(image.qvec, image.tvec))

#     _, _, _, _, tx, ty, tz = gps_priors[name]
#     gps_centers.append(np.array([tx, ty, tz]))

# sparse_centers = np.array(sparse_centers)
# gps_centers = np.array(gps_centers)

# print("🔍 正在估计相似变换 Sim(3)...")
# scale, R_est, t_est = estimate_similarity_transform(sparse_centers, gps_centers)
# print(f"✨ 估计得到：scale={scale:.6f}")

# # 保存 R_est 旋转矩阵
# r_est_save_path = os.path.join(OUTPUT_DIR, "R_est_transform.txt")
# np.savetxt(r_est_save_path, R_est)
# print(f"💾 相似变换的旋转矩阵 R_est 已保存到: {r_est_save_path}")

# # 变换 points3D
# print("🔧 应用变换到 3D点...")
# new_points = {}
# for pid, point in points3D.items():  # 注意这里获取的是整个 Point3D 对象
#     xyz = point.xyz  # 获取 3D 坐标数组
#     xyz_new = scale * (R_est @ xyz) + t_est
#     new_points[pid] = point._replace(xyz=xyz_new)  


# new_images = images


# # # 变换 images
# # print("🔧 应用变换到相机位姿...")

# # new_images = {}
# # OUTPUT_POSES_FILE = "span_poses_ecef.txt" 

# # with open(OUTPUT_POSES_FILE, 'w') as f:
# #     # 可选：写入表头
# #     f.write("# image_name qw qx qy qz pos_x pos_y pos_z\n")

# #     for image_id, image in images.items():
# #         qvec = image.qvec
# #         tvec = image.tvec
# #         camera_id = image.camera_id

# #         R_orig = qvec2rotmat(qvec)
# #         center_orig = compute_camera_center(qvec, tvec)
# #         center_new = scale * (R_est @ center_orig) + t_est
# #         R_new = R_est @ R_orig
# #         t_new = -R_new @ center_new
# #         qvec_new = rotmat2qvec(R_new)

# #         new_images[image_id] = Image(
# #             id=image_id,
# #             qvec=qvec_new,
# #             tvec=t_new,
# #             camera_id=camera_id,
# #             name=image.name,  # 正确写入图像名
# #             xys=image.xys,
# #             point3D_ids=image.point3D_ids
# #         )

# #         # 准备写入文件的数据
# #         # 四元数：原始的 qvec 就是 (w, x, y, z) 顺序
# #         qw, qx, qy, qz = qvec_new[0], qvec_new[1], qvec_new[2], qvec_new[3]
# #         # 位置：使用变换后的相机中心 center_new_ecef (x, y, z)
# #         pos_x, pos_y, pos_z = center_new[0], center_new[1], center_new[2]

# #         # 将图像名、四元数和位置写入文件
# #         f.write(f"{image.name} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {pos_x:.3f} {pos_y:.3f} {pos_z:.3f}\n")

# # print(f"🔧 相机位姿已保存到 {OUTPUT_POSES_FILE}")





# # new_images = {}

# # # 定义输出文件名
# # OUTPUT_POSES_FILE = "span_poses_ecef.txt" # 您可以根据需要更改文件名

# # with open(OUTPUT_POSES_FILE, 'w') as f:
# #     # 可选：写入表头
# #     f.write("# image_name qw qx qy qz pos_x pos_y pos_z\n")

# #     for image_id, image in images.items():
# #         qvec = image.qvec # 原始 COLMAP 四元数 (w, x, y, z)
# #         tvec = image.tvec # 原始 COLMAP 平移向量
# #         camera_id = image.camera_id

# #         R_orig = qvec2rotmat(qvec) # 原始 COLMAP 旋转矩阵
# #         center_orig = compute_camera_center(qvec, tvec) # 原始 COLMAP 相机中心

# #         # 1. 计算新的相机中心位置 (在 ECEF 坐标系下)
# #         #    这里应用完整的 Sim(3) 变换，因为 R_est, t_est, scale 是将整个 COLMAP 模型
# #         #    作为一个刚体（加缩放）对齐到 ECEF 的参数。
# #         #    center_new_ecef 代表了原始相机在对齐后的 ECEF 坐标系中的位置。
# #         center_new_ecef = scale * center_orig + t_est
# #         # center_new_ecef = scale * (R_est @ center_orig) + t_est

# #         # 2. 对于输出到 span_poses_ecef.txt 和新的 .bin 模型：
# #         #    - 位置：使用上面计算的 center_new_ecef
# #         # 准备更新 new_images (用于 .bin 模型)
# #         # 方向：使用原始的 qvec (即 R_orig)
# #         qvec_for_bin = qvec
# #         R_for_bin = R_orig # 对应的旋转矩阵
# #         # R_for_bin = R_est @ R_orig
# #         # 位置：需要计算与 qvec_for_bin (即 R_orig) 和 center_new_ecef 对应的 tvec
# #         # COLMAP中 tvec = -R_world_to_camera @ C_world
# #         # 这里 R_for_bin (即 R_orig) 是从（现在被认为是ECEF对齐的）世界到相机的旋转
# #         tvec_for_bin = -R_for_bin @ center_new_ecef

# #         new_images[image_id] = Image(
# #             id=image_id,
# #             qvec=qvec_for_bin,   # 原始方向
# #             tvec=tvec_for_bin,   # 与原始方向和新的ECEF中心对应的新tvec
# #             camera_id=camera_id,
# #             name=image.name,
# #             xys=image.xys,
# #             point3D_ids=image.point3D_ids
# #         )

# #         # 准备写入文件的数据
# #         # 四元数：原始的 qvec 就是 (w, x, y, z) 顺序
# #         qw, qx, qy, qz = qvec[0], qvec[1], qvec[2], qvec[3]
# #         # 位置：使用变换后的相机中心 center_new_ecef (x, y, z)
# #         pos_x, pos_y, pos_z = center_new_ecef[0], center_new_ecef[1], center_new_ecef[2]

# #         # 将图像名、四元数和位置写入文件
# #         f.write(f"{image.name} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {pos_x:.3f} {pos_y:.3f} {pos_z:.3f}\n")

# # print(f"🔧 相机位姿已保存到 {OUTPUT_POSES_FILE}")





# # new_images = {}

# # # 定义输出文件名
# # OUTPUT_POSES_FILE = "span_poses_ecef.txt" # 您可以根据需要更改文件名

# # with open(OUTPUT_POSES_FILE, 'w') as f:
# #     # 可选：写入表头
# #     f.write("# image_name qw qx qy qz pos_x pos_y pos_z\n")

# #     for image_id, image in images.items():
# #         qvec = image.qvec # COLMAP 格式 (w, x, y, z)
# #         tvec = image.tvec
# #         camera_id = image.camera_id
# #         image_name = image.name # 获取图像文件名

# #         # 保留原始旋转矩阵
# #         R_orig = qvec2rotmat(qvec)

# #         # 计算原始相机中心
# #         center_orig = compute_camera_center(qvec, tvec)

# #         # 应用缩放和平移到相机中心 (这是变换后的ECEF位置)
# #         center_new_ecef = scale * center_orig + t_est # 注意：这里使用的是只应用了缩放和平移的相机中心

# #         # 重新计算平移向量，但保持旋转矩阵不变 (这是COLMAP坐标系下的tvec)
# #         t_new_colmap = -R_orig @ center_new_ecef # 这个t_new是用于更新COLMAP模型的

# #         # 保持原始旋转矩阵对应的四元数 (COLMAP格式 w, x, y, z)
# #         qvec_new_colmap = qvec # 这个qvec_new是用于更新COLMAP模型的

# #         # 创建新的相机对象 (用于更新COLMAP模型)
# #         new_images[image_id] = Image(
# #             id=image_id,
# #             qvec=qvec_new_colmap,  # 保持原始方向 (COLMAP格式)
# #             tvec=t_new_colmap,     # 更新位置 (COLMAP格式)
# #             camera_id=camera_id,
# #             name=image_name,
# #             xys=image.xys,
# #             point3D_ids=image.point3D_ids
# #         )

# #         # 准备写入文件的数据
# #         # 四元数：原始的 qvec 就是 (w, x, y, z) 顺序
# #         qw, qx, qy, qz = qvec_new_colmap[0], qvec_new_colmap[1], qvec_new_colmap[2], qvec_new_colmap[3]
# #         # 位置：使用变换后的相机中心 center_new_ecef (x, y, z)
# #         pos_x, pos_y, pos_z = center_new_ecef[0], center_new_ecef[1], center_new_ecef[2]

# #         # 将图像名、四元数和位置写入文件
# #         f.write(f"{image_name} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {pos_x:.3f} {pos_y:.3f} {pos_z:.3f}\n")

# # print(f"🔧 相机位姿已保存到 {OUTPUT_POSES_FILE}")


# # 保存新的模型 (这部分保持不变)
# print("🔧 保存新的模型到目录...")
# write_model(cameras, new_images, new_points, OUTPUT_DIR, ext=".bin")
# print(f"🎯 完成！新的稀疏模型已保存到 {OUTPUT_DIR}")





# # ===== 不同相机的先验位姿对齐稀疏模型 =====

# import csv # 新增导入 csv 模块
# import os
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# from hloc.utils.read_write_model import read_model, write_model, Image
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # ========== 配置 ==========
# IMAGES_BIN_PATH = "datasets/SensorKit/colmap_seque/sparse/1"  # 原 images.bin
# GPS_PRIOR_PATH = "datasets/SensorKit/mapping_camera_ecef_poses.csv" # 不同相机的ecef先验位姿csv文件路径
# OUTPUT_DIR = "datasets/SensorKit/colmap_seque_prior/sparse_aligned/1/camera_scale"  # 输出文件夹

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ========== 辅助函数 ==========
# def qvec2rotmat(qvec):
#     """Convert COLMAP qvec format to rotation matrix."""
#     return R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

# def rotmat2qvec(Rmat):
#     """Convert rotation matrix to COLMAP qvec format."""
#     quat = R.from_matrix(Rmat).as_quat()  # (x, y, z, w)
#     return np.array([quat[3], quat[0], quat[1], quat[2]])  # (w, x, y, z)

# def compute_camera_center(qvec, tvec):
#     """C = -R^T * t"""
#     Rmat = qvec2rotmat(qvec)
#     center = -Rmat.T @ tvec
#     return center

# def estimate_similarity_transform(source, target):
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
#     scale = np.sum(S) / var_src
#     t_est = mu_tgt - scale * R_est @ mu_src
#     return scale, R_est, t_est

# def read_prior_gps(path):
#     gps = {}
#     with open(path, 'r', newline='') as f: # 修改打开方式以正确处理CSV
#         reader = csv.DictReader(f, delimiter=',') # 使用逗号作为分隔符
#         for row in reader:
#             # 从CSV行中提取所需数据
#             # 图像路径作为键
#             image_full_path = row['image_path']
#             # 从 image_path 中提取图像文件名，用于后续与 COLMAP 模型中的 image.name 匹配
#             # 假设 COLMAP 模型中的 image.name 是不带路径的文件名
#             name = os.path.basename(image_full_path)

#             try:
#                 qw = float(row['quat_w_c2e'])
#                 qx = float(row['quat_x_c2e'])
#                 qy = float(row['quat_y_c2e'])
#                 qz = float(row['quat_z_c2e'])
#                 tx = float(row['pos_x_ecef'])
#                 ty = float(row['pos_y_ecef'])
#                 tz = float(row['pos_z_ecef'])
#                 # 存储时，仍然使用图像文件名作为键，以便与 COLMAP 的 images.name 匹配
#                 gps[name] = (qw, qx, qy, qz, tx, ty, tz)
#             except ValueError as e:
#                 print(f"Warning: Skipping row due to data conversion error: {row} - {e}")
#                 continue
#             except KeyError as e:
#                 print(f"Warning: Skipping row due to missing key: {e} in row {row}")
#                 continue
#     return gps


# # ========== 主程序 ==========

# print("📥 读取数据中...")
# # 读取 .bin 文件
# cameras, images, points3D = read_model(IMAGES_BIN_PATH)
# # GPS_PRIOR_PATH 现在指向CSV文件
# gps_priors = read_prior_gps(GPS_PRIOR_PATH) # GPS_PRIOR_PATH 应设为您的CSV文件路径


# # 提取 images 的文件名作为键
# # COLMAP 模型中的 image.name 通常是不带路径的文件名
# image_names = {image.name: image_id for image_id, image in images.items()}

# # 提取匹配的图片
# # common_names 将包含那些同时存在于 COLMAP 模型和 GPS 先验文件中的图像文件名
# common_names = sorted(set(image_names.keys()) & set(gps_priors.keys()))
# if not common_names:
#     print("❌ 错误：在 COLMAP 模型和 GPS 先验文件中没有找到共同的图像。请检查：")
#     print(f"  1. GPS_PRIOR_PATH ('{GPS_PRIOR_PATH}') 是否正确指向了CSV文件。")
#     print(f"  2. CSV文件中的 'image_path' 列是否包含正确的图像路径，并且其文件名部分能与COLMAP模型中的图像名匹配。")
#     print(f"  3. COLMAP模型路径 IMAGES_BIN_PATH ('{IMAGES_BIN_PATH}') 是否正确。")
#     print(f"  COLMAP中的图像名示例: {list(image_names.keys())[:5] if image_names else '无'}")
#     print(f"  GPS先验中的图像名示例 (从image_path提取): {[os.path.basename(k) for k in list(gps_priors.keys())[:5]] if gps_priors else '无'}") # 这里假设gps_priors的键是文件名
#     exit()
# print(f"✅ 匹配到 {len(common_names)} 张共同图片用于对齐。")


# # 提取 camera 中心 & GPS 中心
# sparse_centers = []
# gps_centers = []
# for name in common_names: # name 是图像文件名
#     image_id = image_names[name]
#     image = images[image_id]
#     sparse_centers.append(compute_camera_center(image.qvec, image.tvec))

#     # 从 gps_priors 中使用图像文件名获取位姿
#     qw, qx, qy, qz, tx, ty, tz = gps_priors[name]
#     gps_centers.append(np.array([tx, ty, tz]))

# sparse_centers = np.array(sparse_centers)
# gps_centers = np.array(gps_centers)

# if sparse_centers.shape[0] < 2 or gps_centers.shape[0] < 2:
#     print(f"❌ 错误：用于估计相似变换的匹配点数量不足 ({sparse_centers.shape[0]}个)。至少需要2个点。")
#     exit()

# print("🔍 正在估计相似变换 Sim(3)...")
# # ...rest of the main program (estimate_similarity_transform, transform points3D, transform images, write_model)...
# # ...existing code...
# scale, R_est, t_est = estimate_similarity_transform(sparse_centers, gps_centers)
# print(f"✨ 估计得到：scale={scale:.6f}")

# # 变换 points3D
# print("🔧 应用变换到 3D点...")
# new_points = {}
# for pid, point in points3D.items():  # 注意这里获取的是整个 Point3D 对象
#     xyz = point.xyz  # 获取 3D 坐标数组
#     xyz_new = scale * (R_est @ xyz) + t_est
#     new_points[pid] = point._replace(xyz=xyz_new)  

# # 变换 images
# print("🔧 应用变换到相机位姿...")
# new_images = {}
# quaternions = []
# positions = []
# for image_id, image in images.items():
#     qvec = image.qvec
#     tvec = image.tvec
#     camera_id = image.camera_id

#     R_orig = qvec2rotmat(qvec)
#     center_orig = compute_camera_center(qvec, tvec)
#     center_new = scale * (R_est @ center_orig) + t_est
#     R_new = R_est @ R_orig
#     t_new = -R_new @ center_new
#     qvec_new = rotmat2qvec(R_new)

#     new_images[image_id] = Image(
#         id=image_id,
#         qvec=qvec_new,
#         tvec=t_new,
#         camera_id=camera_id,
#         name=image.name,  # 正确写入图像名
#         xys=image.xys,
#         point3D_ids=image.point3D_ids
#     )

#     # 读取数据并解析
#     quaternions.append(np.array(qvec_new)) # 获取四元数
#     positions.append(np.array(center_new))  # 获取位置

# quaternions_np = np.array(quaternions)
# positions_np = np.array(positions)
# # 保存到文件
# np.savetxt("quaternions.txt", quaternions_np, fmt="%.6f", delimiter=" ", header="q1 q2 q3 q4")
# np.savetxt("positions.txt", positions_np, fmt="%.6f", delimiter=" ", header="x y z")

# print("🔧 保存相机姿态成功！")

# # 保存新的模型
# print("🔧 保存新的模型到目录...")
# write_model(cameras, new_images, new_points, OUTPUT_DIR, ext=".bin")
# print(f"🎯 完成！新的稀疏模型已保存到 {OUTPUT_DIR}")




























# import h5py

# features_path = 'outputs/SensorKit_hloc/feats-superpoint-n4096.h5'
# matches_path = 'outputs/SensorKit_hloc/feats-superpoint-n4096-r1024_matches-superglue_pairs-db-covis20.h5'
# filtered_features_path = 'outputs/SensorKit_hloc/features_filtered.h5'
# filtered_matches_path = 'outputs/SensorKit_hloc/matches_filtered.h5'

# # 1. 收集所有有匹配的图像名
# connected_images = set()
# with h5py.File(matches_path, 'r') as f:
#     for pair in f.keys():
#         img1, img2 = pair.split(' ')
#         connected_images.add(img1)
#         connected_images.add(img2)

# print(f"Connected images: {len(connected_images)}")

# # 2. 过滤 features.h5
# with h5py.File(features_path, 'r') as fin, h5py.File(filtered_features_path, 'w') as fout:
#     for name in fin.keys():
#         if name in connected_images:
#             fin.copy(name, fout)

# print(f"Filtered features saved to {filtered_features_path}")









# import h5py

# features_path = 'outputs/SensorKit_hloc/feats-superpoint-n4096-r1024.h5' 
# matches_path = 'outputs/SensorKit_hloc/feats-superpoint-n4096-r1024_matches-superglue_pairs-db-covis20.h5'   

# # features_path = 'outputs/cmu_extended/slice2/feats-superpoint-n4096-r1024.h5' 
# # matches_path = 'outputs/cmu_extended/slice2/feats-superpoint-n4096-r1024_matches-superglue_pairs-db-covis20.h5'    

# def print_h5_keys(h5_path, title):
#     print(f'\n=== {title} ===')
#     with h5py.File(h5_path, 'r') as f:
#         keys = list(f.keys())
#         for i, k in enumerate(keys[:10]):  
#             print(f'{i+1:2d}: {k}')
#         print(f'Total: {len(keys)}')

# print_h5_keys(features_path, 'FEATURES.H5 KEYS')
# print_h5_keys(matches_path, 'MATCHES.H5 KEYS (image pairs)')


# import h5py

# matches_path = 'outputs/SensorKit_hloc/feats-superpoint-n4096-r1024_matches-superglue_pairs-db-covis20.h5'  
# new_matches_path = 'outputs/SensorKit_hloc/feats-superpoint-n4096-r1024_matches-superglue_pairs-db-covis20_.h5'  

# def format_key(key):
#     """将键格式化为新的路径（将 - 替换为 /）"""
#     return key.replace('-', '/')

# def format_and_save_keys(h5_path, new_h5_path):
#     with h5py.File(h5_path, 'r') as f:
#         # 获取原始 H5 文件中的所有键
#         keys = list(f.keys())
        
#         # 创建一个新的 H5 文件来保存数据
#         with h5py.File(new_h5_path, 'w') as new_f:
#             for key in keys:
#                 # 格式化键
#                 new_key = format_key(key)
#                 data = f[key]
                
#                 # 检查数据类型，如果是字符串类型，则跳过或进行处理
#                 if isinstance(data, h5py.Group):
#                     # 如果是组（Group），直接跳过
#                     continue
#                 elif isinstance(data, np.ndarray):
#                     try:
#                         # 尝试将数据保存到新文件中
#                         new_f.create_dataset(new_key, data=data)
#                     except TypeError:
#                         print(f"Skipping dataset {key} due to unsupported data type.")
#                         continue
#                 else:
#                     print(f"Skipping dataset {key} due to unsupported data type.")

# # 调用函数，读取原 H5 文件并创建新的 H5 文件
# format_and_save_keys(matches_path, new_matches_path)

# print(f"Formatted H5 file saved as {new_matches_path}")









