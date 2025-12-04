import os
import subprocess
import sqlite3
import csv

# ==============================================================================
# 路径设置 (*** 请根据您的项目修改此部分 ***)
# ==============================================================================

# 1. COLMAP 项目的主输出目录
#    这个目录应该包含 database.db 和 sparse/ 文件夹
PROJECT_DIR = "datasets/SensorKit/colmap_seque"

# 2. 包含 ENU 真值位姿的 CSV 文件路径
ENU_POSES_CSV_PATH = "datasets/SensorKit/enu_mapping_gt.csv" # 假设这是您的新CSV文件名

# --- 以下路径通常不需要修改 ---

# 输入: 包含 COLMAP 数据库和未对齐模型的路径
DB_PATH = os.path.join(PROJECT_DIR, "database.db")
# *** 修改: 明确指定输入模型为 sparse/1 ***
INPUT_SPARSE_DIR = os.path.join(PROJECT_DIR, "sparse/1")

# 输出: 对齐后模型的保存路径
OUTPUT_SPARSE_DIR = os.path.join(PROJECT_DIR, "sparse_aligned_enu")

# 中间文件: 为对齐器生成的参考文件
REF_IMAGES_TXT_PATH = os.path.join(PROJECT_DIR, "ref_images_for_aligner_enu.txt")

# ==============================================================================
# 脚本主逻辑
# ==============================================================================

def create_reference_file_for_aligner():
    """
    从数据库和ENU CSV文件生成 model_aligner 所需的参考文件。
    文件格式: image_name tx ty tz
    """
    print("--- Step 1: Generating reference file for COLMAP model_aligner (ENU) ---")

    # 检查所需文件是否存在
    if not os.path.exists(DB_PATH):
        print(f"错误: 找不到数据库文件: {DB_PATH}")
        return False
    if not os.path.exists(ENU_POSES_CSV_PATH):
        print(f"错误: 找不到 ENU 位姿 CSV 文件: {ENU_POSES_CSV_PATH}")
        return False
    if not os.path.isdir(INPUT_SPARSE_DIR):
        print(f"错误: 找不到输入的稀疏模型目录: {INPUT_SPARSE_DIR}")
        return False

    # 读取 ENU CSV 并直接写入参考文件
    print(f"Processing ENU priors from: {ENU_POSES_CSV_PATH}")
    lines_written = 0
    
    # 从数据库中读取所有图像名称，用于验证
    conn = sqlite3.connect(DB_PATH)
    db_image_names = set(row[0] for row in conn.cursor().execute("SELECT name FROM images;"))
    conn.close()
    print(f"Found {len(db_image_names)} images in the database for validation.")

    with open(ENU_POSES_CSV_PATH, 'r') as f_csv, open(REF_IMAGES_TXT_PATH, 'w') as f_txt:
        # 跳过标题行
        next(f_csv) 
        reader = csv.reader(f_csv, delimiter=',') # 使用制表符作为分隔符
        
        for row in reader:
            try:
                # 解析CSV行
                # camera_id, timestamp_ns, image_id, X_enu, Y_enu, Z_enu
                if len(row) < 6: continue # 跳过格式不正确的行
                
                image_name_with_cam = f"{row[2]}" #  "image_name.jpg"
                
                # 验证图像是否存在于数据库中
                if image_name_with_cam not in db_image_names:
                    # print(f"警告: 图像 '{image_name_with_cam}' 在CSV中，但不在数据库中。跳过。")
                    continue

                tx = float(row[3])
                ty = float(row[4])
                tz = float(row[5])
                
                # 写入格式化后的行: image_name tx ty tz
                f_txt.write(f"{image_name_with_cam} {tx} {ty} {tz}\n")
                lines_written += 1
            except (ValueError, IndexError) as e:
                print(f"警告: 处理CSV行时出错: {e}. 行内容: {row}. 跳过。")
                continue
    
    if lines_written > 0:
        print(f"Successfully wrote {lines_written} entries to reference file: {REF_IMAGES_TXT_PATH}")
        return True
    else:
        print("错误: 未能从CSV文件中写入任何有效的参考位姿。")
        return False


def run_model_aligner():
    """
    执行 colmap model_aligner 命令。
    """
    print("\n--- Step 2: Running COLMAP model_aligner ---")
    
    os.makedirs(OUTPUT_SPARSE_DIR, exist_ok=True)

    align_command = [
        "colmap", "model_aligner",
        "--input_path", INPUT_SPARSE_DIR,
        "--output_path", OUTPUT_SPARSE_DIR,
        "--ref_images_path", REF_IMAGES_TXT_PATH,
        "--ref_is_gps", "0",  # *** 修改: 0 表示非GPS/ECEF坐标 ***
        "--alignment_type", "custom",
        "--alignment_max_error", "1"# *** 修改: custom 对应非ECEF的3D坐标 ***
    ]

    print("Executing command:")
    print(" ".join(align_command))
    
    try:
        subprocess.run(align_command, check=True)
        print("\nModel alignment successful!")
        print(f"Aligned model saved to: {OUTPUT_SPARSE_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"\n错误: colmap model_aligner 执行失败，返回码: {e.returncode}")
    except FileNotFoundError:
        print("\n错误: 'colmap' 命令未找到。请确保 COLMAP 已安装并在系统 PATH 中。")

if __name__ == "__main__":
    if create_reference_file_for_aligner():
        run_model_aligner()
    else:
        print("\n由于未能成功生成参考文件，对齐过程已中止。")