import argparse
import logging
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """设置随机种子以保证可复现性。"""
    random.seed(seed)
    logger.info(f"随机种子已设置为: {seed}")


def split_data(image_root: Path, output_dir: Path, train_split_ratio: float) -> (Path, Path):
    """
    遍历图像根目录，递归查找所有图像，并按比例划分为训练集和验证集。
    将文件路径列表写入 .txt 文件。
    """
    logger.info(f"正在从 '{image_root}' 递归扫描图像文件...")
    image_paths = sorted(list(image_root.rglob("*.jpg")) + list(image_root.rglob("*.png")))
    if not image_paths:
        raise FileNotFoundError(f"在目录 '{image_root}' 中未找到任何 .jpg 或 .png 图像文件。")

    logger.info(f"共找到 {len(image_paths)} 个图像文件。")
    
    # 打乱顺序以进行随机划分
    random.shuffle(image_paths)
    
    split_idx = int(len(image_paths) * train_split_ratio)
    train_files = image_paths[:split_idx]
    val_files = image_paths[split_idx:]

    logger.info(f"数据划分完成: {len(train_files)} 个训练样本, {len(val_files)} 个验证样本。")

    # 将文件列表写入临时文件
    train_list_path = output_dir / "tmp_train_files.txt"
    val_list_path = output_dir / "tmp_val_files.txt"

    with open(train_list_path, "w") as f:
        f.write("\n".join(map(str, train_files)))
    with open(val_list_path, "w") as f:
        f.write("\n".join(map(str, val_files)))

    logger.info(f"训练集和验证集文件列表已保存至: {train_list_path} 和 {val_list_path}")
    return train_list_path, val_list_path


def generate_captions_for_list(
    image_list_path: Path,
    output_csv_path: Path,
    model_name: str,
    caption_script_path: Path,
):
    """
    调用外部脚本为给定的图像列表生成描述。
    注意：这要求 generate_captions.py 支持 --image_list 参数。
    """
    if output_csv_path.exists():
        logger.info(f"描述文件 '{output_csv_path}' 已存在，跳过生成。")
        return

    logger.info(f"--- 正在使用模型 '{model_name}' 为列表 '{image_list_path.name}' 生成描述 ---")
    
    # 假设 generate_captions.py 接受 --image_list 参数
    cmd = [
        sys.executable,  # 用当前正在运行的 Python 解释器（通常是 python3）
        str(caption_script_path),
        "--image_list", str(image_list_path),
        "--output_csv", str(output_csv_path),
        "--model", model_name,
    ]
    
    try:
        # 使用 tqdm 手动包装 subprocess，提供更好的进度反馈
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 实时读取标准错误流以显示可能的进度条或日志
        with tqdm(desc=f"Captioning with {model_name}", unit="img") as pbar:
             for line in iter(process.stderr.readline, ''):
                # 这是一个简化的进度条更新，实际效果取决于 caption 脚本的输出
                if "Processing image" in line:
                    pbar.update(1)
                sys.stderr.write(line) # 打印原始日志
        
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout, stderr=stderr)

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"使用模型 '{model_name}' 生成描述失败。")
        logger.error(f"执行的命令: {' '.join(cmd)}")
        if hasattr(e, 'stderr'):
            logger.error(f"错误输出:\n{e.stderr}")
        raise


def create_final_csvs(
    train_list_path: Path,
    val_list_path: Path,
    output_dir: Path,
    caption_models: list,
    caption_model_for_val: str,
    caption_script_path: Path,
):
    """
    整合所有生成的描述，创建最终的 train.csv 和 val.csv。
    """
    # --- 1. 处理训练集 ---
    logger.info("开始为训练集生成并合并描述...")
    all_train_dfs = []
    for model_name in caption_models:
        intermediate_csv = output_dir / f"intermediate_captions_{model_name}_train.csv"
        generate_captions_for_list(train_list_path, intermediate_csv, model_name, caption_script_path)
        
        if intermediate_csv.exists():
            df = pd.read_csv(intermediate_csv)
            all_train_dfs.append(df)
        else:
            logger.warning(f"未找到中间描述文件 {intermediate_csv}，将忽略。")

    if not all_train_dfs:
        raise RuntimeError("未能为训练集生成任何描述文件。")

    # 合并所有描述，实现“同图多描述”
    final_train_df = pd.concat(all_train_dfs, ignore_index=True)
    # 重命名列以符合训练脚本的期望
    final_train_df = final_train_df.rename(columns={"image_path": "filepath", "description": "title"})
    
    train_csv_path = output_dir / "train.csv"
    final_train_df.to_csv(train_csv_path, index=False)
    logger.info(f"最终训练集 'train.csv' 已创建，包含 {len(final_train_df)} 个 (图像, 文本) 对。")

    # --- 2. 处理验证集 ---
    logger.info(f"开始为验证集生成描述 (使用模型: {caption_model_for_val})...")
    intermediate_val_csv = output_dir / f"intermediate_captions_{caption_model_for_val}_val.csv"
    generate_captions_for_list(val_list_path, intermediate_val_csv, caption_model_for_val, caption_script_path)
    
    if not intermediate_val_csv.exists():
        raise RuntimeError("未能为验证集生成描述文件。")
        
    final_val_df = pd.read_csv(intermediate_val_csv)
    final_val_df = final_val_df.rename(columns={"image_path": "filepath", "description": "title"})
    
    val_csv_path = output_dir / "val.csv"
    final_val_df.to_csv(val_csv_path, index=False)
    logger.info(f"最终验证集 'val.csv' 已创建，包含 {len(final_val_df)} 个样本。")


def main():
    parser = argparse.ArgumentParser(
        description="数据预处理脚本：划分数据集，生成多模型描述，并创建用于CLIP微调的CSV文件。"
    )
    parser.add_argument(
        "--image_root", type=Path, required=True, help="包含所有待处理图像的根目录。"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="保存所有输出（文件列表、CSV文件）的目录。"
    )
    parser.add_argument(
        "--caption_script_path", type=Path, required=True, help="指向 'generate_captions.py' 脚本的路径。"
    )
    parser.add_argument(
        "--train_split_ratio", type=float, default=0.9, help="训练集所占的比例 (0.0 到 1.0)。"
    )
    parser.add_argument(
        "--caption_models",
        nargs="+",
        default=["llava", "llava-ov", "qwen2-vl"],
        help="用于为训练集生成描述的VLM模型列表。",
    )
    parser.add_argument(
        "--caption_model_for_val",
        type=str,
        default="llava",
        help="用于为验证集生成描述的单一模型。",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="用于数据划分的随机种子。"
    )

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # --- 执行流程 ---
    set_seed(args.seed)
    
    # 1. 划分数据，得到文件列表
    train_list_path, val_list_path = split_data(args.image_root, args.output_dir, args.train_split_ratio)
    
    # 2. 生成描述并创建最终的 CSV 文件
    create_final_csvs(
        train_list_path,
        val_list_path,
        args.output_dir,
        args.caption_models,
        args.caption_model_for_val,
        args.caption_script_path,
    )

    logger.info("数据预处理流程全部完成！")
    logger.info(f"最终的训练和验证CSV文件已保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
