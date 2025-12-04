# python3 -m hloc.pipelines.RobotCar.prepare_finetune_data \
#         --image_root datasets/SensorKit/clipdata \
#         --output_dir datasets/SensorKit/clipdata/finetune_data/all \
#         --caption_script_path hloc/generate_captions.py \
#         --caption_models qwen2-vl \
#         --caption_model_for_val qwen2-vl \
#         --train_split_ratio 0.9 \
#         --seed 42



# python3 train_clip.py \
#     --train-csv datasets/SensorKit/clipdata/finetune_data/astri_412_down/train.csv \
#     --val-csv datasets/SensorKit/clipdata/finetune_data/astri_412_down/val.csv \
#     --out-dir datasets/SensorKit/clipdata/finetune_output/astri_412_down/clip-lora-run1 \
#     --mode lora \
#     --model-name "ViT-H-14" \
#     --pretrained "laion2b_s32b_b79k" \
#     --epochs 20 \
#     --batch-size 32 \
#     --lr 1e-4 \
#     --use-amp \
#     --multi-text \
#     --grouped-sampler


# python3 -m hloc.pipelines.RobotCar.clip_hloc --caption_model qwen-vl --lora_ckpt_path datasets/SensorKit/clipdata/finetune_output/astri_412_down/clip-datacomp-lora0-run1/lora_epoch6.pt


# python3 -m hloc.pipelines.RobotCar.clip_hloc --caption_model qwen-vl --lora_ckpt_path datasets/SensorKit/clipdata/finetune_output/all/clip-datacomp-lora0-run1/lora_epoch6.pt



# python3 train_clip.py \
#     --train-csv datasets/SensorKit/clipdata/finetune_data/astri_412_down_remove/train.csv \
#     --val-csv datasets/SensorKit/clipdata/finetune_data/astri_412_down_remove/val.csv \
#     --out-dir datasets/SensorKit/clipdata/finetune_output/astri_412_down_remove/clip-datacomp-lora0-run1 \
#     --mode lora \
#     --model-name "ViT-L-14" \
#     --pretrained "datacomp_xl_s13b_b90k" \
#     --epochs 20 \
#     --batch-size 32 \
#     --lr 1e-4 \
#     --use-amp \
#     --multi-text \
#     --grouped-sampler






#     python3 train_clip.py \
#     --train-csv datasets/SensorKit/clipdata/finetune_data/all/train_qwen2.csv \
#     --val-csv datasets/SensorKit/clipdata/finetune_data/all/val_qwen2.csv \
#     --out-dir datasets/SensorKit/clipdata/finetune_output/all/clip-datacomp-lora0-run1 \
#     --mode lora \
#     --model-name "ViT-L-14" \
#     --pretrained "datacomp_xl_s13b_b90k" \
#     --epochs 20 \
#     --batch-size 32 \
#     --lr 1e-4 \
#     --use-amp \
#     --multi-text \
#     --grouped-sampler










#!/bin/bash

# ==============================================================================
# 该脚本用于批量执行 CLIP LoRA 微调实验。
# 它会依次运行三种不同的微调配置：
# 1. 微调图像和文本两侧 (lora-target: all)
# 2. 只微调图像侧 (lora-target: vision)
# 3. 只微调文本侧 (lora-target: text)
#
# 使用方法:
# 1. 确保此脚本有执行权限: chmod +x run_clip_training_experiments.sh
# 2. 直接运行: ./run_clip_training_experiments.sh
# ==============================================================================



# # --- 步骤 0: 生成或更新文本描述 ---
# echo "============================================================"
# echo "📝 [Step 0] Generating captions for query images..."
# echo "============================================================"

# python3 -m hloc.generate_captions \
#     --query_dir datasets/SensorKit/query/cam1_all/0_undistorted \
#     --output_csv datasets/SensorKit/query/cam1_all/0_undistorted/val_qwen2.csv \
#     --model qwen-vl

# echo "✅ [Step 0] Caption generation finished."
# echo ""




# --- 设置通用参数 ---
# 使用变量可以方便地修改通用路径，避免重复
TRAIN_CSV="datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/00_train_qwen3.csv"
# VAL_CSV="datasets/SensorKit/clipdata/astri_412_down03/val_qwen3.csv"

VAL_CSV="outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/half_qwen3_query_prompts_qwen-vl_77clip10.csv"
VAL_IMG_ROOT="datasets/SensorKit/query/cam1_all/half/"
# VAL_MASK_FEATURES="datasets/SensorKit/query/cam1_all/0_undistorted/test_mask_features.h5"

TRAIN_IMG_ROOT="datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/"
# VAL_IMG_ROOT="datasets/SensorKit/clipdata/astri_412_down03/"
BASE_OUT_DIR="datasets/SensorKit/clipdata/finetune_output"


TRAIN_MASK_FEATURES="datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/train_mask_features.h5"
VAL_MASK_FEATURES="datasets/SensorKit/clipdata/astri_412_down03/val_mask_features.h5"

TRAIN_SHORT_CSV="datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/short_output_csv.csv"
HARDNEG_CSV="datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/00_train_qwen3_hardneg.csv"


# 通用的训练超参数
EPOCHS=50 # 50
BATCH_SIZE=32
LEARNING_RATE=1e-4
# LEARNING_RATE=5e-5 # 根据您的命令更新
NUM_WORKERS=24

# # --- 实验 1: 微调图像和文本两侧 (all) ---
# echo "============================================================"
# echo "🚀 [Experiment 1/3] Starting training for LoRA target: all"
# echo "============================================================"

# # python3 train_clip0.py \
# #     --train-csv "$TRAIN_CSV" \
# #     --val-csv "$VAL_CSV" \
# #     --train-image-root-dir "$TRAIN_IMG_ROOT" \
# #     --val-image-root-dir "$VAL_IMG_ROOT" \
# #     --out-dir "$BASE_OUT_DIR/00_qwen3_clipdata_prepare_sparse03_remove/" \
# #     --epochs "$EPOCHS" \
# #     --batch-size "$BATCH_SIZE" \
# #     --lr "$LEARNING_RATE" \
# #     --num-workers "$NUM_WORKERS" \
# #     --use-amp \
# #     --lora-target all


# python3 train_localclip.py \
#     --train-csv "$TRAIN_CSV" \
#     --val-csv "$VAL_CSV" \
#     --train-image-root-dir "$TRAIN_IMG_ROOT" \
#     --val-image-root-dir "$VAL_IMG_ROOT" \
#     --train-mask-features "$TRAIN_MASK_FEATURES" \
#     --val-mask-features "$VAL_MASK_FEATURES" \
#     --out-dir "$BASE_OUT_DIR/00_qwen3_clipdata_mask_prepare_sparse03_remove/" \
#     --epochs "$EPOCHS" \
#     --batch-size "$BATCH_SIZE" \
#     --lr "$LEARNING_RATE" \
#     --num-workers "$NUM_WORKERS" \
#     --use-amp \
#     --lora-target all


# echo "✅ [Experiment 1/3] Finished training for LoRA target: all"
# echo ""


# # --- 实验 2: 只微调图像侧 (vision) ---
# echo "============================================================"
# echo "🚀 [Experiment 2/3] Starting training for LoRA target: vision"
# echo "============================================================"

# # python3 train_clip0.py \
# #     --train-csv "$TRAIN_CSV" \
# #     --val-csv "$VAL_CSV" \
# #     --train-image-root-dir "$TRAIN_IMG_ROOT" \
# #     --val-image-root-dir "$VAL_IMG_ROOT" \
# #     --out-dir "$BASE_OUT_DIR/00_qwen3_clipdata_vision_prepare_sparse03_remove/" \
# #     --epochs "$EPOCHS" \
# #     --batch-size "$BATCH_SIZE" \
# #     --lr "$LEARNING_RATE" \
# #     --num-workers "$NUM_WORKERS" \
# #     --use-amp \
# #     --lora-target vision


# python3 train_localclip.py \
#     --train-csv "$TRAIN_CSV" \
#     --val-csv "$VAL_CSV" \
#     --train-image-root-dir "$TRAIN_IMG_ROOT" \
#     --val-image-root-dir "$VAL_IMG_ROOT" \
#     --train-mask-features "$TRAIN_MASK_FEATURES" \
#     --val-mask-features "$VAL_MASK_FEATURES" \
#     --out-dir "$BASE_OUT_DIR/00_qwen3_clipdata_vision_mask_prepare_sparse03_remove/" \
#     --epochs "$EPOCHS" \
#     --batch-size "$BATCH_SIZE" \
#     --lr "$LEARNING_RATE" \
#     --num-workers "$NUM_WORKERS" \
#     --use-amp \
#     --lora-target vision

# echo "✅ [Experiment 2/3] Finished training for LoRA target: vision"
# echo ""


# --- 实验 3: 只微调文本侧 (text) ---
echo "============================================================"
echo "🚀 [Experiment 3/3] Starting training for LoRA target: text"
echo "============================================================"

# python3 train_clip0.py \
#     --train-csv "$TRAIN_CSV" \
#     --val-csv "$VAL_CSV" \
#     --train-image-root-dir "$TRAIN_IMG_ROOT" \
#     --val-image-root-dir "$VAL_IMG_ROOT" \
#     --out-dir "$BASE_OUT_DIR/00_qwen3_clipdata_text_mask_prepare_sparse03_remove/" \
#     --epochs "$EPOCHS" \
#     --batch-size "$BATCH_SIZE" \
#     --lr "$LEARNING_RATE" \
#     --num-workers "$NUM_WORKERS" \
#     --use-amp \
#     --lora-target text

# # 运行 train_localclip.py 脚本
# python3 train_localclip.py \
#     --train-csv "$TRAIN_CSV" \
#     --train-short-csv "$TRAIN_SHORT_CSV" \
#     --training-text-mode long \
#     --val-csv "$VAL_CSV" \
#     --train-image-root-dir "$TRAIN_IMG_ROOT" \
#     --val-image-root-dir "$VAL_IMG_ROOT" \
#     --out-dir "$BASE_OUT_DIR/00_qwen3_clipdata_text_mask_prepare_sparse03_remove/" \
#     --epochs "$EPOCHS" \
#     --batch-size "$BATCH_SIZE" \
#     --lr "$LEARNING_RATE" \
#     --num-workers "$NUM_WORKERS" \
#     --use-amp \
#     --lora-target text 


# echo "✅ [Experiment 3/3] Finished training for LoRA target: text"
# echo ""
# echo "🎉 All experiments completed!"


# # --- 生成长文本描述 ---
# echo "============================================================"
# echo "📝 Generating captions for training data..."
# echo "============================================================"

# python3 -m hloc.generate_captions \
#     --query_dir datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove \
#     --output_csv datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/long_train_qwen3.csv \
#     --model qwen-vl

# echo "✅ Captions generated and saved to long_train_qwen3.csv"
# echo ""

# =================================================================
#  STAGE 2: Fine-tune Spatial Relationship Expert
# =================================================================
echo "🚀 Starting Stage 2: Fine-tuning with hard negatives..."

# 关键：加载第一阶段训练好的模型权重
# 假设第一阶段最好的模型是第15个epoch
PRETRAINED_LORA_PATH="datasets/SensorKit/clipdata/finetune_output/00_qwen3_clipdata_text_prepare_sparse03_remove/lora_epoch_36.pt"

# --- 核心修改：调整输入文件并添加 lambda_hard 参数 ---
python3 train_localclip.py \
    --train-csv "$TRAIN_CSV" \
    --train-hard-negative-csv "$HARDNEG_CSV" \
    --val-csv "$VAL_CSV" \
    --train-image-root-dir "$TRAIN_IMG_ROOT" \
    --val-image-root-dir "$VAL_IMG_ROOT" \
    --out-dir "$BASE_OUT_DIR/stg2_00_qwen3_clipdata_text_mask_prepare_sparse03_remove_long/" \
    --pretrained "datacomp_xl_s13b_b90k" \
    --training-text-mode "hard_negative" \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4\
    --lambda-hard 0.5 \
    --lora-target "vision" \
    --use-amp

echo "✅ Stage 2 finished. Final model saved in $BASE_OUT_DIR/stg2_00_qwen3_clipdata_text_mask_prepare_sparse03_remove/"
echo "🎉 All training stages completed!"

    # --out-dir "$BASE_OUT_DIR/00_qwen3_clipdata_text_prepare_sparse03_remove/" \