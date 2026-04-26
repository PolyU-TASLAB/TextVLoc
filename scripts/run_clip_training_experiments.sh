#!/bin/bash
# train_localclip：文本 LoRA + 长描述 + cross-attn。chmod +x run_clip_training_experiments.sh && ./run_clip_training_experiments.sh

set -e

TRAIN_CSV="datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/00_train_qwen3.csv"
VAL_CSV="datasets/SensorKit/clipdata/astri_412_down03/val_qwen3.csv"
TRAIN_IMG_ROOT="datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/"
VAL_IMG_ROOT="datasets/SensorKit/clipdata/astri_412_down03/"
BASE_OUT_DIR="datasets/SensorKit/clipdata/finetune_output"

EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_WORKERS=12

echo "train_localclip: text LoRA + long captions + cross-attn"
python3 train_localclip.py \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --train-image-root-dir "$TRAIN_IMG_ROOT" \
    --val-image-root-dir "$VAL_IMG_ROOT" \
    --out-dir "$BASE_OUT_DIR/stg2_00_qwen3_clipdata_text_mask_prepare_sparse03_remove_long_/" \
    --pretrained "datacomp_xl_s13b_b90k" \
    --training-text-mode "long" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --num-workers "$NUM_WORKERS" \
    --use-amp \
    --lora-target text \
    --use-cross-attn \
    --cross-attn-type full \
    --num-visual-queries 16 \
    --lambda-cross 0.3

echo "Done. Outputs under $BASE_OUT_DIR/stg2_00_qwen3_clipdata_text_mask_prepare_sparse03_remove_long_/"
