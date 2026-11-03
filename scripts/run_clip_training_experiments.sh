#!/bin/bash
# Paper main model: text-last-8 / visual-last-12 LoRA, global CLIP loss only.

set -e

TRAIN_CSV="datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/00_train_qwen3.csv"
VAL_CSV="datasets/SensorKit/clipdata/astri_412_down03/val_qwen3.csv"
TRAIN_IMG_ROOT="datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/"
VAL_IMG_ROOT="datasets/SensorKit/clipdata/astri_412_down03/"
BASE_OUT_DIR="datasets/SensorKit/clipdata/finetune_output"

EPOCHS=${EPOCHS:-40}
BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_WORKERS=12

OUT_DIR="$BASE_OUT_DIR/text_depth_ablation_v1/text8_visual12"
echo "train_localclip: dual LoRA (text 8, visual 12), main CSV descriptions"
python3 train_localclip.py \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --train-image-root-dir "$TRAIN_IMG_ROOT" \
    --val-image-root-dir "$VAL_IMG_ROOT" \
    --out-dir "$OUT_DIR" \
    --pretrained "datacomp_xl_s13b_b90k" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --num-workers "$NUM_WORKERS" \
    --use-amp \
    --lora-target all \
    --lora-last-n 12 \
    --lora-text-last-n 8 \
    --lora-visual-last-n 12 \
    --lora-r 8 \
    --lora-alpha 16 \
    --seed 42 \
    --skip-hf-baseline

echo "Done. Main checkpoint: $OUT_DIR/lora_epoch_22.pt"
