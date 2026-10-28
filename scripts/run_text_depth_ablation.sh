#!/usr/bin/env bash
# Compare text LoRA depth while keeping the visual LoRA on the final 12 blocks.
set -euo pipefail

TEXT_DEPTH="${1:?Usage: bash scripts/run_text_depth_ablation.sh 4|8}"
case "$TEXT_DEPTH" in
    4|8) ;;
    *) echo "Expected text LoRA depth 4 or 8" >&2; exit 2 ;;
esac

OUT_DIR="datasets/SensorKit/clipdata/finetune_output/text_depth_ablation_v1/text${TEXT_DEPTH}_visual12"
if [[ -e "$OUT_DIR/lora_epoch_1.pt" ]]; then
    echo "Training output already exists: $OUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"
echo "Training text-${TEXT_DEPTH}/visual-12 LoRA from DataComp weights; output $OUT_DIR"

python3 train_localclip.py \
    --train-csv datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/00_train_qwen3.csv \
    --val-csv datasets/SensorKit/clipdata/astri_412_down03/val_qwen3.csv \
    --train-image-root-dir datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove \
    --val-image-root-dir datasets/SensorKit/clipdata/astri_412_down03 \
    --out-dir "$OUT_DIR" \
    --model-name ViT-L-14 \
    --pretrained datacomp_xl_s13b_b90k \
    --epochs 22 \
    --batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-3 \
    --num-workers 12 \
    --seed 42 \
    --lora-last-n 12 \
    --lora-text-last-n "$TEXT_DEPTH" \
    --lora-visual-last-n 12 \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0 \
    --lora-target all \
    --use-amp >"$OUT_DIR/train.log" 2>&1

echo "Completed text-${TEXT_DEPTH}/visual-12; log $OUT_DIR/train.log"
