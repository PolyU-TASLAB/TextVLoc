#!/bin/bash
# filepath: scripts/test_all_clip_models.sh

# 设置通用参数
DATASET="datasets/SensorKit"
OUTPUTS="outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025"
NUM_LOC=20
CAPTION_MODEL="qwen-vl"
POSITIVE_PAIRS="datasets/SensorKit/query/cam1_all/05_allcams_positive_pairs_for_eval.csv"

# ============================================================
# CLIP 系列 (OpenAI & LAION)
# ============================================================
echo "========================================="
echo "🔬 Testing CLIP Models"
echo "========================================="

# 1. OpenAI CLIP-ViT-Large-Patch14
echo "📦 [1/6] Testing: openai/clip-vit-large-patch14"
python3 -m hloc.pipelines.RobotCar.clip_hloc \
    --dataset $DATASET \
    --outputs "$OUTPUTS" \
    --num_loc $NUM_LOC \
    --caption_model $CAPTION_MODEL \
    --clip_model_id "openai/clip-vit-large-patch14" \
    --text_only_loc \
    --positive_pairs_csv $POSITIVE_PAIRS

# 2. OpenAI CLIP-ViT-Base-Patch16
echo "📦 [2/6] Testing: openai/clip-vit-base-patch16"
python3 -m hloc.pipelines.RobotCar.clip_hloc \
    --dataset $DATASET \
    --outputs "$OUTPUTS" \
    --num_loc $NUM_LOC \
    --caption_model $CAPTION_MODEL \
    --clip_model_id "openai/clip-vit-base-patch16" \
    --text_only_loc \
    --positive_pairs_csv $POSITIVE_PAIRS

# 3. LAION CLIP-ViT-L-14-DataComp (推荐)
echo "📦 [3/6] Testing: laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
python3 -m hloc.pipelines.RobotCar.clip_hloc \
    --dataset $DATASET \
    --outputs "$OUTPUTS" \
    --num_loc $NUM_LOC \
    --caption_model $CAPTION_MODEL \
    --clip_model_id "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K" \
    --text_only_loc \
    --positive_pairs_csv $POSITIVE_PAIRS

# 4. LAION CLIP-ViT-H-14
echo "📦 [4/6] Testing: laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
python3 -m hloc.pipelines.RobotCar.clip_hloc \
    --dataset $DATASET \
    --outputs "$OUTPUTS" \
    --num_loc $NUM_LOC \
    --caption_model $CAPTION_MODEL \
    --clip_model_id "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
    --text_only_loc \
    --positive_pairs_csv $POSITIVE_PAIRS

# 5. MetaCLIP (Meta)
echo "📦 [5/6] Testing: facebook/metaclip-l14-fullcc2.5b"
python3 -m hloc.pipelines.RobotCar.clip_hloc \
    --dataset $DATASET \
    --outputs "$OUTPUTS" \
    --num_loc $NUM_LOC \
    --caption_model $CAPTION_MODEL \
    --clip_model_id "facebook/metaclip-l14-fullcc2.5b" \
    --text_only_loc \
    --positive_pairs_csv $POSITIVE_PAIRS

# 6. LongCLIP
echo "📦 [6/6] Testing: zer0int/LongCLIP-GmP-ViT-L-14"
python3 -m hloc.pipelines.RobotCar.clip_hloc \
    --dataset $DATASET \
    --outputs "$OUTPUTS" \
    --num_loc $NUM_LOC \
    --caption_model $CAPTION_MODEL \
    --clip_model_id "zer0int/LongCLIP-GmP-ViT-L-14" \
    --text_only_loc \
    --positive_pairs_csv $POSITIVE_PAIRS

echo "✅ CLIP 系列测试完成!"