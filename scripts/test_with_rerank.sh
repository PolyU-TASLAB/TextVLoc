#!/bin/bash
# filepath: scripts/test_with_rerank.sh

DATASET="datasets/SensorKit"
OUTPUTS="outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025"
NUM_LOC=20
CAPTION_MODEL="qwen-vl"
POSITIVE_PAIRS="datasets/SensorKit/query/cam1_all/05_allcams_positive_pairs_for_eval.csv"

echo "========================================="
echo "🔬 Testing Models with Grid Reranking"
echo "========================================="

# 1. CLIP + Rerank
echo "📦 [1/3] Testing: CLIP + Grid Rerank"
python3 -m hloc.pipelines.RobotCar.clip_hloc \
    --dataset $DATASET \
    --outputs "$OUTPUTS" \
    --num_loc $NUM_LOC \
    --caption_model $CAPTION_MODEL \
    --clip_model_id "zer0int/LongCLIP-GmP-ViT-L-14" \
    --text_only_loc \
    --enable_region_rerank \
    --region_grid 3 \
    --region_top_k 20 \
    --positive_pairs_csv $POSITIVE_PAIRS

# laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
# openai/clip-vit-large-patch14
# QuanSun/EVA-CLIP
# zer0int/LongCLIP-GmP-ViT-L-14


# # 2. SigLIP + Rerank
# echo "📦 [2/3] Testing: SigLIP + Grid Rerank"
# python3 -m hloc.pipelines.RobotCar.clip_hloc \
#     --dataset $DATASET \
#     --outputs "$OUTPUTS" \
#     --num_loc $NUM_LOC \
#     --caption_model $CAPTION_MODEL \
#     --clip_model_id "google/siglip-large-patch16-384" \
#     --text_only_loc \
#     --enable_region_rerank \
#     --region_grid 3 \
#     --region_top_k 20 \
#     --positive_pairs_csv $POSITIVE_PAIRS

# # 3. MetaCLIP + Rerank
# echo "📦 [3/3] Testing: MetaCLIP + Grid Rerank"
# python3 -m hloc.pipelines.RobotCar.clip_hloc \
#     --dataset $DATASET \
#     --outputs "$OUTPUTS" \
#     --num_loc $NUM_LOC \
#     --caption_model $CAPTION_MODEL \
#     --clip_model_id "facebook/metaclip-l14-fullcc2.5b" \
#     --text_only_loc \
#     --enable_region_rerank \
#     --region_grid 3 \
#     --region_top_k 20 \
#     --positive_pairs_csv $POSITIVE_PAIRS

# echo "✅ Reranking 测试完成!"