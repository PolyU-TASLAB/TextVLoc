#!/bin/bash
set -euo pipefail
python3 -m hloc.pipelines.RobotCar.clip_hloc \
    --dataset "${DATASET:-datasets/SensorKit}" \
    --outputs "${OUTPUTS:-outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025}" \
    --num_loc 20 --caption_model qwen-vl \
    --clip_model_id laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K \
    --lora_ckpt_path checkpoints/text8_visual12_epoch22.pt \
    --positive_pairs_csv protocols/urbanv2x_paper_positive_pairs.csv \
    --text_only_loc --enable_region_rerank \
    --region_grid 3 --region_top_k 20 --region_top_m 1 \
    --region_reduce max --region_alpha 0.75 "$@"
