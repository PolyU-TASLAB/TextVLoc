# TextVLoc

![TextVLoc Overview](overview.png)

TextVLoc is a text-guided visual localization pipeline built on top of [Hierarchical-Localization (HLOC)](https://github.com/cvg/Hierarchical-Localization). It uses Urban Street CLIP, a LoRA-adapted CLIP retriever for urban street-view text-image matching, as the coarse retrieval module. This project targets the `SensorKit` setup in this repository and combines:

- VLM caption generation for query images
- CLIP-style text-image retrieval
- LoRA fine-tuning for domain adaptation
- Optional region reranking for better retrieval quality
- Optional geometric localization with local feature matching and PnP

The codebase is currently organized around the `SensorKit` dataset layout already used in this workspace.

## Features

- Text-only localization from retrieved database poses
- Retrieval with pretrained CLIP / SigLIP / MetaCLIP / LongCLIP backbones
- Urban Street CLIP LoRA fine-tuning in `train_localclip.py`
- Region-aware reranking with `--enable_region_rerank`
- Geometric localization using SuperPoint + LightGlue + `pycolmap`
- Recall evaluation with `--positive_pairs_csv`

## Text Used For Retrieval

This project does not retrieve directly from raw query images. It first converts each query image into a text description, then retrieves database images with a CLIP-style model.

### Inference-time text

The current default inference setup uses:

- caption backend: `qwen-vl`
- actual VLM: `Qwen/Qwen3-VL-8B-Instruct`
- generated file pattern: `qwen3_query_prompts_qwen-vl_77clip10.csv`

Each query image is described with a fixed navigation-oriented prompt in [hloc/generate_captions.py](hloc/generate_captions.py). The generated text is intended to be:

- first-person viewpoint
- focused on static landmarks and road structure
- a single complete sentence
- under 77 words

This design is meant to fit CLIP-style text retrieval better than generic image captions.

### Training-time text

The current training script [scripts/run_clip_training_experiments.sh](scripts/run_clip_training_experiments.sh) launches `train_localclip.py` with:

- `TRAIN_CSV=datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/00_train_qwen3.csv`
- `VAL_CSV=datasets/SensorKit/clipdata/astri_412_down03/val_qwen3.csv`
- `--out-dir datasets/SensorKit/clipdata/finetune_output/qwen3_clipdata_text_sparse03_remove/`

The training script now uses only the main CSV `description` field for LoRA fine-tuning, while the online localization pipeline currently uses the short `77`-word navigation descriptions for retrieval.

## Repository Layout

```text
.
├── hloc/                                  # HLOC code and TextVLoc pipeline
├── scripts/                               # training / evaluation helper scripts
├── train_localclip.py                     # LoRA fine-tuning entrypoint
├── requirements-textvloc.txt             # Python dependencies
├── Dockerfile                             # Reproducible CUDA environment
├── datasets/SensorKit/                    # local dataset root (not tracked)
└── outputs/SensorKit_hloc/                # local experiment outputs (not tracked)
```

## Expected Local Data Layout

This project assumes the following directories already exist locally:

```text
datasets/SensorKit/
├── allcams/                               # database / mapping images
├── query/cam1_all/0_undistorted/          # query images
├── 0620_gt/
│   ├── enu_all_mapping_gt.csv
│   └── enu_all_query_gt.csv
├── allcams_spatial_colmap_0630/           # COLMAP assets
└── clipdata/                              # CLIP fine-tuning data and checkpoints
```

The default pipeline also expects an existing reference SfM model under:

```text
outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/sfm_superpointmax+lightglue
```

A minimal ENU-aligned reference SfM package for UrbanV2X fine localization is available from the GitHub Release:

```bash
wget https://github.com/xxw77777kk/TextVLoc/releases/download/textvloc-urbanv2x-sfm-v1/textvloc_urbanv2x_sfm_minimal.tar.gz
tar -xzf textvloc_urbanv2x_sfm_minimal.tar.gz
mkdir -p outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025
cp -r textvloc_urbanv2x_sfm_minimal/sfm_superpointmax+lightglue \
  outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/
```

The package contains `cameras.bin`, `images.bin`, `points3D.bin`, and `model.ply`. It excludes the COLMAP `database.db`, text exports, generated features, raw images, and checkpoints.

## Environment Setup

### Docker

```bash
docker build -t textvloc .
docker run --gpus all -it \
  --name textvloc-dev \
  --shm-size 16g \
  -v $(pwd):/workspace/TextVLoc \
  -w /workspace/TextVLoc \
  -e PYTHONPATH=/workspace/TextVLoc \
  textvloc bash
```

The Dockerfile installs `requirements-textvloc.txt`, `pycolmap==3.11.1`, and a CUDA-enabled COLMAP build. Some CLIP/VLM checkpoints are downloaded from Hugging Face on first use.

### Local Python

Docker is the recommended environment. If you run locally, install the Python dependencies manually and make sure PyTorch, CUDA, COLMAP, and `pycolmap` match your system:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-textvloc.txt
pip install pycolmap==3.11.1
```

## Training

The main training entrypoint is:

```bash
bash scripts/run_clip_training_experiments.sh
```

That script currently launches `train_localclip.py` to train Urban Street CLIP with:

- text-only LoRA adaptation (`--lora-target text`)
- main CSV descriptions as the only training text source
- cross-attention enabled

You can modify dataset paths, output paths, batch size, and epochs directly in `scripts/run_clip_training_experiments.sh`.

## Inference and Evaluation

The current command-line entrypoint still uses the legacy Python module path
`hloc.pipelines.RobotCar.clip_hloc` for compatibility.

### 1. Text-only localization

This mode retrieves database images from text prompts and directly uses the top retrieved database pose.

```bash
python3 -m hloc.pipelines.RobotCar.clip_hloc \
  --dataset datasets/SensorKit \
  --outputs outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025 \
  --num_covis 10 \
  --num_loc 20 \
  --caption_model qwen-vl \
  --clip_model_id laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K \
  --lora_ckpt_path datasets/SensorKit/clipdata/finetune_output/qwen3_clipdata_text_sparse03_remove/lora_epoch_22.pt \
  --positive_pairs_csv datasets/SensorKit/query/cam1_all/05_allcams_positive_pairs_for_eval.csv \
  --text_only_loc
```

### 2. Text-only localization with region reranking

Add region reranking on top of the text retrieval stage:

```bash
python3 -m hloc.pipelines.RobotCar.clip_hloc \
  --dataset datasets/SensorKit \
  --outputs outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025 \
  --num_covis 10 \
  --num_loc 20 \
  --caption_model qwen-vl \
  --clip_model_id laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K \
  --lora_ckpt_path datasets/SensorKit/clipdata/finetune_output/qwen3_clipdata_text_sparse03_remove/lora_epoch_22.pt \
  --positive_pairs_csv datasets/SensorKit/query/cam1_all/05_allcams_positive_pairs_for_eval.csv \
  --text_only_loc \
  --enable_region_rerank \
  --region_grid 3 \
  --region_top_k 20
```

Useful extra flags:

- `--evaluate_rerank_recall`
- `--region_top_m 2`
- `--region_alpha 0.5`

You can also run the helper script:

```bash
bash scripts/test_with_rerank.sh
```

### 3. Geometric localization with PnP

To enable image-based geometric localization, remove `--text_only_loc`. The pipeline will:

1. generate captions for query images
2. retrieve candidate database images with CLIP
3. convert retrieval results to matching pairs
4. run SuperPoint + LightGlue matching
5. estimate poses with `localize_sfm.main(...)`

Example:

```bash
python3 -m hloc.pipelines.RobotCar.clip_hloc \
  --dataset datasets/SensorKit \
  --outputs outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025 \
  --num_covis 10 \
  --num_loc 20 \
  --caption_model qwen-vl \
  --clip_model_id laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K \
  --lora_ckpt_path datasets/SensorKit/clipdata/finetune_output/qwen3_clipdata_text_sparse03_remove/lora_epoch_22.pt \
  --positive_pairs_csv datasets/SensorKit/query/cam1_all/05_allcams_positive_pairs_for_eval.csv
```

If you want both reranking and geometric localization, keep `--enable_region_rerank` and omit `--text_only_loc`.

### 4. Opposite-view retrieval evaluation

Opposite-view experiments are grouped under `scripts/opposite_view/`. They compare image-based retrieval methods such as NetVLAD, MegaLoc, CosPlace, and EigenPlaces with TextVLoc under viewpoint reversal.

UrbanV2X uses `cam6` as the query stream and `cam1` as the reference/text-source stream. The provided UrbanV2X text mapping is:

```text
outputs/SensorKit_hloc/cam6_to_cam1_opposite_view/seq5_stride1/cam6_query_to_cam1_front_text_qwen-vl.csv
```

Run the UrbanV2X opposite-view protocol with:

```bash
python3 scripts/opposite_view/run_sensorkit_cam6_to_cam1_opposite_view_eval.py \
  --dataset datasets/SensorKit \
  --outputs outputs/SensorKit_hloc/cam6_to_cam1_opposite_view \
  --sequence-length 5 \
  --center-stride 1
```

Run the SYNTHIA SEQS-02 opposite-view protocol with:

```bash
bash scripts/opposite_view/run_synthia_seq02_opposite_view_compare.sh
```

For Oxford RobotCar, first prepare the LoST-style front/rear protocol, build the rear-view text queries from rear-view captions, and then run image-based and TextVLoc retrieval:

```bash
python3 scripts/opposite_view/prepare_lost_robotcar_opposite_view.py
python3 scripts/opposite_view/prepare_lost_robotcar_rear_oracle_text_queries.py \
  --prepared-dir datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km_rectified \
  --rear-captions-csv datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km_rectified/rear_qwen3_oracle_captions.csv
python3 scripts/opposite_view/run_lost_robotcar_retrieval_eval.py \
  --prepared-dir datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km_rectified \
  --out-dir outputs/SensorKit_hloc/lost_robotcar_overcast_autumn_2m_2km_rectified \
  --methods netvlad megaloc cosplace eigenplaces
python3 scripts/opposite_view/run_lost_robotcar_clip_text_eval.py \
  --prepared-dir datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km_rectified \
  --query-text-csv datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km_rectified/clip_rear_oracle_text_queries.csv
```

## Helper Scripts

- `scripts/run_clip_training_experiments.sh`: LoRA training
- `scripts/test_all_clip_models.sh`: compare pretrained retrieval backbones
- `scripts/test_all_loras.sh`: sweep LoRA checkpoints
- `scripts/test_with_rerank.sh`: evaluate region reranking
- `scripts/opposite_view/`: opposite-view retrieval protocols for Oxford RobotCar, SYNTHIA SEQS-02, and UrbanV2X

## Reproducibility Notes

- This repository does not track `datasets/`, `outputs/`, checkpoints, or generated features by default.
- The current pipeline is tightly coupled to the `SensorKit` directory structure and several hard-coded paths.
- If you plan to publish the project publicly, document how others can obtain `SensorKit` data or provide a script that rebuilds the required folder layout.
- If you need to version checkpoints or sample outputs, use Git LFS instead of normal Git.

## Acknowledgements

- [Hierarchical-Localization (HLOC)](https://github.com/cvg/Hierarchical-Localization)
- [COLMAP](https://colmap.github.io/)
- OpenCLIP / Hugging Face model ecosystem
