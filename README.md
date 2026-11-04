# TextVLoc

![TextVLoc Overview](figures/overview_re.png)

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
- `--out-dir datasets/SensorKit/clipdata/finetune_output/text_depth_ablation_v1/text8_visual12/`

The paper checkpoint adapts the final 8 text blocks and final 12 visual blocks with LoRA (rank 8, alpha 16). It uses the CSV `description` field and the epoch-22 checkpoint with seed 42. The navigation prompt requests a single sentence under 77 words; CLIP tokenization is limited to 77 tokens. The paper checkpoint is `checkpoints/text8_visual12_epoch22.pt` (SHA-256 `eb8bd538b208bf7baf2a3324b44c61d76fb9d18f83ff6ee3ef8c13ba2487f524`).

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

UrbanV2X image and metadata packages are available from the GitHub Release:

```bash
wget https://github.com/xxw77777kk/TextVLoc/releases/download/textvloc-urbanv2x-data-v1/textvloc_urbanv2x_images.tar.gz
wget https://github.com/xxw77777kk/TextVLoc/releases/download/textvloc-urbanv2x-data-v1/textvloc_urbanv2x_metadata.tar.gz
tar -xzf textvloc_urbanv2x_images.tar.gz
tar -xzf textvloc_urbanv2x_metadata.tar.gz
```

The image package restores:

- `datasets/SensorKit/allcams/`
- `datasets/SensorKit/query/cam1_all/0_undistorted/`

The metadata package restores:

- `datasets/SensorKit/0620_gt/enu_all_mapping_gt.csv`
- `datasets/SensorKit/0620_gt/enu_all_query_gt.csv`
- `datasets/SensorKit/query/cam1_all/05_allcams_positive_pairs_for_eval.csv`
- `datasets/SensorKit/clipdata/finetune_output/qwen3_clipdata_text_sparse03_remove/lora_epoch_22.pt` (older text-only baseline; the paper checkpoint is in `checkpoints/`)

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

- dual-tower LoRA: `--lora-target all --lora-text-last-n 8 --lora-visual-last-n 12`
- main CSV descriptions as the training text source
- no cross-attention module in this training command
- 40 training epochs by default, with epoch 22 used for paper evaluation (`EPOCHS=22` stops at the released checkpoint)

You can modify dataset paths, output paths, batch size, and epochs directly in `scripts/run_clip_training_experiments.sh`.


### Evaluation scripts

The paper model uses text-last-8 / visual-last-12 LoRA, epoch 22, and a 3x3 maximum-region reranker with global weight 0.75. Evaluation reads local captions, image data and pose/positive-pair CSVs; results are written under `outputs/`.

```bash
python3 -m scripts.eval_paper_retrieval check
python3 -m scripts.eval_paper_retrieval evaluate --output-dir outputs/paper_retrieval
```

Use `--variants natural_static brief_static` to select generated query styles, and repeat `--caption VARIANT=CSV` to supply local descriptions for any generator or perturbation. CSVs must contain `image_name,description` for the same query set. Query/reference pose CSVs use `camera_id,image_id,X_enu,Y_enu,Z_enu`; positive-pair CSVs use `anchor,positive`. All input paths can be set through command-line options. The positive-pair list determines retrieval recall; geographic recall is computed independently from ENU coordinates.

The description generators are `scripts.generate_query_prompt_variants`, `scripts.generate_llava_query_captions`, `scripts.generate_cam1_deepseek_descriptions` and `scripts.generate_cam1_gemma3_descriptions`. Style prompts are in `scripts/query_prompt_variants.json`; the casual-description prompt is in `scripts/query_synthetic_user_variants.json`. Claude descriptions can be supplied as a local CSV. `scripts.build_paper_text_perturbations` generates incomplete/color/object perturbations and requires a local `--landmark-edits` CSV for captions without an existing color word.

`scripts.eval_paper_pose` reads a local retrieval run and pose/map inputs; run it with `--help` for options. `scripts.benchmark_online_end_to_end` measures text/image localization using prepared map assets. `scripts.summarize_paper_timing --input outputs/paper_timing/per_query.csv` aggregates a local timing run.

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
  --lora_ckpt_path checkpoints/text8_visual12_epoch22.pt \
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
  --lora_ckpt_path checkpoints/text8_visual12_epoch22.pt \
  --positive_pairs_csv datasets/SensorKit/query/cam1_all/05_allcams_positive_pairs_for_eval.csv \
  --text_only_loc \
  --enable_region_rerank \
  --region_grid 3 \
  --region_top_k 20
```

Useful extra flags:

- `--evaluate_rerank_recall`
- `--region_top_m 1`
- `--region_reduce max`
- `--region_alpha 0.75`

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
  --lora_ckpt_path checkpoints/text8_visual12_epoch22.pt \
  --positive_pairs_csv datasets/SensorKit/query/cam1_all/05_allcams_positive_pairs_for_eval.csv
```

If you want both reranking and geometric localization, keep `--enable_region_rerank` and omit `--text_only_loc`.

### 4. UrbanV2X opposite-view retrieval evaluation

The UrbanV2X opposite-view evaluation compares image retrieval baselines with TextVLoc under viewpoint reversal. The multi-view scripts in `scripts/opposite_view/` accept local reference features through `--source` and a local navigation prompt through `--prompt-file`; their `--help` lists the required inputs.

The archived protocol uses `cam6` as the query stream and `cam1` as the reference/text-source stream. Its text mapping is:

```text
outputs/SensorKit_hloc/cam6_to_cam1_opposite_view/seq5_stride1/cam6_query_to_cam1_front_text_qwen-vl.csv
```

Run it with:

```bash
python3 scripts/opposite_view/run_sensorkit_cam6_to_cam1_opposite_view_eval.py \
  --dataset datasets/SensorKit \
  --outputs outputs/SensorKit_hloc/cam6_to_cam1_opposite_view \
  --sequence-length 5 \
  --center-stride 1
```

## Helper Scripts

- `scripts/run_clip_training_experiments.sh`: LoRA training
- `scripts/test_all_clip_models.sh`: compare pretrained retrieval backbones
- `scripts/test_all_loras.sh`: sweep LoRA checkpoints
- `scripts/test_with_rerank.sh`: evaluate region reranking
- `scripts/opposite_view/run_sensorkit_cam6_to_cam1_opposite_view_eval.py`: UrbanV2X opposite-view retrieval evaluation

## Reproducibility Notes

- Datasets, generated descriptions, feature caches and evaluation results are local inputs/outputs. The main LoRA adapter is under `checkpoints/`.
- The current pipeline is tightly coupled to the `SensorKit` directory structure and several hard-coded paths.
- If you plan to publish the project publicly, document how others can obtain `SensorKit` data or provide a script that rebuilds the required folder layout.
- The 10.7 MB LoRA checkpoint is tracked directly in Git; full backbone weights are downloaded by OpenCLIP.

## Acknowledgements

- [Hierarchical-Localization (HLOC)](https://github.com/cvg/Hierarchical-Localization)
- [COLMAP](https://colmap.github.io/)
- OpenCLIP / Hugging Face model ecosystem
