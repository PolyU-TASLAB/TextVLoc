# Reproduce the current TextVLoc manuscript

This bundle corresponds to `TextVLoc/manuscript/manuscript.tex` in the development workspace. `protocol.json` records its SHA-256, exact query/reference order and caption paths. It includes the experiments in that manuscript. Run commands from the repository root in the project Docker container.

| Manuscript experiment | Entry point | Released evidence |
| --- | --- | --- |
| Text-image retrieval | `scripts.eval_paper_retrieval` | Original Qwen captions; pretrained/global/reranked Top-20; expected R@K |
| Coarse localization | Same entry point | ENU query/reference coordinates; localization recall at 5/10/15m |
| Query style and generator generalization | Same entry point | Nine caption variants and all three methods |
| Incomplete/color/object-error robustness | Same entry point; `scripts.build_paper_text_perturbations` | Three paired perturbation CSVs and landmark edit records |
| Opposite-view retrieval | [Existing three-dataset guide](../opposite_view/README.md) | Captions, manifests, reference features and per-query results |
| Fine localization | `scripts.eval_paper_pose` | Query intrinsics, 493 poses and summary |
| Efficiency | `scripts.benchmark_online_end_to_end` | Original 483 timed queries per method and device/stage summaries |
| Rerank fusion sensitivity | `scripts.eval_paper_retrieval evaluate --alpha-sweep` | Original alpha curve in `results/alpha_curve.csv` |

The main checkpoint is `checkpoints/text8_visual12_epoch22.pt`: DataComp OpenCLIP ViT-L/14, final 8 text blocks and final 12 visual blocks, rank 8, LoRA alpha 16, seed 42. The retriever is trained with global symmetric CLIP contrastive loss. The paper reranker uses the maximum similarity among nine nonoverlapping regions and `0.75 * global + 0.25 * region`. It only reorders the existing Top-20.

## Check the published results without model inference

```bash
python3 -m scripts.eval_paper_retrieval check --output-dir outputs/paper_check
python3 -m scripts.eval_paper_pose check
python3 -m scripts.summarize_paper_timing
```

The retrieval check verifies artifact hashes, 493 unique captions per variant, reference indices, self-match exclusion, identical global/reranked candidate membership, and all 156 R@K values against the manuscript table values. It also recomputes geographic localization recall. No model downloads, original images, GPU, or developer feature caches are required. NumPy is required. The pose check independently computes successful-subset error statistics and success rates from the released poses and coordinates.

There are 13 caption sets: original Qwen; natural-language, third-person, brief, noun-phrase and synthetic casual; LLaVA, Claude, DeepSeek Flash and Gemma 3; and controlled incomplete, color-error and object/detail-error captions. Synthetic casual is model-generated, not a human study.

## Rerun model inference from images

Restore the UrbanV2X image data described in the main README, then run:

```bash
python3 -m scripts.eval_paper_retrieval evaluate \
  --dataset-root datasets/SensorKit \
  --output-dir outputs/paper_retrieval \
  --alpha-sweep
```

This freshly encodes all 2,952 reference images for the pretrained and adapted towers, evaluates all 13 released caption sets and caches candidate region features. It writes `summary.csv`, `rankings.npz`, provenance, feature archives and HLoc pair files. The output directory must be new. Use `--variants natural_static brief_static`, or `--methods global rerank`, to narrow a rerun. `--query-limit` and `--reference-limit` are smoke checks and are marked as such in the manifest. Full-image features use FP32; region features use FP16 autocast and storage, matching the archived protocol. GPU, library and batch-dependent numerical differences can affect close ties.

The original five-fusion-weight curve can be regenerated with `--alpha-sweep`; the released historical curve has a finer step size. This is a parameter sensitivity experiment on the evaluated queries, not evidence of validation-only parameter selection.

## Regenerate query descriptions

Exact evaluated text is released, so API calls and generative-model reruns are optional. Regeneration may produce different text and different results. The source query images are the only visual inputs to the single-view generators.

```bash
# Four Qwen style variants.
python3 -m scripts.generate_query_prompt_variants \
  --baseline-csv repro/paper/captions/original_qwen.csv \
  --query-dir datasets/SensorKit/query/cam1_all/0_undistorted \
  --output-dir outputs/query_text_generalization/qwen3_prompt_v1

# Synthetic casual queries (the only synthetic generator variant in the paper).
python3 -m scripts.generate_query_prompt_variants \
  --baseline-csv repro/paper/captions/original_qwen.csv \
  --query-dir datasets/SensorKit/query/cam1_all/0_undistorted \
  --prompt-config scripts/query_synthetic_user_variants.json \
  --output-dir outputs/query_text_generalization/qwen3_synthetic_user_v1

python3 -m scripts.generate_llava_query_captions
python3 -m scripts.generate_cam1_deepseek_descriptions
python3 -m scripts.generate_cam1_gemma3_descriptions

# The exact text-only robustness edits; no VLM/API calls.
python3 -m scripts.build_paper_text_perturbations \
  --output-dir outputs/paper_text_perturbations
```

DeepSeek requires `DEEPSEEK_API_KEY` in the container environment (or the key file documented in its script). Gemma requires access to its gated Hugging Face weights. Claude captions were generated in a Claude Code session with individual query images and the original navigation prompt. No standalone Claude generator was used; its exact CSV is released under `captions/claude_opus55.csv`. Historical model names and settings are preserved in provenance and should not be interpreted as current service availability. No paid generator is called by the evaluation commands.

To evaluate newly generated captions, use `evaluate --variants natural_static --caption natural_static=outputs/query_text_generalization/qwen3_prompt_v1/natural_static.csv --output-dir outputs/new_captions_eval`. Repeat `--caption VARIANT=CSV` for multiple variants. Each CSV must contain all 493 query names and nonempty descriptions. This records new input hashes without changing the published archive.

## Fine localization

The released pose archive reproduces 473/493 successful queries (95.94%) and a 0.91m median error among successful estimates. Error means, RMSE and percentiles in the paper table use only the APE<10m subset; the check command additionally reports all returned poses. Text-only success is 220/493 (44.62%).

To rerun matching and PnP against the reference reconstruction:

```bash
python3 -m scripts.eval_paper_pose run \
  --reference-sfm outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/sfm_superpointmax+lightglue \
  --reference-features outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/feats-superpoint-n4096-r1024.h5 \
  --output-dir outputs/paper_pose
```

Reference local features must match the keypoint ordering used by the SfM map. They are not included in the minimal SfM release and must be retained from/rebuilt consistently with the mapping stage. The query image directory and intrinsics list are configurable. Existing matching query features can be supplied with `--query-features`. This command uses the published global Top-20 to isolate geometric rerunning. Fresh retrieval emits separate HLoc pair files for the standard pipeline.

SuperPoint downloads its original pretrained weights from the upstream repository on first use, with a SHA-256 check. NetVLAD and LightGlue also require their pretrained weights; prepare them before timing. Model downloads are outside the online timing interval.

## Timing

`results/timing.json` and `timing_per_query.csv` contain the original RTX 4090D measurement: 493 queries, 10 warmups, 483 measured queries for each method with rotated method order. `python3 -m scripts.summarize_paper_timing` reaggregates the raw measurements and checks them against the released summary. Caption generation, speech recognition, model/map loading and output writes are excluded. The timing table has no region reranking.

After fresh retrieval and preparing the HLoc reference/query assets:

```bash
python3 -m scripts.benchmark_online_end_to_end \
  --text-index outputs/paper_retrieval/global_database.npz \
  --query-assets outputs/SensorKit_hloc/netvlad_corruption_stress_full/clean \
  --rotate-method-order --output-dir outputs/paper_timing
```

The query-assets directory needs `queries_with_intrinsics.txt`; the map directory needs reference NetVLAD descriptors, reference SuperPoint descriptors and the SfM model. All preparation is offline and outside the reported timing interval. These large feature files are not in the minimal SfM release. New timings depend on the device and software environment; the original per-query archive permits independent aggregation of the paper timing table.

## Protocol discrepancies preserved for review

The historical positive-pair list `protocols/urbanv2x_paper_positive_pairs.csv` has 7,719 rows and reproduces the current manuscript numbers. The manuscript describes a 20m distance criterion, but that list does not strictly enforce ENU 3D distance <=20m. The separate stricter list has 3,512 pairs. With the same rankings it changes main-table R@1 from 41.38 to 40.97 (global), and 45.03 to 44.83 (reranked); geographic localization recall is unchanged. To inspect it:

```bash
python3 -m scripts.eval_paper_retrieval check \
  --positive-pairs protocols/urbanv2x_positive_pairs_netvlad05_enu3d20m.csv \
  --output-dir outputs/paper_strict20m_check
```

The manuscript states 40 training epochs, whereas the released evaluated adapter is epoch 22. The training launcher defaults to 40 epochs and retains epoch checkpoints; evaluation explicitly selects epoch 22. `EPOCHS=22 bash scripts/run_clip_training_experiments.sh` reproduces the training horizon up to that checkpoint. Older checkpoints contain LoRA tensors but omit trained bias/logit-scale values; the loader reconstructs their pretrained values. All archived evaluation numbers come from reloaded checkpoints. These discrepancies are documented here without changing manuscript text or silently relabeling results.
