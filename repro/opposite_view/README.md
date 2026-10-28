# Complete-surroundings opposite-view retrieval

This directory contains the paper checkpoint's reference features, generation prompt, query captions, pairing manifests, per-query results, and recall summaries. `features/` contains the three dataset reference feature arrays and the original image-baseline retrieval summary. The LoRA adapter is [`../../checkpoints/text8_visual12_epoch22.pt`](../../checkpoints/text8_visual12_epoch22.pt); OpenCLIP downloads the DataComp ViT-L/14 base weights. Image datasets and pose files are distributed separately.

| Dataset | Caption views | Queries | References | R@1 / R@5 / R@10 / R@20 within 10 m (%) |
|---|---|---:|---:|---|
| UrbanV2X paired cam6 | cam0, cam2–cam6; no cam1 | 491 | 493 | 15.89 / 41.96 / 58.25 / 73.52 |
| SYNTHIA dawn | Omni_B, Omni_L, Omni_R, Omni_F | 95 | 95 | 7.37 / 22.11 / 38.95 / 62.11 |
| Oxford RobotCar | stereo_centre, mono_left, mono_right, mono_rear | 866 | 833 | 12.01 / 29.21 / 40.42 / 54.27 |

The UrbanV2X caption run has 493 cam0 anchor groups. The paired subset maps 491 cam6 image queries to nearby cam0 anchors within 600 ms and is used for comparison with the archived image baselines. UrbanV2X cam1 is the reference camera and is never supplied to Qwen for this condition. The source cameras and reference images are from the same capture session.

SYNTHIA uses the same-frame Omni_F view in the four-view caption, and Omni_F is the reference direction. This condition has exact-frame directional overlap and is therefore a privileged complete-surroundings result. RobotCar uses the same traversal for all four views and reference images. Exact database rear frames were excluded from the caption inputs, but adjacent rear frames can be very close in time (up to 100 ms alignment) and position. This is also a privileged condition. These results should not be presented as independent-session localization.

## Files and rerun

- `features/manifest.json` records query/reference names and source paths; `*_features.npz` stores the reference embeddings in the same order.
- Each dataset directory contains `captions.csv`, `caption_manifest.json`, `evaluation_manifest.json`, `summary.csv`, and per-query output. RobotCar also has `alignment_report.json`. SHA-256 hashes in evaluation manifests bind captions and checkpoint to the recorded result.
- `multiview_prompt.txt` is the exact generation prompt. Qwen3-VL-8B-Instruct was loaded in FP16 for RobotCar and the default non-quantized path for UrbanV2X and SYNTHIA; sampling was disabled. The CLIP text length was limited to 77 tokens.

From the repository root, restore the published captions and manifests to the scripts' output paths before evaluation:

```bash
mkdir -p outputs/opposite_view_urban_493_six_view_no_cam1 outputs/opposite_view_synthia_four_view_surroundings outputs/opposite_view_robotcar_four_view_surroundings_fp16
cp repro/opposite_view/urbanv2x/{captions.csv,caption_manifest.json} outputs/opposite_view_urban_493_six_view_no_cam1/
cp repro/opposite_view/synthia/{captions.csv,caption_manifest.json} outputs/opposite_view_synthia_four_view_surroundings/
cp repro/opposite_view/robotcar/{captions.csv,caption_manifest.json} outputs/opposite_view_robotcar_four_view_surroundings_fp16/
python3 scripts/opposite_view/urban_six_view_no_cam1.py evaluate
python3 scripts/opposite_view/synthia_four_view_surroundings.py evaluate
python3 scripts/opposite_view/robotcar_four_view_surroundings.py evaluate
python3 scripts/opposite_view/summarize_surroundings.py
```

To regenerate captions, run the same scripts with `generate` after preparing the dataset images. SYNTHIA Omni_L/Omni_R images must be under `outputs/opposite_view_synthia_three_view_no_front/extracted/` in the same relative structure as Omni_B. RobotCar mono archives from traversal `2014-12-09-13-21-02` are processed with `scripts/opposite_view/prepare_robotcar_side_views.py`; the RobotCar SDK and camera models must be present under `datasets/SensorKit/LoST_Robotcar/robotcar-dataset-sdk/`. Run `robotcar_four_view_surroundings.py check` before caption generation. The RobotCar `generate` command defaults to FP16; leave `--load-in-4bit` unset for this result.

The 10 m recall uses pose distance to any retrieved reference among the first k. `summary.csv` gives additional 5 m and 20 m thresholds. The image baseline table is in `features/summary.csv`; its NetVLAD rows are the image comparisons used in the paper.
