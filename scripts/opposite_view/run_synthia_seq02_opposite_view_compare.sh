#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-datasets/SYNTHIA}"
DATA_ROOT="${DATA_ROOT:-datasets/SYNTHIA/SEQS-02}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-5}"
CENTER_STRIDE="${CENTER_STRIDE:-10}"
PREPARED_DIR="${PREPARED_DIR:-outputs/SYNTHIA/seq02_lost}"
EVAL_DIR="${EVAL_DIR:-outputs/SYNTHIA/seq02_lost_seq${SEQUENCE_LENGTH}_stride${CENTER_STRIDE}_eval}"
CAPTION_MODEL="${CAPTION_MODEL:-qwen-vl}"
LORA_CKPT="${LORA_CKPT:-datasets/SensorKit/clipdata/finetune_output/qwen3_clipdata_text_sparse03_remove_no_robotcar/lora_epoch_8.pt}"
CLIP_METHOD="${CLIP_METHOD:-TextVLoc_front_oracle_text_seq${SEQUENCE_LENGTH}_stride${CENTER_STRIDE}_no_robotcar_epoch8}"
METHODS="${METHODS:-netvlad megaloc cosplace eigenplaces}"
EXPERIMENTS="${EXPERIMENTS:-dawn_back_to_dawn_front fall_back_to_fall_front fall_back_to_dawn_front}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
MIN_DISTANCE_M="${MIN_DISTANCE_M:-}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_CAPTION="${SKIP_CAPTION:-0}"
SKIP_IMAGE_BASELINES="${SKIP_IMAGE_BASELINES:-0}"
SKIP_TRELOC="${SKIP_TRELOC:-0}"

extract_archives() {
  if [[ "${SKIP_EXTRACT}" == "1" ]]; then
    return
  fi

  mkdir -p "${DATA_ROOT}"
  for split in DAWN FALL; do
    local archive="${DATASET_DIR}/SYNTHIA-SEQS-02-${split}.rar"
    local target="${DATA_ROOT}/SYNTHIA-SEQS-02-${split}"
    if [[ -d "${target}/RGB/Stereo_Left/Omni_F" && -d "${target}/RGB/Stereo_Left/Omni_B" ]]; then
      echo "[skip] extracted ${split}: ${target}"
      continue
    fi
    if [[ ! -f "${archive}" ]]; then
      echo "Missing archive: ${archive}" >&2
      exit 1
    fi
    echo "[extract] ${archive} -> ${DATA_ROOT}"
    if command -v bsdtar >/dev/null 2>&1; then
      bsdtar -xf "${archive}" -C "${DATA_ROOT}" \
        "SYNTHIA-SEQS-02-${split}/RGB/Stereo_Left/Omni_F" \
        "SYNTHIA-SEQS-02-${split}/RGB/Stereo_Left/Omni_B"
    elif command -v unrar >/dev/null 2>&1; then
      unrar x -y "${archive}" \
        "SYNTHIA-SEQS-02-${split}/RGB/Stereo_Left/Omni_F/*" \
        "SYNTHIA-SEQS-02-${split}/RGB/Stereo_Left/Omni_B/*" \
        "${DATA_ROOT}/"
    else
      echo "Need bsdtar or unrar to extract this RAR archive; container p7zip cannot decode it." >&2
      exit 1
    fi
  done
}

prepare_protocol() {
  local args=(
    python scripts/opposite_view/prepare_synthia_seq02_lost.py
    --data-root "${DATA_ROOT}"
    --out-dir "${PREPARED_DIR}"
    --sample-every "${SAMPLE_EVERY}"
  )
  if [[ -n "${MAX_FRAMES:-}" ]]; then
    args+=(--max-frames "${MAX_FRAMES}")
  fi
  if [[ -n "${MIN_DISTANCE_M}" ]]; then
    args+=(--min-distance-m "${MIN_DISTANCE_M}")
  fi
  "${args[@]}"
}

caption_and_build_queries() {
  local experiment="$1"
  local exp_dir="${PREPARED_DIR}/${experiment}"
  local captions_csv="${exp_dir}/query_text_source_captions_${CAPTION_MODEL}.csv"
  local query_text_csv="${exp_dir}/query_text_front_oracle_${CAPTION_MODEL}.csv"

  if [[ "${SKIP_CAPTION}" != "1" ]]; then
    python -m hloc.generate_captions \
      --image_list "${exp_dir}/query_text_source_image_paths.txt" \
      --output_csv "${captions_csv}" \
      --model "${CAPTION_MODEL}" \
      --device "${DEVICE}"
  fi

  python scripts/opposite_view/build_synthia_seq02_text_queries.py \
    --experiment-dir "${exp_dir}" \
    --captions-csv "${captions_csv}" \
    --output-csv "${query_text_csv}"
}

run_image_baselines() {
  local experiment="$1"
  if [[ "${SKIP_IMAGE_BASELINES}" == "1" ]]; then
    return
  fi
  local args=(
    python scripts/opposite_view/run_synthia_seq02_retrieval_eval.py
    --data-root "${DATA_ROOT}"
    --prepared-dir "${PREPARED_DIR}"
    --experiment "${experiment}"
    --out-dir "${EVAL_DIR}"
    --sequence-length "${SEQUENCE_LENGTH}"
    --center-stride "${CENTER_STRIDE}"
    --methods ${METHODS}
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  "${args[@]}"
}

run_treloc() {
  local experiment="$1"
  if [[ "${SKIP_TRELOC}" == "1" ]]; then
    return
  fi
  local query_text_csv="${PREPARED_DIR}/${experiment}/query_text_front_oracle_${CAPTION_MODEL}.csv"
  local args=(
    python scripts/opposite_view/run_synthia_seq02_clip_text_eval.py
    --data-root "${DATA_ROOT}"
    --prepared-dir "${PREPARED_DIR}"
    --experiment "${experiment}"
    --query-text-csv "${query_text_csv}"
    --out-dir "${EVAL_DIR}"
    --lora-ckpt "${LORA_CKPT}"
    --method "${CLIP_METHOD}"
    --batch-size "${BATCH_SIZE}"
    --device "${DEVICE}"
    --sequence-length "${SEQUENCE_LENGTH}"
    --center-stride "${CENTER_STRIDE}"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  "${args[@]}"
}

summarize() {
  python scripts/opposite_view/summarize_synthia_seq02_lost.py \
    --eval-dir "${EVAL_DIR}" \
    --experiments ${EXPERIMENTS} \
    --clip-method "${CLIP_METHOD}"
}

extract_archives
prepare_protocol

for experiment in ${EXPERIMENTS}; do
  echo "[experiment] ${experiment}"
  caption_and_build_queries "${experiment}"
  run_image_baselines "${experiment}"
  run_treloc "${experiment}"
done

summarize
