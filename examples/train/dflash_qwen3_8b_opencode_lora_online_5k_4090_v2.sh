#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Qwen3-8B + DFlash-LoRA online training
# Hardware: 2 x RTX 4090
#
# GPU 0: vLLM verifier (Qwen/Qwen3-8B)
# GPU 1: DFlash-LoRA training
#
# Run this script from the root directory of speculators-main-lora:
#   chmod +x dflash_qwen3_8b_opencode_lora_online_5k_2x4090.sh
#   ./dflash_qwen3_8b_opencode_lora_online_5k_2x4090.sh
#
# Or:
#   bash dflash_qwen3_8b_opencode_lora_online_5k_2x4090.sh
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Basic configuration
# ------------------------------------------------------------------------------

# Official verifier/base model
MODEL="Qwen/Qwen3-8B"

# Pretrained DFlash checkpoint corresponding to Qwen3-8B
DRAFT_MODEL="RedHatAI/Qwen3-8B-speculator.dflash"

# Dataset preset in your modified repo
DATASET="opencodeinstruct"

# Only use 5000 samples. It is OK if prepare_data.py shuffles them.
MAX_SAMPLES=5000

# Output paths
OUTPUT_DIR="./output/dflash_qwen3_8b_opencode_lora_online_5k"
DATA_DIR="${OUTPUT_DIR}/data"
SAVE_DIR="${OUTPUT_DIR}/checkpoints"

# =========================
# Training log
# =========================
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_$(date '+%Y%m%d_%H%M%S').log"

echo "Training log will be saved to:"
echo "  ${LOG_FILE}"

# 后续所有 stdout + stderr：
# 1. 正常显示在终端
# 2. 同时写入日志文件
exec > >(tee -a "${LOG_FILE}") 2>&1

# 避免 Python 输出缓冲
export PYTHONUNBUFFERED=1

# vLLM OpenAI-compatible endpoint
VLLM_PORT=18082

# ------------------------------------------------------------------------------
# GPU ASSIGNMENT
#
# This is the main place to change which GPUs are used.
#
# For your 2 x 4090 machine:
#   GPU 0 -> vLLM/Qwen3-8B
#   GPU 1 -> DFlash-LoRA training
# ------------------------------------------------------------------------------
VLLM_GPU="5"
TRAIN_GPU="5"


# ------------------------------------------------------------------------------
# 2. Training configuration
# ------------------------------------------------------------------------------

# Start conservatively for 24 GB RTX 4090.
SEQ_LENGTH=4096
VLLM_MAX_MODEL_LEN=8192
MAX_ANCHORS=512

EPOCHS=3
LR=2e-4
SEED=42

# IMPORTANT:
# These IDs must match the pretrained DFlash checkpoint.
TARGET_LAYER_IDS="2 10 18 26 34"

# LoRA
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.0

# ------------------------------------------------------------------------------
# Conda environments
# ------------------------------------------------------------------------------

VLLM_CONDA_ENV="llm"
SPEC_CONDA_ENV="dflash"

VLLM_PYTHON="$(conda run -n "${VLLM_CONDA_ENV}" python -c 'import sys; print(sys.executable)')"
SPEC_PYTHON="$(conda run -n "${SPEC_CONDA_ENV}" python -c 'import sys; print(sys.executable)')"

# Sanity check
[[ -x "${VLLM_PYTHON}" ]] || {
    echo "ERROR: vLLM Python not found: ${VLLM_PYTHON}"
    exit 1
}

[[ -x "${SPEC_PYTHON}" ]] || {
    echo "ERROR: speculator Python not found: ${SPEC_PYTHON}"
    exit 1
}

mkdir -p "${OUTPUT_DIR}"


# ==============================================================================
# Step 1: Prepare 5000 OpenCodeInstruct samples
# ==============================================================================

echo
echo "============================================================"
echo "[1/3] Preparing OpenCodeInstruct (${MAX_SAMPLES} samples)"
echo "============================================================"

"${SPEC_PYTHON}" scripts/prepare_data.py \
    --model "${MODEL}" \
    --data "${DATASET}" \
    --output "${DATA_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --seq-length "${SEQ_LENGTH}" \
    --seed "${SEED}" \
    --num-preprocessing-workers 8


# ==============================================================================
# Step 2: Start Qwen3-8B vLLM verifier on GPU 0
# ==============================================================================

echo
echo "============================================================"
echo "[2/3] Starting vLLM verifier on physical GPU ${VLLM_GPU}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
"${VLLM_PYTHON}" scripts/launch_vllm.py "${MODEL}" \
    --target-layer-ids ${TARGET_LAYER_IDS} \
    -- \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --max-num-seqs 2 \
    --enforce-eager \
    --gpu-memory-utilization 0.60 \
    --disable-uvicorn-access-log \
    > "${OUTPUT_DIR}/vllm.log" 2>&1 &

VLLM_PID=$!


cleanup() {
    echo
    echo "Stopping vLLM server (PID=${VLLM_PID}) ..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM


echo "Waiting for vLLM server..."
echo "vLLM log: ${OUTPUT_DIR}/vllm.log"

READY=0
for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        READY=1
        break
    fi

    # Fail early if the vLLM process has already crashed.
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo
        echo "ERROR: vLLM exited before becoming ready."
        echo "Last 80 lines of vLLM log:"
        tail -n 80 "${OUTPUT_DIR}/vllm.log" || true
        exit 1
    fi

    sleep 2
done

if [[ "${READY}" -ne 1 ]]; then
    echo
    echo "ERROR: vLLM did not become ready."
    echo "Last 80 lines of vLLM log:"
    tail -n 80 "${OUTPUT_DIR}/vllm.log" || true
    exit 1
fi

echo "vLLM is ready."


# ==============================================================================
# Step 3: Train DFlash LoRA on GPU 1
# ==============================================================================

echo
echo "============================================================"
echo "[3/3] Training DFlash-LoRA on physical GPU ${TRAIN_GPU}"
echo "============================================================"

# Notes:
# - Single training GPU => use normal `python`, NOT torchrun.
# - Load the official pretrained DFlash checkpoint.
# - Do NOT pass --draft-vocab-size when finetuning the pretrained DFlash model.
# - Do NOT pass --num-layers / --draft-arch etc.; use checkpoint configuration.
# - AdamW is used explicitly for this LoRA experiment.
# - The LoRA implementation in your fork saves adapter weights only; no merge.
#
# If you only want a quick smoke test first, temporarily add:
#     --max-steps 5
#
CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
"${SPEC_PYTHON}" scripts/trainlora.py \
    --verifier-name-or-path "${MODEL}" \
    --from-pretrained "${DRAFT_MODEL}" \
    --speculator-type dflash \
    \
    --data-path "${DATA_DIR}" \
    --vllm-endpoint "http://127.0.0.1:${VLLM_PORT}/v1" \
    --on-missing generate \
    --on-generate delete \
    \
    --save-path "${SAVE_DIR}" \
    \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --optimizer adamw \
    --weight-decay 0.0 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.05 \
    \
    --total-seq-len "${SEQ_LENGTH}" \
    --target-layer-ids ${TARGET_LAYER_IDS} \
    --max-anchors "${MAX_ANCHORS}" \
    \
    --num-workers 2 \
    --prefetch-factor 1 \
    --seed "${SEED}" \
    --log-freq 10 \
    \
    --enable-lora \
    --lora-adapter-name code \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --lora-save-freq 1 \
    --no-resume-from-checkpoint \


echo
echo "============================================================"
echo "Training finished."
echo "============================================================"
echo
echo "LoRA adapter directory:"
echo "  ${SAVE_DIR}/lora/final/"
echo
echo "Expected adapter weights:"
echo "  ${SAVE_DIR}/lora/final/adapter_model.safetensors"
echo
echo "Expected adapter config:"
echo "  ${SAVE_DIR}/lora/final/adapter_config.json"
echo
echo "No merge is performed."
