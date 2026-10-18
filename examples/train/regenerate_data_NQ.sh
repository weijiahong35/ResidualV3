#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-natural_questions}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"

# ============================================================
# Natural Questions response regeneration
# ============================================================

# 1. 重生成输出
OUTPUT_DIR="${OUTPUT_DIR:-./output/natural_questions/onpolicy}"
REGEN_DIR="${OUTPUT_DIR}/regenerated"
REGEN_FILE="${REGEN_DIR}/natural_questions_qwen3_8b_onpolicy.jsonl"

# 2. 日志
LOG_ROOT="${OUTPUT_DIR}/regenlogs"
RUN_ID="$(date '+%Y%m%d_%H%M%S')_$$"
RUN_DIR="${LOG_ROOT}/run_${RUN_ID}"
RUN_LOG="${RUN_DIR}/run.log"
VLLM_LOG="${RUN_DIR}/vllm.log"

# 3. vLLM / regeneration 参数
VLLM_PORT="${VLLM_PORT:-18081}"
VLLM_GPU="${VLLM_GPU:-7}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

REGEN_CONCURRENCY="${REGEN_CONCURRENCY:-50}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-50}"
REGEN_GPU_MEM="${REGEN_GPU_MEM:-0.93}"

# 4. Conda 环境
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-llm}"
SPEC_CONDA_ENV="${SPEC_CONDA_ENV:-dflash}"

CONDA_BASE="$(conda info --base)"

VLLM_PYTHON="${CONDA_BASE}/envs/${VLLM_CONDA_ENV}/bin/python"
SPEC_PYTHON="${CONDA_BASE}/envs/${SPEC_CONDA_ENV}/bin/python"

# ============================================================
# Check environments
# ============================================================

[[ -x "${VLLM_PYTHON}" ]] || {
    echo "ERROR: ${VLLM_PYTHON} not found"
    exit 1
}

[[ -x "${SPEC_PYTHON}" ]] || {
    echo "ERROR: ${SPEC_PYTHON} not found"
    exit 1
}

mkdir -p "${REGEN_DIR}" "${RUN_DIR}"

exec > >(tee -a "${RUN_LOG}") 2>&1

export PYTHONUNBUFFERED=1

VLLM_PID=""

# ============================================================
# Cleanup
# ============================================================

cleanup() {
    EXIT_CODE=$?

    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping regeneration vLLM..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    trap - EXIT
    exit "${EXIT_CODE}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ============================================================
# Count already regenerated NQ samples
# ============================================================

count_regenerated() {
    "${SPEC_PYTHON}" - "$1" <<'PY'
import json
import os
import sys

path = sys.argv[1]
seen = set()

if os.path.isfile(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # response_regeneration uses primary_id when available.
            # Natural Questions has its own "id", so this works well
            # with resume/deduplication.
            key = obj.get("primary_id")

            if key in (None, ""):
                key = obj.get("id")

            if key not in (None, ""):
                seen.add(str(key))

print(len(seen))
PY
}

# ============================================================
# Wait for vLLM
# ============================================================

wait_for_vllm() {
    for _ in $(seq 1 180); do
        if curl -sf \
            "http://127.0.0.1:${VLLM_PORT}/health" \
            >/dev/null 2>&1; then
            return 0
        fi

        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM exited before becoming ready"
            tail -n 100 "${VLLM_LOG}" || true
            exit 1
        fi

        sleep 2
    done

    echo "ERROR: vLLM did not become ready"
    tail -n 100 "${VLLM_LOG}" || true
    exit 1
}

# ============================================================
# Run info
# ============================================================

echo "============================================================"
echo "Natural Questions response regeneration"
echo "============================================================"
echo "Run:                 ${RUN_ID}"
echo "Model:               ${MODEL}"
echo "Dataset:             ${DATASET}"
echo "Target samples:      ${MAX_SAMPLES}"
echo "Output file:         ${REGEN_FILE}"
echo "Run log:             ${RUN_LOG}"
echo "vLLM log:            ${VLLM_LOG}"
echo "vLLM GPU:            ${VLLM_GPU}"
echo "vLLM port:           ${VLLM_PORT}"
echo "Concurrency:         ${REGEN_CONCURRENCY}"
echo "============================================================"

# ============================================================
# Resume state
# ============================================================

EXISTING="$(count_regenerated "${REGEN_FILE}")"

echo "Cached conversations: ${EXISTING}"
echo "Target conversations: ${MAX_SAMPLES}"

if (( EXISTING >= MAX_SAMPLES )); then
    echo "Cache is already sufficient. Nothing to generate."
    exit 0
fi

REMAINING=$((MAX_SAMPLES - EXISTING))

echo "Missing conversations: ${REMAINING}"

# ============================================================
# Start vLLM teacher model
# ============================================================

echo
echo "[1/2] Starting Qwen3-8B regeneration server"

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
"${VLLM_PYTHON}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${REGEN_GPU_MEM}" \
    --disable-uvicorn-access-log \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

wait_for_vllm

echo "vLLM is ready."

# ============================================================
# Regenerate Natural Questions responses
# ============================================================

echo
echo "[2/2] Generating ${REMAINING} missing Natural Questions conversations"

"${SPEC_PYTHON}" scripts/response_regeneration/script.py \
    --endpoint "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --limit "${REMAINING}" \
    --concurrency "${REGEN_CONCURRENCY}" \
    --outfile "${REGEN_FILE}" \
    --resume

# ============================================================
# Validate final cache
# ============================================================

TOTAL="$(count_regenerated "${REGEN_FILE}")"

echo
echo "Persistent regenerated conversations: ${TOTAL}"

if (( TOTAL < MAX_SAMPLES )); then
    echo "ERROR: cache is still insufficient after regeneration."
    echo "Required:  ${MAX_SAMPLES}"
    echo "Available: ${TOTAL}"
    exit 1
fi

echo
echo "============================================================"
echo "Natural Questions regeneration complete."
echo "Samples: ${TOTAL}"
echo "Output:  ${REGEN_FILE}"
echo "============================================================"