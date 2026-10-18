#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-openmathinstruct2}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"

# 1. 重生成脚本
OUTPUT_DIR="${OUTPUT_DIR:-./output/openmath/onpolicy}"
REGEN_DIR="${OUTPUT_DIR}/regenerated"
REGEN_FILE="${REGEN_DIR}/openmath_qwen3_8b_onpolicy.jsonl"

LOG_ROOT="${OUTPUT_DIR}/regenlogs"
RUN_ID="$(date '+%Y%m%d_%H%M%S')_$$"
RUN_DIR="${LOG_ROOT}/run_${RUN_ID}"
RUN_LOG="${RUN_DIR}/run.log"
VLLM_LOG="${RUN_DIR}/vllm.log"
CONFIG_FILE="${RUN_DIR}/config.txt"
SCRIPT_SNAPSHOT="${RUN_DIR}/run_script.sh"

ENABLE_SPECULATIVE="${ENABLE_SPECULATIVE:-1}"
SPECULATOR_MODEL="${SPECULATOR_MODEL:-RedHatAI/Qwen3-8B-speculator.dflash}"
SPECULATIVE_TOKENS="${SPECULATIVE_TOKENS:-8}"

VLLM_PORT="${VLLM_PORT:-18081}"
VLLM_GPU="${VLLM_GPU:-7}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
REGEN_CONCURRENCY="${REGEN_CONCURRENCY:-35}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-35}"
REGEN_GPU_MEM="${REGEN_GPU_MEM:-0.93}"

VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-llm_dflash_eval}"
SPEC_CONDA_ENV="${SPEC_CONDA_ENV:-dflash}"
CONDA_BASE="$(conda info --base)"
VLLM_PYTHON="${CONDA_BASE}/envs/${VLLM_CONDA_ENV}/bin/python"
SPEC_PYTHON="${CONDA_BASE}/envs/${SPEC_CONDA_ENV}/bin/python"

[[ -x "${VLLM_PYTHON}" ]] || { echo "ERROR: ${VLLM_PYTHON} not found"; exit 1; }
[[ -x "${SPEC_PYTHON}" ]] || { echo "ERROR: ${SPEC_PYTHON} not found"; exit 1; }

mkdir -p "${REGEN_DIR}" "${RUN_DIR}"
[[ -f "$0" ]] && cp "$0" "${SCRIPT_SNAPSHOT}" 2>/dev/null || true

exec > >(tee -a "${RUN_LOG}") 2>&1
export PYTHONUNBUFFERED=1

cat > "${CONFIG_FILE}" <<EOF2
run_id=${RUN_ID}
start_time=$(date '+%Y-%m-%d %H:%M:%S')
model=${MODEL}
dataset=${DATASET}
dataset_split=train_1M
max_samples=${MAX_SAMPLES}
regen_file=${REGEN_FILE}
enable_speculative=${ENABLE_SPECULATIVE}
speculator_model=${SPECULATOR_MODEL}
speculative_tokens=${SPECULATIVE_TOKENS}
vllm_gpu=${VLLM_GPU}
vllm_port=${VLLM_PORT}
vllm_max_model_len=${VLLM_MAX_MODEL_LEN}
vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS}
regen_concurrency=${REGEN_CONCURRENCY}
gpu_memory_utilization=${REGEN_GPU_MEM}
EOF2

VLLM_PID=""

cleanup() {
    EXIT_CODE=$?

    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping regeneration vLLM..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    {
        echo "end_time=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "exit_code=${EXIT_CODE}"
    } >> "${CONFIG_FILE}" 2>/dev/null || true

    trap - EXIT
    exit "${EXIT_CODE}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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

            key = obj.get("primary_id") or obj.get("id")
            if key not in (None, ""):
                seen.add(str(key))

print(len(seen))
PY
}

wait_for_vllm() {
    echo "Waiting for vLLM server..."

    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "vLLM server is ready."
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

EXISTING="$(count_regenerated "${REGEN_FILE}")"

echo "Run: ${RUN_ID}"
echo "Cached conversations: ${EXISTING}"
echo "Target conversations: ${MAX_SAMPLES}"

if (( EXISTING >= MAX_SAMPLES )); then
    echo "Cache is already sufficient. Nothing to generate."
    exit 0
fi

REMAINING=$((MAX_SAMPLES - EXISTING))
echo "Missing conversations: ${REMAINING}"

VLLM_CMD=(
    "${VLLM_PYTHON}" -m vllm.entrypoints.cli.main serve "${MODEL}"
    --host 127.0.0.1
    --port "${VLLM_PORT}"
    --tensor-parallel-size 1
    --data-parallel-size 1
    --dtype bfloat16
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --gpu-memory-utilization "${REGEN_GPU_MEM}"
    --disable-uvicorn-access-log
)

if [[ "${ENABLE_SPECULATIVE}" == "1" ]]; then
    SPEC_CONFIG="{\"method\":\"dflash\",\"model\":\"${SPECULATOR_MODEL}\",\"num_speculative_tokens\":${SPECULATIVE_TOKENS}}"
    VLLM_CMD+=(--speculative-config "${SPEC_CONFIG}")
fi

echo "[1/2] Starting Qwen3-8B regeneration server"
if [[ "${ENABLE_SPECULATIVE}" == "1" ]]; then
    echo "Speculative decoding: ENABLED (${SPECULATOR_MODEL}, tokens=${SPECULATIVE_TOKENS})"
else
    echo "Speculative decoding: DISABLED"
fi

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
"${VLLM_CMD[@]}" > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
wait_for_vllm

echo "[2/2] Generating ${REMAINING} missing conversations"

GEN_START_TS="$(date +%s)"

"${SPEC_PYTHON}" scripts/response_regeneration/script.py \
    --endpoint "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --limit "${REMAINING}" \
    --concurrency "${REGEN_CONCURRENCY}" \
    --outfile "${REGEN_FILE}" \
    --resume

GEN_END_TS="$(date +%s)"
GEN_SECONDS=$((GEN_END_TS - GEN_START_TS))
TOTAL="$(count_regenerated "${REGEN_FILE}")"

{
    echo "cached_before=${EXISTING}"
    echo "requested_this_run=${REMAINING}"
    echo "cached_after=${TOTAL}"
    echo "generation_elapsed_seconds=${GEN_SECONDS}"
} >> "${CONFIG_FILE}"

echo "Persistent regenerated conversations: ${TOTAL}"
echo "Generation elapsed: ${GEN_SECONDS}s"

if (( TOTAL < MAX_SAMPLES )); then
    echo "ERROR: cache is still insufficient after regeneration."
    echo "Required: ${MAX_SAMPLES}"
    echo "Available: ${TOTAL}"
    exit 1
fi

echo "Regeneration complete."
