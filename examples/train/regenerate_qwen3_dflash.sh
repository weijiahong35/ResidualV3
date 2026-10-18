#!/usr/bin/env bash
set -euo pipefail


# ============================================================
# 1. User configuration
# ============================================================

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-opencodeinstruct}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"

OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_qwen3_8b_opencode_lora_onpolicy}"

# ---------- Speculative decoding ----------
# 1 = enable DFlash
# 0 = disable speculative decoding and use normal vLLM decoding
ENABLE_SPECULATIVE="${ENABLE_SPECULATIVE:-1}"

# Pretrained DFlash speculator for Qwen3-8B
SPECULATOR_MODEL="${SPECULATOR_MODEL:-RedHatAI/Qwen3-8B-speculator.dflash}"

# Number of speculative tokens proposed per step
SPECULATIVE_TOKENS="${SPECULATIVE_TOKENS:-8}"


# ---------- vLLM ----------
VLLM_PORT="${VLLM_PORT:-18081}"
VLLM_GPU="${VLLM_GPU:-7}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

# HTTP-side concurrent requests
REGEN_CONCURRENCY="${REGEN_CONCURRENCY:-35}"

# vLLM scheduler-side maximum active sequences
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-35}"

# Fraction of GPU memory available to vLLM
REGEN_GPU_MEM="${REGEN_GPU_MEM:-0.93}"


# ---------- Conda ----------
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-llm}"
SPEC_CONDA_ENV="${SPEC_CONDA_ENV:-dflash}"


# ============================================================
# 2. Derived paths
# ============================================================

REGEN_DIR="${OUTPUT_DIR}/regenerated"
REGEN_FILE="${REGEN_DIR}/opencode_qwen3_8b_onpolicy.jsonl"

LOG_ROOT="${OUTPUT_DIR}/regenlogs"

# Add PID so two runs started in the same second cannot collide
RUN_ID="$(date '+%Y%m%d_%H%M%S')_$$"
RUN_DIR="${LOG_ROOT}/run_${RUN_ID}"

RUN_LOG="${RUN_DIR}/run.log"
VLLM_LOG="${RUN_DIR}/vllm.log"
CONFIG_FILE="${RUN_DIR}/config.txt"
SCRIPT_SNAPSHOT="${RUN_DIR}/run_script.sh"


# ============================================================
# 3. Python environments
# ============================================================

CONDA_BASE="$(conda info --base)"

VLLM_PYTHON="${CONDA_BASE}/envs/${VLLM_CONDA_ENV}/bin/python"
SPEC_PYTHON="${CONDA_BASE}/envs/${SPEC_CONDA_ENV}/bin/python"

[[ -x "${VLLM_PYTHON}" ]] || {
    echo "ERROR: ${VLLM_PYTHON} not found"
    exit 1
}

[[ -x "${SPEC_PYTHON}" ]] || {
    echo "ERROR: ${SPEC_PYTHON} not found"
    exit 1
}


# ============================================================
# 4. Per-run logging
# ============================================================

mkdir -p "${REGEN_DIR}" "${RUN_DIR}"

# Save a copy of the exact shell script used for this run
if [[ -f "$0" ]]; then
    cp "$0" "${SCRIPT_SNAPSHOT}" 2>/dev/null || true
fi

# Everything printed by this shell script and regeneration client
# goes to both terminal and run.log.
exec > >(tee -a "${RUN_LOG}") 2>&1

export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Qwen3-8B on-policy regeneration"
echo "============================================================"
echo "Run ID:                ${RUN_ID}"
echo "Run directory:         ${RUN_DIR}"
echo "Run log:               ${RUN_LOG}"
echo "vLLM log:              ${VLLM_LOG}"
echo "Config:                ${CONFIG_FILE}"
echo "Regenerated data:      ${REGEN_FILE}"
echo "============================================================"


# ============================================================
# 5. Save effective run configuration
# ============================================================

cat > "${CONFIG_FILE}" <<EOF
run_id=${RUN_ID}
start_time=$(date '+%Y-%m-%d %H:%M:%S')

model=${MODEL}
dataset=${DATASET}
max_samples=${MAX_SAMPLES}

output_dir=${OUTPUT_DIR}
regen_file=${REGEN_FILE}

enable_speculative=${ENABLE_SPECULATIVE}
speculator_model=${SPECULATOR_MODEL}
speculative_tokens=${SPECULATIVE_TOKENS}

vllm_port=${VLLM_PORT}
vllm_gpu=${VLLM_GPU}
vllm_max_model_len=${VLLM_MAX_MODEL_LEN}
vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS}
regen_concurrency=${REGEN_CONCURRENCY}
gpu_memory_utilization=${REGEN_GPU_MEM}

vllm_conda_env=${VLLM_CONDA_ENV}
spec_conda_env=${SPEC_CONDA_ENV}
vllm_python=${VLLM_PYTHON}
spec_python=${SPEC_PYTHON}
EOF

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    {
        echo
        echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
        echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    } >> "${CONFIG_FILE}"
fi


# ============================================================
# 6. Cleanup
# ============================================================

VLLM_PID=""

cleanup() {
    EXIT_CODE=$?

    echo
    echo "============================================================"
    echo "Cleanup"
    echo "============================================================"

    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping regeneration vLLM (PID=${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    {
        echo
        echo "end_time=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "exit_code=${EXIT_CODE}"
    } >> "${CONFIG_FILE}" 2>/dev/null || true

    echo "Run directory: ${RUN_DIR}"
    echo "Exit code: ${EXIT_CODE}"

    exit "${EXIT_CODE}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


# ============================================================
# 7. Count already regenerated conversations
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

            key = obj.get("primary_id")
            if key in (None, ""):
                key = obj.get("id")

            if key not in (None, ""):
                seen.add(str(key))

print(len(seen))
PY
}


# ============================================================
# 8. Wait until vLLM is healthy
# ============================================================

wait_for_vllm() {
    echo "Waiting for vLLM server..."

    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "vLLM server is ready."
            return 0
        fi

        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM exited before becoming ready."
            echo
            echo "Last 100 lines of vLLM log:"
            tail -n 100 "${VLLM_LOG}" || true
            exit 1
        fi

        sleep 2
    done

    echo "ERROR: vLLM did not become ready within timeout."
    echo
    echo "Last 100 lines of vLLM log:"
    tail -n 100 "${VLLM_LOG}" || true
    exit 1
}


# ============================================================
# 9. Check persistent regeneration cache
# ============================================================

EXISTING="$(count_regenerated "${REGEN_FILE}")"

echo
echo "Cached conversations: ${EXISTING}"
echo "Target conversations: ${MAX_SAMPLES}"

if (( EXISTING >= MAX_SAMPLES )); then
    echo "Cache is already sufficient. Nothing to generate."
    exit 0
fi

REMAINING=$((MAX_SAMPLES - EXISTING))

echo "Missing conversations: ${REMAINING}"


# ============================================================
# 10. Build vLLM command
# ============================================================

VLLM_CMD=(
    "${VLLM_PYTHON}"
    -m vllm.entrypoints.cli.main
    serve
    "${MODEL}"

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

    VLLM_CMD+=(
        --speculative-config
        "${SPEC_CONFIG}"
    )
fi


# ============================================================
# 11. Start vLLM
# ============================================================

echo
echo "[1/2] Starting Qwen3-8B regeneration server"

if [[ "${ENABLE_SPECULATIVE}" == "1" ]]; then
    echo "Speculative decoding: ENABLED"
    echo "Speculator:           ${SPECULATOR_MODEL}"
    echo "Speculative tokens:   ${SPECULATIVE_TOKENS}"
else
    echo "Speculative decoding: DISABLED"
fi

echo "GPU:                  ${VLLM_GPU}"
echo "Max model length:     ${VLLM_MAX_MODEL_LEN}"
echo "Max active sequences: ${VLLM_MAX_NUM_SEQS}"
echo "GPU memory util:      ${REGEN_GPU_MEM}"
echo
echo "vLLM server output is being saved to:"
echo "  ${VLLM_LOG}"

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
"${VLLM_CMD[@]}" \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

echo "vLLM PID: ${VLLM_PID}"

wait_for_vllm


# ============================================================
# 12. Regenerate missing responses
# ============================================================

echo
echo "[2/2] Generating ${REMAINING} missing conversations"
echo "HTTP concurrency: ${REGEN_CONCURRENCY}"
echo

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


# ============================================================
# 13. Validate result
# ============================================================

TOTAL="$(count_regenerated "${REGEN_FILE}")"

echo
echo "============================================================"
echo "Regeneration summary"
echo "============================================================"
echo "Before run:              ${EXISTING}"
echo "Requested this run:      ${REMAINING}"
echo "After run:               ${TOTAL}"
echo "Generation elapsed:      ${GEN_SECONDS} s"
echo "Persistent output file:  ${REGEN_FILE}"
echo "Run log:                 ${RUN_LOG}"
echo "vLLM log:                ${VLLM_LOG}"
echo "Config:                  ${CONFIG_FILE}"
echo "============================================================"

{
    echo
    echo "cached_before=${EXISTING}"
    echo "requested_this_run=${REMAINING}"
    echo "cached_after=${TOTAL}"
    echo "generation_elapsed_seconds=${GEN_SECONDS}"
} >> "${CONFIG_FILE}"

if (( TOTAL < MAX_SAMPLES )); then
    echo "ERROR: cache is still insufficient after regeneration."
    echo "Required: ${MAX_SAMPLES}"
    echo "Available: ${TOTAL}"
    exit 1
fi

echo
echo "Regeneration complete."
