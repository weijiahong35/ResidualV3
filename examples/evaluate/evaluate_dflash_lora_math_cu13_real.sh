#!/usr/bin/env bash
set -euo pipefail

SPEC_PYTHON="${SPEC_PYTHON:-/data/weijiahong/miniconda3/envs/dflash/bin/python}"

VLLM_ENV="${VLLM_ENV:-/data/weijiahong/miniconda3/envs/llm_dflash_eval}"
VLLM_PYTHON="${VLLM_PYTHON:-${VLLM_ENV}/bin/python}"

# CUDA 13 forward compatibility
CUDA_COMPAT_DIR="${VLLM_ENV}/cuda-compat"
CUDA_RUNTIME_DIR="${VLLM_ENV}/lib"
VLLM_LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${CUDA_RUNTIME_DIR}"

VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
VLLM_CUDA_COMPATIBILITY_PATH="${VLLM_CUDA_COMPATIBILITY_PATH:-${CUDA_COMPAT_DIR}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/evaluate_dflash_lora.py}"

VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/Qwen3-8B-speculator.dflash}"

USE_LORA="${USE_LORA:-0}"
LORA_PATH="${LORA_PATH:-./output/openmath/onpolicy/checkpoints/lora/final}"
MERGE_DTYPE="${MERGE_DTYPE:-bfloat16}"

DATA="${DATA:-nvidia/OpenMathInstruct-2}"
DATA_FILE="${DATA_FILE:-}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
SPLIT="${SPLIT:-train_1M}"
PROMPT_COLUMN="${PROMPT_COLUMN:-problem}"

START_INDEX="${START_INDEX:-10000}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
CONCURRENCY="${CONCURRENCY:-20}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"

SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

VLLM_GPU="${VLLM_GPU:-0}"
VLLM_PORT="${VLLM_PORT:-18083}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-20}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-55296}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

RESULT_ROOT="${RESULT_ROOT:-./eval_results/openmath}"

RUN_TAG="${RUN_TAG:-openmath_10000_12000}"
RUN_ID="${RUN_TAG}_$(date '+%Y%m%d_%H%M%S')_pid$$"
RUN_DIR="${RESULT_ROOT}/${RUN_ID}"

MERGED_ROOT="${MERGED_ROOT:-./output/eval_merged_dflash_lora/math}"
MERGED_DIR="${MERGED_DIR:-${MERGED_ROOT}/${RUN_ID}}"

[[ -f "${EVAL_PY}" ]] || {
    echo "ERROR: ${EVAL_PY} not found"
    exit 1
}

[[ -x "${SPEC_PYTHON}" ]] || {
    echo "ERROR: ${SPEC_PYTHON} not found"
    exit 1
}

[[ -x "${VLLM_PYTHON}" ]] || {
    echo "ERROR: ${VLLM_PYTHON} not found"
    exit 1
}

[[ -f "${CUDA_COMPAT_DIR}/libcuda.so.1" ]] || {
    echo "ERROR: CUDA compat libcuda.so.1 not found: ${CUDA_COMPAT_DIR}/libcuda.so.1"
    exit 1
}

[[ -f "${CUDA_RUNTIME_DIR}/libcudart.so.13" ]] || {
    echo "ERROR: CUDA 13 libcudart not found: ${CUDA_RUNTIME_DIR}/libcudart.so.13"
    exit 1
}

[[ "${USE_LORA}" == "0" || "${USE_LORA}" == "1" ]] || {
    echo "ERROR: USE_LORA must be 0 or 1"
    exit 1
}

[[ ! -e "${RUN_DIR}" ]] || {
    echo "ERROR: ${RUN_DIR} already exists"
    exit 1
}

mkdir -p "${RUN_DIR}"

RUN_LOG="${RUN_DIR}/run.log"
VLLM_LOG="${RUN_DIR}/vllm.log"
EVAL_LOG="${RUN_DIR}/evaluate.log"
CONFIG_LOG="${RUN_DIR}/run_config.env"

exec > >(tee -a "${RUN_LOG}") 2>&1
export PYTHONUNBUFFERED=1

SERVER_URL="http://127.0.0.1:${VLLM_PORT}"
VLLM_PID=""

cat > "${CONFIG_LOG}" <<EOF
USE_LORA=${USE_LORA}
LORA_PATH=${LORA_PATH}

DATA=${DATA}
SPLIT=${SPLIT}
PROMPT_COLUMN=${PROMPT_COLUMN}
START_INDEX=${START_INDEX}
NUM_SAMPLES=${NUM_SAMPLES}

VERIFIER_MODEL=${VERIFIER_MODEL}
DRAFT_MODEL=${DRAFT_MODEL}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS}

VLLM_ENV=${VLLM_ENV}
VLLM_PYTHON=${VLLM_PYTHON}
VLLM_GPU=${VLLM_GPU}
VLLM_PORT=${VLLM_PORT}

CUDA_COMPAT_DIR=${CUDA_COMPAT_DIR}
CUDA_RUNTIME_DIR=${CUDA_RUNTIME_DIR}
VLLM_LD_LIBRARY_PATH=${VLLM_LD_LIBRARY_PATH}
VLLM_ENABLE_CUDA_COMPATIBILITY=${VLLM_ENABLE_CUDA_COMPATIBILITY}
VLLM_CUDA_COMPATIBILITY_PATH=${VLLM_CUDA_COMPATIBILITY_PATH}

MERGED_DIR=${MERGED_DIR}
EOF

cleanup() {
    EXIT_CODE=$?

    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping vLLM server..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    echo "EXIT_CODE=${EXIT_CODE}" >> "${CONFIG_LOG}" 2>/dev/null || true

    trap - EXIT
    exit "${EXIT_CODE}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


if [[ "${USE_LORA}" == "1" ]]; then

    echo "[1/3] Merging LoRA: ${LORA_PATH}"

    CUDA_VISIBLE_DEVICES="" \
    "${SPEC_PYTHON}" "${EVAL_PY}" inspect-lora \
        --lora-path "${LORA_PATH}" \
        | tee "${RUN_DIR}/lora_inspect.json"

    CUDA_VISIBLE_DEVICES="" \
    "${SPEC_PYTHON}" "${EVAL_PY}" merge \
        --draft-model "${DRAFT_MODEL}" \
        --verifier "${VERIFIER_MODEL}" \
        --lora-path "${LORA_PATH}" \
        --output-dir "${MERGED_DIR}" \
        --dtype "${MERGE_DTYPE}" \
        --overwrite \
        | tee "${RUN_DIR}/merge.log"

    SERVE_DRAFT="${MERGED_DIR}"

else

    echo "[1/3] Using pretrained DFlash"

    SERVE_DRAFT="${DRAFT_MODEL}"

fi


echo "[preflight] Checking CUDA 13 environment"

env -u LD_PRELOAD \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}" \
    VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY}" \
    VLLM_CUDA_COMPATIBILITY_PATH="${VLLM_CUDA_COMPATIBILITY_PATH}" \
    CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
    "${VLLM_PYTHON}" - <<'PY'

import ctypes
import torch
import vllm

print("vLLM       :", vllm.__version__)
print("torch      :", torch.__version__)
print("torch CUDA :", torch.version.cuda)

if torch.version.cuda is None:
    raise RuntimeError("PyTorch has no CUDA support")

if not torch.version.cuda.startswith("13"):
    raise RuntimeError(
        f"Expected CUDA 13 PyTorch build, got CUDA {torch.version.cuda}"
    )

ctypes.CDLL("libcuda.so.1")

loaded = []

with open("/proc/self/maps", "r", encoding="utf-8") as f:
    for line in f:
        if "libcuda.so" in line:
            path = line.split()[-1]
            if path not in loaded:
                loaded.append(path)

print("loaded libcuda:")

for path in loaded:
    print(" ", path)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("CUDA avail :", torch.cuda.is_available())
print("devices    :", torch.cuda.device_count())
print("GPU        :", torch.cuda.get_device_name(0))

x = torch.randn(128, 128, device="cuda")
y = x @ x

torch.cuda.synchronize()

print("CUDA 13 compute OK:", y.shape)

PY


SPEC_CONFIG="$("${VLLM_PYTHON}" - "${SERVE_DRAFT}" "${NUM_SPEC_TOKENS}" <<'PY'

import json
import sys

print(json.dumps({
    "method": "dflash",
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
}))

PY
)"


echo "[2/3] Starting vLLM speculative server"

VLLM_ARGS=(
    -m vllm.entrypoints.cli.main
    serve "${VERIFIER_MODEL}"

    --speculative-config "${SPEC_CONFIG}"

    --port "${VLLM_PORT}"

    --tensor-parallel-size 1

    --dtype bfloat16

    --max-model-len "${MAX_MODEL_LEN}"

    --max-num-seqs "${MAX_NUM_SEQS}"

    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"

    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"

    --no-enable-prefix-caching
)


env -u LD_PRELOAD \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}" \
    VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY}" \
    VLLM_CUDA_COMPATIBILITY_PATH="${VLLM_CUDA_COMPATIBILITY_PATH}" \
    CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
    "${VLLM_PYTHON}" "${VLLM_ARGS[@]}" \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!


READY=0

for _ in $(seq 1 180); do

    if curl -sf "${SERVER_URL}/health" >/dev/null 2>&1; then

        READY=1

        echo "vLLM is READY."

        break

    fi


    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then

        echo "ERROR: vLLM exited during startup"

        tail -n 100 "${VLLM_LOG}" || true

        exit 1

    fi


    sleep 2

done


if [[ "${READY}" != "1" ]]; then

    echo "ERROR: vLLM did not become ready"

    tail -n 100 "${VLLM_LOG}" || true

    exit 1

fi


echo "[3/3] Evaluating OpenMathInstruct-2"


EVAL_ARGS=(

    evaluate

    --endpoint "${SERVER_URL}/v1"

    --data "${DATA}"

    --split "${SPLIT}"

    --prompt-column "${PROMPT_COLUMN}"

    --start-index "${START_INDEX}"

    --num-samples "${NUM_SAMPLES}"

    --output-dir "${RUN_DIR}"

    --max-new-tokens "${MAX_NEW_TOKENS}"

    --concurrency "${CONCURRENCY}"

    --request-timeout "${REQUEST_TIMEOUT}"

    --seed "${SEED}"

    --enable-thinking "${ENABLE_THINKING}"
)


[[ -n "${DATA_FILE}" ]] && \
    EVAL_ARGS+=(--data-file "${DATA_FILE}")


[[ -n "${DATASET_CONFIG}" ]] && \
    EVAL_ARGS+=(--dataset-config "${DATASET_CONFIG}")


[[ -n "${SYSTEM_PROMPT}" ]] && \
    EVAL_ARGS+=(--system-prompt "${SYSTEM_PROMPT}")


"${SPEC_PYTHON}" "${EVAL_PY}" "${EVAL_ARGS[@]}" \
    2>&1 | tee "${EVAL_LOG}"


echo "Evaluation completed."
echo "Results: ${RUN_DIR}"