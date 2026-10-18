#!/usr/bin/env bash
# Isolated vLLM 0.26 native runner. The v0.20 runner is intentionally untouched.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

BENCH_MODE="${BENCH_MODE:-base}"
[[ "${BENCH_MODE}" == "target" || "${BENCH_MODE}" == "base" || "${BENCH_MODE}" == "residual" ]] || {
    echo "ERROR: BENCH_MODE must be target, base, or residual"; exit 1;
}

SPEC_PYTHON="${SPEC_PYTHON:-/data/weijiahong/miniconda3/envs/dflash/bin/python}"
VLLM_ENV="${VLLM_ENV:-/data/weijiahong/miniconda3/envs/llm_dflash_eval_v026}"
VLLM_PYTHON="${VLLM_PYTHON:-${VLLM_ENV}/bin/python}"
CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-${VLLM_ENV}/cuda-compat}"
CUDA_RUNTIME_DIR="${CUDA_RUNTIME_DIR:-${VLLM_ENV}/lib}"
VLLM_LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${CUDA_RUNTIME_DIR}"
EVAL_PY="${EVAL_PY:-${PROJECT_ROOT}/residual/evaluate_native_dflash.py}"
SPECMOLE_LAUNCHER="${SPECMOLE_LAUNCHER:-${PROJECT_ROOT}/scripts/launch_vllm_specmole_v026.py}"

VERIFIER_MODEL="${VERIFIER_MODEL:-/data/weijiahong/huggingface/models/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-/data/weijiahong/huggingface/models/Qwen3-8B-speculator.dflash}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/output/openmath/onpolicy/residual/stage3_joint/latest/checkpoints/final}"
DATA="${DATA:-nvidia/OpenMathInstruct-2}"
DATA_FILE="${DATA_FILE:-}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
SPLIT="${SPLIT:-train_1M}"
PROMPT_COLUMN="${PROMPT_COLUMN:-problem}"
START_INDEX="${START_INDEX:-10000}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
CONCURRENCY="${CONCURRENCY:-18}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

# Preserve the existing project GPU selection semantics.
GPU="${GPU:-7}"
TARGET_GPU="${TARGET_GPU:-${GPU}}"
VLLM_PORT="${VLLM_PORT:-18086}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-18}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-55296}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
DTYPE="${DTYPE:-bfloat16}"
SPECMOLE_PROFILE_PROPOSER="${SPECMOLE_PROFILE_PROPOSER:-0}"
SPECMOLE_PROFILE_WINDOW="${SPECMOLE_PROFILE_WINDOW:-128}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

RESULT_ROOT="${RESULT_ROOT:-./eval_results/residual_runtime_v026}"
RUN_TAG="${RUN_TAG:-openmath_${BENCH_MODE}_v026_${START_INDEX}_$((START_INDEX + NUM_SAMPLES))}"
RUN_ID="${RUN_ID:-${RUN_TAG}_$(date '+%Y%m%d_%H%M%S')_pid$$}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/${RUN_ID}}"

for file in "${EVAL_PY}"; do
    [[ -f "${file}" ]] || { echo "ERROR: missing ${file}"; exit 1; }
done
[[ -x "${SPEC_PYTHON}" ]] || { echo "ERROR: ${SPEC_PYTHON} not found"; exit 1; }
[[ -x "${VLLM_PYTHON}" ]] || { echo "ERROR: ${VLLM_PYTHON} not found"; exit 1; }
[[ -f "${CUDA_COMPAT_DIR}/libcuda.so.1" ]] || { echo "ERROR: missing ${CUDA_COMPAT_DIR}/libcuda.so.1"; exit 1; }
[[ -f "${CUDA_RUNTIME_DIR}/libcudart.so.13" ]] || { echo "ERROR: missing ${CUDA_RUNTIME_DIR}/libcudart.so.13"; exit 1; }
if [[ "${BENCH_MODE}" == "residual" ]]; then
    [[ -f "${SPECMOLE_LAUNCHER}" ]] || { echo "ERROR: missing ${SPECMOLE_LAUNCHER}"; exit 1; }
    [[ -d "${CHECKPOINT}" ]] || { echo "ERROR: checkpoint not found: ${CHECKPOINT}"; exit 1; }
fi

[[ ! -e "${RUN_DIR}" ]] || { echo "ERROR: RUN_DIR already exists: ${RUN_DIR}"; exit 1; }
mkdir -p "${RUN_DIR}"
RUN_LOG="${RUN_DIR}/run.log"
EVAL_LOG="${RUN_DIR}/evaluate.log"
CONFIG_LOG="${RUN_DIR}/run_config.env"
SERVER_URL="http://127.0.0.1:${VLLM_PORT}"
VLLM_PID=""
exec > >(tee -a "${RUN_LOG}") 2>&1
export PYTHONUNBUFFERED=1

cat > "${CONFIG_LOG}" <<EOF
MODE=${BENCH_MODE}
RUNTIME=vllm_0.26_native_dflash
VLLM_ENV=${VLLM_ENV}
VERIFIER_MODEL=${VERIFIER_MODEL}
DRAFT_MODEL=${DRAFT_MODEL}
CHECKPOINT=${CHECKPOINT}
DATA=${DATA}
SPLIT=${SPLIT}
START_INDEX=${START_INDEX}
NUM_SAMPLES=${NUM_SAMPLES}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
CONCURRENCY=${CONCURRENCY}
TARGET_GPU=${TARGET_GPU}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS}
ENFORCE_EAGER=${ENFORCE_EAGER}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
DTYPE=${DTYPE}
SPECMOLE_PROFILE_PROPOSER=${SPECMOLE_PROFILE_PROPOSER}
SPECMOLE_PROFILE_WINDOW=${SPECMOLE_PROFILE_WINDOW}
VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}
EOF

stop_server() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    VLLM_PID=""
}
cleanup() {
    code=$?
    stop_server
    echo "EXIT_CODE=${code}" >> "${CONFIG_LOG}" 2>/dev/null || true
    trap - EXIT
    exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SPEC_CONFIG="$("${VLLM_PYTHON}" - "${DRAFT_MODEL}" "${NUM_SPEC_TOKENS}" <<'PY'
import json, sys
print(json.dumps({
    "method": "dflash",
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
}))
PY
)"

COMMON_VLLM_ARGS=(
    serve "${VERIFIER_MODEL}"
    --port "${VLLM_PORT}"
    --tensor-parallel-size 1
    --dtype "${DTYPE}"
    --generation-config vllm
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --no-enable-prefix-caching
)
if [[ "${ENFORCE_EAGER}" == "true" ]]; then
    COMMON_VLLM_ARGS+=(--enforce-eager)
elif [[ "${ENFORCE_EAGER}" != "false" ]]; then
    echo "ERROR: ENFORCE_EAGER must be true or false"; exit 1
fi
if [[ "${BENCH_MODE}" != "target" ]]; then
    COMMON_VLLM_ARGS+=(--speculative-config "${SPEC_CONFIG}")
fi

if [[ "${BENCH_MODE}" == "residual" ]]; then
    COMMAND=("${VLLM_PYTHON}" "${SPECMOLE_LAUNCHER}")
else
    COMMAND=("${VLLM_PYTHON}" -m vllm.entrypoints.cli.main)
fi

echo "Starting ${BENCH_MODE} vLLM 0.26 on physical GPU ${TARGET_GPU}"
env -u LD_PRELOAD \
    PATH="${VLLM_ENV}/bin:${PATH}" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}" \
    VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
    VLLM_CUDA_COMPATIBILITY_PATH="${CUDA_COMPAT_DIR}" \
    VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER}" \
    CUDA_VISIBLE_DEVICES="${TARGET_GPU}" \
    SPECMOLE_CHECKPOINT="${CHECKPOINT}" \
    SPECMOLE_PROFILE_PROPOSER="${SPECMOLE_PROFILE_PROPOSER}" \
    SPECMOLE_PROFILE_WINDOW="${SPECMOLE_PROFILE_WINDOW}" \
    PYTHONPATH="${PROJECT_ROOT}/hs_connectors/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${COMMAND[@]}" "${COMMON_VLLM_ARGS[@]}" > "${RUN_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!

ready=0
for _ in $(seq 1 240); do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: ${BENCH_MODE} vLLM exited during startup"
        tail -n 220 "${RUN_DIR}/vllm.log" || true
        exit 1
    fi
    if curl -sf "${SERVER_URL}/health" >/dev/null 2>&1; then
        ready=1; break
    fi
    sleep 2
done
[[ "${ready}" == "1" ]] || { echo "ERROR: vLLM did not become ready"; exit 1; }

EVAL_ARGS=(
    evaluate --endpoint "${SERVER_URL}/v1" --data "${DATA}" --split "${SPLIT}"
    --prompt-column "${PROMPT_COLUMN}" --start-index "${START_INDEX}"
    --num-samples "${NUM_SAMPLES}" --output-dir "${RUN_DIR}"
    --max-new-tokens "${MAX_NEW_TOKENS}" --concurrency "${CONCURRENCY}"
    --request-timeout "${REQUEST_TIMEOUT}" --seed "${SEED}"
    --enable-thinking "${ENABLE_THINKING}" --temperature 0 --top-p 1 --top-k 0
)
[[ -n "${DATA_FILE}" ]] && EVAL_ARGS+=(--data-file "${DATA_FILE}")
[[ -n "${DATASET_CONFIG}" ]] && EVAL_ARGS+=(--dataset-config "${DATASET_CONFIG}")
[[ -n "${SYSTEM_PROMPT}" ]] && EVAL_ARGS+=(--system-prompt "${SYSTEM_PROMPT}")
"${SPEC_PYTHON}" "${EVAL_PY}" "${EVAL_ARGS[@]}" 2>&1 | tee "${EVAL_LOG}"

echo "Evaluation completed: ${RUN_DIR}"
