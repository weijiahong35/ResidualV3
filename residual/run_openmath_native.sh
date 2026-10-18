#!/usr/bin/env bash
# Shared native-vLLM runner for Base DFlash and Residual SpecMoLE DFlash.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

BENCH_MODE="${BENCH_MODE:-residual}"
[[ "${BENCH_MODE}" == "target" || "${BENCH_MODE}" == "base" || "${BENCH_MODE}" == "residual" ]] || {
    echo "ERROR: BENCH_MODE must be target, base, or residual"; exit 1;
}

SPEC_PYTHON="${SPEC_PYTHON:-/data/weijiahong/miniconda3/envs/dflash/bin/python}"
VLLM_ENV="${VLLM_ENV:-/data/weijiahong/miniconda3/envs/llm_dflash_eval}"
VLLM_PYTHON="${VLLM_PYTHON:-${VLLM_ENV}/bin/python}"
EVAL_PY="${EVAL_PY:-${PROJECT_ROOT}/residual/evaluate_native_dflash.py}"
SPECMOLE_LAUNCHER="${SPECMOLE_LAUNCHER:-${PROJECT_ROOT}/scripts/launch_vllm_specmole.py}"
BASE_PROFILE_LAUNCHER="${BASE_PROFILE_LAUNCHER:-${PROJECT_ROOT}/scripts/launch_vllm_profile_specmole.py}"

CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-${VLLM_ENV}/cuda-compat}"
CUDA_RUNTIME_DIR="${CUDA_RUNTIME_DIR:-${VLLM_ENV}/lib}"
VLLM_LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${CUDA_RUNTIME_DIR}"
VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
VLLM_CUDA_COMPATIBILITY_PATH="${VLLM_CUDA_COMPATIBILITY_PATH:-${CUDA_COMPAT_DIR}}"

VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/Qwen3-8B-speculator.dflash}"
# Formal tests default to final Stage 3. An explicit CHECKPOINT may select Stage 2.
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

# Edit GPU in either test entrypoint, or override GPU/TARGET_GPU from the shell.
GPU="${GPU:-0}"
TARGET_GPU="${TARGET_GPU:-${GPU}}"
VLLM_PORT="${VLLM_PORT:-18083}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-18}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-55296}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
DTYPE="${DTYPE:-bfloat16}"
SPECMOLE_PROFILE_PROPOSER="${SPECMOLE_PROFILE_PROPOSER:-0}"
SPECMOLE_PROFILE_WINDOW="${SPECMOLE_PROFILE_WINDOW:-128}"
SPECMOLE_REFERENCE_RUNTIME="${SPECMOLE_REFERENCE_RUNTIME:-0}"
SPECMOLE_CUDA_GRAPHS="${SPECMOLE_CUDA_GRAPHS:-1}"

RESULT_ROOT="${RESULT_ROOT:-./eval_results/residual_runtime}"
RUN_TAG="${RUN_TAG:-openmath_${BENCH_MODE}_native_${START_INDEX}_$((START_INDEX + NUM_SAMPLES))}"
RUN_ID="${RUN_ID:-${RUN_TAG}_$(date '+%Y%m%d_%H%M%S')_pid$$}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/${RUN_ID}}"

for file in "${EVAL_PY}" "${SPECMOLE_LAUNCHER}" "${BASE_PROFILE_LAUNCHER}"; do
    [[ -f "${file}" ]] || { echo "ERROR: missing ${file}"; exit 1; }
done
[[ -x "${SPEC_PYTHON}" ]] || { echo "ERROR: ${SPEC_PYTHON} not found"; exit 1; }
[[ -x "${VLLM_PYTHON}" ]] || { echo "ERROR: ${VLLM_PYTHON} not found"; exit 1; }
[[ -f "${CUDA_COMPAT_DIR}/libcuda.so.1" ]] || { echo "ERROR: missing ${CUDA_COMPAT_DIR}/libcuda.so.1"; exit 1; }
[[ -f "${CUDA_RUNTIME_DIR}/libcudart.so.13" ]] || { echo "ERROR: missing ${CUDA_RUNTIME_DIR}/libcudart.so.13"; exit 1; }

if [[ -f "${CHECKPOINT}" ]]; then
    CHECKPOINT="$(cd "$(dirname "${CHECKPOINT}")" && pwd)"
elif [[ -d "${CHECKPOINT}" ]]; then
    CHECKPOINT="$(cd "${CHECKPOINT}" && pwd)"
else
    echo "ERROR: Residual checkpoint not found: ${CHECKPOINT}"; exit 1
fi
CHECKPOINT_STAGE="$("${SPEC_PYTHON}" - "${CHECKPOINT}/residual_mole_config.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(str(payload.get("stage") or (payload.get("residual_config") or {}).get("train_stage") or ""))
PY
)"
[[ "${CHECKPOINT_STAGE}" == "mole" || "${CHECKPOINT_STAGE}" == "joint" ]] || {
    echo "ERROR: checkpoint stage must be mole or joint, got '${CHECKPOINT_STAGE}'"; exit 1;
}
[[ -f "${CHECKPOINT}/v20_model.safetensors" || -f "${CHECKPOINT}/residual_mole.safetensors" ]] || {
    echo "ERROR: missing Residual weights in ${CHECKPOINT}"; exit 1;
}

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
RUNTIME=vllm_native_dflash
VERIFIER_MODEL=${VERIFIER_MODEL}
DRAFT_MODEL=${DRAFT_MODEL}
CHECKPOINT=${CHECKPOINT}
CHECKPOINT_STAGE=${CHECKPOINT_STAGE}
DATA=${DATA}
DATA_FILE=${DATA_FILE}
DATASET_CONFIG=${DATASET_CONFIG}
SPLIT=${SPLIT}
PROMPT_COLUMN=${PROMPT_COLUMN}
START_INDEX=${START_INDEX}
NUM_SAMPLES=${NUM_SAMPLES}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
CONCURRENCY=${CONCURRENCY}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT}
SEED=${SEED}
ENABLE_THINKING=${ENABLE_THINKING}
TARGET_GPU=${TARGET_GPU}
VLLM_PORT=${VLLM_PORT}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS}
ENFORCE_EAGER=${ENFORCE_EAGER}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
DTYPE=${DTYPE}
PREFIX_CACHING=false
SPECMOLE_PROFILE_PROPOSER=${SPECMOLE_PROFILE_PROPOSER}
SPECMOLE_PROFILE_WINDOW=${SPECMOLE_PROFILE_WINDOW}
SPECMOLE_REFERENCE_RUNTIME=${SPECMOLE_REFERENCE_RUNTIME}
SPECMOLE_CUDA_GRAPHS=${SPECMOLE_CUDA_GRAPHS}
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

SPEC_CONFIG="$("${VLLM_PYTHON}" - "${DRAFT_MODEL}" "${NUM_SPEC_TOKENS}" "${ENFORCE_EAGER}" <<'PY'
import json, sys
eager = sys.argv[3].strip().lower()
if eager not in {"true", "false"}:
    raise ValueError("ENFORCE_EAGER must be true or false")
print(json.dumps({
    "method": "dflash",
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
    "enforce_eager": eager == "true",
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

start_server() {
    local mode="$1"
    local log_path="$2"
    local -a command
    local -a mode_args=()
    if [[ "${mode}" != "target" ]]; then
        mode_args+=(--speculative-config "${SPEC_CONFIG}")
    fi
    if [[ "${mode}" == "residual" ]]; then
        command=("${VLLM_PYTHON}" "${SPECMOLE_LAUNCHER}")
    elif [[ "${mode}" == "base" && "${SPECMOLE_PROFILE_PROPOSER}" == "1" ]]; then
        command=("${VLLM_PYTHON}" "${BASE_PROFILE_LAUNCHER}")
    else
        command=("${VLLM_PYTHON}" -m vllm.entrypoints.cli.main)
    fi
    echo "Starting ${mode} native vLLM on physical GPU ${TARGET_GPU}"
    env -u LD_PRELOAD \
        LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}" \
        VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY}" \
        VLLM_CUDA_COMPATIBILITY_PATH="${VLLM_CUDA_COMPATIBILITY_PATH}" \
        CUDA_VISIBLE_DEVICES="${TARGET_GPU}" \
        SPECMOLE_CHECKPOINT="${CHECKPOINT}" \
        SPECMOLE_PROFILE_PROPOSER="${SPECMOLE_PROFILE_PROPOSER}" \
        SPECMOLE_PROFILE_WINDOW="${SPECMOLE_PROFILE_WINDOW}" \
        SPECMOLE_REFERENCE_RUNTIME="${SPECMOLE_REFERENCE_RUNTIME}" \
        SPECMOLE_CUDA_GRAPHS="${SPECMOLE_CUDA_GRAPHS}" \
        PYTHONPATH="${PROJECT_ROOT}/hs_connectors/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${command[@]}" "${COMMON_VLLM_ARGS[@]}" "${mode_args[@]}" > "${log_path}" 2>&1 &
    VLLM_PID=$!
    local ready=0
    for _ in $(seq 1 180); do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: ${mode} vLLM exited during startup"
            tail -n 180 "${log_path}" || true
            exit 1
        fi
        if curl -sf "${SERVER_URL}/health" >/dev/null 2>&1; then
            ready=1; break
        fi
        sleep 2
    done
    [[ "${ready}" == "1" ]] || {
        echo "ERROR: ${mode} vLLM did not become ready"; tail -n 180 "${log_path}" || true; exit 1;
    }
}

start_server "${BENCH_MODE}" "${RUN_DIR}/vllm.log"

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
