#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# DFlash / SpecLoRA evaluation launcher (per-run isolated logs/results)
#
# Put this file next to evaluate_dflash_lora.py, recommended location:
#   <SpecLoRA>/examples/evaluate/evaluate_opencode_pretrained_10000_12000.sh
#   <SpecLoRA>/examples/evaluate/evaluate_dflash_lora.py
#
# Run from the root of the SpecLoRA repository:
#   bash ./examples/evaluate/evaluate_opencode_pretrained_10000_12000.sh
#
# Every invocation creates a NEW unique RUN_DIR under RESULT_ROOT, e.g.:
#   eval_results/dflash/20260824_150812_123456/
#       run.log                 # complete shell/test orchestration log
#       vllm.log                # vLLM server stdout/stderr for this run only
#       evaluate.log            # evaluator stdout/stderr for this run only
#       run_config.env          # exact run settings
#       summary.json
#       selected_prompts.jsonl
#       responses.jsonl
#       metrics_before.prom
#       metrics_after.prom
#       [lora_inspect.json]      # when USE_LORA=1
#       [merge.log]              # when USE_LORA=1
#
# No run directory is reused. If an explicitly supplied RUN_TAG already exists,
# this script exits rather than overwriting an earlier test.
# ==============================================================================

# -------------------- Python environments --------------------
SPEC_PYTHON="${SPEC_PYTHON:-/data/weijiahong/miniconda3/envs/dflash/bin/python}"

# Dedicated vLLM 0.20.0 + PyTorch cu130 environment.
VLLM_ENV="${VLLM_ENV:-/data/weijiahong/miniconda3/envs/llm_dflash_eval}"
VLLM_PYTHON="${VLLM_PYTHON:-${VLLM_ENV}/bin/python}"

# Verified-working CUDA 13 forward-compat setup on A100 + R570.
CUDA_COMPAT_DIR="${VLLM_ENV}/cuda-compat"
CUDA_RUNTIME_DIR="${VLLM_ENV}/lib"
VLLM_LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${CUDA_RUNTIME_DIR}"
VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
VLLM_CUDA_COMPATIBILITY_PATH="${VLLM_CUDA_COMPATIBILITY_PATH:-${CUDA_COMPAT_DIR}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/evaluate_dflash_lora.py}"

# -------------------- Models --------------------
VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/Qwen3-8B-speculator.dflash}"

# -------------------- LoRA --------------------
# 0: directly evaluate pretrained DFlash.
# 1: merge LORA_PATH into DRAFT_MODEL, then evaluate the merged standalone draft.
USE_LORA="${USE_LORA:-1}"
#LORA_PATH="${LORA_PATH:-./output/dflash_qwen3_8b_opencode_lora_onpolicy/checkpoints/lora/final}"
LORA_PATH="${LORA_PATH:-./output/dflash_qwen3_8b_opencode_lora_online_5k/checkpoints/lora/final}"
MERGE_DTYPE="${MERGE_DTYPE:-bfloat16}"

# -------------------- Dataset --------------------
# This dedicated launcher evaluates the Python-style slice [10000:12000],
# i.e. zero-based rows 10000..11999, exactly 2000 samples.
# OpenCodeInstruct uses the `input` field as the user prompt.
DATA="${DATA:-nvidia/OpenCodeInstruct}"
DATA_FILE="${DATA_FILE:-}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
SPLIT="${SPLIT:-train}"
PROMPT_COLUMN="${PROMPT_COLUMN:-input}"
START_INDEX="${START_INDEX:-10000}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"

# -------------------- Generation --------------------
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
CONCURRENCY="${CONCURRENCY:-6}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-auto}"
TOP_K="${TOP_K:-auto}"
SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

# -------------------- vLLM --------------------
# Match the conservative settings used by your older v2 training launcher:
VLLM_GPU="${VLLM_GPU:-7}"
VLLM_PORT="${VLLM_PORT:-18083}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-55296}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# -------------------- Per-run result layout --------------------
# Parent directory only. Never point this at one old run directory.
RESULT_ROOT="${RESULT_ROOT:-./eval_results/dflash}"

# Optional human-readable prefix, e.g. RUN_TAG=humaneval_lora7.
# A timestamp + PID is always appended, so every invocation is unique.
RUN_TAG="${RUN_TAG:-opencode_pretrained_10000_12000}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_TAG}_${TIMESTAMP}_pid$$"
RUN_DIR="${RESULT_ROOT}/${RUN_ID}"

# Keep merged weights outside RUN_DIR by default because they can be large, but
# still isolate them per run. Override MERGED_ROOT or MERGED_DIR if desired.
MERGED_ROOT="${MERGED_ROOT:-./output/eval_merged_dflash_lora}"
MERGED_DIR="${MERGED_DIR:-${MERGED_ROOT}/${RUN_ID}}"

if [[ ! -f "${EVAL_PY}" ]]; then
  echo "ERROR: evaluation python file not found: ${EVAL_PY}" >&2
  echo "Put evaluate_dflash_lora.py next to this shell file or set EVAL_PY=/path/to/file.py" >&2
  exit 1
fi

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "ERROR: vLLM Python not found/executable: ${VLLM_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${CUDA_COMPAT_DIR}/libcuda.so.1" ]]; then
  echo "ERROR: CUDA compat libcuda.so.1 not found: ${CUDA_COMPAT_DIR}/libcuda.so.1" >&2
  exit 1
fi
if [[ ! -f "${CUDA_RUNTIME_DIR}/libcudart.so.13" ]]; then
  echo "ERROR: CUDA 13 libcudart not found: ${CUDA_RUNTIME_DIR}/libcudart.so.13" >&2
  exit 1
fi

if [[ "${USE_LORA}" != "0" && "${USE_LORA}" != "1" ]]; then
  echo "ERROR: USE_LORA must be 0 or 1, got ${USE_LORA}" >&2
  exit 1
fi

if [[ -e "${RUN_DIR}" ]]; then
  echo "ERROR: refusing to overwrite existing run directory: ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}"

RUN_LOG="${RUN_DIR}/run.log"
VLLM_LOG="${RUN_DIR}/vllm.log"
EVAL_LOG="${RUN_DIR}/evaluate.log"
CONFIG_LOG="${RUN_DIR}/run_config.env"

# From here on, every shell/evaluator message is also persisted in run.log.
exec > >(tee -a "${RUN_LOG}") 2>&1

SERVER_URL="http://127.0.0.1:${VLLM_PORT}"
VLLM_PID=""

cat > "${CONFIG_LOG}" <<CFG
RUN_ID=${RUN_ID}
RUN_DIR=${RUN_DIR}
STARTED_AT=$(date -Iseconds)
SPEC_PYTHON=${SPEC_PYTHON}
VLLM_ENV=${VLLM_ENV}
VLLM_PYTHON=${VLLM_PYTHON}
CUDA_COMPAT_DIR=${CUDA_COMPAT_DIR}
CUDA_RUNTIME_DIR=${CUDA_RUNTIME_DIR}
VLLM_LD_LIBRARY_PATH=${VLLM_LD_LIBRARY_PATH}
VLLM_ENABLE_CUDA_COMPATIBILITY=${VLLM_ENABLE_CUDA_COMPATIBILITY}
VLLM_CUDA_COMPATIBILITY_PATH=${VLLM_CUDA_COMPATIBILITY_PATH}
EVAL_PY=${EVAL_PY}
VERIFIER_MODEL=${VERIFIER_MODEL}
DRAFT_MODEL=${DRAFT_MODEL}
USE_LORA=${USE_LORA}
LORA_PATH=${LORA_PATH}
MERGED_DIR=${MERGED_DIR}
MERGE_DTYPE=${MERGE_DTYPE}
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
TEMPERATURE=${TEMPERATURE}
TOP_P=${TOP_P}
TOP_K=${TOP_K}
SEED=${SEED}
ENABLE_THINKING=${ENABLE_THINKING}
VLLM_GPU=${VLLM_GPU}
VLLM_PORT=${VLLM_PORT}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
CFG

cleanup() {
  local exit_code=$?
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM server (pid=${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  echo "FINISHED_AT=$(date -Iseconds)" >> "${CONFIG_LOG}" || true
  echo "EXIT_CODE=${exit_code}" >> "${CONFIG_LOG}" || true
  echo "Run directory: ${RUN_DIR}"
  echo "  run log : ${RUN_LOG}"
  echo "  vLLM log: ${VLLM_LOG}"
  echo "  eval log: ${EVAL_LOG}"
  trap - EXIT INT TERM
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "DFlash evaluation run"
echo "Run ID       : ${RUN_ID}"
echo "Run directory: ${RUN_DIR}"
echo "vLLM log     : ${VLLM_LOG}"
echo "Eval log     : ${EVAL_LOG}"
echo "============================================================"

# ==============================================================================
# Step 1: pretrained-only OR LoRA merge
# ==============================================================================
if [[ "${USE_LORA}" == "1" ]]; then
  echo ""
  echo "============================================================"
  echo "[1/3] Inspecting and merging LoRA"
  echo "Base DFlash : ${DRAFT_MODEL}"
  echo "LoRA        : ${LORA_PATH}"
  echo "Merged out  : ${MERGED_DIR}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="" "${SPEC_PYTHON}" "${EVAL_PY}" inspect-lora \
    --lora-path "${LORA_PATH}" \
    | tee "${RUN_DIR}/lora_inspect.json"

  CUDA_VISIBLE_DEVICES="" "${SPEC_PYTHON}" "${EVAL_PY}" merge \
    --draft-model "${DRAFT_MODEL}" \
    --verifier "${VERIFIER_MODEL}" \
    --lora-path "${LORA_PATH}" \
    --output-dir "${MERGED_DIR}" \
    --dtype "${MERGE_DTYPE}" \
    --overwrite \
    | tee "${RUN_DIR}/merge.log"

  SERVE_DRAFT="${MERGED_DIR}"
else
  echo ""
  echo "============================================================"
  echo "[1/3] LoRA disabled: evaluating pretrained DFlash directly"
  echo "DFlash: ${DRAFT_MODEL}"
  echo "============================================================"
  SERVE_DRAFT="${DRAFT_MODEL}"
fi

echo "SERVE_DRAFT=${SERVE_DRAFT}" >> "${CONFIG_LOG}"

# ==============================================================================
# Verified CUDA 13 compatibility preflight
# ==============================================================================
echo ""
echo "============================================================"
echo "[preflight] CUDA 13 compatibility"
echo "vLLM Python : ${VLLM_PYTHON}"
echo "compat path : ${CUDA_COMPAT_DIR}"
echo "GPU         : ${VLLM_GPU}"
echo "============================================================"

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

print("CUDA avail :", torch.cuda.is_available())
print("devices    :", torch.cuda.device_count())
print("GPU        :", torch.cuda.get_device_name(0))
x = torch.randn(128, 128, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("compute OK :", y.shape)
PY

# ==============================================================================
# Step 2: vLLM server -- its stdout/stderr goes ONLY to this run's vllm.log
# ==============================================================================
SPEC_CONFIG="$(${VLLM_PYTHON} - "${SERVE_DRAFT}" "${NUM_SPEC_TOKENS}" <<'PY'
import json, sys
print(json.dumps({
    "method": "dflash",
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
}))
PY
)"

echo ""
echo "============================================================"
echo "[2/3] Starting vLLM speculative server"
echo "Verifier     : ${VERIFIER_MODEL}"
echo "Draft        : ${SERVE_DRAFT}"
echo "Spec tokens  : ${NUM_SPEC_TOKENS}"
echo "GPU          : ${VLLM_GPU}"
echo "Port         : ${VLLM_PORT}"
echo "vLLM log     : ${VLLM_LOG}"
echo "============================================================"

VLLM_ARGS=(
  -m vllm.entrypoints.cli.main serve "${VERIFIER_MODEL}"
  --speculative-config "${SPEC_CONFIG}"
  --port "${VLLM_PORT}"
  --tensor-parallel-size 1
  --dtype bfloat16
  --generation-config vllm
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --no-enable-prefix-caching
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
)


printf 'VLLM_COMMAND=' >> "${CONFIG_LOG}"
printf '%q ' \
  env -u LD_PRELOAD \
  LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}" \
  VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY}" \
  VLLM_CUDA_COMPATIBILITY_PATH="${VLLM_CUDA_COMPATIBILITY_PATH}" \
  CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
  "${VLLM_PYTHON}" "${VLLM_ARGS[@]}" >> "${CONFIG_LOG}"
printf '\n' >> "${CONFIG_LOG}"

env -u LD_PRELOAD \
  LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}" \
  VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY}" \
  VLLM_CUDA_COMPATIBILITY_PATH="${VLLM_CUDA_COMPATIBILITY_PATH}" \
  CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
  "${VLLM_PYTHON}" "${VLLM_ARGS[@]}" \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "VLLM_PID=${VLLM_PID}" >> "${CONFIG_LOG}"

echo "Waiting for vLLM server..."
READY=0
for _ in $(seq 1 180); do
  if curl -sf "${SERVER_URL}/health" >/dev/null 2>&1; then
    READY=1
    echo "vLLM is READY."
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM exited during startup. Last 100 lines of this run's vLLM log:"
    tail -100 "${VLLM_LOG}" || true
    exit 1
  fi
  sleep 2
done

if [[ "${READY}" != "1" ]]; then
  echo "ERROR: vLLM did not become ready. Last 100 lines of this run's vLLM log:"
  tail -100 "${VLLM_LOG}" || true
  exit 1
fi

# ==============================================================================
# Step 3: exact dataset slice + evaluation
# ==============================================================================
echo ""
echo "============================================================"
echo "[3/3] Evaluating"
echo "Data         : ${DATA}"
echo "Data file    : ${DATA_FILE:-<none>}"
echo "Start index  : ${START_INDEX} (zero-based)"
echo "Num samples  : ${NUM_SAMPLES}"
echo "Thinking     : ${ENABLE_THINKING}"
echo "Results      : ${RUN_DIR}"
echo "Eval log     : ${EVAL_LOG}"
echo "============================================================"

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
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --seed "${SEED}"
  --enable-thinking "${ENABLE_THINKING}"
)

if [[ -n "${DATA_FILE}" ]]; then
  EVAL_ARGS+=(--data-file "${DATA_FILE}")
fi
if [[ -n "${DATASET_CONFIG}" ]]; then
  EVAL_ARGS+=(--dataset-config "${DATASET_CONFIG}")
fi
if [[ -n "${SYSTEM_PROMPT}" ]]; then
  EVAL_ARGS+=(--system-prompt "${SYSTEM_PROMPT}")
fi

printf 'EVAL_COMMAND=' >> "${CONFIG_LOG}"
printf '%q ' "${SPEC_PYTHON}" "${EVAL_PY}" "${EVAL_ARGS[@]}" >> "${CONFIG_LOG}"
printf '\n' >> "${CONFIG_LOG}"

# Evaluator stdout/stderr is saved to evaluate.log AND duplicated into run.log.
"${SPEC_PYTHON}" "${EVAL_PY}" "${EVAL_ARGS[@]}" 2>&1 | tee "${EVAL_LOG}"

echo ""
echo "============================================================"
echo "Evaluation completed"
echo "Summary   : ${RUN_DIR}/summary.json"
echo "Responses : ${RUN_DIR}/responses.jsonl"
echo "vLLM log  : ${VLLM_LOG}"
echo "Eval log  : ${EVAL_LOG}"
echo "Run log   : ${RUN_LOG}"
echo "============================================================"
