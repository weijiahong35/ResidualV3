#!/usr/bin/env bash
set -euo pipefail

SPEC_PYTHON="${SPEC_PYTHON:-/data/weijiahong/miniconda3/envs/dflash/bin/python}"
VLLM_ENV="${VLLM_ENV:-/data/weijiahong/miniconda3/envs/llm_dflash_eval}"
VLLM_PYTHON="${VLLM_PYTHON:-${VLLM_ENV}/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/evaluate_dflash_lora.py}"

VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/Qwen3-8B-speculator.dflash}"

USE_LORA="${USE_LORA:-0}"
LORA_PATH="${LORA_PATH:-./output/cnn_dailymail/onpolicy/checkpoints/lora/final}"
MERGE_DTYPE="${MERGE_DTYPE:-bfloat16}"

DATA="${DATA:-abisee/cnn_dailymail}"
DATASET_CONFIG="${DATASET_CONFIG:-3.0.0}"
SPLIT="${SPLIT:-test}"

START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
CONCURRENCY="${CONCURRENCY:-20}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"

SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

VLLM_GPU="${VLLM_GPU:-7}"
VLLM_PORT="${VLLM_PORT:-18083}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-20}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-55296}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

RESULT_ROOT="${RESULT_ROOT:-./eval_results/cnn_dailymail}"

RUN_TAG="${RUN_TAG:-cnn_dailymail_test_0_2000}"
RUN_ID="${RUN_TAG}_$(date '+%Y%m%d_%H%M%S')_pid$$"
RUN_DIR="${RESULT_ROOT}/${RUN_ID}"

MERGED_ROOT="${MERGED_ROOT:-./output/eval_merged_dflash_lora/summarization}"
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
EVAL_DATA="${RUN_DIR}/cnn_dailymail_eval.jsonl"

exec > >(tee -a "${RUN_LOG}") 2>&1
export PYTHONUNBUFFERED=1

SERVER_URL="http://127.0.0.1:${VLLM_PORT}"
VLLM_PID=""

cat > "${CONFIG_LOG}" <<EOF
USE_LORA=${USE_LORA}
LORA_PATH=${LORA_PATH}
DATA=${DATA}
DATASET_CONFIG=${DATASET_CONFIG}
SPLIT=${SPLIT}
START_INDEX=${START_INDEX}
NUM_SAMPLES=${NUM_SAMPLES}
VERIFIER_MODEL=${VERIFIER_MODEL}
DRAFT_MODEL=${DRAFT_MODEL}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS}
VLLM_GPU=${VLLM_GPU}
VLLM_PORT=${VLLM_PORT}
MERGED_DIR=${MERGED_DIR}
SEED=${SEED}
ENABLE_THINKING=${ENABLE_THINKING}
EOF

cleanup() {
    EXIT_CODE=$?

    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
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

echo "[1/4] Preparing CNN/DailyMail test prompts"

"${SPEC_PYTHON}" - \
    "${DATA}" \
    "${DATASET_CONFIG}" \
    "${SPLIT}" \
    "${START_INDEX}" \
    "${NUM_SAMPLES}" \
    "${EVAL_DATA}" <<'PY'
import itertools
import json
import sys

from datasets import load_dataset

data, config, split, start, count, output = sys.argv[1:]

start = int(start)
count = int(count)

ds = load_dataset(
    data,
    config,
    split=split,
    streaming=True,
)

rows = list(itertools.islice(ds, start, start + count))

if len(rows) != count:
    raise RuntimeError(
        f"Expected {count} samples, got {len(rows)}"
    )

with open(output, "w", encoding="utf-8") as f:
    for i, row in enumerate(rows):
        obj = {
            "id": row.get("id"),
            "dataset_index": start + i,
            "prompt": f"Summarize: {row['article']}",
            "reference": row["highlights"],
        }

        f.write(
            json.dumps(obj, ensure_ascii=False) + "\n"
        )
PY

if [[ "${USE_LORA}" == "1" ]]; then
    echo "[2/4] Merging LoRA"

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
    echo "[2/4] Using pretrained DFlash"
    SERVE_DRAFT="${DRAFT_MODEL}"
fi

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

echo "[3/4] Starting vLLM speculative server"

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

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
"${VLLM_PYTHON}" "${VLLM_ARGS[@]}" \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

READY=0

for _ in $(seq 1 180); do
    if curl -sf "${SERVER_URL}/health" >/dev/null 2>&1; then
        READY=1
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

echo "[4/4] Evaluating CNN/DailyMail test"

EVAL_ARGS=(
    evaluate
    --endpoint "${SERVER_URL}/v1"
    --data "${EVAL_DATA}"
    --split train
    --prompt-column prompt
    --start-index 0
    --num-samples "${NUM_SAMPLES}"
    --output-dir "${RUN_DIR}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --concurrency "${CONCURRENCY}"
    --request-timeout "${REQUEST_TIMEOUT}"
    --seed "${SEED}"
    --enable-thinking "${ENABLE_THINKING}"
)

[[ -n "${SYSTEM_PROMPT}" ]] && \
    EVAL_ARGS+=(--system-prompt "${SYSTEM_PROMPT}")

"${SPEC_PYTHON}" "${EVAL_PY}" "${EVAL_ARGS[@]}" \
    2>&1 | tee "${EVAL_LOG}"

echo "Results: ${RUN_DIR}"