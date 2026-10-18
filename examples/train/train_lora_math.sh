#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/Qwen3-8B-speculator.dflash}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"

# 2. 重生成数据训练脚本
OUTPUT_DIR="${OUTPUT_DIR:-./output/openmath/onpolicy}"
REGEN_DIR="${OUTPUT_DIR}/regenerated"
REGEN_FILE="${REGEN_DIR}/openmath_qwen3_8b_onpolicy.jsonl"
REGEN_RUN_FILE="${REGEN_DIR}/openmath_qwen3_8b_onpolicy.current.jsonl"

DATA_DIR="${OUTPUT_DIR}/data"


LOG_ROOT="${OUTPUT_DIR}/trainlogs"
RUN_ID="$(date '+%Y%m%d_%H%M%S')_$$"
RUN_DIR="${LOG_ROOT}/run_${RUN_ID}"
TRAIN_LOG="${RUN_DIR}/train.log"
VLLM_LOG="${RUN_DIR}/vllm.log"
SAVE_DIR="${RUN_DIR}/checkpoints"

VLLM_PORT="${VLLM_PORT:-18083}"
VLLM_GPU="${VLLM_GPU:-4}"
TRAIN_GPU="${TRAIN_GPU:-4}"

SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MAX_ANCHORS="${MAX_ANCHORS:-1536}"

HS_VLLM_MAX_MODEL_LEN="${HS_VLLM_MAX_MODEL_LEN:-9216}"
HS_VLLM_MAX_BATCHED_TOKENS="${HS_VLLM_MAX_BATCHED_TOKENS:-55296}"
HS_VLLM_MAX_SEQS="${HS_VLLM_MAX_SEQS:-6}"
HS_VLLM_GPU_MEM="${HS_VLLM_GPU_MEM:-0.60}"

EPOCHS="${EPOCHS:-2}"
LR="${LR:-6e-4}"
SEED="${SEED:-42}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 10 18 26 34}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.10}"

NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-llm}"
SPEC_CONDA_ENV="${SPEC_CONDA_ENV:-dflash}"

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

mkdir -p "${REGEN_DIR}" "${RUN_DIR}"

exec > >(tee -a "${TRAIN_LOG}") 2>&1
export PYTHONUNBUFFERED=1

VLLM_PID=""

cleanup() {
    EXIT_CODE=$?

    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping hidden-state vLLM..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

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

            key = obj.get("primary_id")
            if key in (None, ""):
                key = obj.get("id")

            if key not in (None, ""):
                seen.add(str(key))

print(len(seen))
PY
}

wait_for_vllm() {
    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
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

echo "Run: ${RUN_ID}"
echo "Train log: ${TRAIN_LOG}"
echo "vLLM log: ${VLLM_LOG}"

TOTAL="$(count_regenerated "${REGEN_FILE}")"

echo "Cached conversations: ${TOTAL}"
echo "Required conversations: ${MAX_SAMPLES}"

if (( TOTAL < MAX_SAMPLES )); then
    echo "ERROR: regenerated data is insufficient."
    echo "Required: ${MAX_SAMPLES}"
    echo "Available: ${TOTAL}"
    echo "Missing: $((MAX_SAMPLES - TOTAL))"
    exit 1
fi

echo "[1/3] Building ${MAX_SAMPLES}-conversation OpenMath training view"

"${SPEC_PYTHON}" - "${REGEN_FILE}" "${REGEN_RUN_FILE}" "${MAX_SAMPLES}" <<'PY'
import json
import sys

src, dst, limit_s = sys.argv[1:]
limit = int(limit_s)

selected = set()

with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        key = obj.get("primary_id")
        if key in (None, ""):
            key = obj.get("id")

        if key in (None, ""):
            continue

        key = str(key)

        if key not in selected:
            if len(selected) >= limit:
                continue
            selected.add(key)

        if key in selected:
            fout.write(line if line.endswith("\n") else line + "\n")

if len(selected) != limit:
    raise SystemExit(
        f"Expected {limit} conversations, selected only {len(selected)}."
    )

print(f"Selected {len(selected)} conversations.")
PY

echo "[2/3] Preparing OpenMath training data"

"${SPEC_PYTHON}" scripts/prepare_data.py \
    --model "${MODEL}" \
    --data "${REGEN_RUN_FILE}" \
    --output "${DATA_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --seq-length "${SEQ_LENGTH}" \
    --seed "${SEED}" \
    --num-preprocessing-workers 8 \
    --overwrite

echo "[3/3] Starting hidden-state verifier and DFlash-LoRA training"

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
"${VLLM_PYTHON}" scripts/launch_vllm.py "${MODEL}" \
    --target-layer-ids ${TARGET_LAYER_IDS} \
    -- \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --dtype bfloat16 \
    --max-num-seqs "${HS_VLLM_MAX_SEQS}" \
    --max-model-len "${HS_VLLM_MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${HS_VLLM_MAX_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${HS_VLLM_GPU_MEM}" \
    --disable-uvicorn-access-log \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

echo "[$(date '+%F %T')] Waiting for hidden-state vLLM..."
wait_for_vllm
echo "[$(date '+%F %T')] Hidden-state vLLM is READY."

echo "[$(date '+%F %T')] Starting DFlash-LoRA training on GPU ${TRAIN_GPU}..."

CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
"${SPEC_PYTHON}" scripts/trainlora.py \
    --verifier-name-or-path "${MODEL}" \
    --from-pretrained "${DRAFT_MODEL}" \
    --speculator-type dflash \
    --data-path "${DATA_DIR}" \
    --vllm-endpoint "http://127.0.0.1:${VLLM_PORT}/v1" \
    --on-missing generate \
    --on-generate delete \
    --save-path "${SAVE_DIR}" \
    --epochs "${EPOCHS}" \
    --scheduler-type cosine \
    --total-seq-len "${SEQ_LENGTH}" \
    --target-layer-ids ${TARGET_LAYER_IDS} \
    --max-anchors "${MAX_ANCHORS}" \
    --num-workers "${NUM_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --seed "${SEED}" \
    --log-freq 10 \
    --optimizer adamw \
    --lr "${LR}" \
    --weight-decay 0.01 \
    --loss-fn ce \
    --per-position-loss-weight dpace \
    --dpace-alpha 0.5 \
    --enable-lora \
    --lora-adapter-name math \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --lora-save-freq 1 \
    --scheduler-warmup-ratio 0.03 \
    --no-resume-from-checkpoint

echo "Training finished."
echo "Adapter: ${SAVE_DIR}/lora/final/"