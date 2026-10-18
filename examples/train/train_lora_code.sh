#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B + DFlash-LoRA training.
# This script NEVER regenerates responses.
# It exits with an error if the persistent response cache has fewer
# than MAX_SAMPLES unique primary_id/id conversations.

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/Qwen3-8B-speculator.dflash}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"

OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_qwen3_8b_opencode_lora_onpolicy}"
REGEN_DIR="${OUTPUT_DIR}/regenerated"
REGEN_FILE="${REGEN_DIR}/opencode_qwen3_8b_onpolicy.jsonl"
REGEN_RUN_FILE="${REGEN_DIR}/opencode_qwen3_8b_onpolicy.current.jsonl"
DATA_DIR="${OUTPUT_DIR}/data"
SAVE_DIR="${OUTPUT_DIR}/checkpoints"

VLLM_PORT="${VLLM_PORT:-18081}"
VLLM_GPU="${VLLM_GPU:-7}"
TRAIN_GPU="${TRAIN_GPU:-7}"


SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MAX_ANCHORS="${MAX_ANCHORS:-1024}"

HS_VLLM_MAX_MODEL_LEN="${HS_VLLM_MAX_MODEL_LEN:-9216}"
HS_VLLM_MAX_BATCHED_TOKENS="${HS_VLLM_MAX_BATCHED_TOKENS:-55296}"
HS_VLLM_MAX_SEQS="${HS_VLLM_MAX_SEQS:-6}"
HS_VLLM_GPU_MEM="${HS_VLLM_GPU_MEM:-0.60}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
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

[[ -x "${VLLM_PYTHON}" ]] || { echo "ERROR: ${VLLM_PYTHON} not found"; exit 1; }
[[ -x "${SPEC_PYTHON}" ]] || { echo "ERROR: ${SPEC_PYTHON} not found"; exit 1; }

mkdir -p "${REGEN_DIR}" "${OUTPUT_DIR}/logs"
LOG_FILE="${OUTPUT_DIR}/logs/train_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG_FILE}") 2>&1
export PYTHONUNBUFFERED=1

VLLM_PID=""

cleanup() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping hidden-state vLLM..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
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
            tail -n 100 "${OUTPUT_DIR}/vllm_hidden_states.log" || true
            exit 1
        fi
        sleep 2
    done

    echo "ERROR: vLLM did not become ready"
    tail -n 100 "${OUTPUT_DIR}/vllm_hidden_states.log" || true
    exit 1
}

TOTAL="$(count_regenerated "${REGEN_FILE}")"
echo "Cached conversations: ${TOTAL}"
echo "Required conversations: ${MAX_SAMPLES}"

if (( TOTAL < MAX_SAMPLES )); then
    echo "ERROR: regenerated data is insufficient."
    echo "Required: ${MAX_SAMPLES}"
    echo "Available: ${TOTAL}"
    echo "Missing: $((MAX_SAMPLES - TOTAL))"
    echo "Run: MAX_SAMPLES=${MAX_SAMPLES} ./regenerate_data.sh"
    exit 1
fi

echo "[1/3] Building ${MAX_SAMPLES}-conversation training view"
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

echo "[2/3] Preparing data"
"${SPEC_PYTHON}" scripts/prepare_data.py \
    --model "${MODEL}" \
    --data "${REGEN_RUN_FILE}" \
    --output "${DATA_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --seq-length "${SEQ_LENGTH}" \
    --seed "${SEED}" \
    --num-preprocessing-workers 8 \
    --overwrite

echo "[3/3] Starting hidden-state verifier and training"

echo "[$(date '+%F %T')] Starting hidden-state vLLM on GPU ${VLLM_GPU}..."
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
    > "${OUTPUT_DIR}/vllm_hidden_states.log" 2>&1 &

VLLM_PID=$!

echo "[$(date '+%F %T')] Waiting for hidden-state vLLM to become ready..."
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
    --lr "${LR}" \
    --scheduler-type cosine \
    --total-seq-len "${SEQ_LENGTH}" \
    --target-layer-ids ${TARGET_LAYER_IDS} \
    --max-anchors "${MAX_ANCHORS}" \
    --num-workers "${NUM_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --seed "${SEED}" \
    --log-freq 10 \
    --optimizer adamw \
    --weight-decay 0.01 \
    --loss-fn kl_div \
    --enable-lora \
    --lora-adapter-name code \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --lora-save-freq 1 \
    --scheduler-warmup-ratio 0.03 \
    --no-resume-from-checkpoint
    

echo "Training finished. Adapter: ${SAVE_DIR}/lora/final/"
