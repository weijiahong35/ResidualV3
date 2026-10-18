#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/Qwen3-8B-speculator.dflash}"
DATASET="${DATASET:-opencodeinstruct}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"

OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_qwen3_8b_opencode_lora_online_5k}"
DATA_DIR="${OUTPUT_DIR}/data"

VLLM_PORT="${VLLM_PORT:-18082}"
VLLM_GPU="${VLLM_GPU:-5}"
TRAIN_GPU="${TRAIN_GPU:-5}"

SEQ_LENGTH="${SEQ_LENGTH:-4096}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
MAX_ANCHORS="${MAX_ANCHORS:-512}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 10 18 26 34}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
SEED="${SEED:-42}"

OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-cosine}"
SCHEDULER_WARMUP_RATIO="${SCHEDULER_WARMUP_RATIO:-0.05}"

LORA_ADAPTER_NAME="${LORA_ADAPTER_NAME:-code}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_SAVE_FREQ="${LORA_SAVE_FREQ:-1}"

NUM_PREPROCESSING_WORKERS="${NUM_PREPROCESSING_WORKERS:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
LOG_FREQ="${LOG_FREQ:-10}"

VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-llm}"
SPEC_CONDA_ENV="${SPEC_CONDA_ENV:-dflash}"

VLLM_PYTHON="$(
    conda run -n "${VLLM_CONDA_ENV}" python -c 'import sys; print(sys.executable)'
)"
SPEC_PYTHON="$(
    conda run -n "${SPEC_CONDA_ENV}" python -c 'import sys; print(sys.executable)'
)"

[[ -x "${VLLM_PYTHON}" ]] || {
    echo "ERROR: ${VLLM_PYTHON} not found"
    exit 1
}

[[ -x "${SPEC_PYTHON}" ]] || {
    echo "ERROR: ${SPEC_PYTHON} not found"
    exit 1
}

# 每次训练独立目录
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ID="${RUN_ID:-${RUN_TIMESTAMP}_r${LORA_R}_a${LORA_ALPHA}_lr${LR}}"

RUN_DIR="${OUTPUT_DIR}/runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
SAVE_DIR="${RUN_DIR}/checkpoints"

TRAIN_LOG="${LOG_DIR}/train.log"
VLLM_LOG="${LOG_DIR}/vllm.log"
PARAMS_FILE="${RUN_DIR}/params.env"
COMMAND_FILE="${RUN_DIR}/launch_command.txt"

mkdir -p "${DATA_DIR}" "${LOG_DIR}" "${SAVE_DIR}"

exec > >(tee -a "${TRAIN_LOG}") 2>&1
export PYTHONUNBUFFERED=1

echo "Run: ${RUN_ID}"
echo "Dataset: ${DATASET}, samples=${MAX_SAMPLES}"
echo "LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "LR=${LR}, epochs=${EPOCHS}"
echo "GPU: vLLM=${VLLM_GPU}, train=${TRAIN_GPU}"
echo "Dir: ${RUN_DIR}"

cat > "${PARAMS_FILE}" <<EOF
RUN_ID=${RUN_ID}
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

MODEL=${MODEL}
DRAFT_MODEL=${DRAFT_MODEL}

DATASET=${DATASET}
MAX_SAMPLES=${MAX_SAMPLES}
DATA_DIR=${DATA_DIR}

SEQ_LENGTH=${SEQ_LENGTH}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}
MAX_ANCHORS=${MAX_ANCHORS}
TARGET_LAYER_IDS=${TARGET_LAYER_IDS}

EPOCHS=${EPOCHS}
LR=${LR}
SEED=${SEED}

OPTIMIZER=${OPTIMIZER}
WEIGHT_DECAY=${WEIGHT_DECAY}
SCHEDULER_TYPE=${SCHEDULER_TYPE}
SCHEDULER_WARMUP_RATIO=${SCHEDULER_WARMUP_RATIO}

LORA_ADAPTER_NAME=${LORA_ADAPTER_NAME}
LORA_R=${LORA_R}
LORA_ALPHA=${LORA_ALPHA}
LORA_DROPOUT=${LORA_DROPOUT}
LORA_SAVE_FREQ=${LORA_SAVE_FREQ}

NUM_PREPROCESSING_WORKERS=${NUM_PREPROCESSING_WORKERS}
NUM_WORKERS=${NUM_WORKERS}
PREFETCH_FACTOR=${PREFETCH_FACTOR}

VLLM_GPU=${VLLM_GPU}
TRAIN_GPU=${TRAIN_GPU}
VLLM_PORT=${VLLM_PORT}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}
VLLM_DTYPE=${VLLM_DTYPE}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER}

RUN_DIR=${RUN_DIR}
SAVE_DIR=${SAVE_DIR}
TRAIN_LOG=${TRAIN_LOG}
VLLM_LOG=${VLLM_LOG}
EOF

VLLM_PID=""

cleanup() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping vLLM..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 1. Prepare data
echo "[1/3] Preparing data..."

"${SPEC_PYTHON}" scripts/prepare_data.py \
    --model "${MODEL}" \
    --data "${DATASET}" \
    --output "${DATA_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --seq-length "${SEQ_LENGTH}" \
    --seed "${SEED}" \
    --num-preprocessing-workers "${NUM_PREPROCESSING_WORKERS}"

# 2. Start vLLM
echo "[2/3] Starting vLLM..."

VLLM_ARGS=(
    scripts/launch_vllm.py "${MODEL}"
    --target-layer-ids ${TARGET_LAYER_IDS}
    --
    --port "${VLLM_PORT}"
    --tensor-parallel-size 1
    --data-parallel-size 1
    --dtype "${VLLM_DTYPE}"
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --disable-uvicorn-access-log
)

if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
    VLLM_ARGS+=(--enforce-eager)
fi

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
"${VLLM_PYTHON}" "${VLLM_ARGS[@]}" \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

READY=0

for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        READY=1
        break
    fi

    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM exited before becoming ready"
        tail -n 100 "${VLLM_LOG}" || true
        exit 1
    fi

    sleep 2
done

if [[ "${READY}" -ne 1 ]]; then
    echo "ERROR: vLLM did not become ready"
    tail -n 100 "${VLLM_LOG}" || true
    exit 1
fi

echo "vLLM ready."

# 3. Train
echo "[3/3] Starting training..."

TRAIN_CMD=(
    "${SPEC_PYTHON}" scripts/trainlora.py

    --verifier-name-or-path "${MODEL}"
    --from-pretrained "${DRAFT_MODEL}"
    --speculator-type dflash

    --data-path "${DATA_DIR}"
    --vllm-endpoint "http://127.0.0.1:${VLLM_PORT}/v1"
    --on-missing generate
    --on-generate delete

    --save-path "${SAVE_DIR}"

    --epochs "${EPOCHS}"
    --lr "${LR}"
    --optimizer "${OPTIMIZER}"
    --weight-decay "${WEIGHT_DECAY}"
    --scheduler-type "${SCHEDULER_TYPE}"
    --scheduler-warmup-ratio "${SCHEDULER_WARMUP_RATIO}"

    --total-seq-len "${SEQ_LENGTH}"
    --target-layer-ids ${TARGET_LAYER_IDS}
    --max-anchors "${MAX_ANCHORS}"

    --num-workers "${NUM_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --seed "${SEED}"
    --log-freq "${LOG_FREQ}"

    --enable-lora
    --lora-adapter-name "${LORA_ADAPTER_NAME}"
    --lora-r "${LORA_R}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}"
    --lora-save-freq "${LORA_SAVE_FREQ}"

    --no-resume-from-checkpoint
)

{
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${TRAIN_GPU}"
    printf '%q ' "${TRAIN_CMD[@]}"
    printf '\n'
} > "${COMMAND_FILE}"

CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${TRAIN_CMD[@]}"

cat >> "${PARAMS_FILE}" <<EOF
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
STATUS=finished
EOF

echo
echo "Training finished."
echo "Run:   ${RUN_DIR}"
echo "LoRA:  ${SAVE_DIR}/lora/final/"
echo "Train: ${TRAIN_LOG}"
echo "vLLM:  ${VLLM_LOG}"