#!/usr/bin/env bash
# Stage 1 V20.6: Dual-V10 Predictor training (independent Bias/NLA + Repair full V10 branches).
# Prefixes: D1,D2 -> L3; D1..D3 -> L4; D1..D4 -> L5. lm_head remains unchanged.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/Qwen3-8B-speculator.dflash}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/openmath/onpolicy}"
RESIDUAL_ROOT="${RESIDUAL_ROOT:-${OUTPUT_DIR}/residual}"
DATA_DIR="${DATA_DIR:-${RESIDUAL_ROOT}/data}"
STAGE_ROOT="${STAGE_DIR:-${RESIDUAL_ROOT}/stage1_predictor}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')_$$}"
RUN_DIR="${STAGE_ROOT}/runs/${RUN_ID}"
[[ ! -e "${RUN_DIR}" ]] || { echo "ERROR: run already exists: ${RUN_DIR}"; exit 1; }
mkdir -p "${RUN_DIR}"

SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MAX_ANCHORS="${MAX_ANCHORS:-1536}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 10 18 26 34}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SEED="${SEED:-42}"
LOG_FREQ="${LOG_FREQ:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"
MISMATCH_NUM_HIDDEN_LAYERS="${MISMATCH_NUM_HIDDEN_LAYERS:-5}"
MISMATCH_HIDDEN_DIM="${MISMATCH_HIDDEN_DIM:-256}"
PREDICTOR_DEPTH="${PREDICTOR_DEPTH:-4}"
PREDICTOR_ATTN_HEADS="${PREDICTOR_ATTN_HEADS:-4}"
PREDICTOR_FFN_EXPANSION="${PREDICTOR_FFN_EXPANSION:-4}"
MISMATCH_DIRECTION_DIM="${MISMATCH_DIRECTION_DIM:-256}"
REPAIR_POSITION_DIM="${REPAIR_POSITION_DIM:-256}"
DISTRIBUTION_TEMPERATURE="${DISTRIBUTION_TEMPERATURE:-1.0}"
MISMATCH_LAMBDA_SEVERITY="${MISMATCH_LAMBDA_SEVERITY:-1.0}"
MISMATCH_LAMBDA_REPAIR="${MISMATCH_LAMBDA_REPAIR:-1.0}"

VLLM_PORT="${VLLM_PORT:-18083}"
VLLM_GPU="${VLLM_GPU:-6}"
TRAIN_GPU="${TRAIN_GPU:-7}"
HS_VLLM_MAX_MODEL_LEN="${HS_VLLM_MAX_MODEL_LEN:-9216}"
HS_VLLM_MAX_BATCHED_TOKENS="${HS_VLLM_MAX_BATCHED_TOKENS:-55296}"
HS_VLLM_MAX_SEQS="${HS_VLLM_MAX_SEQS:-6}"
HS_VLLM_GPU_MEM="${HS_VLLM_GPU_MEM:-0.50}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-llm}"
SPEC_CONDA_ENV="${SPEC_CONDA_ENV:-dflash}"
CONDA_BASE="$(conda info --base)"
VLLM_PYTHON="${VLLM_PYTHON:-${CONDA_BASE}/envs/${VLLM_CONDA_ENV}/bin/python}"
SPEC_PYTHON="${SPEC_PYTHON:-${CONDA_BASE}/envs/${SPEC_CONDA_ENV}/bin/python}"
VLLM_LOG="${RUN_DIR}/vllm.log"
TRAIN_LOG="${RUN_DIR}/train.log"
VLLM_PID=""

cat > "${RUN_DIR}/run_info.txt" <<EOF
stage=stage1_predictor
run_id=${RUN_ID}
predictor_variant=v20_6_dual_v10_softlogit_train
model=${MODEL}
draft_model=${DRAFT_MODEL}
data_dir=${DATA_DIR}
contract=V10 prev-hT/prev-hD/current-prefix information; p2/p3/p4 each have independent full Bias/NLA and Repair V10 fusion branches; training uses raw-logit soft targets only; greedy is evaluation-only; Base D1+D2 and D3..D5 fused
predictor_depth=${PREDICTOR_DEPTH}
predictor_attention_heads=${PREDICTOR_ATTN_HEADS}
predictor_ffn_expansion=${PREDICTOR_FFN_EXPANSION}
started_at=$(date -Is)
EOF

cleanup_vllm(){ local code=$?; if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true; fi; return "${code}"; }
trap cleanup_vllm EXIT INT TERM
wait_for_vllm(){ local f="$1"; for i in $(seq 1 180); do if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then return 0; fi; if ! kill -0 "${VLLM_PID}" 2>/dev/null; then echo "ERROR: Target vLLM exited before ready"; tail -n 120 "$f" || true; exit 1; fi; if (( i % 15 == 0 )); then echo "Waiting for Target vLLM..."; fi; sleep 2; done; echo "ERROR: Target vLLM did not become ready"; exit 1; }

echo "Stage 1: Hybrid Predictor | run=${RUN_ID}"
[[ -x "${VLLM_PYTHON}" ]] || { echo "ERROR: Python not found: ${VLLM_PYTHON}"; exit 1; }
[[ -x "${SPEC_PYTHON}" ]] || { echo "ERROR: Python not found: ${SPEC_PYTHON}"; exit 1; }
[[ -d "${DATA_DIR}" ]] || { echo "ERROR: prepared data missing: ${DATA_DIR}"; exit 1; }
if [[ -d "${DATA_DIR}/hidden_states" ]]; then rm -rf "${DATA_DIR}/hidden_states"; fi

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${VLLM_PYTHON}" scripts/launch_vllm.py "${MODEL}" --target-layer-ids ${TARGET_LAYER_IDS} -- \
  --port "${VLLM_PORT}" --tensor-parallel-size 1 --data-parallel-size 1 --dtype bfloat16 \
  --max-num-seqs "${HS_VLLM_MAX_SEQS}" --max-model-len "${HS_VLLM_MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${HS_VLLM_MAX_BATCHED_TOKENS}" --gpu-memory-utilization "${HS_VLLM_GPU_MEM}" \
  --disable-uvicorn-access-log > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
wait_for_vllm "${VLLM_LOG}"

echo "Target vLLM ready. Starting Predictor training..."
CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${SPEC_PYTHON}" residual/train_residual.py \
  --verifier-name-or-path "${MODEL}" --from-pretrained "${DRAFT_MODEL}" --speculator-type dflash \
  --data-path "${DATA_DIR}" --vllm-endpoint "http://127.0.0.1:${VLLM_PORT}/v1" \
  --on-missing generate --on-generate delete --save-path "${RUN_DIR}/base_unused" \
  --total-seq-len "${SEQ_LENGTH}" --target-layer-ids ${TARGET_LAYER_IDS} --max-anchors "${MAX_ANCHORS}" \
  --num-workers "${NUM_WORKERS}" --prefetch-factor "${PREFETCH_FACTOR}" --seed "${SEED}" \
  --log-freq "${LOG_FREQ}" --weight-decay "${WEIGHT_DECAY}" --no-resume-from-checkpoint \
  --train-stage predictor --epochs "${EPOCHS}" --lr "${LR}" \
  --mismatch-num-hidden-layers "${MISMATCH_NUM_HIDDEN_LAYERS}" \
  --mismatch-hidden-dim "${MISMATCH_HIDDEN_DIM}" --mismatch-direction-dim "${MISMATCH_DIRECTION_DIM}" --repair-position-dim "${REPAIR_POSITION_DIM}" \
  --predictor-depth "${PREDICTOR_DEPTH}" --predictor-attention-heads "${PREDICTOR_ATTN_HEADS}" --predictor-ffn-expansion "${PREDICTOR_FFN_EXPANSION}" \
  --mismatch-lambda-severity "${MISMATCH_LAMBDA_SEVERITY}" --mismatch-lambda-repair "${MISMATCH_LAMBDA_REPAIR}" \
  --distribution-temperature "${DISTRIBUTION_TEMPERATURE}" \
  --teacher-trajectory reference_off_policy \
  --residual-save-path "${RUN_DIR}/checkpoints" \
  2>&1 | tee "${TRAIN_LOG}"

[[ -f "${RUN_DIR}/checkpoints/final/residual_mole.safetensors" ]] || { echo "ERROR: final checkpoint missing"; exit 1; }
ln -sfn "runs/${RUN_ID}" "${STAGE_ROOT}/latest"
echo "Stage 1 finished: ${RUN_DIR}/checkpoints/final"
