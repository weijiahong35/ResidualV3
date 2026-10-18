#!/usr/bin/env bash
# Stage 3 V20.6: joint base/corrected Predictor calibration with residual MoLE.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/Qwen3-8B-speculator.dflash}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/openmath/onpolicy}"
RESIDUAL_ROOT="${RESIDUAL_ROOT:-${OUTPUT_DIR}/residual}"
DATA_DIR="${DATA_DIR:-${RESIDUAL_ROOT}/data}"
STAGE_ROOT="${STAGE_DIR:-${RESIDUAL_ROOT}/stage3_joint}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${RESIDUAL_ROOT}/stage2_mole/latest/checkpoints/final}"
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
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-4}"
PREDICTOR_LR_SCALE="${PREDICTOR_LR_SCALE:-0.1}"
ACCEPT_LOSS="${ACCEPT_LOSS:-nla}"
LAMBDA_PRED="${LAMBDA_PRED:-0.15}"
LAMBDA_PRED_ONLINE="${LAMBDA_PRED_ONLINE:-0.10}"
LAMBDA_REG="${LAMBDA_REG:-0.0}"
LAMBDA_SOFT_ALIGNMENT="${LAMBDA_SOFT_ALIGNMENT:-1.0}"
MISMATCH_HIDDEN_DIM="${MISMATCH_HIDDEN_DIM:-256}"
PREDICTOR_DEPTH="${PREDICTOR_DEPTH:-4}"
PREDICTOR_ATTN_HEADS="${PREDICTOR_ATTN_HEADS:-4}"
PREDICTOR_FFN_EXPANSION="${PREDICTOR_FFN_EXPANSION:-4}"
MISMATCH_DIRECTION_DIM="${MISMATCH_DIRECTION_DIM:-256}"
REPAIR_POSITION_DIM="${REPAIR_POSITION_DIM:-256}"
DISTRIBUTION_TEMPERATURE="${DISTRIBUTION_TEMPERATURE:-1.0}"
MISMATCH_LAMBDA_SEVERITY="${MISMATCH_LAMBDA_SEVERITY:-1.0}"
MISMATCH_LAMBDA_REPAIR="${MISMATCH_LAMBDA_REPAIR:-1.0}"
MOLE_NUM_EXPERTS="${MOLE_NUM_EXPERTS:-8}"
MOLE_TOP_K="${MOLE_TOP_K:-2}"
MOLE_RANK="${MOLE_RANK:-8}"
MOLE_ALPHA="${MOLE_ALPHA:-16}"

MOLE_MAX_RELATIVE_UPDATE="${MOLE_MAX_RELATIVE_UPDATE:-0.20}"
ROUTER_GATE_INIT_BIAS="${ROUTER_GATE_INIT_BIAS:--1.5}"
ROUTER_BALANCE_MODE="${ROUTER_BALANCE_MODE:-none}"
ROUTER_BIAS_UPDATE_RATE="${ROUTER_BIAS_UPDATE_RATE:-0.0}"
ROUTER_LOAD_EMA="${ROUTER_LOAD_EMA:-0.95}"
ROUTER_BIAS_CLIP="${ROUTER_BIAS_CLIP:-2.0}"
LAMBDA_ROUTER_Z="${LAMBDA_ROUTER_Z:-1e-4}"
LAMBDA_TAIL_NLA="${LAMBDA_TAIL_NLA:-0.02}"
TAIL_NLA_FRACTION="${TAIL_NLA_FRACTION:-0.10}"

VLLM_PORT="${VLLM_PORT:-18083}"
VLLM_GPU="${VLLM_GPU:-7}"
TRAIN_GPU="${TRAIN_GPU:-6}"
HS_VLLM_MAX_MODEL_LEN="${HS_VLLM_MAX_MODEL_LEN:-9216}"
HS_VLLM_MAX_BATCHED_TOKENS="${HS_VLLM_MAX_BATCHED_TOKENS:-55296}"
HS_VLLM_MAX_SEQS="${HS_VLLM_MAX_SEQS:-6}"
HS_VLLM_GPU_MEM="${HS_VLLM_GPU_MEM:-0.60}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-llm}"
SPEC_CONDA_ENV="${SPEC_CONDA_ENV:-dflash}"
CONDA_BASE="$(conda info --base)"
VLLM_PYTHON="${VLLM_PYTHON:-${CONDA_BASE}/envs/${VLLM_CONDA_ENV}/bin/python}"
SPEC_PYTHON="${SPEC_PYTHON:-${CONDA_BASE}/envs/${SPEC_CONDA_ENV}/bin/python}"
VLLM_LOG="${RUN_DIR}/vllm.log"
TRAIN_LOG="${RUN_DIR}/train.log"
VLLM_PID=""

cat > "${RUN_DIR}/run_info.txt" <<EOF
stage=stage3_joint
run_id=${RUN_ID}
predictor_variant=v20_6_dual_v10_softlogit_train
stage2_checkpoint=${STAGE2_CHECKPOINT}
model=${MODEL}
draft_model=${DRAFT_MODEL}
data_dir=${DATA_DIR}
predictor_depth=${PREDICTOR_DEPTH}
predictor_attention_heads=${PREDICTOR_ATTN_HEADS}
predictor_ffn_expansion=${PREDICTOR_FFN_EXPANSION}
lambda_pred_base=${LAMBDA_PRED}
lambda_pred_online=${LAMBDA_PRED_ONLINE}
router_balance_mode=${ROUTER_BALANCE_MODE}
router_gate_init_bias=${ROUTER_GATE_INIT_BIAS}
mole_max_relative_update=${MOLE_MAX_RELATIVE_UPDATE}
lambda_tail_nla=${LAMBDA_TAIL_NLA}
tail_nla_fraction=${TAIL_NLA_FRACTION}
started_at=$(date -Is)
EOF

cleanup_vllm(){ local code=$?; if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true; fi; return "${code}"; }
trap cleanup_vllm EXIT INT TERM
wait_for_vllm(){ local f="$1"; for i in $(seq 1 180); do if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then return 0; fi; if ! kill -0 "${VLLM_PID}" 2>/dev/null; then echo "ERROR: Target vLLM exited before ready"; tail -n 120 "$f" || true; exit 1; fi; if (( i % 15 == 0 )); then echo "Waiting for Target vLLM..."; fi; sleep 2; done; echo "ERROR: Target vLLM did not become ready"; exit 1; }

echo "Stage 3: Joint Hybrid Residual | run=${RUN_ID}"
[[ -x "${VLLM_PYTHON}" ]] || { echo "ERROR: Python not found: ${VLLM_PYTHON}"; exit 1; }
[[ -x "${SPEC_PYTHON}" ]] || { echo "ERROR: Python not found: ${SPEC_PYTHON}"; exit 1; }
[[ -d "${DATA_DIR}" ]] || { echo "ERROR: prepared data missing: ${DATA_DIR}"; exit 1; }
[[ -f "${STAGE2_CHECKPOINT}/residual_mole.safetensors" ]] || { echo "ERROR: Stage 2 checkpoint missing: ${STAGE2_CHECKPOINT}"; exit 1; }
if [[ -d "${DATA_DIR}/hidden_states" ]]; then rm -rf "${DATA_DIR}/hidden_states"; fi

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${VLLM_PYTHON}" scripts/launch_vllm.py "${MODEL}" --target-layer-ids ${TARGET_LAYER_IDS} -- \
  --port "${VLLM_PORT}" --tensor-parallel-size 1 --data-parallel-size 1 --dtype bfloat16 \
  --max-num-seqs "${HS_VLLM_MAX_SEQS}" --max-model-len "${HS_VLLM_MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${HS_VLLM_MAX_BATCHED_TOKENS}" --gpu-memory-utilization "${HS_VLLM_GPU_MEM}" \
  --disable-uvicorn-access-log > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
wait_for_vllm "${VLLM_LOG}"

echo "Target vLLM ready. Starting joint training..."
CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${SPEC_PYTHON}" residual/train_residual.py \
  --verifier-name-or-path "${MODEL}" --from-pretrained "${DRAFT_MODEL}" --speculator-type dflash \
  --data-path "${DATA_DIR}" --vllm-endpoint "http://127.0.0.1:${VLLM_PORT}/v1" \
  --on-missing generate --on-generate delete --save-path "${RUN_DIR}/base_unused" \
  --total-seq-len "${SEQ_LENGTH}" --target-layer-ids ${TARGET_LAYER_IDS} --max-anchors "${MAX_ANCHORS}" \
  --num-workers "${NUM_WORKERS}" --prefetch-factor "${PREFETCH_FACTOR}" --seed "${SEED}" \
  --log-freq "${LOG_FREQ}" --weight-decay "${WEIGHT_DECAY}" --no-resume-from-checkpoint \
  --train-stage joint --stage2-checkpoint "${STAGE2_CHECKPOINT}" \
  --epochs "${EPOCHS}" --lr "${LR}" --predictor-lr-scale "${PREDICTOR_LR_SCALE}" \
  --accept-loss "${ACCEPT_LOSS}" --lambda-pred "${LAMBDA_PRED}" --lambda-pred-online "${LAMBDA_PRED_ONLINE}" --lambda-reg "${LAMBDA_REG}" \
  --lambda-router-z "${LAMBDA_ROUTER_Z}" --lambda-tail-nla "${LAMBDA_TAIL_NLA}" --tail-nla-fraction "${TAIL_NLA_FRACTION}" \
  --mismatch-hidden-dim "${MISMATCH_HIDDEN_DIM}" --mismatch-direction-dim "${MISMATCH_DIRECTION_DIM}" --repair-position-dim "${REPAIR_POSITION_DIM}" \
  --predictor-depth "${PREDICTOR_DEPTH}" --predictor-attention-heads "${PREDICTOR_ATTN_HEADS}" --predictor-ffn-expansion "${PREDICTOR_FFN_EXPANSION}" \
  --mismatch-lambda-severity "${MISMATCH_LAMBDA_SEVERITY}" --mismatch-lambda-repair "${MISMATCH_LAMBDA_REPAIR}" \
  --mole-num-experts "${MOLE_NUM_EXPERTS}" --mole-top-k "${MOLE_TOP_K}" --mole-rank "${MOLE_RANK}" --mole-alpha "${MOLE_ALPHA}" \
  --mole-max-relative-update "${MOLE_MAX_RELATIVE_UPDATE}" \
  --router-gate-init-bias "${ROUTER_GATE_INIT_BIAS}" --router-balance-mode "${ROUTER_BALANCE_MODE}" \
  --router-bias-update-rate "${ROUTER_BIAS_UPDATE_RATE}" --router-load-ema "${ROUTER_LOAD_EMA}" --router-bias-clip "${ROUTER_BIAS_CLIP}" \
  --distribution-temperature "${DISTRIBUTION_TEMPERATURE}" \
  --lambda-soft-alignment "${LAMBDA_SOFT_ALIGNMENT}" \
  --teacher-trajectory reference_off_policy \
  --residual-save-path "${RUN_DIR}/checkpoints" \
  2>&1 | tee "${TRAIN_LOG}"

[[ -f "${RUN_DIR}/checkpoints/final/residual_mole.safetensors" ]] || { echo "ERROR: final checkpoint missing"; exit 1; }
ln -sfn "runs/${RUN_ID}" "${STAGE_ROOT}/latest"
echo "Stage 3 finished: ${RUN_DIR}/checkpoints/final"
