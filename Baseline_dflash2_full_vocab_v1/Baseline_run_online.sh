#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
FAMILY="dflash2"
OUT="$ROOT/eval_results/Baseline_${FAMILY}_full_vocab_bf16_v2_seed42/${FAMILY}/online"
TARGET="/data/weijiahong/huggingface/models/Qwen3-8B"
SPEC_PY="/data/weijiahong/miniconda3/envs/dflash/bin/python"
VLLM_ENV="/data/weijiahong/miniconda3/envs/llm_dflash_eval_v026"
VLLM_PY="$VLLM_ENV/bin/python"
: "${SERVER_GPU:?Set SERVER_GPU to a free physical GPU}"
: "${TRAIN_GPU:?Set TRAIN_GPU to a different free physical GPU}"
[[ "$SERVER_GPU" != "$TRAIN_GPU" ]] || { echo 'SERVER_GPU and TRAIN_GPU must differ' >&2; exit 2; }
[[ -z "${CUDA_VISIBLE_DEVICES+x}" ]] || { echo 'Unset CUDA_VISIBLE_DEVICES' >&2; exit 2; }
PORT="${PORT:-18561}"
EPOCHS="${EPOCHS:-16}"
MAX_GROUPS="${MAX_GROUPS:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
SERVER_MEM_UTIL="${SERVER_MEM_UTIL:-0.4}"
HS_DIR="$OUT/hidden_states_tmp"
mkdir -p "$OUT" "$HS_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$ROOT"
"$SPEC_PY" -m py_compile "$HERE"/Baseline_*.py
bash -n "$HERE/Baseline_run_online.sh"
"$SPEC_PY" "$HERE/Baseline_online_contract_test.py"
if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Port ${PORT} already has a server" >&2
  exit 2
fi
if find "$HS_DIR" -type f -print -quit | grep -q .; then
  echo "Ephemeral hidden directory is not empty: $HS_DIR" >&2
  exit 2
fi
SERVER_PID=''
stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 -- "-$SERVER_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Server still responds on port ${PORT} after cleanup" >&2
    return 1
  fi
  SERVER_PID=''
}
cleanup() {
  local code=$?
  trap - EXIT
  stop_server
  exit "$code"
}
trap cleanup EXIT INT TERM
setsid env -u LD_PRELOAD \
  LD_LIBRARY_PATH="$VLLM_ENV/cuda-compat:$VLLM_ENV/lib" \
  VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  VLLM_CUDA_COMPATIBILITY_PATH="$VLLM_ENV/cuda-compat" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLLM_PY" scripts/launch_vllm.py "$TARGET" \
  --hidden-states-path "$HS_DIR" \
  --target-layer-ids 2 10 18 26 34 -- \
  --port "$PORT" --device-ids "$SERVER_GPU" \
  --tensor-parallel-size 1 --data-parallel-size 1 \
  --dtype bfloat16 --max-model-len 9216 \
  --max-num-batched-tokens 9216 --max-num-seqs 1 \
  --gpu-memory-utilization "$SERVER_MEM_UTIL" --disable-uvicorn-access-log \
  > "$OUT/vllm.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Target server failed; see $OUT/vllm.log" >&2
    exit 1
  fi
  sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null
"$SPEC_PY" "$HERE/Baseline_train_online.py" \
  --gpu "$TRAIN_GPU" --vllm-endpoint "http://127.0.0.1:${PORT}/v1" \
  --hidden-states-path "$HS_DIR" --on-missing generate --on-generate delete \
  --epochs "$EPOCHS" --max-groups "$MAX_GROUPS" --max-seq-len "$MAX_SEQ_LEN" > "$OUT/train.log" 2>&1
"$SPEC_PY" "$HERE/Baseline_test_trajectory.py" \
  --gpu "$TRAIN_GPU" --vllm-endpoint "http://127.0.0.1:${PORT}/v1" \
  --hidden-states-path "$HS_DIR" --on-missing generate --on-generate delete \
  --max-groups "$MAX_GROUPS" > "$OUT/test.log" 2>&1

# Free the owned hidden-state server before native decode starts.
stop_server
CKPT_DIR="$OUT"
if [[ "$MAX_GROUPS" != 0 ]]; then CKPT_DIR="$OUT/smoke_${MAX_GROUPS}_groups"; fi
CKPT="$CKPT_DIR/epoch_$(printf '%02d' "$EPOCHS").safetensors"
NATIVE_OUT="$CKPT_DIR/native_bf16_v2"
"$SPEC_PY" "$HERE/Baseline_test_online.py" --gpu "$TRAIN_GPU" --checkpoint "$CKPT" \
  --output-dir "$NATIVE_OUT" --smoke --allow-shared-gpu > "$CKPT_DIR/native_smoke.log" 2>&1
if [[ "$MAX_GROUPS" == 0 ]]; then
  "$SPEC_PY" "$HERE/Baseline_test_online.py" --gpu "$TRAIN_GPU" --checkpoint "$CKPT" \
    --output-dir "$NATIVE_OUT" --allow-shared-gpu > "$CKPT_DIR/native_runtime.log" 2>&1
fi
