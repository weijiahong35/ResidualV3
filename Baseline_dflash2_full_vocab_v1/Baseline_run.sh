#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PYTHON="/data/weijiahong/miniconda3/envs/dflash/bin/python"
GPU="${GPU:-2}"
EPOCHS="${EPOCHS:-16}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  echo 'Unset CUDA_VISIBLE_DEVICES; GPU is a physical index.' >&2
  exit 2
fi
cd "$ROOT"
"$PYTHON" -m py_compile "$HERE"/Baseline_*.py
"$PYTHON" "$HERE/Baseline_contract_test.py"
"$PYTHON" "$HERE/Baseline_prepare_cache.py" --gpu "$GPU"
"$PYTHON" "$HERE/Baseline_train.py" --gpu "$GPU" --epochs "$EPOCHS"
"$PYTHON" "$HERE/Baseline_test.py" --gpu "$GPU" --epochs "$EPOCHS"
"$PYTHON" "$HERE/Baseline_finish.py"
