#!/usr/bin/env bash
# Prepare the already-regenerated OpenMath pool for Residual MoLE training.
# This script does NOT regenerate model responses and does NOT start training.
set -euo pipefail

# ==============================================================================
# 1. Project / environment parameters (owned by this script)
# ==============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

SPEC_CONDA_ENV="${SPEC_CONDA_ENV:-dflash}"
CONDA_BASE="$(conda info --base)"
SPEC_PYTHON="${SPEC_PYTHON:-${CONDA_BASE}/envs/${SPEC_CONDA_ENV}/bin/python}"
[[ -x "${SPEC_PYTHON}" ]] || { echo "ERROR: Python not found: ${SPEC_PYTHON}"; exit 1; }

# ==============================================================================
# 2. OpenMath data parameters
# ==============================================================================
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/openmath/onpolicy}"
RESIDUAL_ROOT="${RESIDUAL_ROOT:-${OUTPUT_DIR}/residual}"
REGEN_FILE="${REGEN_FILE:-${OUTPUT_DIR}/regenerated/openmath_qwen3_8b_onpolicy.jsonl}"
RUN_FILE="${RUN_FILE:-${RESIDUAL_ROOT}/openmath_qwen3_8b_onpolicy.residual.current.jsonl}"
DATA_DIR="${DATA_DIR:-${RESIDUAL_ROOT}/data}"

# Number of distinct regenerated conversations selected as this run's current set.
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
SEED="${SEED:-42}"
NUM_PREPROCESSING_WORKERS="${NUM_PREPROCESSING_WORKERS:-8}"

mkdir -p "${DATA_DIR}" "${RESIDUAL_ROOT}"

[[ -f "${REGEN_FILE}" ]] || {
  echo "ERROR: regenerated OpenMath file not found: ${REGEN_FILE}"
  echo "This script only prepares already-regenerated data; it does not regenerate responses."
  exit 1
}

# Select all rows belonging to the first MAX_SAMPLES distinct conversations.
"${SPEC_PYTHON}" - "${REGEN_FILE}" "${RUN_FILE}" "${MAX_SAMPLES}" <<'PY'
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

        key = obj.get("primary_id") or obj.get("id")
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
        f"Need {limit} regenerated conversations, found {len(selected)}. "
        "Set MAX_SAMPLES to the number you actually want to use."
    )

print(f"Selected {len(selected)} conversations -> {dst}")
PY

# Reuse the original project's DFlash data preprocessing pipeline.
"${SPEC_PYTHON}" scripts/prepare_data.py \
  --model "${MODEL}" \
  --data "${RUN_FILE}" \
  --output "${DATA_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --seq-length "${SEQ_LENGTH}" \
  --seed "${SEED}" \
  --num-preprocessing-workers "${NUM_PREPROCESSING_WORKERS}" \
  --overwrite

echo "Prepared residual OpenMath data: ${DATA_DIR}"
echo "Current JSONL: ${RUN_FILE}"
