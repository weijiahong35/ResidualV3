#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ENV="${VLLM_ENV_DIR:-/data/weijiahong/miniconda3/envs/llm_dflash_eval_v026}"
exec "$TASK_ENV/bin/python" "$TASK_DIR/scheduler.py" "$@"
