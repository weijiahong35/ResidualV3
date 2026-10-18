#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${VLLM_ENV_DIR:-/data/weijiahong/miniconda3/envs/llm_dflash_eval_v026}"
exec "$ENV_DIR/bin/python" "$HERE/download_datasets.py" "$@"
