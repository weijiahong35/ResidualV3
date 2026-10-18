#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${VLLM_ENV_DIR:-/data/weijiahong/miniconda3/envs/llm_dflash_eval_v026}"
# No changes to torch/vllm/CUDA or the shared conda site-packages.
CUDA_VISIBLE_DEVICES="" "$ENV_DIR/bin/python" -m pip install \
  --target "$HERE/.client_deps" --no-deps -r "$HERE/requirements-client.txt"
