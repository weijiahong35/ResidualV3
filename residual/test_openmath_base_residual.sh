#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Change this default directly, or run with GPU=3 / TARGET_GPU=3.
GPU="${GPU:-7}"
exec env PROJECT_ROOT="${PROJECT_ROOT}" GPU="${GPU}" BENCH_MODE=base \
    bash "${PROJECT_ROOT}/residual/run_openmath_native.sh" "$@"
