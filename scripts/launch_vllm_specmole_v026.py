#!/usr/bin/env python3
"""Launch vLLM 0.26 after process-local native SpecMoLE registration."""

from __future__ import annotations

import sys
from pathlib import Path


# Import the adapter as a top-level module.  This avoids importing the project's
# optional training package (and its datasets dependency) in the lean v0.26 env.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = PROJECT_ROOT / "src" / "speculators" / "models" / "dflash"
sys.path.insert(0, str(ADAPTER_DIR))

from vllm_native_specmole_v026 import install_vllm_specmole_v026


install_vllm_specmole_v026()


if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
