#!/usr/bin/env python3
"""Launch vLLM after installing the process-local native SpecMoLE adapter."""

from speculators.models.dflash.vllm_native_specmole import install_vllm_specmole


# Keep registration at module import time so multiprocessing ``spawn`` workers
# install the same model/proposer overrides before constructing GPUModelRunner.
install_vllm_specmole()


if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
