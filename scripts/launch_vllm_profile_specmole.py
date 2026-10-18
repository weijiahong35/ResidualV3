"""Launch stock vLLM DFlash with optional proposer CUDA-event profiling.

This process-local hook does not change the stock model or proposal math.  It
is used only to obtain a directly comparable Base proposer latency.
"""

from __future__ import annotations

from speculators.models.dflash.vllm_native_specmole import (
    ProposerCudaTimerSpecMoLE,
)
from vllm.entrypoints.cli.main import main
from vllm.v1.spec_decode.dflash import DFlashProposer


class ProfiledDFlashProposerSpecMoLE(DFlashProposer):
    def __init__(self, *args, **kwargs):
        self._specmole_timer = ProposerCudaTimerSpecMoLE("base")
        super().__init__(*args, **kwargs)

    def propose(self, *args, **kwargs):
        cpu_started = self._specmole_timer.begin()
        output = super().propose(*args, **kwargs)
        self._specmole_timer.finish(cpu_started)
        return output


def install_profile_hook_specmole() -> None:
    from vllm.v1.worker import gpu_model_runner

    gpu_model_runner.DFlashProposer = ProfiledDFlashProposerSpecMoLE


# Multiprocessing spawn imports this launcher before constructing GPUModelRunner.
install_profile_hook_specmole()


if __name__ == "__main__":
    main()
