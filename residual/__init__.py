"""Mismatch-aware Residual MoLE extensions for this SpecLoRA checkout.

Keep package import light so standalone tensor/unit tests do not require the full
training stack (transformers/vLLM) to be importable in the inspection runtime.
"""

from .config import ResidualConfig

RESIDUAL_VERSION = "V20.6-dual-v10-softlogit-train"

__all__ = ["ResidualConfig", "ResidualDFlashModel", "RESIDUAL_VERSION"]


def __getattr__(name):
    if name == "ResidualDFlashModel":
        from .model_residual import ResidualDFlashModel
        return ResidualDFlashModel
    raise AttributeError(name)
