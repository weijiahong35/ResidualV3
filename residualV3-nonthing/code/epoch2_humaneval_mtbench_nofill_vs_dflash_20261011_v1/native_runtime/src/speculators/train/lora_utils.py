"""PEFT-LoRA helpers for fine-tuning DFlash draft projections.

This module intentionally uses PEFT's low-level adapter injection API so the
original ``SpeculatorModel`` / DFlash class is preserved. The base dense DFlash
checkpoint must already be loaded before :func:`inject_dflash_lora` is called.

The fixed LoRA target set used here is:

- Attention: ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``
- MLP: ``gate_proj``, ``up_proj``, ``down_proj``
- Fusion: root ``fc`` projection used to fuse verifier hidden states

Only LoRA A/B parameters on those projections are allowed to remain trainable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model
from safetensors.torch import save_file

logger = logging.getLogger("speculators")


DFLASH_LORA_ATTN_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)

DFLASH_LORA_MLP_TARGET_MODULES: tuple[str, ...] = (
    "gate_proj",
    "up_proj",
    "down_proj",
)

DFLASH_LORA_FUSION_TARGET_MODULES: tuple[str, ...] = (
    "fc",
)

DFLASH_LORA_TARGET_MODULES: tuple[str, ...] = (
    *DFLASH_LORA_ATTN_TARGET_MODULES,
    *DFLASH_LORA_MLP_TARGET_MODULES,
    *DFLASH_LORA_FUSION_TARGET_MODULES,
)


def _is_dflash_lora_target(module_name: str) -> bool:
    """Return True only for requested DFlash LoRA projections.

    Accepted module shapes are:

    - ``*.self_attn.{q,k,v,o}_proj``
    - ``*.mlp.{gate,up,down}_proj``
    - root ``fc`` fusion projection

    PEFT target module names are suffix matches, so this stricter structural
    check prevents an unrelated module with the same leaf name from silently
    receiving a LoRA adapter.
    """
    if module_name == "fc":
        return True

    parts = module_name.split(".")
    if len(parts) < 2:
        return False

    parent = parts[-2]
    leaf = parts[-1]

    if parent == "self_attn":
        return leaf in DFLASH_LORA_ATTN_TARGET_MODULES
    if parent == "mlp":
        return leaf in DFLASH_LORA_MLP_TARGET_MODULES
    return False


def _is_requested_adapter_ab(name: str, adapter_name: str) -> bool:
    """Match PEFT's in-model parameter names for one named LoRA adapter."""
    return (
        f".lora_A.{adapter_name}." in f".{name}"
        or f".lora_B.{adapter_name}." in f".{name}"
    )


def _is_lora_param_under_requested_target(name: str) -> bool:
    """Return True if a LoRA parameter belongs to one requested projection."""
    dotted = f".{name}"

    if any(
        f".self_attn.{target}.lora_" in dotted
        for target in DFLASH_LORA_ATTN_TARGET_MODULES
    ):
        return True

    if any(
        f".mlp.{target}.lora_" in dotted
        for target in DFLASH_LORA_MLP_TARGET_MODULES
    ):
        return True

    if ".fc.lora_" in dotted:
        return True

    return False


def _validate_complete_projection_sets(target_module_names: list[str]) -> None:
    """Validate complete attention/MLP sets and exactly one root fusion FC."""
    attn_prefix_to_targets: dict[str, set[str]] = {}
    mlp_prefix_to_targets: dict[str, set[str]] = {}
    fusion_targets: list[str] = []

    for module_name in target_module_names:
        if module_name == "fc":
            fusion_targets.append(module_name)
            continue

        prefix, leaf = module_name.rsplit(".", 1)
        parent = prefix.rsplit(".", 1)[-1]

        if parent == "self_attn":
            attn_prefix_to_targets.setdefault(prefix, set()).add(leaf)
        elif parent == "mlp":
            mlp_prefix_to_targets.setdefault(prefix, set()).add(leaf)

    if not attn_prefix_to_targets:
        raise RuntimeError(
            "No DFlash self_attn q_proj/k_proj/v_proj/o_proj modules were found. "
            "This attention+MLP+fusion LoRA patch expects attention projections in the "
            "draft model."
        )

    if not mlp_prefix_to_targets:
        raise RuntimeError(
            "No DFlash mlp gate_proj/up_proj/down_proj modules were found. "
            "This attention+MLP+fusion LoRA patch expects SwiGLU MLP projections in the "
            "draft model."
        )

    expected_attn = set(DFLASH_LORA_ATTN_TARGET_MODULES)
    incomplete_attn = {
        prefix: sorted(expected_attn - leaves)
        for prefix, leaves in attn_prefix_to_targets.items()
        if leaves != expected_attn
    }
    if incomplete_attn:
        raise RuntimeError(
            f"Incomplete DFlash attention target set: {incomplete_attn}"
        )

    expected_mlp = set(DFLASH_LORA_MLP_TARGET_MODULES)
    incomplete_mlp = {
        prefix: sorted(expected_mlp - leaves)
        for prefix, leaves in mlp_prefix_to_targets.items()
        if leaves != expected_mlp
    }
    if incomplete_mlp:
        raise RuntimeError(
            f"Incomplete DFlash SwiGLU target set: {incomplete_mlp}"
        )

    if fusion_targets != ["fc"]:
        raise RuntimeError(
            "Expected exactly one root DFlash fusion projection named 'fc', "
            f"found: {fusion_targets}"
        )


def inject_dflash_lora(
    model: torch.nn.Module,
    *,
    adapter_name: str = "code",
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
) -> LoraConfig:
    """Inject LoRA into DFlash attention, MLP, and fusion FC projections.

    The caller is responsible for loading the dense DFlash checkpoint first.
    This function then:

    1. finds modules matching q/k/v/o, gate/up/down, and root ``fc``;
    2. verifies attention/MLP targets belong to the expected DFlash blocks and
       the fusion target is exactly the root ``fc`` module;
    3. verifies each attention block exposes q/k/v/o and each MLP exposes
       gate/up/down;
    4. injects one named PEFT LoRA adapter into all eight projection types;
    5. freezes all model parameters;
    6. re-enables gradients only for this adapter's ``lora_A`` / ``lora_B``.

    The pre/post injection checks are deliberate. PEFT target module lists are
    suffix matches, so these checks prevent an unrelated future module with a
    matching leaf name from silently receiving trainable LoRA weights.
    """
    if not adapter_name or not adapter_name.strip():
        raise ValueError("adapter_name must be a non-empty string")
    if r <= 0:
        raise ValueError(f"LoRA rank r must be > 0, got {r}")
    if alpha <= 0:
        raise ValueError(f"LoRA alpha must be > 0, got {alpha}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")

    target_module_names = [
        name
        for name, _module in model.named_modules()
        if name.rsplit(".", 1)[-1] in DFLASH_LORA_TARGET_MODULES
    ]
    if not target_module_names:
        raise RuntimeError(
            "No requested q/k/v/o/gate/up/down/fc projection modules were found. "
            "Expected DFlash Qwen3 attention, MLP, and root fusion FC modules "
            "before LoRA injection."
        )

    unexpected_targets = [
        name for name in target_module_names if not _is_dflash_lora_target(name)
    ]
    if unexpected_targets:
        preview = ", ".join(unexpected_targets[:8])
        raise RuntimeError(
            "Refusing LoRA injection because a requested target suffix exists "
            "outside DFlash self_attn/mlp projections or the root fusion fc: "
            f"{preview}"
        )

    _validate_complete_projection_sets(target_module_names)

    attn_target_count = sum(
        1 for name in target_module_names if ".self_attn." in f".{name}."
    )
    mlp_target_count = sum(
        1 for name in target_module_names if ".mlp." in f".{name}."
    )
    fusion_target_count = sum(1 for name in target_module_names if name == "fc")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(DFLASH_LORA_TARGET_MODULES),
        bias="none",
        inference_mode=False,
    )

    # Low-level PEFT injection mutates and returns the original model class.
    injected = inject_adapter_in_model(
        lora_config,
        model,
        adapter_name=adapter_name,
    )
    if injected is not model:
        raise RuntimeError(
            "inject_adapter_in_model unexpectedly replaced the model object; "
            "this patch requires in-place low-level PEFT injection."
        )

    # Explicitly freeze *everything*, then opt in only this named adapter's A/B.
    for param in model.parameters():
        param.requires_grad = False

    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        if _is_requested_adapter_ab(name, adapter_name):
            if not _is_lora_param_under_requested_target(name):
                raise RuntimeError(
                    "LoRA parameter appeared outside requested DFlash "
                    f"attention/MLP/fusion targets: {name}"
                )
            param.requires_grad = True
            trainable_names.append(name)

    if not trainable_names:
        raise RuntimeError(
            f"PEFT injected no trainable lora_A/lora_B parameters for {adapter_name!r}."
        )

    bad_trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and not _is_requested_adapter_ab(name, adapter_name)
    ]
    if bad_trainable:
        raise RuntimeError(
            "Non-LoRA parameters remained trainable after freezing: "
            + ", ".join(bad_trainable[:8])
        )

    logger.info(
        "Injected DFlash PEFT-LoRA adapter %r into %d projection modules "
        "(%d attention + %d MLP + %d fusion FC; "
        "r=%d, alpha=%d, dropout=%s).",
        adapter_name,
        len(target_module_names),
        attn_target_count,
        mlp_target_count,
        fusion_target_count,
        r,
        alpha,
        dropout,
    )
    return lora_config


def save_lora_adapter(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    lora_config: LoraConfig,
    adapter_name: str = "code",
) -> None:
    """Save only one LoRA adapter in standard PEFT filenames.

    ``model`` must be the unwrapped module (``ddp_model.module`` under DDP).
    The resulting directory contains:

    - ``adapter_model.safetensors`` -- adapter tensors only;
    - ``adapter_config.json`` -- serialized :class:`LoraConfig`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = get_peft_model_state_dict(
        model,
        adapter_name=adapter_name,
        save_embedding_layers=False,
    )
    if not state_dict:
        raise RuntimeError(
            f"No PEFT state was returned for adapter {adapter_name!r}; not saving."
        )

    # Normal LoRA with bias='none' and no modules_to_save must contain only A/B.
    unexpected = [
        key
        for key in state_dict
        if ".lora_A." not in f".{key}." and ".lora_B." not in f".{key}."
    ]
    if unexpected:
        raise RuntimeError(
            "Adapter-only state unexpectedly contains non-LoRA tensors: "
            + ", ".join(unexpected[:8])
        )

    cpu_state = {
        key: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
    }
    save_file(
        cpu_state,
        str(output_dir / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    lora_config.save_pretrained(output_dir)

    config_path = output_dir / "adapter_config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"PEFT config save did not create expected file: {config_path}"
        )

    logger.info("Saved adapter-only LoRA checkpoint to %s", output_dir)


def log_trainable_parameters(model: torch.nn.Module) -> dict[str, int | float]:
    """Log and return total/trainable parameter counts.

    Raises if any trainable parameter is not a LoRA A/B parameter. This makes the
    startup log an executable safety check rather than a cosmetic statistic.
    """
    total = 0
    trainable = 0
    trainable_names: list[str] = []
    non_lora_trainable: list[str] = []

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            trainable_names.append(name)
            if ".lora_A." not in f".{name}." and ".lora_B." not in f".{name}.":
                non_lora_trainable.append(name)

    if non_lora_trainable:
        raise RuntimeError(
            "Expected only LoRA A/B to be trainable, but found: "
            + ", ".join(non_lora_trainable[:8])
        )
    if trainable == 0:
        raise RuntimeError("No trainable LoRA parameters found.")

    pct = 100.0 * trainable / total if total else 0.0
    logger.info(
        "Trainable parameters: %d / %d (%.6f%%), tensors=%d",
        trainable,
        total,
        pct,
        len(trainable_names),
    )
    for name in trainable_names:
        logger.debug("LoRA trainable: %s", name)

    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": pct,
        "trainable_tensors": len(trainable_names),
    }
