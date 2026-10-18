from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

try:
    from safetensors.torch import load_file, save_file
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Residual MoLE checkpointing requires safetensors") from exc


def residual_state_dict(model: torch.nn.Module, stage: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        keep = False
        if "predictor." in name and stage in {"predictor", "mole", "joint", "nomole", "nomole_joint", "eval", "eval_nomole"}:
            keep = True
        if "routers." in name and stage in {"mole", "joint", "eval"}:
            keep = True
        if "residual_mole." in name and stage in {"mole", "joint", "eval"}:
            keep = True
        if "dense_gates." in name and stage in {"nomole", "nomole_joint", "eval_nomole"}:
            keep = True
        if "direction_projector" in name:
            keep = True
        if keep:
            out[name] = tensor.detach().cpu().contiguous()
    return out


def save_residual_checkpoint(
    model: torch.nn.Module,
    out_dir: str | Path,
    *,
    stage: str,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "v20_model.safetensors"
    state = residual_state_dict(model, stage)
    save_file(state, str(weights))
    # Compatibility name for existing launchers in this checkout.
    save_file(state, str(out / "residual_mole.safetensors"))
    payload = {"stage": stage, "residual_config": config}
    if extra:
        payload.update(extra)
    (out / "residual_mole_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (out / "v20_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    runtime_policy = {
        "target_temperature": config.get("sampling_temperature", 0.0),
        "target_top_k": config.get("sampling_top_k", 0),
        "target_top_p": config.get("sampling_top_p", 1.0),
        "draft_mode": config.get("draft_mode", "greedy"),
        "distribution_temperature": config.get("distribution_temperature", 1.0),
    }
    (out / "runtime_policy.json").write_text(
        json.dumps(runtime_policy, indent=2), encoding="utf-8"
    )
    stage_state = {"stage": stage}
    if extra:
        stage_state.update(extra)
    (out / "stage_state.json").write_text(
        json.dumps(stage_state, indent=2), encoding="utf-8"
    )
    base_config = getattr(getattr(model, "base_model", None), "config", None)
    base_ref = {
        "name_or_path": getattr(base_config, "_name_or_path", None),
        "model_type": getattr(base_config, "model_type", None),
    }
    (out / "base_model_reference.json").write_text(
        json.dumps(base_ref, indent=2), encoding="utf-8"
    )
    return weights


def load_residual_checkpoint(
    model: torch.nn.Module, checkpoint: str | Path, *, strict: bool = False
) -> tuple[list[str], list[str]]:
    p = Path(checkpoint)
    if p.is_dir():
        canonical = p / "v20_model.safetensors"
        p = canonical if canonical.exists() else p / "residual_mole.safetensors"
    state = load_file(str(p))

    # Compatibility with the first V20 implementation, where SpecMoLE modules
    # were registered directly under base_model.layers.<idx>.residual_mole.
    # The isolated layout owns them under residual_mole.<idx> instead.
    migrated: dict[str, torch.Tensor] = {}
    marker = "base_model.layers."
    middle = ".residual_mole."
    for key, value in state.items():
        if key.startswith(marker) and middle in key:
            prefix, suffix = key.split(middle, 1)
            layer_idx = prefix[len(marker):]
            if layer_idx.isdigit():
                key = f"residual_mole.{layer_idx}.{suffix}"
        migrated[key] = value
    result = model.load_state_dict(migrated, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
