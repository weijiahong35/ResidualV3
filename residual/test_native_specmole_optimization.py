#!/usr/bin/env python3
"""Checkpoint-backed exact parity test for the native SpecMoLE fast path.

The reference side is the original Predictor.forward_from_history -> Router ->
ResidualMoLE.forward implementation.  The optimized side is the inference-only
runtime helper, both eagerly and through the production-style CUDA graph.  No
hidden states or tensor dumps are written to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
import torch.nn.functional as F

from residual.mismatch_predictor import MismatchPredictor
from residual.mole_linear import ResidualMoLE
from residual.router import ResidualRouter
from speculators.models.dflash.vllm_native_specmole import (
    _apply_prediction_valid_mask,
    _checkpoint_metadata,
    _router_inputs,
    _specmole_control_runtime,
    _weights_path,
)


class _ControlModules(nn.Module):
    def __init__(self, cfg, *, hidden_size: int, block_size: int) -> None:
        super().__init__()
        self.predictor = MismatchPredictor(
            hidden_size=hidden_size,
            hidden_dim=cfg.mismatch_hidden_dim,
            direction_dim=cfg.mismatch_direction_dim,
            block_size=block_size,
            max_target_layers=cfg.mismatch_num_hidden_layers,
            max_drafter_layers=5,
            predictor_depth=cfg.predictor_depth,
            attention_heads=cfg.predictor_attention_heads,
            ffn_expansion=cfg.predictor_ffn_expansion,
        )
        self.routers = nn.ModuleDict()
        self.residual_mole = nn.ModuleDict()
        for layer_idx in cfg.correction_layer_indices:
            self.routers[f"p{layer_idx}"] = ResidualRouter(
                cfg.mismatch_hidden_dim + cfg.mismatch_direction_dim,
                risk_input_dim=cfg.mismatch_hidden_dim + cfg.router_context_dim,
                num_experts=cfg.mole_num_experts,
                top_k=cfg.mole_top_k,
                gate_init_bias=cfg.router_gate_init_bias,
                balance_mode=cfg.router_balance_mode,
                bias_update_rate=cfg.router_bias_update_rate,
                load_ema=cfg.router_load_ema,
                bias_clip=cfg.router_bias_clip,
                temperature=cfg.router_temperature_start,
                routing_mode="topk",
            )
            self.residual_mole[str(layer_idx)] = ResidualMoLE(
                hidden_size,
                num_experts=cfg.mole_num_experts,
                rank=cfg.mole_rank,
                alpha=cfg.mole_alpha,
                shared_rank=cfg.mole_shared_rank,
                dropout=0.0,
                max_relative_update=cfg.mole_max_relative_update,
            )


def _load_modules(checkpoint: Path, device: torch.device):
    root, stage, cfg = _checkpoint_metadata(checkpoint)
    modules = _ControlModules(cfg, hidden_size=4096, block_size=8)
    state = load_file(str(_weights_path(root)), device="cpu")
    migrated = {}
    for key, value in state.items():
        marker = "base_model.layers."
        middle = ".residual_mole."
        if key.startswith(marker) and middle in key:
            prefix, suffix = key.split(middle, 1)
            layer = prefix[len(marker) :]
            if layer.isdigit():
                key = f"residual_mole.{layer}.{suffix}"
        migrated[key] = value
    result = modules.load_state_dict(migrated, strict=False)
    critical_missing = [
        key
        for key in result.missing_keys
        if key.startswith(("predictor.", "routers.", "residual_mole."))
    ]
    unexpected = [key for key in result.unexpected_keys if key != "direction_projector"]
    if critical_missing or unexpected:
        raise RuntimeError(
            f"checkpoint audit failed: missing={critical_missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    return modules.to(device=device, dtype=torch.bfloat16).eval(), cfg, stage


def _reference(
    modules: _ControlModules,
    previous_target: torch.Tensor,
    previous_draft: torch.Tensor,
    current: torch.Tensor,
    mlp_input: torch.Tensor,
    valid: torch.Tensor,
):
    predictor = modules.predictor
    prepared = predictor.prepare_history(previous_target, previous_draft)
    pred = predictor.forward_from_history(prepared, current)
    pred = _apply_prediction_valid_mask(pred, valid)
    repair_input, risk_input = _router_inputs(pred)
    layer_idx = current.shape[1]
    effective, route_logits, gate = modules.routers[f"p{layer_idx}"](
        repair_input, risk_input
    )
    gate = gate * valid.unsqueeze(-1).to(gate.dtype)
    weights = effective / gate.clamp_min(1.0e-9)
    weights = torch.where(gate > 0, weights, torch.zeros_like(weights))
    correction = modules.residual_mole[str(layer_idx)](
        mlp_input.reshape(1, -1, mlp_input.shape[-1]),
        route_weights=weights,
        gate=gate,
        valid_mask=valid,
    ).reshape_as(mlp_input)
    return (
        pred["severity"],
        pred["repair_direction"],
        route_logits,
        weights,
        gate,
        correction,
    )


def _compare(name: str, reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "name": name,
        "shape": list(reference.shape),
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="output/openmath/onpolicy/residual/stage3_joint/latest/checkpoints/final",
    )
    parser.add_argument(
        "--draft-checkpoint",
        default="/data/weijiahong/huggingface/models/Qwen3-8B-speculator.dflash",
    )
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("CUDA is required because production uses CUDA graphs")
    torch.manual_seed(20260904)
    torch.cuda.manual_seed_all(20260904)
    modules, cfg, stage = _load_modules(Path(args.checkpoint), device)
    batch, block, hidden = args.batch_size, 8, 4096
    previous_target = torch.randn(
        batch, cfg.mismatch_num_hidden_layers, hidden,
        device=device, dtype=torch.bfloat16,
    )
    previous_draft = torch.randn(
        batch, 5, hidden, device=device, dtype=torch.bfloat16,
    )
    all_current = torch.randn(batch, 5, hidden, device=device, dtype=torch.bfloat16)
    all_mlp = torch.randn(
        3, batch, block, hidden, device=device, dtype=torch.bfloat16,
    )
    valid = torch.ones(batch, block, device=device, dtype=torch.bool)
    valid[::3, -2:] = False

    prepared = modules.predictor.prepare_history(previous_target, previous_draft)
    field_names = (
        "predictor_severity",
        "predictor_repair_direction",
        "router_logits",
        "router_weights",
        "gate",
        "mole_correction",
    )
    report: dict[str, object] = {
        "checkpoint_stage": stage,
        "batch_size": batch,
        "dtype": "bfloat16",
        "prefixes": {},
    }

    # Load the real DFlash LM head, so the final comparison is in actual draft
    # logit space rather than through a synthetic projection.
    draft_weights = Path(args.draft_checkpoint) / "model.safetensors"
    with safe_open(str(draft_weights), framework="pt", device="cpu") as handle:
        lm_head = handle.get_tensor("lm_head.weight").to(device=device)

    all_pass = True
    graph_pool = torch.cuda.graph_pool_handle()
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    for slot, layer_idx in enumerate(cfg.correction_layer_indices):
        current = all_current[:, :layer_idx, :]
        mlp_input = all_mlp[slot]
        reference = _reference(
            modules, previous_target, previous_draft, current, mlp_input, valid
        )

        def fast(*inputs):
            return _specmole_control_runtime(
                modules.predictor,
                modules.predictor.risk_branches[f"p{layer_idx}"],
                modules.predictor.repair_branches[f"p{layer_idx}"],
                modules.routers[f"p{layer_idx}"],
                modules.residual_mole[str(layer_idx)],
                *inputs,
            )

        inputs = (
            prepared["d_prev_tokens"],
            prepared["history"],
            prepared["history_sources"],
            modules.predictor.project_current(current),
            mlp_input,
            valid,
        )
        eager = fast(*inputs)
        static_inputs = tuple(value.clone() for value in inputs)
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                graph_outputs = fast(*static_inputs)
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream, pool=graph_pool):
            graph_outputs = fast(*static_inputs)
        for destination, source in zip(static_inputs, inputs):
            destination.copy_(source)
        graph.replay()
        torch.cuda.synchronize()
        replay = tuple(value.clone() for value in graph_outputs)

        rows = []
        for name, ref, eager_value, replay_value in zip(
            field_names, reference, eager, replay
        ):
            eager_result = _compare(f"{name}:eager", ref, eager_value)
            graph_result = _compare(f"{name}:cuda_graph", ref, replay_value)
            rows.extend([eager_result, graph_result])
            all_pass &= eager_result["bitwise_equal"] and graph_result["bitwise_equal"]

        ref_corrected = mlp_input + reference[5]
        eager_corrected = mlp_input + eager[5]
        graph_corrected = mlp_input + replay[5]
        for label, candidate in (
            ("corrected_hidden:eager", eager_corrected),
            ("corrected_hidden:cuda_graph", graph_corrected),
        ):
            result = _compare(label, ref_corrected, candidate)
            rows.append(result)
            all_pass &= result["bitwise_equal"]

        ref_logits = F.linear(ref_corrected[:, -1, :], lm_head)
        for label, corrected in (
            ("draft_logits:eager", eager_corrected),
            ("draft_logits:cuda_graph", graph_corrected),
        ):
            result = _compare(label, ref_logits, F.linear(corrected[:, -1, :], lm_head))
            rows.append(result)
            all_pass &= result["bitwise_equal"]
        report["prefixes"][f"p{layer_idx}"] = rows

    report["pass"] = bool(all_pass)
    print(json.dumps(report, indent=2))
    if not all_pass:
        raise SystemExit("Native SpecMoLE optimization tensor parity: FAIL")
    print("Native SpecMoLE optimization tensor parity: PASS (all fields bitwise exact)")


if __name__ == "__main__":
    main()
