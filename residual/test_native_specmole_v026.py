#!/usr/bin/env python3
"""Checkpoint-backed tensor parity for the vLLM 0.26 SpecMoLE adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "speculators" / "models" / "dflash"))

import vllm_native_specmole_v026 as runtime
from residual.mismatch_predictor import MismatchPredictor
from residual.mole_linear import ResidualMoLE
from residual.router import ResidualRouter


class Controls(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.predictor = MismatchPredictor(
            hidden_size=4096,
            hidden_dim=cfg.mismatch_hidden_dim,
            direction_dim=cfg.mismatch_direction_dim,
            block_size=8,
            max_target_layers=cfg.mismatch_num_hidden_layers,
            max_drafter_layers=5,
            predictor_depth=cfg.predictor_depth,
            attention_heads=cfg.predictor_attention_heads,
            ffn_expansion=cfg.predictor_ffn_expansion,
        )
        self.routers = nn.ModuleDict()
        self.moles = nn.ModuleDict()
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
            self.moles[str(layer_idx)] = ResidualMoLE(
                4096,
                num_experts=cfg.mole_num_experts,
                rank=cfg.mole_rank,
                alpha=cfg.mole_alpha,
                shared_rank=cfg.mole_shared_rank,
                dropout=0.0,
                max_relative_update=cfg.mole_max_relative_update,
            )

    def control_eager(
        self,
        layer_idx,
        previous_target,
        previous_draft,
        current,
        mlp_input,
        valid,
    ):
        prepared = self.predictor.prepare_history(previous_target, previous_draft)
        return runtime.specmole_control_exact(
            self.predictor,
            self.predictor.risk_branches[f"p{layer_idx}"],
            self.predictor.repair_branches[f"p{layer_idx}"],
            self.routers[f"p{layer_idx}"],
            self.moles[str(layer_idx)],
            prepared["d_prev_tokens"],
            prepared["history"],
            prepared["history_sources"],
            self.predictor.project_current(current),
            mlp_input,
            valid,
        )


def load_controls(checkpoint: Path, device: torch.device):
    root, stage, cfg = runtime.checkpoint_metadata(checkpoint)
    controls = Controls(cfg)
    state = load_file(str(runtime.weights_path(root)), device="cpu")
    migrated = {}
    for key, value in state.items():
        marker = "base_model.layers."
        middle = ".residual_mole."
        if key.startswith(marker) and middle in key:
            prefix, suffix = key.split(middle, 1)
            key = f"moles.{prefix[len(marker):]}.{suffix}"
        elif key.startswith("residual_mole."):
            key = "moles." + key[len("residual_mole.") :]
        migrated[key] = value
    result = controls.load_state_dict(migrated, strict=False)
    missing = [
        key
        for key in result.missing_keys
        if key.startswith(("predictor.", "routers.", "moles."))
    ]
    unexpected = [key for key in result.unexpected_keys if key != "direction_projector"]
    if missing or unexpected:
        raise RuntimeError(f"audit failed: missing={missing[:8]} unexpected={unexpected[:8]}")
    return controls.to(device=device, dtype=torch.bfloat16).eval(), cfg, stage


def reference(controls, previous_target, previous_draft, current, mlp_input, valid):
    predictor = controls.predictor
    pred = predictor.forward_from_history(
        predictor.prepare_history(previous_target, previous_draft), current
    )
    valid_bool = valid.bool()
    severity = torch.where(
        valid_bool, pred["severity"], torch.zeros_like(pred["severity"])
    )
    overlap = torch.exp(-severity.float()).clamp(1.0e-6, 1.0).to(severity.dtype)
    overlap = torch.where(valid_bool, overlap, torch.ones_like(overlap))
    reach = torch.cat(
        [torch.ones_like(overlap[:, :1]), torch.cumprod(overlap[:, :-1], dim=-1)],
        dim=-1,
    )
    first_reject = reach * (1.0 - overlap) * valid_bool.float()
    reach = reach * valid_bool.float()
    repair_latent = pred["repair_latent"]
    repair = torch.cat(
        [repair_latent, pred["repair_direction_per_pos"].to(repair_latent.dtype)],
        dim=-1,
    )
    physical = torch.stack(
        [severity, 1.0 - overlap, first_reject, reach], dim=-1
    ).to(pred["risk_latent"].dtype)
    risk = torch.cat([pred["risk_latent"], physical], dim=-1)
    layer_idx = current.shape[1]
    effective, logits, gate = controls.routers[f"p{layer_idx}"](repair, risk)
    gate = gate * valid_bool.unsqueeze(-1).to(gate.dtype)
    weights = effective / gate.clamp_min(1.0e-9)
    weights = torch.where(gate > 0, weights, torch.zeros_like(weights))
    correction = controls.moles[str(layer_idx)](
        mlp_input.reshape(1, -1, 4096),
        route_weights=weights,
        gate=gate,
        valid_mask=valid,
    ).reshape_as(mlp_input)
    return severity, pred["repair_direction"], logits, weights, gate, correction


def stats(name, ref, got):
    delta = (ref.float() - got.float()).abs()
    return {
        "tensor": name,
        "bitwise_equal": bool(torch.equal(ref, got)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
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
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(20260904)
    controls, cfg, stage = load_controls(Path(args.checkpoint), device)
    batch = args.batch_size
    previous_target = torch.randn(batch, 5, 4096, device=device, dtype=torch.bfloat16)
    previous_draft = torch.randn(batch, 5, 4096, device=device, dtype=torch.bfloat16)
    all_current = torch.randn(batch, 5, 4096, device=device, dtype=torch.bfloat16)
    all_mlp = torch.randn(3, batch, 8, 4096, device=device, dtype=torch.bfloat16)
    valid = torch.ones(batch, 8, device=device, dtype=torch.bool)
    valid[::3, -2:] = False
    with safe_open(
        str(Path(args.draft_checkpoint) / "model.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        lm_head = handle.get_tensor("lm_head.weight").to(device=device)

    names = (
        "predictor_severity",
        "predictor_repair_direction",
        "router_logits",
        "router_weights",
        "gate",
        "mole_correction",
    )
    report = {"stage": stage, "batch": batch, "prefixes": {}}
    passed = True
    for slot, layer_idx in enumerate(cfg.correction_layer_indices):
        current = all_current[:, :layer_idx]
        mlp_input = all_mlp[slot]
        ref = reference(
            controls, previous_target, previous_draft, current, mlp_input, valid
        )
        eager = controls.control_eager(
            layer_idx, previous_target, previous_draft, current, mlp_input, valid
        )
        # Validate the actual production custom-op dispatcher too.  It is this
        # node that vLLM AOT retains and its kernels are captured by FULL CG.
        class Adapter:
            def _control_eager(self, *op_args):
                return controls.control_eager(*op_args)[5]

        runtime._ACTIVE_CONTROL_MODEL = Adapter()
        custom_correction = runtime._control_custom_op(
            previous_target,
            previous_draft,
            current,
            mlp_input,
            valid,
            layer_idx,
        )
        rows = []
        for name, expected, candidate in zip(names, ref, eager):
            row = stats(name, expected, candidate)
            rows.append(row)
            passed &= row["bitwise_equal"]
        row = stats("custom_op_correction", ref[5], custom_correction)
        rows.append(row)
        passed &= row["bitwise_equal"]
        corrected_ref = mlp_input + ref[5]
        corrected = mlp_input + custom_correction
        row = stats("corrected_hidden", corrected_ref, corrected)
        rows.append(row)
        passed &= row["bitwise_equal"]
        ref_logits = F.linear(corrected_ref[:, -1], lm_head)
        logits = F.linear(corrected[:, -1], lm_head)
        row = stats("draft_logits", ref_logits, logits)
        rows.append(row)
        passed &= row["bitwise_equal"]
        ref_topology = torch.topk(ref[3], cfg.mole_top_k, dim=-1).indices
        got_topology = torch.topk(eager[3], cfg.mole_top_k, dim=-1).indices
        topology_equal = bool(torch.equal(ref_topology, got_topology))
        rows.append({"tensor": "router_topology", "bitwise_equal": topology_equal})
        passed &= topology_equal
        report["prefixes"][f"p{layer_idx}"] = rows
    report["pass"] = passed
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("v0.26 SpecMoLE tensor parity: FAIL")
    print("v0.26 SpecMoLE tensor parity: PASS (all tensors/topology bitwise exact)")


if __name__ == "__main__":
    main()
