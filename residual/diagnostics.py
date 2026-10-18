from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .mismatch import MismatchTargets, expected_accepted_length


def _prefix(pred: dict[str, torch.Tensor]) -> int:
    value = pred.get("prefix_layers")
    if value is None:
        raise KeyError("progressive prediction is missing prefix_layers")
    return int(value.detach().item()) if torch.is_tensor(value) else int(value)


@torch.no_grad()
def predictor_target_diagnostics(
    pred: dict[str, torch.Tensor],
    target: MismatchTargets,
) -> dict[str, torch.Tensor]:
    """V10-comparable soft-logit predictor diagnostics."""
    k = _prefix(pred)
    out: dict[str, torch.Tensor] = {}
    valid = target.valid_mask.bool()

    if valid.any():
        pred_sev = pred["severity"].float()
        target_sev = target.severity.float()
        err = (pred_sev - target_sev)[valid]
        out[f"pred_nla_mae_p{k}"] = err.abs().mean()
        out[f"pred_nla_rmse_p{k}"] = err.square().mean().sqrt()
        out[f"pred_nla_bias_p{k}"] = err.mean()
        out[f"pred_nla_pred_mean_p{k}"] = pred_sev[valid].mean()
        out[f"pred_nla_target_mean_p{k}"] = target_sev[valid].mean()

        pred_overlap = torch.exp(-pred_sev).clamp(0.0, 1.0)
        overlap_err = (pred_overlap - target.overlap.float())[valid]
        out[f"pred_overlap_mae_p{k}"] = overlap_err.abs().mean()
        out[f"pred_overlap_bias_p{k}"] = overlap_err.mean()

        pred_eal = expected_accepted_length(pred_overlap, valid)
        eal_err = pred_eal - target.expected_accepted_length.float()
        out[f"pred_eal_mae_p{k}"] = eal_err.abs().mean()
        out[f"pred_eal_bias_p{k}"] = eal_err.mean()

    reject_mass = target.first_reject_prob.float().sum(dim=-1)
    block_valid = reject_mass > 1.0e-8
    if block_valid.any():
        cos = F.cosine_similarity(
            pred["repair_direction"].float()[block_valid],
            target.repair_direction.float()[block_valid],
            dim=-1,
        )
        weights = reject_mass[block_valid]
        out[f"pred_repair_cos_p{k}"] = cos.mean()
        out[f"pred_repair_cos_weighted_p{k}"] = (
            cos * weights
        ).sum() / weights.sum().clamp_min(1.0e-8)

    return {name: value.detach() for name, value in out.items()}


@torch.no_grad()
def progressive_prediction_diagnostics(
    predictions: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Track how independent p2/p3/p4 Risk and Repair branches evolve."""
    if not predictions:
        return {}

    out: dict[str, torch.Tensor] = {}
    for pred in predictions:
        k = _prefix(pred)
        for task in ("risk", "repair"):
            gate = pred.get(f"{task}_fusion_gate")
            if gate is not None:
                g = gate.float().reshape(-1)
                out[f"{task}_fusion_gate_mean_p{k}"] = g.mean()
                out[f"{task}_fusion_gate_std_p{k}"] = g.std(unbiased=False)
            for key in (
                "cross_update_ratio",
                "self_update_ratio",
                "ffn_update_ratio",
                "latent_diversity",
            ):
                value = pred.get(f"{task}_{key}")
                if value is not None:
                    out[f"{task}_{key}_p{k}"] = value.float().reshape(())

    # Aggregate headline diagnostics for the trainer's compact first log line.
    for key in ("fusion_gate", "cross_update_ratio", "self_update_ratio", "ffn_update_ratio", "latent_diversity"):
        vals = [pred.get(key) for pred in predictions if pred.get(key) is not None]
        if vals:
            flat = [v.float().mean() for v in vals]
            out[f"{key}_mean" if key == "fusion_gate" else key] = torch.stack(flat).mean()

    for left, right in zip(predictions[:-1], predictions[1:], strict=True):
        k1, k2 = _prefix(left), _prefix(right)
        rz1 = left["risk_block_latent"].float()
        rz2 = right["risk_block_latent"].float()
        pz1 = left["repair_block_latent"].float()
        pz2 = right["repair_block_latent"].float()
        out[f"risk_latent_cos_p{k1}_p{k2}"] = F.cosine_similarity(
            rz1, rz2, dim=-1
        ).mean()
        out[f"repair_latent_cos_p{k1}_p{k2}"] = F.cosine_similarity(
            pz1, pz2, dim=-1
        ).mean()
        # Keep the old logger key pointed at Risk only for compatibility.
        out[f"latent_cos_p{k1}_p{k2}"] = out[f"risk_latent_cos_p{k1}_p{k2}"]
        out[f"severity_change_mae_p{k1}_p{k2}"] = (
            right["severity"].float() - left["severity"].float()
        ).abs().mean()
        out[f"repair_cos_p{k1}_p{k2}"] = F.cosine_similarity(
            left["repair_direction"].float(),
            right["repair_direction"].float(),
            dim=-1,
        ).mean()

    return {name: value.detach() for name, value in out.items()}


@torch.no_grad()
def routing_diagnostics(
    prefix_lengths: list[int] | tuple[int, ...],
    routes: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not routes:
        return {}
    if len(prefix_lengths) != len(routes):
        raise ValueError("prefix_lengths and routes must have the same length")

    out: dict[str, torch.Tensor] = {}
    eps = 1.0e-9
    for k, route in zip(prefix_lengths, routes, strict=True):
        effective = route.float().reshape(-1, route.shape[-1]).clamp_min(0.0)
        gate = effective.sum(dim=-1)
        out[f"route_gate_mean_p{k}"] = gate.mean()
        out[f"route_gate_std_p{k}"] = gate.std(unbiased=False)
        out[f"route_noop_frac_p{k}"] = (gate < 0.10).float().mean()
        p = effective / gate.unsqueeze(-1).clamp_min(eps)
        entropy_per_anchor = -(p * p.clamp_min(eps).log()).sum(dim=-1)
        entropy = entropy_per_anchor.mean()
        num_experts = p.shape[-1]
        out[f"route_entropy_p{k}"] = entropy
        out[f"route_entropy_norm_p{k}"] = entropy / max(math.log(num_experts), eps)
        out[f"route_effective_experts_p{k}"] = entropy.exp()
        out[f"route_active_experts_p{k}"] = (p > eps).float().sum(dim=-1).mean()
        sorted_p, _ = torch.sort(p, dim=-1, descending=True)
        out[f"route_max_prob_p{k}"] = sorted_p[:, 0].mean()
        if num_experts > 1:
            out[f"route_margin_p{k}"] = (sorted_p[:, 0] - sorted_p[:, 1]).mean()
        top1 = p.argmax(dim=-1)
        for expert in range(num_experts):
            out[f"route_weight_p{k}_e{expert}"] = p[:, expert].mean()
            out[f"route_top1_frac_p{k}_e{expert}"] = (top1 == expert).float().mean()

    for k1, k2, r1, r2 in zip(
        prefix_lengths[:-1], prefix_lengths[1:], routes[:-1], routes[1:], strict=True
    ):
        p = r1.float().reshape(-1, r1.shape[-1]).clamp_min(0.0)
        q = r2.float().reshape(-1, r2.shape[-1]).clamp_min(0.0)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
        out[f"route_cos_p{k1}_p{k2}"] = F.cosine_similarity(p, q, dim=-1).mean()
        out[f"route_l1_p{k1}_p{k2}"] = (p - q).abs().sum(dim=-1).mean()
        out[f"route_top1_change_p{k1}_p{k2}"] = (
            p.argmax(dim=-1) != q.argmax(dim=-1)
        ).float().mean()
        m = 0.5 * (p + q)
        kl_pm = (p * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
        kl_qm = (q * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
        out[f"route_js_p{k1}_p{k2}"] = (0.5 * (kl_pm + kl_qm)).mean()

    return {name: value.detach() for name, value in out.items()}
