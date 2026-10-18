from __future__ import annotations

import torch
import torch.nn.functional as F

from .mismatch import MismatchTargets


def predictor_loss(
    pred: dict[str, torch.Tensor],
    target: MismatchTargets,
    *,
    lambda_s: float = 1.0,
    lambda_r: float = 1.0,
    severity_calibration_weight: float = 0.25,
    severity_mean_weight: float = 0.05,
    eps: float = 1e-8,
    **_: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dual-V10 Stage-1 objective with V10 soft logit-derived supervision.

    Risk/Bias Predictor: V10 NLA severity regression, weighted by exact EAL
    sensitivity from soft Target/Draft overlap.

    Repair Predictor: V10 block repair direction cosine loss, weighted by the
    probability that the block contains a rejection under the soft overlap view.
    """
    pred_sev = pred["severity"].float()
    target_sev = target.severity.float()
    valid = target.valid_mask.bool()
    w = target.severity_weight.float() * target.valid_mask.float()
    denom = w.sum().clamp_min(1.0)
    sev_elem = F.smooth_l1_loss(
        pred_sev, target_sev, reduction="none"
    )
    sev_weighted = (sev_elem * w).sum() / denom

    if valid.any():
        sev_cal = F.smooth_l1_loss(pred_sev[valid], target_sev[valid])
        sev_mean = F.smooth_l1_loss(
            pred_sev[valid].mean(), target_sev[valid].mean()
        )
    else:
        sev_cal = pred_sev.sum() * 0.0
        sev_mean = pred_sev.sum() * 0.0

    sev = (
        sev_weighted
        + float(severity_calibration_weight) * sev_cal
        + float(severity_mean_weight) * sev_mean
    )

    block_reject_mass = target.first_reject_prob.float().sum(dim=-1)
    block_valid = block_reject_mass > eps
    if block_valid.any():
        repair_elem = 1.0 - F.cosine_similarity(
            pred["repair_direction"].float()[block_valid],
            target.repair_direction.float()[block_valid],
            dim=-1,
        )
        repair_w = block_reject_mass[block_valid].detach()
        repair = (repair_elem * repair_w).sum() / repair_w.sum().clamp_min(eps)
    else:
        repair = pred["repair_direction"].sum() * 0.0

    total = float(lambda_s) * sev + float(lambda_r) * repair
    zero = total.detach() * 0.0
    return total, {
        "pred_risk": sev_weighted.detach(),
        "pred_severity": sev_weighted.detach(),
        "pred_severity_calibration": sev_cal.detach(),
        "pred_severity_mean_match": sev_mean.detach(),
        "pred_repair": repair.detach(),
        "pred_repair_direction": repair.detach(),
        "pred_repair_magnitude": zero,
        "reject_mass_mean": block_reject_mass.mean().detach(),
    }


def soft_logit_alignment_loss(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    token_weight: torch.Tensor | None = None,
    distribution_temperature: float = 1.0,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure soft Target->Draft KL used for Stage2/3 correction training.

    No argmax, no hard first mismatch, no sampling.  token_weight can be the
    detached V10 severity_weight so positions with larger EAL sensitivity matter
    more while the entire objective remains distribution-derived.
    """
    if draft_logits.shape != target_logits.shape:
        raise ValueError("draft_logits and target_logits must have the same shape")
    if distribution_temperature <= 0:
        raise ValueError("distribution_temperature must be > 0")

    vocab = draft_logits.shape[-1]
    d = draft_logits.reshape(-1, vocab).float()
    t = target_logits.reshape(-1, vocab).float()
    valid = valid_mask.reshape(-1).bool()

    temp = float(distribution_temperature)
    p = torch.softmax(t / temp, dim=-1)
    log_p = torch.log(p.clamp_min(eps))
    log_q = F.log_softmax(d / temp, dim=-1)
    token_kl = (p * (log_p - log_q)).sum(dim=-1) * (temp * temp)

    if token_weight is None:
        w = valid.float()
    else:
        if token_weight.shape != valid_mask.shape:
            raise ValueError("token_weight must match valid_mask")
        w = token_weight.reshape(-1).float().detach() * valid.float()
    loss = (token_kl * w).sum() / w.sum().clamp_min(1.0)
    active_fraction = (w > 0).float().sum() / valid.float().sum().clamp_min(1.0)
    return loss, active_fraction
