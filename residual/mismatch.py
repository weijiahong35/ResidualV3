from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MismatchTargets:
    """Pure logit-derived training targets.

    No argmax, sampling, top-p or top-k is used here.  The training teacher is
    the full Target/Draft distribution geometry, matching the V10 contract.
    """

    tv: torch.Tensor                    # [A,B]
    severity: torch.Tensor              # [A,B] = -log(overlap)
    repair_direction: torch.Tensor      # [A,D], soft first-reject pooled
    valid_mask: torch.Tensor            # [A,B]
    overlap: torch.Tensor               # [A,B] = sum_v min(p_v,q_v)
    reach_prob: torch.Tensor            # [A,B] = prod_{i<j} overlap_i
    first_reject_prob: torch.Tensor     # [A,B] = reach_j * (1-overlap_j)
    severity_weight: torch.Tensor       # [A,B] = suffix sum of survival-after-j
    expected_accepted_length: torch.Tensor  # [A]
    no_reject_prob: torch.Tensor        # [A,1]
    first_reject_distribution: torch.Tensor  # [A,B+1]
    repair_direction_per_pos: torch.Tensor   # [A,B,D]
    repair_magnitude: torch.Tensor      # [A,B] == TV, compatibility/diagnostics


def acceptance_structure(
    overlap: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft speculative structure from overlap probabilities.

    reach_j        = prod_{i<j} alpha_i
    first_reject_j = reach_j * (1-alpha_j)
    survival_j     = prod_{i<=j} alpha_i
    EAL            = sum_j survival_j
    severity_w_j   = sum_{m>=j} survival_m

    Invalid slots are neutral in prefix products and zero in outputs.
    """
    if overlap.shape != valid_mask.shape:
        raise ValueError(
            f"overlap/valid_mask shape mismatch: {overlap.shape} vs {valid_mask.shape}"
        )
    valid = valid_mask.bool()
    alpha = overlap.float().clamp(0.0, 1.0)
    alpha_eff = torch.where(valid, alpha, torch.ones_like(alpha))

    survival_after_raw = torch.cumprod(alpha_eff, dim=-1)
    reach_before_raw = torch.cat(
        [torch.ones_like(survival_after_raw[..., :1]), survival_after_raw[..., :-1]],
        dim=-1,
    )
    first_reject = reach_before_raw * (1.0 - alpha) * valid.float()
    survival_after = survival_after_raw * valid.float()
    severity_weight = torch.flip(
        torch.cumsum(torch.flip(survival_after, dims=[-1]), dim=-1), dims=[-1]
    ) * valid.float()
    reach_before = reach_before_raw * valid.float()
    eal = survival_after.sum(dim=-1)
    no_reject = survival_after_raw[..., -1:].clamp(0.0, 1.0)
    return reach_before, first_reject, severity_weight, eal, no_reject


def expected_accepted_length(
    overlap: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    return acceptance_structure(overlap, valid_mask)[3]


def mean_valid_block_eal(
    block_eal: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average precomputed EAL over blocks containing at least one valid slot."""
    if block_eal.ndim != 1:
        raise ValueError(f"Expected block_eal [A], got {tuple(block_eal.shape)}")
    if valid_mask.ndim != 2 or valid_mask.shape[0] != block_eal.shape[0]:
        raise ValueError(
            f"Expected valid_mask [A,B] aligned with block_eal, got "
            f"{tuple(valid_mask.shape)} vs {tuple(block_eal.shape)}"
        )
    block_valid = valid_mask.bool().any(dim=-1)
    num_valid = block_valid.sum()
    if block_valid.any():
        return block_eal[block_valid].mean(), num_valid
    return block_eal.sum() * 0.0, num_valid


def exact_mismatch_targets(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    block_size: int,
    direction_projector: torch.Tensor,
    sample_from_anchor: bool = False,
    distribution_temperature: float = 1.0,
    eps: float = 1e-6,
) -> MismatchTargets:
    """Build V10-style soft mismatch targets directly from raw logits.

    Training intentionally does NOT use greedy argmax or sampling.  With
    distribution_temperature=1, p/q are the ordinary full softmax distributions.

    Repair geometry follows V10:
      positive residual = [p-q]_+
      repair distribution per position = [p-q]_+ / TV
      block repair direction = soft-first-reject weighted pooled direction
    """
    del sample_from_anchor
    if distribution_temperature <= 0:
        raise ValueError("distribution_temperature must be > 0")
    if target_logits.shape != draft_logits.shape:
        raise ValueError(
            f"target/draft logits differ: {target_logits.shape} vs {draft_logits.shape}"
        )
    vocab = target_logits.shape[-1]
    if lm_head_weight.shape[0] != vocab:
        raise ValueError(f"lm_head vocab={lm_head_weight.shape[0]} but logits vocab={vocab}")

    if target_logits.ndim == 3 and target_logits.shape[0] == 1:
        n = target_logits.shape[1]
        if n % block_size:
            raise ValueError(f"flattened length {n} not divisible by block_size={block_size}")
        anchors = n // block_size
        target_logits = target_logits.reshape(anchors, block_size, vocab)
        draft_logits = draft_logits.reshape(anchors, block_size, vocab)
        loss_mask = loss_mask.reshape(anchors, block_size)
    elif target_logits.ndim != 3 or target_logits.shape[1] != block_size:
        raise ValueError("Expected [1,A*B,V] or [A,B,V] logits")

    valid = loss_mask.bool()
    temp = float(distribution_temperature)
    p = F.softmax(target_logits.float() / temp, dim=-1)
    q = F.softmax(draft_logits.float() / temp, dim=-1)
    residual = p - q
    tv = 0.5 * residual.abs().sum(dim=-1)
    overlap = torch.minimum(p, q).sum(dim=-1).clamp(0.0, 1.0)
    severity = -torch.log(overlap.clamp_min(eps))

    reach_prob, first_reject_prob, severity_weight, eal, no_reject = acceptance_structure(
        overlap, valid
    )
    first_reject_distribution = torch.cat([first_reject_prob, no_reject], dim=-1)
    first_reject_distribution = first_reject_distribution / first_reject_distribution.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)

    positive = residual.clamp_min(0.0)
    repair_magnitude = positive.sum(dim=-1)  # mathematically equal to TV
    e_positive = torch.matmul(positive, lm_head_weight.float())
    repair_hidden = e_positive / repair_magnitude.clamp_min(eps).unsqueeze(-1)
    repair_hidden = torch.where(
        (repair_magnitude > eps).unsqueeze(-1),
        repair_hidden,
        torch.zeros_like(repair_hidden),
    )
    low_per_pos = repair_hidden @ direction_projector.float()
    repair_direction_per_pos = F.normalize(low_per_pos, dim=-1, eps=eps)

    repair_w = first_reject_prob
    repair_denom = repair_w.sum(dim=-1, keepdim=True).clamp_min(eps)
    pooled = (repair_hidden * repair_w.unsqueeze(-1)).sum(dim=1) / repair_denom
    low = pooled @ direction_projector.float()
    repair_direction = F.normalize(low, dim=-1, eps=eps)

    return MismatchTargets(
        tv=tv,
        severity=severity,
        repair_direction=repair_direction,
        valid_mask=valid,
        overlap=overlap,
        reach_prob=reach_prob,
        first_reject_prob=first_reject_prob,
        severity_weight=severity_weight,
        expected_accepted_length=eal,
        no_reject_prob=no_reject,
        first_reject_distribution=first_reject_distribution,
        repair_direction_per_pos=repair_direction_per_pos,
        repair_magnitude=repair_magnitude,
    )


def sparse_union_residual_projection(
    p: torch.Tensor,
    q: torch.Tensor,
    lm_head_weight: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("p and q must be same-shape 1D probability tensors")
    k = min(topk, p.numel())
    ids = torch.unique(torch.cat([torch.topk(p, k).indices, torch.topk(q, k).indices]))
    return (p[ids] - q[ids]) @ lm_head_weight[ids].float()


def sparse_union_repair_projection(
    p: torch.Tensor,
    q: torch.Tensor,
    lm_head_weight: torch.Tensor,
    topk: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("p and q must be same-shape 1D probability tensors")
    k = min(topk, p.numel())
    ids = torch.unique(torch.cat([torch.topk(p, k).indices, torch.topk(q, k).indices]))
    positive = (p[ids] - q[ids]).clamp_min(0.0)
    mass = positive.sum()
    if float(mass) <= eps:
        return torch.zeros(
            lm_head_weight.shape[1], device=lm_head_weight.device, dtype=torch.float32
        )
    return (positive / mass) @ lm_head_weight[ids].float()


def detach_mismatch_targets(target: MismatchTargets) -> MismatchTargets:
    return MismatchTargets(**{
        name: getattr(target, name).detach()
        for name in MismatchTargets.__dataclass_fields__
    })
