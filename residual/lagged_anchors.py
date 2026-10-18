from __future__ import annotations

import torch


def select_lagged_anchor_pairs(
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
    num_pairs: int,
    block_size: int,
    *,
    round_stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select dense multi-anchor lag-1 training pairs.

    The regenerated OpenMath rows contain token sequences and ``loss_mask`` but do
    not contain runtime speculative-round boundaries.  Therefore this offline
    training path represents one previous round by a fixed token stride.  The
    default stride is ``block_size`` (one full DFlash block).  The function keeps
    the existing random multi-anchor spirit: current anchors are randomly sampled
    from all valid lagged pairs and then sorted for attention efficiency.

    Returns:
        prev_anchors: [A]
        current_anchors: [A]
        pair_valid: [A] bool; padded entries are False
    """
    if loss_mask.ndim != 2 or loss_mask.shape[0] != 1:
        raise ValueError(f"Expected loss_mask [1,T], got {tuple(loss_mask.shape)}")
    if document_ids.shape != loss_mask.shape:
        raise ValueError(
            f"document_ids must match loss_mask, got {tuple(document_ids.shape)} vs {tuple(loss_mask.shape)}"
        )
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    stride = block_size if round_stride is None else int(round_stride)
    if stride <= 0:
        raise ValueError("round_stride must be > 0")

    device = loss_mask.device
    total = loss_mask.shape[1]
    positions = torch.arange(total, device=device, dtype=torch.long)
    prev = positions - stride

    # Both previous and current blocks must fit completely inside a single packed
    # document. This is stricter than generic DFlash anchor sampling because a
    # lagged pair is intended to represent two consecutive round states.
    current_end = positions + block_size - 1
    prev_end = prev + block_size - 1
    in_range = (prev >= 0) & (current_end < total) & (prev_end < total)

    safe_prev = prev.clamp(min=0, max=max(total - 1, 0))
    safe_cur_end = current_end.clamp(min=0, max=max(total - 1, 0))
    safe_prev_end = prev_end.clamp(min=0, max=max(total - 1, 0))

    docs = document_ids[0]
    valid = in_range & loss_mask[0].bool() & loss_mask[0, safe_prev].bool()
    valid &= docs >= 0
    valid &= docs == docs[safe_prev]
    valid &= docs == docs[safe_cur_end]
    valid &= docs[safe_prev] == docs[safe_prev_end]

    candidates = positions[valid]
    out_prev = torch.zeros(num_pairs, dtype=torch.long, device=device)
    out_cur = torch.zeros(num_pairs, dtype=torch.long, device=device)
    pair_valid = torch.zeros(num_pairs, dtype=torch.bool, device=device)

    k = min(num_pairs, int(candidates.numel()))
    if k:
        perm = torch.randperm(candidates.numel(), device=device)[:k]
        cur = torch.sort(candidates[perm]).values
        out_cur[:k] = cur
        out_prev[:k] = cur - stride
        pair_valid[:k] = True
    return out_prev, out_cur, pair_valid
