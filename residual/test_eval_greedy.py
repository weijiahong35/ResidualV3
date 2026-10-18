"""Evaluation-only greedy semantics smoke test.

Training code MUST NOT import this module.  Greedy argmax equality is retained
only to document/check the final speculative-decoding evaluation protocol.
"""
from __future__ import annotations

import torch

from residual.evaluate_residual import offline_greedy_metrics
from residual.mismatch import mean_valid_block_eal
from residual.trainer_residual import _training_acceptance_metrics


def _logits(token_ids: torch.Tensor, vocab_size: int = 10) -> torch.Tensor:
    return torch.nn.functional.one_hot(token_ids, num_classes=vocab_size).float()


def run_all() -> None:
    target_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])

    # Test 1/3: slot0 differs, but all seven proposals match and fully accept.
    full_draft_ids = torch.tensor([[9, 1, 2, 3, 4, 5, 6, 7]])
    full = offline_greedy_metrics(
        _logits(target_ids),
        _logits(full_draft_ids).requires_grad_(True),
        block_size=8,
        sample_from_anchor=False,
        block_valid=torch.tensor([True]),
    )
    assert full["offline_greedy_eal"].item() == 7.0
    assert full["offline_greedy_accept_rate"].item() == 1.0
    assert full["offline_greedy_full_block_accept_rate"].item() == 1.0
    assert full["offline_num_valid_blocks"].item() == 1
    assert "offline_greedy_position_accept_p7" in full
    assert "offline_greedy_position_accept_p8" not in full
    assert all(not value.requires_grad for value in full.values())

    # Training aliases come from the exact same helper and inputs as validation.
    training = _training_acceptance_metrics(
        _logits(target_ids),
        _logits(full_draft_ids).requires_grad_(True),
        block_size=8,
        sample_from_anchor=False,
        block_valid=torch.tensor([True]),
    )
    assert training["accept_eal"].item() == full["offline_greedy_eal"].item()
    assert training["accept_num_valid_blocks"].item() == 1
    for pos in range(1, 8):
        assert training[f"accept_position_{pos}"].item() == full[
            f"offline_greedy_position_accept_p{pos}"
        ].item()
    assert all(not value.requires_grad for value in training.values())

    # Test 2: later equality cannot revive acceptance after proposal-position 3.
    reject_draft_ids = torch.tensor([[9, 1, 2, 8, 4, 5, 6, 7]])
    rejected = offline_greedy_metrics(
        _logits(target_ids),
        _logits(reject_draft_ids),
        block_size=8,
        sample_from_anchor=False,
        block_valid=torch.tensor([True]),
    )
    assert rejected["offline_greedy_eal"].item() == 2.0
    expected_positions = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for pos, expected in enumerate(expected_positions, start=1):
        assert rejected[f"offline_greedy_position_accept_p{pos}"].item() == expected

    # Test 4: an invalid second block contributes to neither numerator nor denominator.
    padded_target = target_ids.repeat(2, 1)
    padded_draft = torch.cat(
        [full_draft_ids, torch.full_like(full_draft_ids, 9)], dim=0
    )
    padded_target_logits = _logits(padded_target).reshape(1, -1, 10)
    padded_draft_logits = _logits(padded_draft).reshape(1, -1, 10)
    padded = offline_greedy_metrics(
        padded_target_logits,
        padded_draft_logits,
        block_size=8,
        sample_from_anchor=False,
        block_valid=torch.tensor([True, False]),
    )
    assert padded["offline_greedy_eal"].item() == 7.0
    assert padded["offline_num_valid_blocks"].item() == 1

    # Test 5: soft EAL aggregation excludes invalid padding blocks.
    valid_mask = torch.zeros(2, 8, dtype=torch.bool)
    valid_mask[0, 1:] = True
    soft_eal, soft_count = mean_valid_block_eal(
        torch.tensor([6.0, 0.0]), valid_mask
    )
    assert soft_eal.item() == 6.0
    assert soft_count.item() == 1

    # Test 6: no valid blocks returns explicit zero sentinels, never NaN/crash.
    empty = offline_greedy_metrics(
        padded_target_logits,
        padded_draft_logits,
        block_size=8,
        sample_from_anchor=False,
        block_valid=torch.tensor([False, False]),
    )
    assert empty["offline_num_valid_blocks"].item() == 0
    assert all(
        torch.isfinite(value) and value.item() == 0.0 for value in empty.values()
    )
    empty_soft_eal, empty_soft_count = mean_valid_block_eal(
        torch.tensor([0.0, 0.0]), torch.zeros(2, 8, dtype=torch.bool)
    )
    assert empty_soft_eal.item() == 0.0
    assert empty_soft_count.item() == 0
    print("Evaluation-only greedy semantics: PASS")


if __name__ == "__main__":
    run_all()
