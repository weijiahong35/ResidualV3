"""Evaluation-only greedy semantics smoke test.

Training code MUST NOT import this module.  Greedy argmax equality is retained
only to document/check the final speculative-decoding evaluation protocol.
"""
from __future__ import annotations

import torch


def run_all() -> None:
    target_logits = torch.tensor([[[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]]])
    draft_logits = torch.tensor([[[0.0, 3.0, 1.0], [0.0, 2.0, 1.0]]])
    target_ids = target_logits.argmax(-1)
    draft_ids = draft_logits.argmax(-1)
    accepted = target_ids.eq(draft_ids)
    assert torch.equal(accepted, torch.tensor([[True, False]]))
    print("Evaluation-only greedy semantics: PASS")


if __name__ == "__main__":
    run_all()
