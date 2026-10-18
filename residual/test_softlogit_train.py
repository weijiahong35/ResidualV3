"""Standalone V20.6 soft-logit training tests; no transformers required."""
from __future__ import annotations

import torch

from residual.losses import predictor_loss, soft_logit_alignment_loss
from residual.mismatch import exact_mismatch_targets
from residual.mismatch_predictor import MismatchPredictor


def run_all() -> None:
    torch.manual_seed(0)
    a, b, v, h, d = 3, 8, 101, 32, 16
    target = torch.randn(a, b, v)
    draft = torch.randn(a, b, v)
    mask = torch.ones(a, b, dtype=torch.bool)
    mask[:, 0] = False
    lm = torch.randn(v, h)
    proj = torch.randn(h, d) / h**0.5

    t = exact_mismatch_targets(
        target,
        draft,
        lm,
        mask,
        block_size=b,
        direction_projector=proj,
        distribution_temperature=1.0,
    )
    assert t.severity.shape == (a, b)
    assert t.repair_direction.shape == (a, d)
    assert t.repair_direction_per_pos.shape == (a, b, d)
    assert torch.allclose(t.repair_magnitude, t.tv, atol=2e-5, rtol=2e-5)
    assert torch.allclose(t.first_reject_distribution.sum(-1), torch.ones(a), atol=1e-5)

    # Distribution identity should produce zero mismatch without any argmax path.
    z = exact_mismatch_targets(
        target,
        target.clone(),
        lm,
        mask,
        block_size=b,
        direction_projector=proj,
    )
    assert z.tv[mask].abs().max() < 1e-6
    assert z.severity[mask].abs().max() < 1e-6

    predictor = MismatchPredictor(
        hidden_size=h,
        hidden_dim=32,
        direction_dim=d,
        block_size=b,
        max_target_layers=5,
        max_drafter_layers=5,
        predictor_depth=2,
        attention_heads=4,
        ffn_expansion=4,
    )
    prev_t = torch.randn(a, 5, h)
    prev_d = torch.randn(a, 5, h)
    cur_d = torch.randn(a, 4, h)
    hist = predictor.prepare_history(prev_t, prev_d)
    preds = predictor.forward_prefixes_from_history(hist, cur_d, prefix_lengths=(2, 3, 4))

    total = 0.0
    for pred in preds:
        assert pred["severity"].shape == (a, b)
        assert pred["repair_direction"].shape == (a, d)
        assert pred["risk_latent"].shape == (a, b, 32)
        assert pred["repair_latent"].shape == (a, b, 32)
        loss, _ = predictor_loss(pred, t)
        total = total + loss

    # Every pK has two fully independent complete V10 branches.
    for k in ("p2", "p3", "p4"):
        risk_ids = {id(x) for x in predictor.risk_branches[k].parameters()}
        repair_ids = {id(x) for x in predictor.repair_branches[k].parameters()}
        assert not (risk_ids & repair_ids)

    total.backward()
    for k in ("p2", "p3", "p4"):
        assert any(x.grad is not None and x.grad.abs().sum() > 0 for x in predictor.risk_branches[k].parameters())
        assert any(x.grad is not None and x.grad.abs().sum() > 0 for x in predictor.repair_branches[k].parameters())

    dlog = draft.clone().requires_grad_(True)
    align, frac = soft_logit_alignment_loss(
        dlog,
        target,
        mask,
        token_weight=t.severity_weight,
        distribution_temperature=1.0,
    )
    assert 0.0 < float(frac) <= 1.0
    align.backward()
    assert dlog.grad is not None and torch.isfinite(dlog.grad).all()
    print("V20.6 soft-logit training tests: PASS")


if __name__ == "__main__":
    run_all()
