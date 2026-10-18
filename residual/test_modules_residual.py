from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import Qwen3Config

from residual.config import ResidualConfig
from residual.losses import predictor_loss, soft_logit_alignment_loss
from residual.mismatch import exact_mismatch_targets
from residual.mismatch_predictor import MismatchPredictor
from residual.mole_linear import ResidualMoLE
from residual.router import ResidualRouter
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash.model_definitions_specmole import specmole_decoder_layer_forward
from residual.model_residual import ResidualDFlashModel, _AnchorBatch


class _ZeroAttention(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return torch.zeros_like(hidden_states), None


class _TinyDFlash(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab: int, layers: int, block_size: int):
        super().__init__()
        qcfg = Qwen3Config(
            hidden_size=hidden_size,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        self.config = SimpleNamespace(
            transformer_layer_config=qcfg,
            sample_from_anchor=False,
        )
        self.block_size = block_size
        self.target_layer_ids = list(range(layers))
        self.layers = torch.nn.ModuleList(
            [Qwen3DFlashDecoderLayer(qcfg, i) for i in range(layers)]
        )
        for layer in self.layers:
            layer.self_attn = _ZeroAttention()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab, bias=False)
        self.use_draft_vocab = False
        self.sliding_window_indices = []


def run_all() -> None:
    torch.manual_seed(0)
    A, B, V, H, D = 3, 8, 101, 32, 16
    target = torch.randn(A, B, V)
    draft = torch.randn(A, B, V)
    mask = torch.ones(A, B, dtype=torch.bool)
    mask[:, 0] = False
    lm = torch.randn(V, H)
    proj = torch.randn(H, D) / H**0.5

    m = exact_mismatch_targets(
        target,
        draft,
        lm,
        mask,
        block_size=B,
        direction_projector=proj,
        distribution_temperature=1.0,
    )
    assert torch.allclose(m.repair_magnitude, m.tv, atol=2e-5, rtol=2e-5)
    assert m.repair_direction_per_pos.shape == (A, B, D)
    assert torch.allclose(m.first_reject_distribution.sum(-1), torch.ones(A), atol=1e-5)

    predictor = MismatchPredictor(
        hidden_size=H,
        hidden_dim=32,
        direction_dim=D,
        block_size=B,
        max_target_layers=5,
        max_drafter_layers=5,
        predictor_depth=2,
    )
    prev_t = torch.randn(A, 5, H)
    prev_d = torch.randn(A, 5, H)
    cur_d = torch.randn(A, 4, H)
    history = predictor.prepare_history(prev_t, prev_d)
    preds = predictor.forward_prefixes_from_history(
        history, cur_d, prefix_lengths=(2, 3, 4)
    )
    for pred in preds:
        assert pred["severity"].shape == (A, B)
        assert pred["repair_direction"].shape == (A, D)
        assert pred["risk_latent"].shape == (A, B, 32)
        assert pred["repair_latent"].shape == (A, B, 32)
    for k in ("p2", "p3", "p4"):
        risk_ids = {id(p) for p in predictor.risk_branches[k].parameters()}
        repair_ids = {id(p) for p in predictor.repair_branches[k].parameters()}
        assert not (risk_ids & repair_ids)

    total = sum(predictor_loss(p, m)[0] for p in preds)
    total.backward()
    for k in ("p2", "p3", "p4"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.risk_branches[k].parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.repair_branches[k].parameters())

    # Stage2/3 correction training is soft-KL only.
    dlog = draft.clone().requires_grad_(True)
    kl, _ = soft_logit_alignment_loss(
        dlog, target, mask, token_weight=m.severity_weight
    )
    kl.backward()
    assert dlog.grad is not None and torch.isfinite(dlog.grad).all()

    repair_in = torch.randn(A, B, 32 + D)
    risk_in = torch.randn(A, B, 32 + 4)
    router = ResidualRouter(
        repair_in.shape[-1],
        risk_input_dim=risk_in.shape[-1],
        num_experts=8,
        top_k=2,
        routing_mode="soft",
    )
    w, _, g = router(repair_in, risk_in)
    assert torch.allclose(w.sum(-1, keepdim=True), g, atol=1e-5)
    router.set_routing_mode("topk", 0.7)
    w, _, g = router(repair_in, risk_in)
    assert (w > 0).sum(-1).max() <= 2

    mole = ResidualMoLE(H, num_experts=8, rank=4, alpha=8, shared_rank=8)
    x = torch.randn(1, A * B, H)
    probs = w / g.clamp_min(1e-9)
    y = mole(x, route_weights=probs, gate=g, valid_mask=mask)
    assert y.shape == x.shape
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)

    layer_cfg = Qwen3Config(
        hidden_size=H,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    layer = Qwen3DFlashDecoderLayer(layer_cfg, 0)
    layer.self_attn = _ZeroAttention()
    layer_input = torch.randn(1, A * B, H)
    base_out = layer(hidden_states=layer_input, target_hidden=layer_input)
    layer_mole = ResidualMoLE(H, num_experts=8, rank=4, alpha=8)
    corrected_out = specmole_decoder_layer_forward(
        layer,
        residual_mole=layer_mole,
        residual_control={"route_weights": probs, "gate": g, "valid_mask": mask},
        hidden_states=layer_input,
        target_hidden=layer_input,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )
    assert torch.allclose(corrected_out, base_out, atol=0, rtol=0)

    tiny = _TinyDFlash(H, V, 5, B)
    wrapper = ResidualDFlashModel(
        tiny,
        ResidualConfig(
            train_stage="mole",
            mismatch_hidden_dim=32,
            repair_position_dim=32,
            mismatch_direction_dim=D,
            predictor_depth=1,
            mole_num_experts=8,
            correction_layer_indices=(2, 3, 4),
            teacher_trajectory="reference_off_policy",
        ),
    )
    assert all(not hasattr(layer, "residual_mole") for layer in wrapper.base_model.layers)
    assert all(not p.requires_grad for p in wrapper.predictor.parameters())
    assert all(p.requires_grad for p in wrapper.routers.parameters())
    assert all(p.requires_grad for module in wrapper._mole_modules for p in module.parameters())

    anchor_batch = _AnchorBatch(
        noise=torch.randn(1, A * B, H),
        fc_output=torch.randn(1, 4, H),
        full_mask=None,
        sliding_mask=None,
        all_position_ids=torch.arange(A * B).unsqueeze(0),
        position_embeddings=None,
        anchor_positions=torch.arange(A),
        anchor_valid=torch.ones(A, dtype=torch.bool),
        target_layers=torch.randn(A, 5, H),
        targets=torch.randn(1, A * B, V),
        aligned_mask=mask.reshape(1, A * B).float(),
    )
    tiny_history = wrapper.predictor.prepare_history(
        torch.randn(A, 5, H), torch.randn(A, 5, H)
    )
    base_logits = wrapper._rollout(
        anchor_batch, correction=False, predictor_history=tiny_history
    )[0]
    zero_corrected_logits = wrapper._rollout(
        anchor_batch, correction=True, predictor_history=tiny_history
    )[0]
    assert torch.allclose(zero_corrected_logits, base_logits, atol=0, rtol=0)
    zero_corrected_logits.square().mean().backward()
    assert any(module.expert_B.grad is not None for module in wrapper._mole_modules)
    assert all(
        not parameter.requires_grad
        for decoder_layer in wrapper.base_model.layers
        for parameter in decoder_layer.mlp.parameters()
    )
    print("Residual V20.6 Dual-V10 soft-logit module tests: PASS")


if __name__ == "__main__":
    run_all()
