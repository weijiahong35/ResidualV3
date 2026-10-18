from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.model_definitions_specmole import specmole_decoder_layer_forward
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks, select_anchors
from speculators.models.metrics import resolve_loss_config
from speculators.models.utils import conditional_torch_compile

from .config import ResidualConfig
from .diagnostics import predictor_target_diagnostics, progressive_prediction_diagnostics, routing_diagnostics
from .hidden_selector import select_hidden_layer_indices
from .lagged_anchors import select_lagged_anchor_pairs
from .losses import predictor_loss, soft_logit_alignment_loss
from .mismatch import (
    MismatchTargets,
    detach_mismatch_targets,
    exact_mismatch_targets,
    mean_valid_block_eal,
)
from .mismatch_predictor import MismatchPredictor
from .mole_linear import ResidualMoLE
from .router import ResidualRouter, route_consistency_loss, router_z_loss
from .state_cache import PreviousRoundState

logger = logging.getLogger(__name__)


@dataclass
class ResidualForwardOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor | float]
    logits: torch.Tensor
    targets: torch.Tensor
    routing_weights: torch.Tensor | None
    prediction: dict[str, torch.Tensor] | None = None
    predictions: list[dict[str, torch.Tensor]] | None = None
    mismatch: MismatchTargets | None = None
    previous_anchors: torch.Tensor | None = None
    current_anchors: torch.Tensor | None = None
    pair_valid: torch.Tensor | None = None


@dataclass
class _AnchorBatch:
    noise: torch.Tensor
    fc_output: torch.Tensor
    full_mask: object
    sliding_mask: object
    all_position_ids: torch.Tensor
    position_embeddings: object
    anchor_positions: torch.Tensor
    anchor_valid: torch.Tensor
    target_layers: torch.Tensor
    targets: torch.Tensor
    aligned_mask: torch.Tensor


class ResidualDFlashModel(nn.Module):
    """Source-aligned Risk/Repair Residual MoLE for DFlash.

    V20.6 keeps the real V10 Predictor information contract: previous-round
    Target/Draft states plus the current progressive Draft prefix.  The raw V10
    information stem is shared, but every p2/p3/p4 decision owns TWO complete,
    independent V10 fusion branches: one Bias/NLA branch and one Repair branch.

    Training supervision is purely logit-derived and soft (V10 overlap/NLA,
    soft first-reject, soft repair geometry).  Greedy argmax is reserved for
    runtime/evaluation.  Residual MoLE remains external after the frozen MLP.
    """

    def __init__(self, base_model: nn.Module, cfg: ResidualConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg
        self.block_size = int(base_model.block_size)
        self.hidden_size = int(base_model.config.transformer_layer_config.hidden_size)
        self.num_target_layers = len(base_model.target_layer_ids)
        self.num_draft_layers = len(base_model.layers)
        cfg.validate(self.num_draft_layers)
        self.correction_layer_indices = tuple(cfg.correction_layer_indices)
        requested = min(cfg.mismatch_num_hidden_layers, self.num_target_layers)
        self.target_hidden_indices = select_hidden_layer_indices(self.num_target_layers, requested)
        self.drafter_hidden_indices = list(range(self.num_draft_layers))
        self.gate_cap = 0.0 if cfg.train_stage == "mole" else 1.0
        if cfg.train_stage != "eval_base" and cfg.teacher_trajectory == "on_policy_replay":
            raise NotImplementedError(
                "teacher_trajectory='on_policy_replay' requires proposal-conditioned "
                "verifier records from the hidden-state backend; this checkout uses "
                "reference_off_policy Target/Draft logits for training."
            )
        if cfg.train_stage != "eval_base" and cfg.teacher_trajectory == "reference_off_policy":
            logger.info(
                "Training teacher is reference-trajectory soft Target/Draft logit geometry; "
                "no greedy argmax labels or sampling are used in training."
            )
        if cfg.train_stage != "eval_base" and getattr(base_model, "use_draft_vocab", False):
            logger.warning(
                "Verifier logits are restricted to the reduced Draft vocabulary. Target "
                "probabilities are conditional on that support; exact full-policy V20 labels "
                "require backend top-k token IDs/probabilities."
            )

        self.predictor: MismatchPredictor | None = None
        if cfg.train_stage != "eval_base":
            self.predictor = MismatchPredictor(
                hidden_size=self.hidden_size,
                hidden_dim=cfg.mismatch_hidden_dim,
                direction_dim=cfg.mismatch_direction_dim,
                block_size=self.block_size,
                max_target_layers=max(len(self.target_hidden_indices), 1),
                max_drafter_layers=self.num_draft_layers,
                predictor_depth=cfg.predictor_depth,
                attention_heads=cfg.predictor_attention_heads,
                ffn_expansion=cfg.predictor_ffn_expansion,
            )
            self.predictor.to(dtype=base_model.lm_head.weight.dtype)

        self.routers = nn.ModuleDict()
        # SpecMoLE modules are owned by this wrapper, not injected into the base
        # DFlash decoder layers.  This keeps ordinary DFlash architecture/state
        # dicts/source completely independent from V20.
        self.residual_mole = nn.ModuleDict()
        self._mole_modules: list[ResidualMoLE] = []
        self._mole_modules_by_layer: dict[int, list[ResidualMoLE]] = {}
        if cfg.train_stage in {"mole", "joint", "eval"}:
            for idx in self.correction_layer_indices:
                self.routers[f"p{idx}"] = ResidualRouter(
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
                    routing_mode="soft" if cfg.train_stage == "mole" else "topk",
                ).to(dtype=base_model.lm_head.weight.dtype)
                module = ResidualMoLE(
                    self.hidden_size,
                    num_experts=cfg.mole_num_experts,
                    rank=cfg.mole_rank,
                    alpha=cfg.mole_alpha,
                    shared_rank=cfg.mole_shared_rank,
                    dropout=cfg.mole_dropout,
                    max_relative_update=cfg.mole_max_relative_update,
                ).to(dtype=base_model.lm_head.weight.dtype)
                self.residual_mole[str(idx)] = module
                self._mole_modules.append(module)
                self._mole_modules_by_layer[idx] = [module]

        # Kept as empty compatibility containers for old evaluation/logging code.
        self.dense_gates = nn.ModuleDict()
        self._dense_modules: list[nn.Module] = []
        self._dense_modules_by_layer: dict[int, list[nn.Module]] = {}
        self.stage2_router_teacher = nn.ModuleDict()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(cfg.residual_seed)
        direction_projector = torch.randn(
            self.hidden_size, cfg.mismatch_direction_dim, generator=generator
        ) / self.hidden_size**0.5
        self.register_buffer("direction_projector", direction_projector, persistent=True)
        self.freeze_for_stage(cfg.train_stage)

    @property
    def target_layer_ids(self):
        return self.base_model.target_layer_ids

    @property
    def config(self):
        return self.base_model.config

    @property
    def round_stride(self) -> int:
        # V10 information contract: previous round is represented by one full
        # DFlash block stride in the offline packed training sequence.
        return self.block_size

    def _routing_prefix_lengths(self) -> list[int]:
        return list(self.correction_layer_indices)

    def set_training_progress(self, fraction: float) -> None:
        if self.cfg.train_stage == "mole":
            warmup = max(self.cfg.gate_warmup_ratio, 1e-9)
            self.gate_cap = min(max(float(fraction) / warmup, 0.0), 1.0)
        else:
            self.gate_cap = 1.0

    def snapshot_stage2_routers(self) -> None:
        if self.cfg.train_stage != "joint" or not self.routers:
            return
        self.stage2_router_teacher = copy.deepcopy(self.routers)
        self.stage2_router_teacher.eval()
        for parameter in self.stage2_router_teacher.parameters():
            parameter.requires_grad_(False)

    def freeze_for_stage(self, stage: str) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        if self.predictor is not None:
            for parameter in self.predictor.parameters():
                parameter.requires_grad_(False)
        for parameter in self.routers.parameters():
            parameter.requires_grad_(False)
        if stage == "predictor":
            if self.predictor is None:
                raise RuntimeError("Stage 1 requires Predictors")
            for parameter in self.predictor.parameters():
                parameter.requires_grad_(True)
        elif stage == "mole":
            for parameter in self.routers.parameters():
                parameter.requires_grad_(True)
            for module in self._mole_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        elif stage == "joint":
            if self.predictor is None:
                raise RuntimeError("Stage 3 requires Predictors")
            for parameter in self.predictor.parameters():
                parameter.requires_grad_(True)
            for parameter in self.routers.parameters():
                parameter.requires_grad_(True)
            for module in self._mole_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        elif stage in {"eval", "eval_base"}:
            pass
        else:
            raise ValueError(
                f"V20 supports predictor/mole/joint/eval/eval_base; got {stage!r}"
            )

    def trainable_parameter_groups(self, base_lr: float) -> list[dict]:
        groups: list[dict] = []
        if self.predictor is not None:
            params = [p for p in self.predictor.parameters() if p.requires_grad]
            if params:
                scale = self.cfg.predictor_lr_scale if self.cfg.train_stage == "joint" else 1.0
                groups.append({"params": params, "lr": base_lr * scale, "name": "predictor"})
        params = [p for p in self.routers.parameters() if p.requires_grad]
        if params:
            scale = (
                self.cfg.router_lr_scale_stage3
                if self.cfg.train_stage == "joint"
                else self.cfg.router_lr_scale_stage2
            )
            groups.append({"params": params, "lr": base_lr * scale, "name": "router"})
        params = [p for module in self._mole_modules for p in module.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": base_lr, "name": "residual_mole"})
        if not groups:
            raise RuntimeError("No trainable V20 parameters")
        return groups

    def _policy_features(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [
                self.cfg.sampling_temperature,
                float(self.cfg.sampling_top_k) / 100.0,
                self.cfg.sampling_top_p,
                self.cfg.distribution_temperature,
            ],
            device=device,
            dtype=dtype,
        )

    def _target_layers_at_anchors(self, hidden_states: torch.Tensor,
                                  anchor_positions: torch.Tensor) -> torch.Tensor:
        expected = self.num_target_layers * self.hidden_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(f"Expected target feature dim {expected}, got {hidden_states.shape[-1]}")
        split = hidden_states.reshape(1, hidden_states.shape[1], self.num_target_layers, self.hidden_size)
        return split[0, anchor_positions][:, self.target_hidden_indices]

    def _prepare_for_anchors(self, hidden_states, input_ids, loss_mask,
                             verifier_last_hidden_states, document_ids, position_ids,
                             anchor_positions, anchor_valid) -> _AnchorBatch:
        base = self.base_model
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(total_seq_len, device=device).unsqueeze(0)
        full_mask = base._create_attention_mask(
            document_ids, total_seq_len, anchor_positions, device, sliding_window=None
        ) if base.uses_full_attn else None
        sliding_mask = base._create_attention_mask(
            document_ids, total_seq_len, anchor_positions, device,
            sliding_window=base.sliding_window,
            sliding_window_non_causal=base.sliding_window_non_causal,
        ) if base.uses_sliding_window_attn else None
        mask_ids = torch.full(
            (1, anchor_positions.numel() * self.block_size), base.mask_token_id,
            dtype=torch.long, device=device,
        )
        mask_ids[:, ::self.block_size] = input_ids[:, anchor_positions]
        noise = base.embed_tokens(mask_ids)
        fc_output = base.hidden_norm(base.fc(hidden_states))
        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        all_position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        position_embeddings = base.rotary_emb(hidden_states, all_position_ids)
        anchored_idx = get_base_indices_for_anchored_blocks(anchor_positions, self.block_size)
        with torch.no_grad():
            if anchored_idx.numel() < total_seq_len:
                target_idx = anchored_idx if base.config.sample_from_anchor else (anchored_idx - 1) % total_seq_len
                targets = base.verifier_lm_head(base.verifier_norm(verifier_last_hidden_states[:, target_idx]))
            else:
                target_logits = base.verifier_lm_head(base.verifier_norm(verifier_last_hidden_states))
                if not base.config.sample_from_anchor:
                    target_logits = torch.roll(target_logits, 1, dims=1)
                targets = target_logits[:, anchored_idx]
        aligned = loss_mask[:, anchored_idx].clone()
        aligned = aligned * anchor_valid.repeat_interleave(self.block_size).unsqueeze(0).to(aligned.dtype)
        if not base.config.sample_from_anchor:
            aligned[:, ::self.block_size] = 0
        return _AnchorBatch(
            noise, fc_output, full_mask, sliding_mask, all_position_ids,
            position_embeddings, anchor_positions, anchor_valid,
            self._target_layers_at_anchors(hidden_states, anchor_positions),
            targets, aligned,
        )

    def _prepare_runtime_anchor(self, hidden_states: torch.Tensor, input_ids: torch.Tensor,
                                position_ids: torch.Tensor | None = None) -> _AnchorBatch:
        if hidden_states.shape[0] != 1 or input_ids.shape[0] != 1:
            raise ValueError("V20 reference runtime currently supports batch size 1")
        device = hidden_states.device
        context_length = hidden_states.shape[1]
        if input_ids.shape[1] != context_length + 1:
            raise ValueError(
                "Runtime DFlash requires Target context states [0:P] and input_ids "
                f"[0:P+1] ending in the Target-sampled anchor; got hidden length "
                f"{context_length}, input length {input_ids.shape[1]}"
            )
        if position_ids is None:
            context_position_ids = torch.arange(
                context_length, device=device
            ).unsqueeze(0)
            anchor_position = context_length
        else:
            if position_ids.shape != (1, context_length + 1):
                raise ValueError(
                    "Runtime position_ids must include context plus anchor; got "
                    f"{tuple(position_ids.shape)}"
                )
            context_position_ids = position_ids[:, :-1]
            anchor_position = int(position_ids[0, -1].item())
        anchors = torch.tensor([anchor_position], device=device, dtype=torch.long)
        valid_anchor = torch.ones(1, device=device, dtype=torch.bool)
        # document_ids also contains the synthetic anchor so mask construction
        # can resolve its document, while total_seq_len remains the number of
        # real Target context K/V rows.
        document_ids = torch.zeros(
            (1, context_length + 1), device=device, dtype=torch.long
        )
        # Runtime has no future verifier rows; construct the non-teacher tensors directly.
        base = self.base_model
        full = base._create_attention_mask(
            document_ids, context_length, anchors, device, sliding_window=None
        ) if base.uses_full_attn else None
        sliding = base._create_attention_mask(
            document_ids, context_length, anchors, device,
            sliding_window=base.sliding_window,
            sliding_window_non_causal=base.sliding_window_non_causal,
        ) if base.uses_sliding_window_attn else None
        ids = torch.full((1, self.block_size), base.mask_token_id, device=device, dtype=torch.long)
        ids[:, 0] = input_ids[:, -1]
        noise = base.embed_tokens(ids)
        fc_output = base.hidden_norm(base.fc(hidden_states))
        anchor_pos_tensor = torch.tensor(
            [anchor_position], device=device, dtype=torch.long
        )
        mask_pos = get_base_indices_for_anchored_blocks(
            anchor_pos_tensor, self.block_size
        )
        all_pos = torch.cat([context_position_ids, mask_pos.unsqueeze(0)], dim=1)
        pos_emb = base.rotary_emb(hidden_states, all_pos)
        valid = torch.ones((1, self.block_size), device=device, dtype=hidden_states.dtype)
        if not base.config.sample_from_anchor:
            valid[:, 0] = 0
        # The caller replaces this placeholder with the current anchor's Target
        # state after the verifier block has run. It is not used for this round.
        context_tail = torch.tensor(
            [context_length - 1], device=device, dtype=torch.long
        )
        return _AnchorBatch(
            noise, fc_output, full, sliding, all_pos, pos_emb, anchors, valid_anchor,
            self._target_layers_at_anchors(hidden_states, context_tail),
            torch.empty((1, self.block_size, 0), device=device, dtype=hidden_states.dtype),
            valid,
        )

    def runtime_target_state(
        self, hidden_states: torch.Tensor, token_index: int = -1
    ) -> torch.Tensor:
        """Select Predictor Target history from concatenated runtime aux states."""
        length = hidden_states.shape[1]
        index = token_index if token_index >= 0 else length + token_index
        if index < 0 or index >= length:
            raise IndexError(
                f"runtime target-state index {token_index} outside length {length}"
            )
        anchor = torch.tensor([index], device=hidden_states.device, dtype=torch.long)
        return self._target_layers_at_anchors(hidden_states, anchor).detach()

    def _router_inputs(self, pred: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Repair chooses expert identity from its own full V10 predictor only.
        repair_latent = pred["repair_latent"]
        repair = torch.cat(
            [
                repair_latent,
                pred["repair_direction_per_pos"].to(repair_latent.dtype),
            ],
            dim=-1,
        )

        # Bias/Risk controls correction strength using only continuous signals
        # derived from its V10 NLA prediction.  No greedy acceptance appears.
        risk_latent = pred["risk_latent"]
        physical = torch.stack(
            [
                pred["severity"],
                pred["tv"],
                pred["first_reject"],
                pred["reach_prob"],
            ],
            dim=-1,
        ).to(risk_latent.dtype)
        return repair, torch.cat([risk_latent, physical], dim=-1)

    @conditional_torch_compile
    def _layer_compiled(self, x: torch.Tensor, fc_output: torch.Tensor,
                        full_mask: object, sliding_mask: object,
                        all_position_ids: torch.Tensor, position_embeddings: object,
                        layer_idx: int, residual_control: dict | None = None,
                        **kwargs) -> torch.Tensor:
        layer = self.base_model.layers[layer_idx]
        attention_mask = (
            sliding_mask
            if layer_idx in self.base_model.sliding_window_indices
            else full_mask
        )
        if residual_control is None:
            # Exact ordinary DFlash path. No SpecMoLE argument reaches the base
            # decoder layer and model_definitions.py stays upstream-compatible.
            return layer(
                hidden_states=x, target_hidden=fc_output,
                attention_mask=attention_mask,
                position_ids=all_position_ids, use_cache=False,
                position_embeddings=position_embeddings, **kwargs,
            )

        key = str(layer_idx)
        if key not in self.residual_mole:
            raise RuntimeError(
                f"SpecMoLE control supplied for layer {layer_idx}, but no correction module exists"
            )
        return specmole_decoder_layer_forward(
            layer,
            residual_mole=self.residual_mole[key],
            residual_control=residual_control,
            hidden_states=x,
            target_hidden=fc_output,
            attention_mask=attention_mask,
            position_ids=all_position_ids,
            use_cache=False,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    def _layer(self, batch: _AnchorBatch, x: torch.Tensor, layer_idx: int,
               residual_control: dict | None = None, **kwargs) -> torch.Tensor:
        return self._layer_compiled(
            x, batch.fc_output, batch.full_mask, batch.sliding_mask,
            batch.all_position_ids, batch.position_embeddings, layer_idx,
            residual_control, **kwargs,
        )

    @staticmethod
    def _apply_prediction_valid_mask(
        pred: dict[str, torch.Tensor], valid_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Neutralize invalid DFlash slots under the SOFT V10 prediction view."""
        valid = valid_mask.bool()
        severity = torch.where(valid, pred["severity"], torch.zeros_like(pred["severity"]))
        overlap = torch.exp(-severity.float()).clamp(1.0e-6, 1.0).to(severity.dtype)
        overlap = torch.where(valid, overlap, torch.ones_like(overlap))

        survival_raw = torch.cumprod(overlap, dim=-1)
        survival = survival_raw * valid.float()
        reach_raw = torch.cat(
            [torch.ones_like(overlap[:, :1]), torch.cumprod(overlap[:, :-1], dim=-1)],
            dim=-1,
        )
        reach = reach_raw * valid.float()
        first_reject = reach_raw * (1.0 - overlap) * valid.float()
        no_reject = survival_raw[:, -1:]

        pred["severity"] = severity
        pred["distribution_nla"] = severity
        pred["runtime_nla"] = severity
        pred["distribution_overlap"] = overlap
        pred["runtime_acceptance"] = overlap
        pred["tv"] = 1.0 - overlap
        pred["survival"] = survival
        pred["reach_prob"] = reach
        pred["first_reject"] = first_reject
        pred["no_reject"] = no_reject
        pred["first_reject_with_no_reject"] = torch.cat([first_reject, no_reject], dim=-1)
        pred["expected_accepted_length"] = survival.sum(dim=-1)
        pred["repair_magnitude"] = pred["tv"]  # compatibility only; not a Repair head
        return pred

    def _rollout(self, batch: _AnchorBatch, *, correction: bool,
                 collect_predictions: bool = True,
                 predictor_history: dict[str, torch.Tensor] | None = None, **kwargs):
        """Progressive rollout with V10-style fused D1+D2 probe.

        D1/D2 never carry residual correction, so they are always executed in the
        same compiled graph as V10.  Dynamic Predictor/Router decisions begin only
        at D3.  This is important for DFlash FlexAttention: compiling D1 and D2 as
        separate large anchored-attention calls can fall back to a very large
        dense attention-score tensor at high anchor counts.

        The returned state stack is anchor-level [A,L,H], which is exactly the V10
        lagged Predictor history contract; no [A,B,L,H] hidden-state stack is kept.
        """
        if collect_predictions and self.predictor is None:
            raise RuntimeError("Predictor is unavailable")

        # Pure Base runtime/evaluation can use the fully fused V10 path.
        if not correction and not collect_predictions:
            logits, anchor_states = self._base_rollout_v10_fused(batch, **kwargs)
            return logits, anchor_states, [], [], []

        x, d1, d2 = self._run_base_probe_layers_compiled(
            batch.noise,
            batch.fc_output,
            batch.full_mask,
            batch.sliding_mask,
            batch.all_position_ids,
            batch.position_embeddings,
            **kwargs,
        )
        anchor_states: list[torch.Tensor] = [d1, d2]
        predictions: list[dict[str, torch.Tensor]] = []
        effective_routes: list[torch.Tensor] = []
        route_logits: list[torch.Tensor] = []
        valid = batch.aligned_mask.reshape(-1, self.block_size).bool()

        # D3..DN are progressive because each corrected layer changes the prefix
        # available to the next Predictor.  One tail layer at a time mirrors the
        # V10 routed-tail execution while preserving the external SpecMoLE path.
        for layer_idx in range(2, self.num_draft_layers):
            control = None
            if layer_idx in self.correction_layer_indices and collect_predictions:
                if predictor_history is None:
                    raise RuntimeError(
                        "Dual-V10 Predictor requires previous-round history"
                    )
                current_prefix = torch.stack(anchor_states, dim=1)
                if self.cfg.train_stage == "mole":
                    with torch.no_grad():
                        pred = self.predictor.forward_from_history(
                            predictor_history, current_prefix.detach()
                        )
                else:
                    pred = self.predictor.forward_from_history(
                        predictor_history, current_prefix.detach()
                    )
                pred = self._apply_prediction_valid_mask(pred, valid)

                if correction:
                    repair_in, risk_in = self._router_inputs(pred)
                    if self.cfg.train_stage == "mole":
                        repair_in, risk_in = repair_in.detach(), risk_in.detach()
                    effective, logits_r, gate = self.routers[f"p{layer_idx}"](
                        repair_in, risk_in
                    )
                    gate = gate * valid.unsqueeze(-1).to(gate.dtype)
                    probs = effective / gate.clamp_min(1e-9)
                    probs = torch.where(
                        gate > 0, probs, torch.zeros_like(probs)
                    )
                    control = {
                        "route_weights": probs,
                        "gate": gate,
                        "valid_mask": valid,
                        "gate_cap": self.gate_cap,
                    }
                    pred["route_gate"] = gate * self.gate_cap
                    pred["route_logits"] = logits_r
                    teacher_key = f"p{layer_idx}"
                    if (
                        self.cfg.train_stage == "joint"
                        and teacher_key in self.stage2_router_teacher
                    ):
                        with torch.no_grad():
                            _, teacher_logits, _ = self.stage2_router_teacher[teacher_key](
                                repair_in.detach(), risk_in.detach()
                            )
                        pred["stage2_route_logits"] = teacher_logits
                    effective_routes.append(probs * gate * self.gate_cap)
                    route_logits.append(logits_r)
                predictions.append(pred)

            x = self._layer(batch, x, layer_idx, control, **kwargs)
            block = x.reshape(-1, self.block_size, self.hidden_size)
            anchor_states.append(block[:, 0, :])

        logits = self.base_model.lm_head(self.base_model.norm(x))
        return (
            logits,
            torch.stack(anchor_states, dim=1),
            predictions,
            effective_routes,
            route_logits,
        )

    @conditional_torch_compile
    def _run_base_probe_layers_compiled(
        self,
        noise: torch.Tensor,
        fc_output: torch.Tensor,
        full_mask: object,
        sliding_mask: object,
        all_position_ids: torch.Tensor,
        position_embeddings: object,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """V10 fused Base-DFlash probe: run D1/D2 in one compiled graph.

        This path is used only when no Residual-MoLE correction is applied.
        Keeping the two probe layers inside one compiled function preserves the
        upstream/V10 FlexAttention lowering and avoids per-layer dense-attention
        fallback at large anchored sequence lengths.
        """
        base = self.base_model
        x = noise

        layer_idx = 0
        layer = base.layers[layer_idx]
        x = layer(
            hidden_states=x,
            target_hidden=fc_output,
            attention_mask=(
                sliding_mask if layer_idx in base.sliding_window_indices else full_mask
            ),
            position_ids=all_position_ids,
            use_cache=False,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        block = x.reshape(-1, self.block_size, self.hidden_size)
        d1 = block[:, 0, :]

        layer_idx = 1
        layer = base.layers[layer_idx]
        x = layer(
            hidden_states=x,
            target_hidden=fc_output,
            attention_mask=(
                sliding_mask if layer_idx in base.sliding_window_indices else full_mask
            ),
            position_ids=all_position_ids,
            use_cache=False,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        block = x.reshape(-1, self.block_size, self.hidden_size)
        d2 = block[:, 0, :]
        return x, d1, d2

    @conditional_torch_compile
    def _run_base_tail_layers_compiled(
        self,
        x: torch.Tensor,
        fc_output: torch.Tensor,
        full_mask: object,
        sliding_mask: object,
        all_position_ids: torch.Tensor,
        position_embeddings: object,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact V10-style fused Base tail: compile only D3..DN.

        The final norm/lm_head deliberately remain outside this compiled graph,
        matching V10.  The purpose of this graph is only to keep DFlash
        FlexAttention fused while collecting anchor-level D3..DN states.
        """
        base = self.base_model
        layer_states: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(base.layers[2:], start=2):
            x = layer(
                hidden_states=x,
                target_hidden=fc_output,
                attention_mask=(
                    sliding_mask
                    if layer_idx in base.sliding_window_indices
                    else full_mask
                ),
                position_ids=all_position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            block = x.reshape(-1, self.block_size, self.hidden_size)
            layer_states.append(block[:, 0, :])
        tail = torch.stack(layer_states, dim=1)
        return x, tail

    def _base_rollout_v10_fused(
        self,
        batch: _AnchorBatch,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uncorrected Base DFlash rollout using the exact V10 fused layout.

        D1+D2 are one compiled graph; D3..DN are another compiled graph; the
        final norm/lm_head is outside both.  Only anchor-level Draft states
        [A,L,H] are retained for the lagged Predictor contract.
        """
        x, d1, d2 = self._run_base_probe_layers_compiled(
            batch.noise,
            batch.fc_output,
            batch.full_mask,
            batch.sliding_mask,
            batch.all_position_ids,
            batch.position_embeddings,
            **kwargs,
        )
        x, tail = self._run_base_tail_layers_compiled(
            x,
            batch.fc_output,
            batch.full_mask,
            batch.sliding_mask,
            batch.all_position_ids,
            batch.position_embeddings,
            **kwargs,
        )
        logits = self.base_model.lm_head(self.base_model.norm(x))
        drafter_layers = torch.cat(
            [d1.unsqueeze(1), d2.unsqueeze(1), tail], dim=1
        )
        return logits, drafter_layers

    def _previous_state(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        prev_anchors: torch.Tensor,
        pair_valid: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V10 information set: previous Target stack + complete base Draft stack."""
        prev_target = self._target_layers_at_anchors(hidden_states, prev_anchors)
        prev_batch = self._prepare_for_anchors(
            hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
            document_ids, position_ids, prev_anchors, pair_valid,
        )
        with torch.no_grad():
            _, prev_drafter = self._base_rollout_v10_fused(prev_batch, **kwargs)
        return prev_target.detach(), prev_drafter.detach()

    def _mismatch(self, batch: _AnchorBatch, logits: torch.Tensor) -> MismatchTargets:
        return exact_mismatch_targets(
            batch.targets,
            logits,
            self.base_model.lm_head.weight,
            batch.aligned_mask,
            block_size=self.block_size,
            direction_projector=self.direction_projector,
            sample_from_anchor=self.base_model.config.sample_from_anchor,
            distribution_temperature=self.cfg.distribution_temperature,
        )

    def _predictor_objective(
        self,
        predictions: list[dict[str, torch.Tensor]],
        target: MismatchTargets,
        prefix: str = "",
    ):
        losses: list[torch.Tensor] = []
        metrics: dict[str, torch.Tensor] = {}
        for pred in predictions:
            loss, values = predictor_loss(
                pred,
                target,
                lambda_s=self.cfg.mismatch_lambda_severity,
                lambda_r=self.cfg.mismatch_lambda_repair,
                severity_calibration_weight=self.cfg.severity_calibration_weight,
                severity_mean_weight=self.cfg.severity_mean_weight,
            )
            losses.append(loss)
            idx = int(pred["prefix_layers"].item())
            metrics.update(
                {f"{prefix}{key}_p{idx}": value for key, value in values.items()}
            )
            metrics.update(
                {
                    f"{prefix}{key}": value
                    for key, value in predictor_target_diagnostics(pred, target).items()
                }
            )
        total = torch.stack(losses).mean()
        metrics[f"{prefix}pred_risk"] = torch.stack(
            [
                metrics[f"{prefix}pred_risk_p{int(p['prefix_layers'].item())}"]
                for p in predictions
            ]
        ).mean()
        metrics[f"{prefix}pred_repair"] = torch.stack(
            [
                metrics[f"{prefix}pred_repair_p{int(p['prefix_layers'].item())}"]
                for p in predictions
            ]
        ).mean()
        for key in (
            "pred_severity",
            "pred_severity_calibration",
            "pred_severity_mean_match",
        ):
            metrics[f"{prefix}{key}"] = torch.stack(
                [
                    metrics[f"{prefix}{key}_p{int(p['prefix_layers'].item())}"]
                    for p in predictions
                ]
            ).mean()
        valid = target.valid_mask.bool()
        if valid.any():
            metrics[f"{prefix}tv_mean"] = target.tv[valid].mean().detach()
            metrics[f"{prefix}overlap_mean"] = target.overlap[valid].mean().detach()
        metrics[f"{prefix}reject_mass_mean"] = (
            target.first_reject_prob.float().sum(dim=-1).mean().detach()
        )
        return total, metrics

    def _correction_objective(
        self,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        target: MismatchTargets,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Stage2/3 correction objective from raw Target/Draft logits only.

        We optimize soft Target->Draft KL on all valid positions, weighted by
        the detached V10 EAL-sensitivity weights.  No argmax/AUF is used during
        training; greedy verification remains an evaluation-only protocol.
        """
        soft_kl, active_fraction = soft_logit_alignment_loss(
            logits,
            target_logits,
            target.valid_mask,
            token_weight=target.severity_weight,
            distribution_temperature=self.cfg.distribution_temperature,
        )
        total = self.cfg.lambda_soft_alignment * soft_kl
        return total, {
            "soft_alignment_kl": soft_kl.detach(),
            "soft_weight_fraction": active_fraction.detach(),
        }

    def _correction_metrics(self) -> dict[str, torch.Tensor]:
        ratios = [m.last_update_ratio for m in self._mole_modules if m.last_update_ratio is not None]
        gates = [m.last_gate_mean for m in self._mole_modules if m.last_gate_mean is not None]
        output_cosines = [
            m.last_expert_output_cosine
            for m in self._mole_modules
            if m.last_expert_output_cosine is not None
        ]
        out: dict[str, torch.Tensor] = {}
        if ratios:
            out["mole_update_ratio"] = torch.stack(ratios).mean()
        if gates:
            out["mole_gate_mean"] = torch.stack(gates).mean()
        if output_cosines:
            out["expert_output_cosine"] = torch.stack(output_cosines).mean()
        if self._mole_modules:
            weight_cosines = [m.expert_weight_cosines() for m in self._mole_modules]
            out["expert_A_weight_cosine"] = torch.stack([x[0] for x in weight_cosines]).mean()
            out["expert_B_weight_cosine"] = torch.stack([x[1] for x in weight_cosines]).mean()
        return out

    def forward_residual(self, hidden_states: torch.Tensor, input_ids: torch.Tensor,
                         loss_mask: torch.Tensor, verifier_last_hidden_states: torch.Tensor,
                         document_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
                         *, max_anchors: int = 3072, gamma: float = 4.0, **kwargs) -> ResidualForwardOutput:
        if self.cfg.train_stage == "eval_base":
            anchors, anchor_valid = select_anchors(loss_mask, max_anchors, self.block_size)
            batch = self._prepare_for_anchors(
                hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
                document_ids, position_ids, anchors, anchor_valid,
            )
            logits, _ = self._base_rollout_v10_fused(batch, **kwargs)
            mismatch = self._mismatch(batch, logits)
            loss, metrics = compute_metrics(
                logits, batch.targets, batch.aligned_mask, self.block_size, gamma=gamma,
                loss_config=resolve_loss_config(self.cfg.accept_loss),
                per_position_loss_weight="dpace", dpace_alpha=0.0,
                sample_from_anchor=self.base_model.config.sample_from_anchor,
            )
            metrics["loss"] = loss.detach()
            return ResidualForwardOutput(
                loss, metrics, logits, batch.targets, None, mismatch=mismatch,
                current_anchors=anchors, pair_valid=anchor_valid,
            )

        prev_anchors, anchors, anchor_valid = select_lagged_anchor_pairs(
            loss_mask, document_ids, max_anchors, self.block_size,
            round_stride=self.round_stride,
        )
        batch = self._prepare_for_anchors(
            hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
            document_ids, position_ids, anchors, anchor_valid,
        )
        prev_target, prev_drafter = self._previous_state(
            hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
            document_ids, position_ids, prev_anchors, anchor_valid, **kwargs
        )
        if self.predictor is None:
            raise RuntimeError("Dual-V10 training requires Predictor")
        predictor_history = self.predictor.prepare_history(prev_target, prev_drafter)

        # Base rollout supplies Stage1 supervision and Stage3 base-trajectory records.
        need_base = self.cfg.train_stage in {"predictor", "joint", "eval"}
        base_logits = base_states = base_preds = base_target = None
        if need_base:
            # Base/reference trajectory uses the original V10 fused DFlash path.
            # Predictor supervision is evaluated afterwards from anchor-level
            # D1..D5 states, rather than interleaving a Python/compiled call at
            # every layer.  This keeps FlexAttention fused and avoids the very
            # large dense-attention temporary seen in the V20 generic rollout.
            base_logits, base_states = self._base_rollout_v10_fused(batch, **kwargs)
            base_preds = self.predictor.forward_prefixes_from_history(
                predictor_history,
                base_states,
                prefix_lengths=list(self.predictor.prefix_lengths),
            )
            valid = batch.aligned_mask.reshape(-1, self.block_size).bool()
            base_preds = [self._apply_prediction_valid_mask(p, valid) for p in base_preds]
            base_target = self._mismatch(batch, base_logits)

        if self.cfg.train_stage == "predictor":
            assert base_logits is not None and base_preds is not None and base_target is not None
            total, metrics = self._predictor_objective(base_preds, base_target)
            metrics.update(progressive_prediction_diagnostics(base_preds))
            metrics["loss"] = total.detach()
            metrics["valid_anchors"] = anchor_valid.sum().detach()
            metrics["valid_lagged_pairs"] = anchor_valid.sum().detach()
            metrics["teacher_off_policy_proxy"] = 1.0
            metrics["teacher_reduced_vocab_proxy"] = float(
                getattr(self.base_model, "use_draft_vocab", False)
            )
            return ResidualForwardOutput(
                total, metrics, base_logits, batch.targets, None,
                prediction=base_preds[0], predictions=base_preds,
                mismatch=base_target, previous_anchors=prev_anchors,
                current_anchors=anchors, pair_valid=anchor_valid,
            )

        logits, _, online_preds, routes, route_logits = self._rollout(
            batch, correction=True, collect_predictions=True,
            predictor_history=predictor_history, **kwargs
        )
        # Teacher geometry is a detached view of the current corrected distribution.
        # Gradients for Stage2/3 flow only through soft_logit_alignment_loss(logits, target).
        corrected_target = self._mismatch(batch, logits.detach())
        correction_loss, correction_loss_metrics = self._correction_objective(
            logits, batch.targets, corrected_target
        )
        metrics: dict[str, torch.Tensor | float] = {
            **progressive_prediction_diagnostics(online_preds),
            **routing_diagnostics(self.correction_layer_indices, routes),
            **self._correction_metrics(),
            **correction_loss_metrics,
            "correction_kl": correction_loss_metrics["soft_alignment_kl"],
            "valid_anchors": anchor_valid.sum().detach(),
            "routing_points": float(len(routes)),
            "teacher_off_policy_proxy": float(
                self.cfg.teacher_trajectory == "reference_off_policy"
            ),
            "teacher_reduced_vocab_proxy": float(
                getattr(self.base_model, "use_draft_vocab", False)
            ),
        }
        z_loss = torch.stack([router_z_loss(x) for x in route_logits]).mean() if route_logits else correction_loss.new_zeros(())
        metrics["router_z_loss"] = z_loss.detach()
        route_consistency_values = [
            route_consistency_loss(p["route_logits"], p["stage2_route_logits"])
            for p in online_preds if "stage2_route_logits" in p
        ]
        route_consistency = torch.stack(route_consistency_values).mean() if route_consistency_values else correction_loss.new_zeros(())
        metrics["route_consistency_loss"] = route_consistency.detach()
        total = correction_loss + self.cfg.lambda_router_z * z_loss + self.cfg.lambda_route_consistency * route_consistency

        supervised_preds = online_preds
        supervised_target = detach_mismatch_targets(corrected_target)
        if self.cfg.train_stage == "joint":
            assert base_preds is not None and base_target is not None
            base_pred_loss, base_pred_metrics = self._predictor_objective(base_preds, base_target)
            online_pred_loss, online_pred_metrics = self._predictor_objective(
                online_preds, detach_mismatch_targets(corrected_target), prefix="online_"
            )
            metrics.update(base_pred_metrics)
            metrics.update(online_pred_metrics)
            total = total + self.cfg.lambda_pred * base_pred_loss + self.cfg.lambda_pred_online * online_pred_loss
            supervised_preds, supervised_target = base_preds, base_target
        else:
            with torch.no_grad():
                _, pred_metrics = self._predictor_objective(online_preds, corrected_target)
            metrics.update(pred_metrics)
        metrics["loss"] = total.detach()
        valid = corrected_target.valid_mask
        soft_eal, soft_eal_valid_blocks = mean_valid_block_eal(
            corrected_target.expected_accepted_length,
            valid,
        )
        metrics["soft_eal"] = soft_eal.detach()
        metrics["soft_eal_valid_blocks"] = soft_eal_valid_blocks.detach()
        if valid.any():
            metrics["tv_mean"] = corrected_target.tv[valid].mean().detach()
            metrics["overlap_mean"] = corrected_target.overlap[valid].mean().detach()
            for pos in range(self.block_size):
                pos_valid = valid[:, pos]
                if pos_valid.any():
                    metrics[f"overlap_position_{pos}"] = (
                        corrected_target.overlap[:, pos][pos_valid].mean().detach()
                    )
        route_stack = torch.stack(routes) if routes else None
        return ResidualForwardOutput(
            total, metrics, logits, batch.targets, route_stack,
            prediction=supervised_preds[0] if supervised_preds else None,
            predictions=supervised_preds, mismatch=supervised_target,
            previous_anchors=prev_anchors, current_anchors=anchors, pair_valid=anchor_valid,
        )

    @torch.inference_mode()
    def propose_runtime(self, hidden_states: torch.Tensor, input_ids: torch.Tensor,
                        previous_state: PreviousRoundState | None = None,
                        position_ids: torch.Tensor | None = None, *,
                        build_base_next_state: bool = True):
        del build_base_next_state
        batch = self._prepare_runtime_anchor(hidden_states, input_ids, position_ids)
        predictor_history = None
        if previous_state is not None:
            if self.predictor is None and self.cfg.train_stage != "eval_base":
                raise RuntimeError("Runtime Dual-V10 correction requires Predictor")
            if self.predictor is not None:
                predictor_history = self.predictor.prepare_history(
                    previous_state.target_layers, previous_state.drafter_layers
                )
        # First speculative round has no lag-1 state, so it stays pure Base DFlash.
        correction = (
            previous_state is not None
            and self.cfg.train_stage == "eval"
            and bool(self.routers)
        )
        logits, states, predictions, routes, _ = self._rollout(
            batch, correction=correction, collect_predictions=correction,
            predictor_history=predictor_history,
        )
        target_state = batch.target_layers.detach()
        # _rollout already returns the V10 anchor-level state stack [A,L,H].
        draft_state = states.detach()
        state = PreviousRoundState(target_state, draft_state)
        diagnostics: dict[str, torch.Tensor] = {
            "runtime_base_history_replay": torch.tensor(0.0, device=hidden_states.device),
            # Make it observable that a runtime round actually traversed the
            # Stage-2 Predictor -> Router -> MoLE path. The first round has no
            # lagged state and is intentionally Base DFlash.
            "runtime_residual_correction_active": torch.tensor(
                float(correction), device=hidden_states.device
            ),
        }
        if predictions:
            diagnostics.update(progressive_prediction_diagnostics(predictions))
        if routes:
            diagnostics.update(routing_diagnostics(self.correction_layer_indices, routes))
        diagnostics.update(self._correction_metrics())
        block_logits = logits.reshape(1, self.block_size, -1)
        if self.base_model.config.sample_from_anchor:
            draft_logits = block_logits
        else:
            draft_logits = block_logits[:, 1:]
        return draft_logits, state, diagnostics

    def forward(self, *args, **kwargs):
        out = self.forward_residual(*args, **kwargs)
        return None, out.loss, out.metrics
