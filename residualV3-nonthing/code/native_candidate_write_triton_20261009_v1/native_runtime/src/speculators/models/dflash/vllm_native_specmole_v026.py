"""Native SpecMoLE adapter for the vLLM 0.26 V2 DFlash runner.

This module is intentionally independent from the vLLM 0.20 adapter.  It keeps
the stock v0.26 DFlash model, attention/KV-cache, sampler, V2 scheduler and
dedicated FULL CUDA graph manager, and inserts only the trained lag-one
Predictor -> Router -> MoLE correction into draft decoder layers p2/p3/p4.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from residual.config import ResidualConfig
from residual.hidden_selector import select_hidden_layer_indices
from residual.mismatch_predictor import MismatchPredictor
from residual.mole_linear import ResidualMoLE
from residual.router import ResidualRouter

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


logger = init_logger(__name__)
_CHECKPOINT_ENV = "SPECMOLE_CHECKPOINT"
_SUPPORTED_STAGES = {"mole", "joint"}
_ACTIVE_CONTROL_MODEL: Any | None = None


@torch.library.custom_op("specmole_v026::control", mutates_args=())
def _control_custom_op(
    previous_target: torch.Tensor,
    previous_draft: torch.Tensor,
    current_prefix: torch.Tensor,
    mlp_input: torch.Tensor,
    valid_mask: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Opaque AOT boundary whose eager CUDA kernels are captured by DFlash CG.

    Compiling the trained attention-heavy controls as part of the DFlash
    Inductor graph both changes bf16 reduction topology and creates an
    impractically large compiler graph.  A registered custom op is visible to
    Dynamo as one traceable node (not a graph break).  Its exact eager kernels
    run during DFlash FULL graph capture; steady-state replay therefore has no
    Python dispatch while preserving the validated checkpoint arithmetic.
    """
    model = _ACTIVE_CONTROL_MODEL
    if model is None:
        raise RuntimeError("SpecMoLE control model is not initialized")
    return model._control_eager(
        layer_idx,
        previous_target,
        previous_draft,
        current_prefix,
        mlp_input,
        valid_mask,
    )


@_control_custom_op.register_fake
def _control_custom_op_fake(
    previous_target: torch.Tensor,
    previous_draft: torch.Tensor,
    current_prefix: torch.Tensor,
    mlp_input: torch.Tensor,
    valid_mask: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    return torch.empty_like(mlp_input)


def checkpoint_metadata(
    checkpoint: str | os.PathLike[str],
) -> tuple[Path, str, ResidualConfig]:
    root = Path(checkpoint).expanduser().resolve()
    if root.is_file():
        root = root.parent
    config_path = root / "residual_mole_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"SpecMoLE config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw = dict(payload.get("residual_config") or {})
    stage = str(payload.get("stage") or raw.get("train_stage") or "")
    if stage not in _SUPPORTED_STAGES:
        raise ValueError(
            "Native SpecMoLE requires a Stage-2/Stage-3 checkpoint; "
            f"got stage={stage!r}"
        )
    allowed = {item.name for item in fields(ResidualConfig)}
    raw = {key: value for key, value in raw.items() if key in allowed}
    raw["train_stage"] = "eval"
    return root, stage, ResidualConfig(**raw)


def weights_path(root: Path) -> Path:
    for name in ("v20_model.safetensors", "residual_mole.safetensors"):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No SpecMoLE safetensors found under {root}")


def _weighted_depth_pool(tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    weights = F.softmax(scores, dim=0)
    return (tokens * weights.view(1, -1, 1)).sum(dim=1)


def _fusion_block(
    block: nn.Module,
    latents: torch.Tensor,
    source_tokens: torch.Tensor,
) -> torch.Tensor:
    # Do not merge these identical-looking norm calls: the reference checkpoint
    # was validated with this precise bf16 execution order.
    cross_out, _ = block.cross_attn(
        query=block.cross_norm(latents),
        key=block.source_norm(source_tokens),
        value=block.source_norm(source_tokens),
        need_weights=False,
    )
    latents = latents + cross_out
    self_out, _ = block.self_attn(
        query=block.self_norm(latents),
        key=block.self_norm(latents),
        value=block.self_norm(latents),
        need_weights=False,
    )
    latents = latents + self_out
    return latents + block.ffn(block.ffn_norm(latents))


def _predictor_branch(
    branch: nn.Module,
    history: torch.Tensor,
    d_cur: torch.Tensor,
    temporal_delta: torch.Tensor,
    depth_delta: torch.Tensor,
    source_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_summary = _weighted_depth_pool(d_cur, branch.current_layer_scores)
    temporal_summary = _weighted_depth_pool(
        temporal_delta, branch.temporal_layer_scores
    )
    depth_summary = _weighted_depth_pool(depth_delta, branch.depth_layer_scores)
    temporal_abs_summary = _weighted_depth_pool(
        temporal_delta.abs(), branch.temporal_layer_scores
    )
    depth_abs_summary = _weighted_depth_pool(
        depth_delta.abs(), branch.depth_layer_scores
    )
    last_current = d_cur[:, -1, :]
    temporal = branch.temporal_encoder(
        torch.cat(
            [
                last_current,
                current_summary,
                temporal_summary,
                depth_summary,
                temporal_abs_summary,
                depth_abs_summary,
            ],
            dim=-1,
        )
    )
    current = branch.current_encoder(
        torch.cat(
            [d_cur[:, 0, :], last_current, last_current - d_cur[:, 0, :]],
            dim=-1,
        )
    )
    base_gate = torch.sigmoid(
        branch.base_fusion_gate(torch.cat([history, temporal], dim=-1))
    )
    repair_seed = branch.base_fusion_norm(history + base_gate * temporal)
    latents = torch.stack([history, temporal, current, repair_seed], dim=1)
    latents = branch.latent_input_norm(latents + branch.role_embed.unsqueeze(0))
    for block in branch.fusion_blocks:
        latents = _fusion_block(block, latents, source_tokens)
    latent = branch.output_norm(latents)[:, 3, :]
    if branch.task == "risk":
        output = F.softplus(branch.severity_head(latent))
    else:
        output = F.normalize(branch.repair_head(latent), dim=-1, eps=1.0e-6)
    return latent, output


def mole_forward_exact(
    module: ResidualMoLE,
    x: torch.Tensor,
    route_weights: torch.Tensor,
    gate: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Vectorized, diagnostic-free equivalent of ResidualMoLE.forward."""
    batch, block, hidden = x.shape
    xd = x.reshape(-1, hidden)
    weights = route_weights.float()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
    g = gate.float().clamp(0.0, 1.0) * valid_mask.unsqueeze(-1).float()

    shared = F.linear(F.linear(xd, module.shared_A), module.shared_B)
    shared = shared * module.shared_scaling
    low = torch.einsum("nh,erh->ner", xd, module.expert_A)
    routed_all = torch.einsum("ner,ehr->neh", low, module.expert_B)
    routed = (
        routed_all
        * weights.reshape(-1, module.num_experts, 1).to(routed_all.dtype)
    ).sum(dim=1)
    correction = (shared + routed * module.scaling) * g.reshape(-1, 1).to(
        shared.dtype
    )
    if module.max_relative_update > 0:
        input_norm = xd.float().norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        corr_norm = correction.float().norm(dim=-1, keepdim=True)
        cap = module.max_relative_update * input_norm
        correction = correction * torch.minimum(
            torch.ones_like(corr_norm), cap / corr_norm.clamp_min(1.0e-6)
        ).to(correction.dtype)
    return correction.reshape(batch, block, hidden)


def specmole_control_exact(
    predictor: MismatchPredictor,
    risk_branch: nn.Module,
    repair_branch: nn.Module,
    router: ResidualRouter,
    mole: ResidualMoLE,
    d_prev_tokens: torch.Tensor,
    history: torch.Tensor,
    history_sources: torch.Tensor,
    d_cur: torch.Tensor,
    mlp_input: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Exact inference tensors used both by runtime and parity tests."""
    k = d_cur.shape[1]
    temporal_delta = d_cur - d_prev_tokens[:, :k, :]
    depth_delta = d_cur[:, 1:, :] - d_cur[:, :-1, :]
    current_tokens = (
        d_cur
        + predictor.drafter_layer_embed[:k].unsqueeze(0)
        + predictor.drafter_source_embed.unsqueeze(1)
        + predictor.current_time_embed.unsqueeze(1)
    )
    temporal_tokens = (
        temporal_delta
        + predictor.drafter_layer_embed[:k].unsqueeze(0)
        + predictor.temporal_delta_source_embed.unsqueeze(1)
        + predictor.current_time_embed.unsqueeze(1)
    )
    depth_tokens = (
        depth_delta
        + predictor.drafter_layer_embed[1:k].unsqueeze(0)
        + predictor.depth_delta_source_embed.unsqueeze(1)
        + predictor.current_time_embed.unsqueeze(1)
    )
    source_tokens = torch.cat(
        [history_sources, current_tokens, temporal_tokens, depth_tokens], dim=1
    )
    risk_latent, severity = _predictor_branch(
        risk_branch,
        history,
        d_cur,
        temporal_delta,
        depth_delta,
        source_tokens,
    )
    repair_latent, repair_direction = _predictor_branch(
        repair_branch,
        history,
        d_cur,
        temporal_delta,
        depth_delta,
        source_tokens,
    )

    valid = valid_mask.bool()
    severity = torch.where(valid, severity, torch.zeros_like(severity))
    overlap = torch.exp(-severity.float()).clamp(1.0e-6, 1.0).to(severity.dtype)
    overlap = torch.where(valid, overlap, torch.ones_like(overlap))
    reach = torch.cat(
        [torch.ones_like(overlap[:, :1]), torch.cumprod(overlap[:, :-1], dim=-1)],
        dim=-1,
    )
    first_reject = reach * (1.0 - overlap) * valid.float()
    reach = reach * valid.float()
    block_size = predictor.block_size
    repair_per_pos = repair_direction.unsqueeze(1).expand(-1, block_size, -1)
    repair_input = torch.cat(
        [repair_latent.unsqueeze(1).expand(-1, block_size, -1), repair_per_pos],
        dim=-1,
    )
    physical = torch.stack(
        [severity, 1.0 - overlap, first_reject, reach], dim=-1
    ).to(risk_latent.dtype)
    risk_input = torch.cat(
        [risk_latent.unsqueeze(1).expand(-1, block_size, -1), physical], dim=-1
    )
    effective, router_logits, gate = router(repair_input, risk_input)
    gate = gate * valid.unsqueeze(-1).to(gate.dtype)
    router_weights = effective / gate.clamp_min(1.0e-9)
    router_weights = torch.where(
        gate > 0, router_weights, torch.zeros_like(router_weights)
    )
    correction = mole_forward_exact(
        mole, mlp_input, router_weights, gate, valid_mask
    )
    return (
        severity,
        repair_direction,
        router_logits,
        router_weights,
        gate,
        correction,
    )


class DFlashQwen3ModelSpecMoLEV026(DFlashQwen3Model):
    """Compiled v0.26 DFlash backbone plus p2/p3/p4 residual controls.

    DFlashQwen3Model is decorated with vLLM's support_torch_compile.  Its
    wrapper compiles the runtime subclass' overridden forward, so this class
    deliberately does not opt out with ignore_torch_compile.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        checkpoint_root: Path,
        checkpoint_stage: str,
        residual_config: ResidualConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        self.specmole_checkpoint_root = checkpoint_root
        self.specmole_checkpoint_stage = checkpoint_stage
        self.residual_config = residual_config
        self.block_size = (
            int(vllm_config.speculative_config.num_speculative_tokens) + 1
        )
        self.hidden_size = int(self.config.hidden_size)
        self.num_draft_layers = len(self.layers)
        residual_config.validate(self.num_draft_layers)
        self.correction_layer_indices = tuple(
            residual_config.correction_layer_indices
        )
        dtype = vllm_config.model_config.dtype

        dflash_cfg = getattr(self.config, "dflash_config", {}) or {}
        num_target_layers = len(dflash_cfg.get("target_layer_ids") or [])
        if num_target_layers <= 0:
            raise ValueError("DFlash target_layer_ids are required by SpecMoLE")
        requested = min(
            residual_config.mismatch_num_hidden_layers, num_target_layers
        )
        self.target_hidden_indices = tuple(
            select_hidden_layer_indices(num_target_layers, requested)
        )
        self.num_target_layers = num_target_layers

        self.predictor = MismatchPredictor(
            hidden_size=self.hidden_size,
            hidden_dim=residual_config.mismatch_hidden_dim,
            direction_dim=residual_config.mismatch_direction_dim,
            block_size=self.block_size,
            max_target_layers=max(requested, 1),
            max_drafter_layers=self.num_draft_layers,
            predictor_depth=residual_config.predictor_depth,
            attention_heads=residual_config.predictor_attention_heads,
            ffn_expansion=residual_config.predictor_ffn_expansion,
        ).to(dtype=dtype)
        self.routers = nn.ModuleDict()
        self.residual_mole = nn.ModuleDict()
        for layer_idx in self.correction_layer_indices:
            self.routers[f"p{layer_idx}"] = ResidualRouter(
                residual_config.mismatch_hidden_dim
                + residual_config.mismatch_direction_dim,
                risk_input_dim=(
                    residual_config.mismatch_hidden_dim
                    + residual_config.router_context_dim
                ),
                num_experts=residual_config.mole_num_experts,
                top_k=residual_config.mole_top_k,
                gate_init_bias=residual_config.router_gate_init_bias,
                balance_mode=residual_config.router_balance_mode,
                bias_update_rate=residual_config.router_bias_update_rate,
                load_ema=residual_config.router_load_ema,
                bias_clip=residual_config.router_bias_clip,
                temperature=residual_config.router_temperature_start,
                routing_mode="topk",
            ).to(dtype=dtype)
            self.residual_mole[str(layer_idx)] = ResidualMoLE(
                self.hidden_size,
                num_experts=residual_config.mole_num_experts,
                rank=residual_config.mole_rank,
                alpha=residual_config.mole_alpha,
                shared_rank=residual_config.mole_shared_rank,
                dropout=0.0,
                max_relative_update=residual_config.mole_max_relative_update,
            ).to(dtype=dtype)

        max_batch = vllm_config.scheduler_config.max_num_seqs
        self.register_buffer(
            "_specmole_previous_target",
            torch.zeros(max_batch, requested, self.hidden_size, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_specmole_previous_draft",
            torch.zeros(
                max_batch, self.num_draft_layers, self.hidden_size, dtype=dtype
            ),
            persistent=False,
        )
        self.register_buffer(
            "_specmole_history_valid",
            torch.zeros(max_batch, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_specmole_current_draft",
            torch.zeros(
                max_batch, self.num_draft_layers, self.hidden_size, dtype=dtype
            ),
            persistent=False,
        )
        position_valid = torch.ones(max_batch, self.block_size, dtype=torch.bool)
        if not bool(getattr(self.config, "sample_from_anchor", False)):
            position_valid[:, 0] = False
        self.register_buffer(
            "_specmole_position_valid", position_valid, persistent=False
        )
        global _ACTIVE_CONTROL_MODEL
        _ACTIVE_CONTROL_MODEL = self

    def prepare_specmole_batch(
        self,
        previous_target: torch.Tensor,
        previous_draft: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> None:
        batch = previous_target.shape[0]
        self._specmole_previous_target[:batch].copy_(previous_target)
        self._specmole_previous_draft[:batch].copy_(previous_draft)
        self._specmole_history_valid.zero_()
        self._specmole_history_valid[:batch].copy_(history_valid)

    def current_draft_states(self, batch: int) -> torch.Tensor:
        return self._specmole_current_draft[:batch]

    def load_specmole_checkpoint(self) -> set[str]:
        from safetensors.torch import load_file

        state = load_file(str(weights_path(self.specmole_checkpoint_root)), device="cpu")
        own: dict[str, torch.Tensor] = dict(self.named_parameters())
        own.update(dict(self.named_buffers()))
        loaded: set[str] = set()
        unexpected: list[str] = []
        for original_key, value in state.items():
            key = original_key
            marker = "base_model.layers."
            middle = ".residual_mole."
            if key.startswith(marker) and middle in key:
                layer, suffix = key.split(middle, 1)
                layer_idx = layer[len(marker) :]
                if layer_idx.isdigit():
                    key = f"residual_mole.{layer_idx}.{suffix}"
            target = own.get(key)
            if target is None:
                if key != "direction_projector":
                    unexpected.append(original_key)
                continue
            if tuple(target.shape) != tuple(value.shape):
                raise ValueError(
                    f"SpecMoLE shape mismatch {key}: "
                    f"checkpoint={tuple(value.shape)} runtime={tuple(target.shape)}"
                )
            target.data.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded.add(key)
        critical = {
            key
            for key in own
            if key.startswith(("predictor.", "routers.", "residual_mole."))
        }
        missing = sorted(critical - loaded)
        if missing:
            raise RuntimeError(f"Incomplete SpecMoLE checkpoint: {missing[:16]}")
        if unexpected:
            raise RuntimeError(f"Unexpected SpecMoLE tensors: {unexpected[:16]}")
        logger.warning(
            "v0.26 SpecMoLE checkpoint audit PASS: loaded=%d expected=688 stage=%s",
            len(loaded),
            self.specmole_checkpoint_stage,
        )
        return loaded

    @torch.compiler.disable
    def _control_eager(
        self,
        layer_idx: int,
        previous_target: torch.Tensor,
        previous_draft: torch.Tensor,
        current_prefix: torch.Tensor,
        mlp_input: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        predictor = self.predictor
        prepared = predictor.prepare_history(previous_target, previous_draft)
        return specmole_control_exact(
            predictor,
            predictor.risk_branches[f"p{layer_idx}"],
            predictor.repair_branches[f"p{layer_idx}"],
            self.routers[f"p{layer_idx}"],
            self.residual_mole[str(layer_idx)],
            prepared["d_prev_tokens"],
            prepared["history"],
            prepared["history_sources"],
            predictor.project_current(current_prefix),
            mlp_input,
            valid,
        )[5]

    def _correction(
        self,
        layer_idx: int,
        anchors: list[torch.Tensor],
        mlp_input: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        return _control_custom_op(
            self._specmole_previous_target[: mlp_input.shape[0]],
            self._specmole_previous_draft[: mlp_input.shape[0]],
            torch.stack(anchors, dim=1).detach(),
            mlp_input,
            valid,
            layer_idx,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The profile-only one-token invocation is not a parallel DFlash query.
        if input_ids.shape[0] % self.block_size:
            return super().forward(input_ids, positions, input_embeds)
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        hidden_states = input_embeds
        residual = None
        batch = hidden_states.shape[0] // self.block_size
        valid = (
            self._specmole_position_valid[:batch]
            & self._specmole_history_valid[:batch, None]
        )
        anchors: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            if residual is not None:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            else:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = layer.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            mlp_input, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = layer.mlp(mlp_input)
            if layer_idx in self.correction_layer_indices:
                correction = self._correction(
                    layer_idx,
                    anchors,
                    mlp_input.reshape(batch, self.block_size, self.hidden_size),
                    valid,
                )
                hidden_states = hidden_states + correction.reshape_as(hidden_states)
            complete = hidden_states + residual
            anchors.append(
                complete.reshape(batch, self.block_size, self.hidden_size)[:, 0, :]
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        self._specmole_current_draft[:batch].copy_(torch.stack(anchors, dim=1))
        return hidden_states


class DFlashQwen3ForCausalLMSpecMoLEV026(DFlashQwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        checkpoint = os.environ.get(_CHECKPOINT_ENV)
        if not checkpoint:
            raise RuntimeError(f"{_CHECKPOINT_ENV} must name a SpecMoLE checkpoint")
        root, stage, cfg = checkpoint_metadata(checkpoint)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlashQwen3ModelSpecMoLEV026(
            vllm_config=vllm_config,
            checkpoint_root=root,
            checkpoint_stage=stage,
            residual_config=cfg,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        super().load_weights(weights)
        self.model.load_specmole_checkpoint()
        return None

    def prepare_specmole_batch(
        self,
        previous_target: torch.Tensor,
        previous_draft: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        self.model.prepare_specmole_batch(previous_target, previous_draft, valid)

    def current_draft_states(self, batch: int) -> torch.Tensor:
        return self.model.current_draft_states(batch)


class DFlashSpecMoLEV026(DFlashSpeculator):
    """V2 speculator with bounded request-id state outside the GPU hot graph."""

    _speculator_name = "SpecMoLE"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self._history: dict[str, torch.Tensor] = {}
        self._profile_enabled = os.environ.get(
            "SPECMOLE_PROFILE_PROPOSER", "0"
        ).lower() in {"1", "true", "yes"}
        self._profile_window = max(
            8, int(os.environ.get("SPECMOLE_PROFILE_WINDOW", "128"))
        )
        self._profile_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._profile_index = 0
        self._profile_cpu_ms = 0.0
        super().__init__(vllm_config, device)
        if self._profile_enabled:
            self._profile_events = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(self._profile_window)
            ]

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if not isinstance(self.model, DFlashQwen3ForCausalLMSpecMoLEV026):
            raise TypeError(f"Unexpected v0.26 DFlash model: {type(self.model).__name__}")
        self._zero_history = self.model.current_draft_states(1)[0].detach().clone()
        self._zero_history.zero_()

    def _prepare_history(
        self,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> list[str]:
        if not aux_hidden_states:
            raise RuntimeError("SpecMoLE requires DFlash Target auxiliary hidden states")
        batch = input_batch.num_reqs
        req_ids = list(input_batch.req_ids[:batch])
        starts = input_batch.query_start_loc[:batch].long()
        selected = [
            aux_hidden_states[i].index_select(0, starts)
            for i in self.model.model.target_hidden_indices
        ]
        previous_target = torch.stack(selected, dim=1)
        rows: list[torch.Tensor] = []
        flags: list[bool] = []
        for req_id in req_ids:
            cached = self._history.get(req_id)
            rows.append(self._zero_history if cached is None else cached)
            flags.append(cached is not None)
        previous_draft = torch.stack(rows, dim=0)
        valid = torch.tensor(flags, device=self.device, dtype=torch.bool)
        self.model.prepare_specmole_batch(previous_target, previous_draft, valid)

        # InputBatch contains all currently live V2 requests.  Bound the Python
        # map without touching scheduler, token commit or KV-cache state.
        live = set(req_ids)
        for stale in self._history.keys() - live:
            del self._history[stale]
        return req_ids

    @torch.inference_mode()
    def propose(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        event_pair = None
        cpu_start = 0.0
        if self._profile_enabled:
            event_pair = self._profile_events[self._profile_index]
            event_pair[0].record()
            cpu_start = time.perf_counter()
        input_batch: InputBatch = kwargs.get("input_batch", args[0] if args else None)
        aux_hidden_states = kwargs.get(
            "aux_hidden_states", args[4] if len(args) > 4 else None
        )
        req_ids = self._prepare_history(input_batch, aux_hidden_states)
        draft_tokens = super().propose(*args, **kwargs)
        current = self.model.current_draft_states(len(req_ids)).detach().clone()
        for row, req_id in enumerate(req_ids):
            self._history[req_id] = current[row]
        if event_pair is not None:
            event_pair[1].record()
            self._profile_cpu_ms += (time.perf_counter() - cpu_start) * 1000.0
            self._profile_index += 1
            if self._profile_index == self._profile_window:
                event_pair[1].synchronize()
                durations = [a.elapsed_time(b) for a, b in self._profile_events]
                ordered = sorted(durations)
                logger.warning(
                    "SPECMOLE_V026_PROPOSER_PROFILE calls=%d gpu_mean_ms=%.4f "
                    "gpu_p50_ms=%.4f gpu_p95_ms=%.4f cpu_submit_mean_ms=%.4f",
                    self._profile_window,
                    sum(durations) / self._profile_window,
                    ordered[self._profile_window // 2],
                    ordered[min(self._profile_window - 1, int(self._profile_window * 0.95))],
                    self._profile_cpu_ms / self._profile_window,
                )
                self._profile_index = 0
                self._profile_cpu_ms = 0.0
        return draft_tokens


def install_vllm_specmole_v026() -> None:
    """Install v0.26-only model/speculator overrides in this process."""
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.worker.gpu import model_runner
    from vllm.v1.worker.gpu import spec_decode

    ModelRegistry.register_model(
        "DFlashDraftModel", DFlashQwen3ForCausalLMSpecMoLEV026
    )

    def init_speculator(vllm_config: VllmConfig, device: torch.device):
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        if speculative_config.method == "dflash":
            return DFlashSpecMoLEV026(vllm_config, device)
        return spec_decode._specmole_original_init_speculator(vllm_config, device)

    if not hasattr(spec_decode, "_specmole_original_init_speculator"):
        spec_decode._specmole_original_init_speculator = spec_decode.init_speculator
    spec_decode.init_speculator = init_speculator
    model_runner.init_speculator = init_speculator
