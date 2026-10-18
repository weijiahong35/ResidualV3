"""Native vLLM DFlash adapter for V20 Residual/SpecMoLE checkpoints.

This module deliberately lives beside, rather than modifies, the stock DFlash
implementation.  It reuses vLLM's DFlash attention, KV cache, logits head and
proposer protocol, and only adds the trained lag-one Predictor -> Router ->
SpecMoLE computation at the decoder layer's second residual addition.

The adapter is installed process-locally by ``scripts/launch_vllm_specmole.py``.
No vLLM installation files are patched.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from residual.config import ResidualConfig
from residual.hidden_selector import select_hidden_layer_indices
from residual.mismatch_predictor import MismatchPredictor
from residual.mole_linear import ResidualMoLE
from residual.router import ResidualRouter

from vllm.config import VllmConfig
from vllm.compilation.decorators import ignore_torch_compile
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.spec_decode.dflash import DFlashProposer


_CHECKPOINT_ENV = "SPECMOLE_CHECKPOINT"
_SUPPORTED_STAGES = {"mole", "joint"}
logger = init_logger(__name__)


class ProposerCudaTimerSpecMoLE:
    """Optional low-overhead CUDA-event window timer for proposer profiling."""

    def __init__(self, label: str) -> None:
        self.enabled = (
            os.environ.get("SPECMOLE_PROFILE_PROPOSER", "0").strip().lower()
            in {"1", "true", "yes"}
        )
        self.label = label
        self.window = max(
            8, int(os.environ.get("SPECMOLE_PROFILE_WINDOW", "128"))
        )
        self.index = 0
        self.cpu_submit_ms = 0.0
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        if self.enabled:
            self.events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(self.window)
            ]

    def begin(self) -> float:
        if not self.enabled:
            return 0.0
        self.events[self.index][0].record()
        return time.perf_counter()

    def finish(self, cpu_started: float) -> None:
        if not self.enabled:
            return
        start, end = self.events[self.index]
        end.record()
        self.cpu_submit_ms += (time.perf_counter() - cpu_started) * 1000.0
        self.index += 1
        if self.index != self.window:
            return
        end.synchronize()
        durations = [a.elapsed_time(b) for a, b in self.events]
        ordered = sorted(durations)
        logger.warning(
            "SPECMOLE_PROPOSER_PROFILE label=%s calls=%d gpu_mean_ms=%.4f "
            "gpu_p50_ms=%.4f gpu_p95_ms=%.4f cpu_submit_mean_ms=%.4f",
            self.label,
            self.window,
            sum(durations) / self.window,
            ordered[self.window // 2],
            ordered[min(self.window - 1, int(self.window * 0.95))],
            self.cpu_submit_ms / self.window,
        )
        self.index = 0
        self.cpu_submit_ms = 0.0


def _checkpoint_metadata(checkpoint: str | os.PathLike[str]) -> tuple[Path, str, ResidualConfig]:
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
            f"Native SpecMoLE requires a Stage-2/Stage-3 checkpoint "
            f"(stage one of {sorted(_SUPPORTED_STAGES)}), got {stage!r}"
        )
    allowed = {item.name for item in fields(ResidualConfig)}
    raw = {key: value for key, value in raw.items() if key in allowed}
    # Runtime uses the trained modules from either Stage 2 or Stage 3.  The
    # existing reference evaluator likewise selects eval/top-k routing here.
    raw["train_stage"] = "eval"
    cfg = ResidualConfig(**raw)
    return root, stage, cfg


def _weights_path(root: Path) -> Path:
    for name in ("v20_model.safetensors", "residual_mole.safetensors"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No V20 SpecMoLE safetensors found under {root}")


def _apply_prediction_valid_mask(
    pred: dict[str, torch.Tensor], valid_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Match ResidualDFlashModel's soft runtime masking exactly."""
    valid = valid_mask.bool()
    severity = torch.where(valid, pred["severity"], torch.zeros_like(pred["severity"]))
    overlap = torch.exp(-severity.float()).clamp(1.0e-6, 1.0).to(severity.dtype)
    overlap = torch.where(valid, overlap, torch.ones_like(overlap))
    survival_raw = torch.cumprod(overlap, dim=-1)
    reach_raw = torch.cat(
        [torch.ones_like(overlap[:, :1]), torch.cumprod(overlap[:, :-1], dim=-1)],
        dim=-1,
    )
    pred["severity"] = severity
    pred["tv"] = 1.0 - overlap
    pred["reach_prob"] = reach_raw * valid.float()
    pred["first_reject"] = reach_raw * (1.0 - overlap) * valid.float()
    pred["survival"] = survival_raw * valid.float()
    return pred


def _router_inputs(pred: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    repair_latent = pred["repair_latent"]
    repair = torch.cat(
        [repair_latent, pred["repair_direction_per_pos"].to(repair_latent.dtype)],
        dim=-1,
    )
    risk_latent = pred["risk_latent"]
    physical = torch.stack(
        [pred["severity"], pred["tv"], pred["first_reject"], pred["reach_prob"]],
        dim=-1,
    ).to(risk_latent.dtype)
    return repair, torch.cat([risk_latent, physical], dim=-1)


def _mole_forward_native(
    module: ResidualMoLE,
    x: torch.Tensor,
    route_weights: torch.Tensor,
    gate: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Exact ResidualMoLE inference math without its diagnostic side effects."""
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
        routed_all * weights.reshape(-1, module.num_experts, 1).to(routed_all.dtype)
    ).sum(dim=1)
    correction = (shared + routed * module.scaling) * g.reshape(-1, 1).to(shared.dtype)

    if module.max_relative_update > 0:
        input_norm = xd.float().norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        corr_norm = correction.float().norm(dim=-1, keepdim=True)
        cap = module.max_relative_update * input_norm
        correction = correction * torch.minimum(
            torch.ones_like(corr_norm), cap / corr_norm.clamp_min(1.0e-6)
        ).to(correction.dtype)
    return correction.reshape(batch, block, hidden)


def _weighted_depth_pool_runtime(
    tokens: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    """Inference-only equivalent of MismatchPredictor._weighted_depth_pool."""
    weights = F.softmax(scores, dim=0)
    return (tokens * weights.view(1, -1, 1)).sum(dim=1)


def _fusion_block_runtime(
    block: nn.Module,
    latents: torch.Tensor,
    source_tokens: torch.Tensor,
) -> torch.Tensor:
    """Run one trained fusion block without unused training diagnostics.

    Keep the original normalization calls and MHA argument identities exactly:
    packing identical Q/K/V or K/V is algebraically equivalent, but can change
    bf16 GEMM accumulation enough to flip a close top-k Router decision.
    """
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


def _predictor_branch_runtime(
    branch: nn.Module,
    history: torch.Tensor,
    d_cur: torch.Tensor,
    temporal_delta: torch.Tensor,
    depth_delta: torch.Tensor,
    source_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Essential V20 branch outputs, excluding metric-only fp32 reductions."""
    current_summary = _weighted_depth_pool_runtime(
        d_cur, branch.current_layer_scores
    )
    temporal_summary = _weighted_depth_pool_runtime(
        temporal_delta, branch.temporal_layer_scores
    )
    depth_summary = _weighted_depth_pool_runtime(
        depth_delta, branch.depth_layer_scores
    )
    temporal_abs_summary = _weighted_depth_pool_runtime(
        temporal_delta.abs(), branch.temporal_layer_scores
    )
    depth_abs_summary = _weighted_depth_pool_runtime(
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
        latents = _fusion_block_runtime(block, latents, source_tokens)
    latent = branch.output_norm(latents)[:, 3, :]
    if branch.task == "risk":
        task_output = F.softplus(branch.severity_head(latent))
    else:
        task_output = F.normalize(
            branch.repair_head(latent), dim=-1, eps=1.0e-6
        )
    return latent, task_output


def _specmole_control_runtime(
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Pure-tensor Predictor -> Router -> MoLE path for one progressive prefix.

    Extra outputs are intentional: they support exact before/after tensor
    parity tests. Production uses only the final correction, and Inductor
    eliminates output-only materialization from its correction-only wrapper.
    """
    k = d_cur.shape[1]
    d_prev_aligned = d_prev_tokens[:, :k, :]
    temporal_delta = d_cur - d_prev_aligned
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
    risk_latent, severity = _predictor_branch_runtime(
        risk_branch,
        history,
        d_cur,
        temporal_delta,
        depth_delta,
        source_tokens,
    )
    repair_latent, repair_direction = _predictor_branch_runtime(
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
    block = predictor.block_size
    repair_per_pos = repair_direction.unsqueeze(1).expand(-1, block, -1)
    repair_input = torch.cat(
        [repair_latent.unsqueeze(1).expand(-1, block, -1), repair_per_pos], dim=-1
    )
    physical = torch.stack(
        [severity, 1.0 - overlap, first_reject, reach], dim=-1
    ).to(risk_latent.dtype)
    risk_input = torch.cat(
        [risk_latent.unsqueeze(1).expand(-1, block, -1), physical], dim=-1
    )
    effective, route_logits, gate = router(repair_input, risk_input)
    gate = gate * valid.unsqueeze(-1).to(gate.dtype)
    probabilities = effective / gate.clamp_min(1.0e-9)
    probabilities = torch.where(
        gate > 0, probabilities, torch.zeros_like(probabilities)
    )
    correction = _mole_forward_native(
        mole, mlp_input, probabilities, gate, valid_mask
    )
    return (
        severity,
        repair_direction,
        route_logits,
        probabilities,
        gate,
        correction,
    )


@ignore_torch_compile
class DFlashQwen3ModelSpecMoLE(DFlashQwen3Model):
    """Stock vLLM DFlash model with only the trained residual branch added."""

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
        # vLLM flattens the original Speculators config into a Qwen config and
        # does not retain every top-level field.  DFlash's native contract is
        # one Target anchor plus K parallel draft queries.
        self.block_size = int(vllm_config.speculative_config.num_speculative_tokens) + 1
        self.hidden_size = int(self.config.hidden_size)
        self.num_draft_layers = len(self.layers)
        residual_config.validate(self.num_draft_layers)
        self.correction_layer_indices = tuple(residual_config.correction_layer_indices)
        dtype = vllm_config.model_config.dtype

        dflash_cfg = getattr(self.config, "dflash_config", {}) or {}
        num_target_layers = len(dflash_cfg.get("target_layer_ids") or [])
        if num_target_layers <= 0:
            raise ValueError("DFlash target_layer_ids are required by native SpecMoLE")
        requested = min(residual_config.mismatch_num_hidden_layers, num_target_layers)
        self.target_hidden_indices = select_hidden_layer_indices(num_target_layers, requested)
        self.num_target_layers = num_target_layers

        self.predictor = MismatchPredictor(
            hidden_size=self.hidden_size,
            hidden_dim=residual_config.mismatch_hidden_dim,
            direction_dim=residual_config.mismatch_direction_dim,
            block_size=self.block_size,
            max_target_layers=max(len(self.target_hidden_indices), 1),
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
        # Stable-address buffers are copied before each proposal, so CUDA graph
        # replay sees new per-request histories without Python state in forward.
        self.register_buffer(
            "_specmole_previous_target",
            torch.zeros(max_batch, requested, self.hidden_size, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_specmole_previous_draft",
            torch.zeros(max_batch, self.num_draft_layers, self.hidden_size, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_specmole_history_valid",
            torch.zeros(max_batch, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_specmole_current_draft",
            torch.zeros(max_batch, self.num_draft_layers, self.hidden_size, dtype=dtype),
            persistent=False,
        )
        position_template = torch.ones(max_batch, self.block_size, dtype=torch.bool)
        if not bool(getattr(self.config, "sample_from_anchor", False)):
            position_template[:, 0] = False
        self.register_buffer(
            "_specmole_position_template", position_template, persistent=False
        )
        self._specmole_capture_controls = (
            os.environ.get("SPECMOLE_CUDA_GRAPHS", "1").strip().lower()
            not in {"0", "false", "no"}
        )
        self._specmole_reference_runtime = (
            os.environ.get("SPECMOLE_REFERENCE_RUNTIME", "0").strip().lower()
            in {"1", "true", "yes"}
        )
        if self._specmole_reference_runtime:
            self._specmole_capture_controls = False
        self._specmole_disable_graph_replay = False
        # Keep runtime callables/graphs outside nn.Module registration. They share
        # the already-loaded Predictor/Router/MoLE parameters without creating
        # duplicate state-dict names or weights.
        object.__setattr__(self, "_specmole_control_functions", {})
        object.__setattr__(self, "_specmole_control_graphs", {})

    def prepare_specmole_batch(
        self,
        previous_target: torch.Tensor,
        previous_draft: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> None:
        batch = int(previous_target.shape[0])
        if batch > self._specmole_previous_target.shape[0]:
            raise ValueError(f"SpecMoLE batch {batch} exceeds configured max_num_seqs")
        self._specmole_previous_target[:batch].copy_(previous_target)
        self._specmole_previous_draft[:batch].copy_(previous_draft)
        self._specmole_history_valid.zero_()
        self._specmole_history_valid[:batch].copy_(history_valid)

    def prepare_specmole_dummy(self, batch_size: int) -> None:
        batch = min(max(int(batch_size), 0), self._specmole_previous_target.shape[0])
        self._specmole_previous_target.zero_()
        self._specmole_previous_draft.zero_()
        self._specmole_history_valid.zero_()

    def current_draft_states(self, batch_size: int) -> torch.Tensor:
        return self._specmole_current_draft[:batch_size]

    def load_specmole_checkpoint(self) -> set[str]:
        from safetensors.torch import load_file

        state = load_file(str(_weights_path(self.specmole_checkpoint_root)), device="cpu")
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
                # direction_projector is a training-target projection and is not
                # consumed by the inference Predictor/Router/MoLE path.
                if key != "direction_projector":
                    unexpected.append(original_key)
                continue
            if tuple(target.shape) != tuple(value.shape):
                raise ValueError(
                    f"SpecMoLE tensor shape mismatch for {key}: "
                    f"checkpoint={tuple(value.shape)} runtime={tuple(target.shape)}"
                )
            target.data.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded.add(key)

        critical = {
            key
            for key in own
            if key.startswith("predictor.")
            or key.startswith("routers.")
            or key.startswith("residual_mole.")
        }
        missing = sorted(critical - loaded)
        if missing:
            raise RuntimeError(f"SpecMoLE checkpoint is incomplete; missing={missing[:16]}")
        if unexpected:
            raise RuntimeError(f"Unexpected SpecMoLE checkpoint tensors: {unexpected[:16]}")
        logger.warning(
            "Native SpecMoLE checkpoint audit PASS: loaded=%d path=%s stage=%s",
            len(loaded),
            self.specmole_checkpoint_root,
            self.specmole_checkpoint_stage,
        )
        self._build_specmole_control_functions()
        return loaded

    def _build_specmole_control_functions(self) -> None:
        functions: dict[int, Any] = {}
        for layer_idx in self.correction_layer_indices:
            predictor = self.predictor
            risk_branch = predictor.risk_branches[f"p{layer_idx}"]
            repair_branch = predictor.repair_branches[f"p{layer_idx}"]
            router = self.routers[f"p{layer_idx}"]
            mole = self.residual_mole[str(layer_idx)]

            def correction_only(
                d_prev_tokens: torch.Tensor,
                history: torch.Tensor,
                history_sources: torch.Tensor,
                current_prefix: torch.Tensor,
                mlp_input: torch.Tensor,
                valid_mask: torch.Tensor,
                *,
                _predictor=predictor,
                _risk_branch=risk_branch,
                _repair_branch=repair_branch,
                _router=router,
                _mole=mole,
            ) -> torch.Tensor:
                return _specmole_control_runtime(
                    _predictor,
                    _risk_branch,
                    _repair_branch,
                    _router,
                    _mole,
                    d_prev_tokens,
                    history,
                    history_sources,
                    _predictor.project_current(current_prefix),
                    mlp_input,
                    valid_mask,
                )[5]

            functions[layer_idx] = correction_only
        object.__setattr__(self, "_specmole_control_functions", functions)
        if self._specmole_capture_controls:
            self._capture_specmole_control_graphs()

    @torch.inference_mode()
    def _capture_specmole_control_graphs(self) -> None:
        """Capture exact eager kernels once for the sustained max-size batch."""
        batch = self._specmole_previous_target.shape[0]
        started = time.perf_counter()
        prepared = self.predictor.prepare_history(
            self._specmole_previous_target,
            self._specmole_previous_draft,
        )
        max_prefix = max(self.correction_layer_indices)
        current = self._specmole_previous_draft[:, :max_prefix, :]
        mlp_input = torch.zeros(
            batch,
            self.block_size,
            self.hidden_size,
            device=self._specmole_previous_target.device,
            dtype=self._specmole_previous_target.dtype,
        )
        valid = self._specmole_position_template
        capture_stream = torch.cuda.Stream(device=self._specmole_previous_target.device)
        capture_stream.wait_stream(torch.cuda.current_stream())
        graph_pool = torch.cuda.graph_pool_handle()
        graphs: dict[int, tuple[Any, tuple[torch.Tensor, ...], torch.Tensor]] = {}
        for layer_idx in self.correction_layer_indices:
            inputs = (
                prepared["d_prev_tokens"],
                prepared["history"],
                prepared["history_sources"],
                current[:, :layer_idx, :],
                mlp_input,
                valid,
            )
            static_inputs = tuple(t.clone() for t in inputs)
            function = self._specmole_control_functions[layer_idx]
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    static_output = function(*static_inputs)
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream, pool=graph_pool):
                static_output = function(*static_inputs)
            graphs[layer_idx] = (graph, static_inputs, static_output)
        torch.cuda.synchronize(self._specmole_previous_target.device)
        object.__setattr__(self, "_specmole_control_graphs", graphs)
        logger.warning(
            "Native SpecMoLE control CUDA graphs ready: batch=%d prefixes=%s capture_s=%.2f",
            batch,
            list(self.correction_layer_indices),
            time.perf_counter() - started,
        )

    def set_specmole_graph_replay(self, enabled: bool) -> None:
        self._specmole_disable_graph_replay = not enabled

    def _control_correction(
        self,
        prepared_history: dict[str, torch.Tensor],
        anchor_states: list[torch.Tensor],
        mlp_input: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        layer_idx = len(anchor_states)
        inputs = (
            prepared_history["d_prev_tokens"],
            prepared_history["history"],
            prepared_history["history_sources"],
            torch.stack(anchor_states, dim=1).detach(),
            mlp_input,
            valid,
        )
        captured = self._specmole_control_graphs.get(layer_idx)
        if (
            captured is not None
            and not self._specmole_disable_graph_replay
            and inputs[0].shape[0] == self._specmole_previous_target.shape[0]
        ):
            graph, static_inputs, static_output = captured
            for destination, source in zip(static_inputs, inputs):
                destination.copy_(source)
            graph.replay()
            return static_output
        return self._specmole_control_functions[layer_idx](*inputs)

    def _reference_control_correction(
        self,
        anchor_states: list[torch.Tensor],
        mlp_input: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Original pre-optimization runtime, retained only for A/B profiling."""
        batch = valid.shape[0]
        history = self.predictor.prepare_history(
            self._specmole_previous_target[:batch],
            self._specmole_previous_draft[:batch],
        )
        current_prefix = torch.stack(anchor_states, dim=1)
        pred = self.predictor.forward_from_history(
            history, current_prefix.detach()
        )
        pred = _apply_prediction_valid_mask(pred, valid)
        repair_input, risk_input = _router_inputs(pred)
        layer_idx = len(anchor_states)
        effective, _, gate = self.routers[f"p{layer_idx}"](
            repair_input, risk_input
        )
        gate = gate * valid.unsqueeze(-1).to(gate.dtype)
        probabilities = effective / gate.clamp_min(1.0e-9)
        probabilities = torch.where(
            gate > 0, probabilities, torch.zeros_like(probabilities)
        )
        return _mole_forward_native(
            self.residual_mole[str(layer_idx)],
            mlp_input,
            probabilities,
            gate,
            valid,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # vLLM memory profiling deliberately invokes DFlash with arbitrary
        # token counts (including a single query).  Only real parallel-draft
        # calls are composed of complete [anchor + 7 masks] request blocks.
        if input_ids.shape[0] % self.block_size:
            return super().forward(input_ids, positions, input_embeds)
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        hidden_states = input_embeds
        residual = None
        # Derive the padded request count from the graph input.  Keeping a
        # mutable Python batch-size attribute in the compiled forward causes
        # TorchInductor to lift an ``int`` into a tensor-only standalone graph.
        batch = hidden_states.shape[0] // self.block_size
        position_valid = (
            self._specmole_position_template[:batch]
            & self._specmole_history_valid[:batch, None]
        )
        prepared_history: dict[str, torch.Tensor] | None = None
        if not self._specmole_reference_runtime:
            prepared_history = self.predictor.prepare_history(
                self._specmole_previous_target[:batch],
                self._specmole_previous_draft[:batch],
            )

        anchor_states: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            if residual is not None:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            else:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = layer.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            mlp_input, residual = layer.post_attention_layernorm(hidden_states, residual)
            hidden_states = layer.mlp(mlp_input)

            if layer_idx in self.correction_layer_indices:
                shaped_mlp_input = mlp_input.reshape(
                    batch, self.block_size, self.hidden_size
                )
                if self._specmole_reference_runtime:
                    correction = self._reference_control_correction(
                        anchor_states, shaped_mlp_input, position_valid
                    )
                else:
                    assert prepared_history is not None
                    correction = self._control_correction(
                        prepared_history,
                        anchor_states,
                        shaped_mlp_input,
                        position_valid,
                    )
                correction = correction.reshape(
                    hidden_states.shape[0], self.hidden_size
                )
                hidden_states = hidden_states + correction

            complete = hidden_states + residual
            anchor = complete.reshape(
                batch, self.block_size, self.hidden_size
            )[:, 0, :]
            anchor_states.append(anchor)

        hidden_states, _ = self.norm(hidden_states, residual)
        self._specmole_current_draft[:batch].copy_(torch.stack(anchor_states, dim=1))
        return hidden_states


class DFlashQwen3ForCausalLMSpecMoLE(DFlashQwen3ForCausalLM):
    """vLLM-loadable DFlash architecture backed by a separate V20 checkpoint."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        checkpoint = os.environ.get(_CHECKPOINT_ENV)
        if not checkpoint:
            raise RuntimeError(f"{_CHECKPOINT_ENV} must name a Stage-2/Stage-3 checkpoint")
        checkpoint_root, checkpoint_stage, cfg = _checkpoint_metadata(checkpoint)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
        self.model = DFlashQwen3ModelSpecMoLE(
            vllm_config=vllm_config,
            checkpoint_root=checkpoint_root,
            checkpoint_stage=checkpoint_stage,
            residual_config=cfg,
            prefix="model",
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
        # Stock DFlash intentionally returns ``None`` here because its checkpoint
        # names are remapped before AutoWeightsLoader runs. Preserve that contract
        # (which disables the generic pre-remap strict-name check); the external
        # SpecMoLE loader performs its own exact missing/unexpected/shape audit.
        super().load_weights(weights)
        self.model.load_specmole_checkpoint()
        return None

    def prepare_specmole_batch(
        self,
        previous_target: torch.Tensor,
        previous_draft: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> None:
        self.model.prepare_specmole_batch(previous_target, previous_draft, history_valid)

    def prepare_specmole_dummy(self, batch_size: int) -> None:
        self.model.prepare_specmole_dummy(batch_size)

    def current_draft_states(self, batch_size: int) -> torch.Tensor:
        return self.model.current_draft_states(batch_size)

    def set_specmole_graph_replay(self, enabled: bool) -> None:
        self.model.set_specmole_graph_replay(enabled)


class DFlashSpecMoLEProposer(DFlashProposer):
    """DFlash proposer hook that maintains lag-one state per vLLM request."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        self._specmole_runner = runner
        self._specmole_history: dict[str, torch.Tensor] = {}
        self._specmole_timer = ProposerCudaTimerSpecMoLE("residual")
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)

    def load_model(self, target_model: nn.Module) -> None:
        """Let vLLM create/load the draft model before validating our override."""
        super().load_model(target_model)
        if not isinstance(self.model, DFlashQwen3ForCausalLMSpecMoLE):
            raise TypeError(
                "SpecMoLE proposer did not receive DFlashQwen3ForCausalLMSpecMoLE; "
                f"got {type(self.model).__name__}"
            )
        # One immutable row is enough for every request without lag-one history.
        # Keeping it here avoids allocating and zeroing BxLxH on every proposal.
        self._specmole_zero_history = (
            self.model.current_draft_states(1)[0].detach().clone().zero_()
        )

    def _request_ids(self, batch_size: int) -> list[str]:
        if self._specmole_runner is None:
            raise RuntimeError("Native SpecMoLE requires the vLLM GPU runner")
        req_ids = list(self._specmole_runner.input_batch.req_ids[:batch_size])
        if len(req_ids) != batch_size:
            raise RuntimeError(
                f"DFlash batch has {batch_size} requests but runner exposes {len(req_ids)}"
            )
        live = set(self._specmole_runner.input_batch.req_id_to_index)
        for stale in set(self._specmole_history) - live:
            del self._specmole_history[stale]
        return req_ids

    def _prepare_history(
        self,
        req_ids: list[str],
        target_hidden_states: torch.Tensor,
        common_attn_metadata: Any,
    ) -> None:
        batch = len(req_ids)
        starts = common_attn_metadata.query_start_loc[:batch].long()
        target_at_previous_anchor = target_hidden_states.index_select(0, starts)
        hidden = self.model.model.hidden_size
        expected = self.model.model.num_target_layers * hidden
        if target_at_previous_anchor.shape[-1] != expected:
            raise ValueError(
                f"Expected concatenated Target hidden width {expected}, got "
                f"{target_at_previous_anchor.shape[-1]}"
            )
        target_layers = target_at_previous_anchor.reshape(
            batch, self.model.model.num_target_layers, hidden
        )[:, self.model.model.target_hidden_indices, :]
        history_rows: list[torch.Tensor] = []
        valid_flags: list[bool] = []
        for req_id in req_ids:
            cached = self._specmole_history.get(req_id)
            history_rows.append(
                cached if cached is not None else self._specmole_zero_history
            )
            valid_flags.append(cached is not None)
        # These replace B indexed GPU copies and B scalar GPU writes with one
        # device stack plus one tiny H2D copy.  No arithmetic is changed.
        previous_draft = torch.stack(history_rows, dim=0)
        valid = torch.tensor(
            valid_flags, device=target_hidden_states.device, dtype=torch.bool
        )
        self.model.prepare_specmole_batch(target_layers, previous_draft, valid)

    @torch.inference_mode()
    def propose(self, *args, **kwargs):
        cpu_started = self._specmole_timer.begin()
        common_attn_metadata = kwargs["common_attn_metadata"]
        target_hidden_states = kwargs["target_hidden_states"]
        batch_size = common_attn_metadata.batch_size()
        req_ids = self._request_ids(batch_size)
        self._prepare_history(
            req_ids,
            target_hidden_states,
            common_attn_metadata,
        )
        draft_ids = super().propose(*args, **kwargs)
        current = self.model.current_draft_states(batch_size)
        # Clone the contiguous batch once.  Per-request row views retain this
        # allocation until their request is evicted, while avoiding B clone
        # kernels in every proposer invocation.
        current_batch = current.detach().clone()
        for row, req_id in enumerate(req_ids):
            self._specmole_history[req_id] = current_batch[row]
        self._specmole_timer.finish(cpu_started)
        return draft_ids

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int, *args, **kwargs) -> None:
        batch = min(
            self.max_batch_size,
            max(0, min(num_tokens, self.max_query_tokens) // self.model.model.block_size),
        )
        self.model.prepare_specmole_dummy(batch)
        self.model.set_specmole_graph_replay(False)
        try:
            return super().dummy_run(num_tokens, *args, **kwargs)
        finally:
            self.model.set_specmole_graph_replay(True)


def install_vllm_specmole() -> None:
    """Register the custom model and proposer in the current vLLM process."""
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.worker import gpu_model_runner

    ModelRegistry.register_model(
        "DFlashDraftModel",
        "speculators.models.dflash.vllm_native_specmole:DFlashQwen3ForCausalLMSpecMoLE",
    )
    gpu_model_runner.DFlashProposer = DFlashSpecMoLEProposer
