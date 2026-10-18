from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SharedLayerProjector(nn.Module):
    """LayerNorm + shared projection for one hidden-state source.

    All Drafter states across previous/current time and Layer1/Layer2 use the
    same projector, so temporal and within-block deltas live in one learned
    coordinate system.  V7 intentionally keeps this V6 alignment contract
    unchanged so the architecture change is isolated to the fusion stage.
    """

    def __init__(self, hidden_size: int, common_dim: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.common_dim = common_dim
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, common_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in {2, 3} or x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected [...,{self.hidden_size}] with rank 2 or 3, got {tuple(x.shape)}"
            )
        return self.proj(self.norm(x))


class SwiGLUFeedForward(nn.Module):
    """Standard SwiGLU feed-forward branch."""

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        if expansion < 1:
            raise ValueError("expansion must be >= 1")
        inner = dim * expansion
        self.in_proj = nn.Linear(dim, inner * 2, bias=False)
        self.out_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(value * F.silu(gate))


class StructuredLatentFusionBlock(nn.Module):
    """One normal Transformer-style fusion block over structured latent tokens.

    V7 deliberately removes ReZero and the learned hop gate.  Every deep branch
    therefore receives gradients from the first optimizer step.

    Block order:
      1) Pre-Norm cross-attention: 4 latent tokens -> source-token bank
      2) Pre-Norm self-attention among the 4 latent tokens
      3) Pre-Norm SwiGLU FFN

    Residual output projections use a small *static initialization* scaled by
    depth.  This keeps the initial residual update moderate without introducing
    a learnable scalar that can delay training or drift during long runs.
    """

    def __init__(self, dim: int, heads: int, expansion: int, depth: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.cross_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = SwiGLUFeedForward(dim, expansion=expansion)

        # Static residual-branch initialization.  Unlike ReZero, this never
        # multiplies a branch by zero, so all branch weights learn from step 1.
        # 0.02/sqrt(2N) follows the common residual-depth scaling principle.
        resid_std = 0.02 / math.sqrt(2.0 * depth)
        nn.init.normal_(self.cross_attn.out_proj.weight, mean=0.0, std=resid_std)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.normal_(self.self_attn.out_proj.weight, mean=0.0, std=resid_std)
        nn.init.zeros_(self.self_attn.out_proj.bias)
        nn.init.normal_(self.ffn.out_proj.weight, mean=0.0, std=resid_std)

    @staticmethod
    def _relative_update(update: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
        # Metric only; detached to keep diagnostics out of the backward graph.
        with torch.no_grad():
            num = update.float().norm(dim=-1)
            den = before.float().norm(dim=-1).clamp_min(1.0e-6)
            return (num / den).mean()

    def forward(
        self,
        latents: torch.Tensor,
        source_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if latents.ndim != 3:
            raise ValueError(f"Expected latents [A,Q,D], got {tuple(latents.shape)}")
        if source_tokens.ndim != 3:
            raise ValueError(
                f"Expected source_tokens [A,S,D], got {tuple(source_tokens.shape)}"
            )

        before_cross = latents
        cross_out, _ = self.cross_attn(
            query=self.cross_norm(latents),
            key=self.source_norm(source_tokens),
            value=self.source_norm(source_tokens),
            need_weights=False,
        )
        cross_ratio = self._relative_update(cross_out, before_cross)
        latents = before_cross + cross_out

        before_self = latents
        self_out, _ = self.self_attn(
            query=self.self_norm(latents),
            key=self.self_norm(latents),
            value=self.self_norm(latents),
            need_weights=False,
        )
        self_ratio = self._relative_update(self_out, before_self)
        latents = before_self + self_out

        before_ffn = latents
        ffn_out = self.ffn(self.ffn_norm(latents))
        ffn_ratio = self._relative_update(ffn_out, before_ffn)
        latents = before_ffn + ffn_out

        return latents, cross_ratio, self_ratio, ffn_ratio


class ProgressivePredictorBranch(nn.Module):
    """One complete V10-style predictor branch for one task and one prefix.

    Each instance owns the full V10 branch structure: branch-specific depth
    pooling, temporal/current encoders, fusion gate, role latents, four
    cross-attention + self-attention + SwiGLU fusion blocks, output norm, and
    its own task heads.  Risk and Repair use different instances with fully
    independent weights, while consuming the SAME V10 information prepared by
    the shared stem.
    """

    NUM_ROLE_LATENTS = 4

    def __init__(
        self,
        prefix_layers: int,
        hidden_dim: int,
        direction_dim: int,
        block_size: int,
        attention_heads: int,
        predictor_depth: int,
        ffn_expansion: int,
        *,
        task: str,
    ) -> None:
        super().__init__()
        if prefix_layers < 2:
            raise ValueError("prefix_layers must be >= 2")
        if task not in {"risk", "repair"}:
            raise ValueError("task must be 'risk' or 'repair'")
        self.prefix_layers = int(prefix_layers)
        self.hidden_dim = int(hidden_dim)
        self.block_size = int(block_size)
        self.task = task

        # Full V10 branch capacity. None of these weights are shared between
        # Risk and Repair, or between p2/p3/p4.
        self.current_layer_scores = nn.Parameter(torch.zeros(prefix_layers))
        self.temporal_layer_scores = nn.Parameter(torch.zeros(prefix_layers))
        self.depth_layer_scores = nn.Parameter(torch.zeros(prefix_layers - 1))

        self.temporal_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.current_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.base_fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.constant_(self.base_fusion_gate.bias, -1.0)
        self.base_fusion_norm = nn.LayerNorm(hidden_dim)

        self.role_embed = nn.Parameter(torch.zeros(self.NUM_ROLE_LATENTS, hidden_dim))
        self.latent_input_norm = nn.LayerNorm(hidden_dim)
        self.fusion_blocks = nn.ModuleList(
            [
                StructuredLatentFusionBlock(
                    hidden_dim,
                    heads=attention_heads,
                    expansion=ffn_expansion,
                    depth=predictor_depth,
                )
                for _ in range(predictor_depth)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        if task == "risk":
            # Exact V10 bias/severity head: positive NLA prediction per slot.
            self.severity_head = nn.Linear(hidden_dim, block_size)
        else:
            # Exact V10 repair head: one rejection-conditioned direction per block.
            self.repair_head = nn.Linear(hidden_dim, direction_dim)

        nn.init.normal_(self.role_embed, std=0.02)

    @staticmethod
    def _weighted_depth_pool(tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != scores.numel():
            raise ValueError(
                f"Branch expected {scores.numel()} tokens, got {tokens.shape[1]}"
            )
        weights = F.softmax(scores, dim=0)
        return (tokens * weights.view(1, -1, 1)).sum(dim=1)

    @staticmethod
    def _latent_diversity(latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = F.normalize(latents.float(), dim=-1, eps=1.0e-6)
            sim = torch.matmul(z, z.transpose(1, 2))
            q = sim.shape[1]
            if q <= 1:
                return torch.zeros((), device=latents.device, dtype=torch.float32)
            mask = ~torch.eye(q, device=latents.device, dtype=torch.bool).unsqueeze(0)
            offdiag = sim.masked_select(mask.expand_as(sim))
            return 1.0 - offdiag.mean()

    def forward(
        self,
        *,
        history: torch.Tensor,
        d_cur: torch.Tensor,
        temporal_delta: torch.Tensor,
        depth_delta: torch.Tensor,
        source_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        k = self.prefix_layers
        if d_cur.ndim != 3 or d_cur.shape[1] != k:
            raise ValueError(f"p{k} branch requires d_cur [A,{k},D], got {tuple(d_cur.shape)}")
        if temporal_delta.shape != d_cur.shape:
            raise ValueError("temporal_delta must match d_cur")
        if depth_delta.ndim != 3 or depth_delta.shape[1] != k - 1:
            raise ValueError(f"p{k} branch requires {k-1} depth deltas")

        current_summary = self._weighted_depth_pool(d_cur, self.current_layer_scores)
        temporal_summary = self._weighted_depth_pool(
            temporal_delta, self.temporal_layer_scores
        )
        depth_summary = self._weighted_depth_pool(depth_delta, self.depth_layer_scores)
        temporal_abs_summary = self._weighted_depth_pool(
            temporal_delta.abs(), self.temporal_layer_scores
        )
        depth_abs_summary = self._weighted_depth_pool(
            depth_delta.abs(), self.depth_layer_scores
        )
        last_current = d_cur[:, -1, :]

        temporal = self.temporal_encoder(
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
        current = self.current_encoder(
            torch.cat(
                [d_cur[:, 0, :], last_current, last_current - d_cur[:, 0, :]],
                dim=-1,
            )
        )

        base_gate = torch.sigmoid(
            self.base_fusion_gate(torch.cat([history, temporal], dim=-1))
        )
        repair_seed = self.base_fusion_norm(history + base_gate * temporal)

        latents = torch.stack([history, temporal, current, repair_seed], dim=1)
        latents = self.latent_input_norm(latents + self.role_embed.unsqueeze(0))

        cross_ratios: list[torch.Tensor] = []
        self_ratios: list[torch.Tensor] = []
        ffn_ratios: list[torch.Tensor] = []
        for block in self.fusion_blocks:
            latents, cross_r, self_r, ffn_r = block(latents, source_tokens)
            cross_ratios.append(cross_r)
            self_ratios.append(self_r)
            ffn_ratios.append(ffn_r)

        latents = self.output_norm(latents)
        latent = latents[:, 3, :]
        out: dict[str, torch.Tensor] = {
            "latent": latent,
            "fusion_gate": base_gate,
            "cross_update_ratio": torch.stack(cross_ratios).mean(),
            "self_update_ratio": torch.stack(self_ratios).mean(),
            "ffn_update_ratio": torch.stack(ffn_ratios).mean(),
            "latent_diversity": self._latent_diversity(latents),
            "prefix_layers": torch.tensor(k, device=latent.device),
        }

        if self.task == "risk":
            severity = F.softplus(self.severity_head(latent))
            out["severity"] = severity
        else:
            out["repair_direction"] = F.normalize(
                self.repair_head(latent), dim=-1, eps=1.0e-6
            )
        return out


class MismatchPredictor(nn.Module):
    """V20.6 Dual-V10 soft-logit Predictor.

    The information contract is exactly the V10-style lagged contract:
    previous Target layers + previous Draft layers + current progressive Draft
    prefix + temporal/depth deltas.  That information is projected/prepared
    once by the shared V10 stem.

    For EACH progressive decision p2/p3/p4 there are then TWO complete,
    independent V10 fusion predictors:

      * Risk/Bias Predictor: full V10 branch, trained only to predict V10 NLA
        severity from the soft Target/Draft logit geometry.

      * Repair Predictor: another full V10 branch with its OWN independent
        weights and the same V10 inputs, trained only for the V10 soft
        rejection-conditioned repair direction.

    Therefore only the INFORMATION/STEM is shared.  Risk and Repair do NOT
    share a latent, fusion stack, attention block, or task head.
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_dim: int,
        direction_dim: int,
        block_size: int,
        max_target_layers: int = 5,
        max_drafter_layers: int = 5,
        attention_heads: int = 4,
        predictor_depth: int = 4,
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        if max_drafter_layers < 3:
            raise ValueError("Progressive MoLE requires at least three Drafter layers")
        if hidden_dim % attention_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by attention_heads={attention_heads}"
            )
        if predictor_depth < 1:
            raise ValueError("predictor_depth must be >= 1")
        if ffn_expansion < 1:
            raise ValueError("ffn_expansion must be >= 1")

        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim
        self.block_size = int(block_size)
        self.max_target_layers = max_target_layers
        self.max_drafter_layers = max_drafter_layers
        self.predictor_depth = predictor_depth
        self.prefix_lengths = tuple(range(2, max_drafter_layers))

        # Shared stem: only components whose semantics are identical for all
        # decisions live here.
        self.target_projector = SharedLayerProjector(hidden_size, hidden_dim)
        self.drafter_projector = SharedLayerProjector(hidden_size, hidden_dim)

        self.target_layer_embed = nn.Parameter(torch.zeros(max_target_layers, hidden_dim))
        self.drafter_layer_embed = nn.Parameter(torch.zeros(max_drafter_layers, hidden_dim))
        self.target_source_embed = nn.Parameter(torch.zeros(1, hidden_dim))
        self.drafter_source_embed = nn.Parameter(torch.zeros(1, hidden_dim))
        self.temporal_delta_source_embed = nn.Parameter(torch.zeros(1, hidden_dim))
        self.depth_delta_source_embed = nn.Parameter(torch.zeros(1, hidden_dim))
        self.previous_time_embed = nn.Parameter(torch.zeros(1, hidden_dim))
        self.current_time_embed = nn.Parameter(torch.zeros(1, hidden_dim))

        self.target_layer_scores = nn.Parameter(torch.zeros(max_target_layers))
        self.drafter_layer_scores = nn.Parameter(torch.zeros(max_drafter_layers))
        self.history_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Same V10 information, two FULL independent predictors per pK.
        self.risk_branches = nn.ModuleDict(
            {
                f"p{k}": ProgressivePredictorBranch(
                    prefix_layers=k,
                    hidden_dim=hidden_dim,
                    direction_dim=direction_dim,
                    block_size=block_size,
                    attention_heads=attention_heads,
                    predictor_depth=predictor_depth,
                    ffn_expansion=ffn_expansion,
                    task="risk",
                )
                for k in self.prefix_lengths
            }
        )
        self.repair_branches = nn.ModuleDict(
            {
                f"p{k}": ProgressivePredictorBranch(
                    prefix_layers=k,
                    hidden_dim=hidden_dim,
                    direction_dim=direction_dim,
                    block_size=block_size,
                    attention_heads=attention_heads,
                    predictor_depth=predictor_depth,
                    ffn_expansion=ffn_expansion,
                    task="repair",
                )
                for k in self.prefix_lengths
            }
        )

        for p in (
            self.target_layer_embed,
            self.drafter_layer_embed,
            self.target_source_embed,
            self.drafter_source_embed,
            self.temporal_delta_source_embed,
            self.depth_delta_source_embed,
            self.previous_time_embed,
            self.current_time_embed,
        ):
            nn.init.normal_(p, std=0.02)

    @staticmethod
    def _weighted_depth_pool(tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        l = tokens.shape[1]
        if l == 0:
            raise ValueError("Cannot pool an empty layer sequence")
        weights = F.softmax(scores[:l], dim=0)
        return (tokens * weights.view(1, l, 1)).sum(dim=1)

    def branch(
        self, prefix_layers: int, *, task: str
    ) -> ProgressivePredictorBranch:
        key = f"p{int(prefix_layers)}"
        if task == "risk":
            table = self.risk_branches
        elif task == "repair":
            table = self.repair_branches
        else:
            raise ValueError("task must be 'risk' or 'repair'")
        if key not in table:
            raise ValueError(
                f"No {task} Predictor branch for prefix {prefix_layers}; "
                f"available={list(self.prefix_lengths)}"
            )
        return table[key]

    def project_current(self, current_drafter_layers: torch.Tensor) -> torch.Tensor:
        """Shared projection for current Drafter states; accepts [A,H] or [A,K,H]."""
        return self.drafter_projector(current_drafter_layers)

    def prepare_history(
        self,
        previous_target_layers: torch.Tensor,
        previous_drafter_layers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Project/encode lag-1 history once for reuse by all branches."""
        if previous_target_layers.ndim != 3 or previous_target_layers.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected previous_target_layers [A,L,{self.hidden_size}], "
                f"got {tuple(previous_target_layers.shape)}"
            )
        if previous_drafter_layers.ndim != 3 or previous_drafter_layers.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected previous_drafter_layers [A,L,{self.hidden_size}], "
                f"got {tuple(previous_drafter_layers.shape)}"
            )
        if previous_target_layers.shape[1] > self.max_target_layers:
            raise ValueError(
                f"Got {previous_target_layers.shape[1]} Target layers; max={self.max_target_layers}"
            )
        if previous_drafter_layers.shape[1] > self.max_drafter_layers:
            raise ValueError(
                f"Got {previous_drafter_layers.shape[1]} Drafter layers; max={self.max_drafter_layers}"
            )
        if previous_target_layers.shape[0] != previous_drafter_layers.shape[0]:
            raise ValueError("previous Target/Drafter stacks must have the same anchor count")

        t_prev_tokens = self.target_projector(previous_target_layers)
        d_prev_tokens = self.drafter_projector(previous_drafter_layers)
        t_summary = self._weighted_depth_pool(t_prev_tokens, self.target_layer_scores)
        d_summary = self._weighted_depth_pool(d_prev_tokens, self.drafter_layer_scores)
        history = self.history_encoder(torch.cat([t_summary, d_summary], dim=-1))

        lt = t_prev_tokens.shape[1]
        ld = d_prev_tokens.shape[1]
        t_history = (
            t_prev_tokens
            + self.target_layer_embed[:lt].unsqueeze(0)
            + self.target_source_embed.unsqueeze(1)
            + self.previous_time_embed.unsqueeze(1)
        )
        d_history = (
            d_prev_tokens
            + self.drafter_layer_embed[:ld].unsqueeze(0)
            + self.drafter_source_embed.unsqueeze(1)
            + self.previous_time_embed.unsqueeze(1)
        )
        return {
            "d_prev_tokens": d_prev_tokens,
            "history": history,
            "history_sources": torch.cat([t_history, d_history], dim=1),
        }

    def _forward_projected_prefix(
        self,
        prepared: dict[str, torch.Tensor],
        d_cur: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if d_cur.ndim != 3 or d_cur.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected projected current prefix [A,K,{self.hidden_dim}], got {tuple(d_cur.shape)}"
            )
        k = int(d_cur.shape[1])
        risk_branch = self.branch(k, task="risk")
        repair_branch = self.branch(k, task="repair")
        d_prev_tokens = prepared["d_prev_tokens"]
        if d_prev_tokens.shape[0] != d_cur.shape[0]:
            raise ValueError("previous/current states must have the same anchor count")
        if d_prev_tokens.shape[1] < k:
            raise ValueError(
                f"Need previous Drafter Layers 1..{k} for temporal deltas, "
                f"but only {d_prev_tokens.shape[1]} were provided"
            )

        d_prev_aligned = d_prev_tokens[:, :k, :]
        temporal_delta = d_cur - d_prev_aligned
        depth_delta = d_cur[:, 1:, :] - d_cur[:, :-1, :]

        current_tokens = (
            d_cur
            + self.drafter_layer_embed[:k].unsqueeze(0)
            + self.drafter_source_embed.unsqueeze(1)
            + self.current_time_embed.unsqueeze(1)
        )
        temporal_tokens = (
            temporal_delta
            + self.drafter_layer_embed[:k].unsqueeze(0)
            + self.temporal_delta_source_embed.unsqueeze(1)
            + self.current_time_embed.unsqueeze(1)
        )
        depth_tokens = (
            depth_delta
            + self.drafter_layer_embed[1:k].unsqueeze(0)
            + self.depth_delta_source_embed.unsqueeze(1)
            + self.current_time_embed.unsqueeze(1)
        )
        source_tokens = torch.cat(
            [prepared["history_sources"], current_tokens, temporal_tokens, depth_tokens],
            dim=1,
        )

        common = dict(
            history=prepared["history"],
            d_cur=d_cur,
            temporal_delta=temporal_delta,
            depth_delta=depth_delta,
            source_tokens=source_tokens,
        )
        risk = risk_branch(**common)
        repair = repair_branch(**common)

        b = self.block_size
        risk_latent = risk["latent"].unsqueeze(1).expand(-1, b, -1)
        repair_latent = repair["latent"].unsqueeze(1).expand(-1, b, -1)
        repair_direction_per_pos = repair["repair_direction"].unsqueeze(1).expand(-1, b, -1)

        # Convert predicted V10 NLA back to a continuous overlap estimate.
        # This remains soft; no argmax or sampling is used.
        severity = risk["severity"]
        overlap = torch.exp(-severity.float()).clamp(1.0e-6, 1.0).to(severity.dtype)
        survival = torch.cumprod(overlap, dim=-1)
        reach_before = torch.cat(
            [
                torch.ones_like(overlap[:, :1]),
                torch.cumprod(overlap[:, :-1], dim=-1),
            ],
            dim=-1,
        )
        first_reject = reach_before * (1.0 - overlap)
        no_reject = survival[:, -1:]

        out = {
            "severity": severity,
            "distribution_nla": severity,
            "distribution_overlap": overlap,
            "runtime_nla": severity,          # compatibility alias; SOFT, not greedy
            "runtime_acceptance": overlap,    # compatibility alias; SOFT, not greedy
            "tv": 1.0 - overlap,
            "survival": survival,
            "reach_prob": reach_before,
            "first_reject": first_reject,
            "no_reject": no_reject,
            "first_reject_with_no_reject": torch.cat([first_reject, no_reject], dim=-1),
            "expected_accepted_length": survival.sum(dim=-1),
            "repair_direction": repair["repair_direction"],
            "repair_direction_per_pos": repair_direction_per_pos,
            "repair_magnitude": 1.0 - overlap,  # derived compatibility value only
            "latent": risk["latent"],
            "risk_block_latent": risk["latent"],
            "repair_block_latent": repair["latent"],
            "risk_latent": risk_latent,
            "repair_latent": repair_latent,
            "repair_position_latent": repair_latent,
            "fusion_gate": 0.5 * (risk["fusion_gate"] + repair["fusion_gate"]),
            "cross_update_ratio": 0.5 * (
                risk["cross_update_ratio"] + repair["cross_update_ratio"]
            ),
            "self_update_ratio": 0.5 * (
                risk["self_update_ratio"] + repair["self_update_ratio"]
            ),
            "ffn_update_ratio": 0.5 * (
                risk["ffn_update_ratio"] + repair["ffn_update_ratio"]
            ),
            "latent_diversity": 0.5 * (
                risk["latent_diversity"] + repair["latent_diversity"]
            ),
            "risk_fusion_gate": risk["fusion_gate"],
            "repair_fusion_gate": repair["fusion_gate"],
            "risk_cross_update_ratio": risk["cross_update_ratio"],
            "repair_cross_update_ratio": repair["cross_update_ratio"],
            "risk_self_update_ratio": risk["self_update_ratio"],
            "repair_self_update_ratio": repair["self_update_ratio"],
            "risk_ffn_update_ratio": risk["ffn_update_ratio"],
            "repair_ffn_update_ratio": repair["ffn_update_ratio"],
            "risk_latent_diversity": risk["latent_diversity"],
            "repair_latent_diversity": repair["latent_diversity"],
            "prefix_layers": torch.tensor(k, device=d_cur.device),
        }
        return out

    def forward_from_projected_history(
        self,
        prepared: dict[str, torch.Tensor],
        projected_current_layers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._forward_projected_prefix(prepared, projected_current_layers)

    def forward_from_history(
        self,
        prepared: dict[str, torch.Tensor],
        current_drafter_layers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Select the independent pK branch from one raw current prefix [A,K,H]."""
        if current_drafter_layers.ndim != 3 or current_drafter_layers.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected current_drafter_layers [A,K,{self.hidden_size}], "
                f"got {tuple(current_drafter_layers.shape)}"
            )
        k = int(current_drafter_layers.shape[1])
        if k not in self.prefix_lengths:
            raise ValueError(
                f"Current prefix must be one of {list(self.prefix_lengths)}, got {k}"
            )
        return self._forward_projected_prefix(
            prepared, self.project_current(current_drafter_layers)
        )

    def forward_prefixes_from_history(
        self,
        prepared: dict[str, torch.Tensor],
        current_drafter_layers: torch.Tensor,
        *,
        prefix_lengths: list[int] | tuple[int, ...] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Evaluate p2/p3/... while projecting the current layer stack only once."""
        if current_drafter_layers.ndim != 3 or current_drafter_layers.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected current_drafter_layers [A,L,{self.hidden_size}], got {tuple(current_drafter_layers.shape)}"
            )
        max_k = int(current_drafter_layers.shape[1])
        if prefix_lengths is None:
            prefix_lengths = [k for k in self.prefix_lengths if k <= max_k]
        if not prefix_lengths:
            return []
        max_requested = max(int(k) for k in prefix_lengths)
        if max_requested > max_k:
            raise ValueError(
                f"Requested prefix {max_requested}, but only {max_k} current layers exist"
            )
        # Do not project the final full-depth state when it is not a routing
        # prefix (e.g. D5 for a five-layer model).
        projected = self.project_current(current_drafter_layers[:, :max_requested, :])
        out: list[dict[str, torch.Tensor]] = []
        for k in prefix_lengths:
            if k not in self.prefix_lengths:
                raise ValueError(f"No branch for prefix {k}")
            if k > max_k:
                raise ValueError(f"Requested prefix {k}, but only {max_k} current layers exist")
            out.append(self._forward_projected_prefix(prepared, projected[:, :k, :]))
        return out

    def forward_prefixes(
        self,
        previous_target_layers: torch.Tensor,
        previous_drafter_layers: torch.Tensor,
        current_drafter_layers: torch.Tensor,
        *,
        prefix_lengths: list[int] | tuple[int, ...] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        prepared = self.prepare_history(previous_target_layers, previous_drafter_layers)
        return self.forward_prefixes_from_history(
            prepared,
            current_drafter_layers,
            prefix_lengths=prefix_lengths,
        )

    def forward(
        self,
        previous_target_layers: torch.Tensor,
        previous_drafter_layers: torch.Tensor,
        current_drafter_layers: torch.Tensor,
        current_drafter_second: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """One-prefix entry point; legacy (D1,D2) two-tensor call is supported."""
        if current_drafter_second is not None:
            if current_drafter_layers.ndim != 2 or current_drafter_second.ndim != 2:
                raise ValueError("Legacy current_drafter_first/second inputs must both be [A,H]")
            current_drafter_layers = torch.stack(
                [current_drafter_layers, current_drafter_second], dim=1
            )
        elif current_drafter_layers.ndim == 2:
            raise ValueError(
                "A single [A,H] current state is insufficient; provide [A,K,H] with a routed K"
            )
        prepared = self.prepare_history(previous_target_layers, previous_drafter_layers)
        return self.forward_from_history(prepared, current_drafter_layers)
