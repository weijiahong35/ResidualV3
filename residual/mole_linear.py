from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ResidualMoLE(nn.Module):
    """Independent shared+routed low-rank residual after a complete frozen MLP.

    This module does not wrap or modify a base projection. It maps the decoder
    layer's normalized MLP input H->H and its
    output is added immediately before the layer's second residual add.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        num_experts: int = 8,
        rank: int = 8,
        alpha: float = 16.0,
        shared_rank: int | None = None,
        dropout: float = 0.0,
        max_relative_update: float = 0.20,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.rank = int(rank)
        self.shared_rank = int(rank if shared_rank is None else shared_rank)
        self.scaling = float(alpha) / float(rank)
        self.shared_scaling = float(alpha) / float(self.shared_rank)
        self.max_relative_update = float(max_relative_update)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.shared_A = nn.Parameter(torch.empty(self.shared_rank, self.hidden_size))
        self.shared_B = nn.Parameter(torch.zeros(self.hidden_size, self.shared_rank))
        self.expert_A = nn.Parameter(
            torch.empty(self.num_experts, self.rank, self.hidden_size)
        )
        self.expert_B = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size, self.rank)
        )
        nn.init.kaiming_uniform_(self.shared_A, a=math.sqrt(5))
        for expert in range(self.num_experts):
            nn.init.kaiming_uniform_(self.expert_A[expert], a=math.sqrt(5))

        self._last_update_ratio: torch.Tensor | None = None
        self._last_gate_mean: torch.Tensor | None = None
        self._last_expert_output_cosine: torch.Tensor | None = None

    @property
    def last_update_ratio(self) -> torch.Tensor | None:
        return self._last_update_ratio

    @property
    def last_gate_mean(self) -> torch.Tensor | None:
        return self._last_gate_mean

    @property
    def last_expert_output_cosine(self) -> torch.Tensor | None:
        return self._last_expert_output_cosine

    @torch.no_grad()
    def expert_weight_cosines(self) -> tuple[torch.Tensor, torch.Tensor]:
        def off_diagonal_mean(x: torch.Tensor) -> torch.Tensor:
            if self.num_experts < 2:
                return x.new_zeros((), dtype=torch.float32)
            z = F.normalize(x.float().reshape(self.num_experts, -1), dim=-1, eps=1e-6)
            sim = z @ z.T
            mask = ~torch.eye(self.num_experts, device=sim.device, dtype=torch.bool)
            return sim[mask].mean()
        return off_diagonal_mean(self.expert_A), off_diagonal_mean(self.expert_B)

    def forward(
        self,
        x: torch.Tensor,
        *,
        route_weights: torch.Tensor,
        gate: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        gate_cap: float = 1.0,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected [1,A*B,{self.hidden_size}], got {tuple(x.shape)}")
        if route_weights.ndim != 3 or route_weights.shape[-1] != self.num_experts:
            raise ValueError("route_weights must be [A,B,E]")
        if gate.shape != (*route_weights.shape[:2], 1):
            raise ValueError("gate must be [A,B,1]")
        if x.shape[1] != route_weights.shape[0] * route_weights.shape[1]:
            raise ValueError("flattened DFlash length does not match [A,B] control")

        weights = route_weights.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        g = gate.float().clamp(0.0, 1.0) * float(max(0.0, min(gate_cap, 1.0)))
        if valid_mask is not None:
            if valid_mask.shape != route_weights.shape[:2]:
                raise ValueError("valid_mask must be [A,B]")
            g = g * valid_mask.unsqueeze(-1).float()

        xd = self.dropout(x).reshape(-1, self.hidden_size)
        shared = F.linear(F.linear(xd, self.shared_A), self.shared_B)
        shared = shared * self.shared_scaling
        # [N,E,R] then [N,E,H].  This keeps expert routing position-aware.
        low = torch.einsum("nh,erh->ner", xd, self.expert_A)
        routed_all = torch.einsum("ner,ehr->neh", low, self.expert_B)
        routed = (
            routed_all
            * weights.reshape(-1, self.num_experts, 1).to(routed_all.dtype)
        ).sum(dim=1)
        routed = routed * self.scaling
        correction = (shared + routed) * g.reshape(-1, 1).to(shared.dtype)

        if self.max_relative_update > 0:
            input_norm = xd.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            corr_norm = correction.float().norm(dim=-1, keepdim=True)
            cap = self.max_relative_update * input_norm
            correction = correction * torch.minimum(
                torch.ones_like(corr_norm), cap / corr_norm.clamp_min(1e-6)
            ).to(correction.dtype)

        with torch.no_grad():
            denom = xd.float().norm(dim=-1).clamp_min(1e-6)
            self._last_update_ratio = (
                correction.float().norm(dim=-1) / denom
            ).mean().detach()
            self._last_gate_mean = g.mean().detach()
            sample = routed_all[: min(routed_all.shape[0], 64)]
            expert_vectors = sample.permute(1, 0, 2).reshape(self.num_experts, -1).float()
            expert_vectors = F.normalize(expert_vectors, dim=-1, eps=1e-6)
            similarity = expert_vectors @ expert_vectors.T
            off_diag = ~torch.eye(
                self.num_experts, device=similarity.device, dtype=torch.bool
            )
            self._last_expert_output_cosine = (
                similarity[off_diag].mean().detach()
                if self.num_experts > 1
                else similarity.new_zeros(())
            )
        return correction.reshape_as(x)
