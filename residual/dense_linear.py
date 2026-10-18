from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class DenseCorrectionGate(nn.Module):
    """One scalar correction gate per anchor; no expert routing."""

    def __init__(self, input_dim: int, init_bias: float = -1.5) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, float(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(x.float())).to(dtype=x.dtype)


class DenseLoRALinear(nn.Module):
    """Frozen Linear + one gated low-rank residual (strict no-MoLE ablation).

    There are no experts and no Router. A per-anchor scalar gate controls whether
    correction is applied, and the same token-wise relative-update safety cap as
    the MoLE path prevents correction blow-ups.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        block_size: int,
        max_relative_update: float = 0.20,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"DenseLoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / rank
        self.block_size = int(block_size)
        self.max_relative_update = float(max_relative_update)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self._gate: torch.Tensor | None = None
        self._last_update_ratio: torch.Tensor | None = None
        self._last_gate_mean: torch.Tensor | None = None

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def last_update_ratio(self) -> torch.Tensor | None:
        return self._last_update_ratio

    @property
    def last_gate_mean(self) -> torch.Tensor | None:
        return self._last_gate_mean

    def set_gate(self, gate: torch.Tensor | None) -> None:
        self._gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        gate = self._gate
        if gate is None:
            self._last_update_ratio = None
            self._last_gate_mean = None
            return out
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected DFlash [1,A*B,H] activations, got {tuple(x.shape)}")
        if gate.ndim == 1:
            gate = gate.unsqueeze(-1)
        if gate.shape[-1] != 1:
            raise ValueError(f"Dense correction gate must be [A,1], got {tuple(gate.shape)}")
        if x.shape[1] != gate.shape[0] * self.block_size:
            raise ValueError(
                f"gate anchors={gate.shape[0]}, block_size={self.block_size} do not match T={x.shape[1]}"
            )

        token_gate = gate.repeat_interleave(self.block_size, dim=0).unsqueeze(0)
        low = F.linear(self.dropout(x), self.lora_A)
        correction = F.linear(low, self.lora_B) * self.scaling
        correction = correction * token_gate.to(correction.dtype)

        if self.max_relative_update > 0:
            base_norm = out.float().norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            corr_norm = correction.float().norm(dim=-1, keepdim=True)
            cap = self.max_relative_update * base_norm
            scale = torch.minimum(torch.ones_like(corr_norm), cap / corr_norm.clamp_min(1.0e-6))
            correction = correction * scale.to(correction.dtype)

        with torch.no_grad():
            base_norm = out.float().norm(dim=-1).clamp_min(1.0e-6)
            corr_norm = correction.float().norm(dim=-1)
            self._last_update_ratio = (corr_norm / base_norm).mean().detach()
            self._last_gate_mean = gate.float().mean().detach()
        return out + correction
