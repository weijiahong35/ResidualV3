from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass



class SwiGLUBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        inner = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, inner * 2)
        self.down = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, g = self.up(h).chunk(2, dim=-1)
        h = self.down(F.silu(g) * a)
        return x + self.dropout(h)


def normalized_rows(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return x / x.float().norm(dim=-1, keepdim=True).clamp_min(eps).to(x.dtype)


@dataclass
class Rollout:
    logits: torch.Tensor                 # [1,7,V], final BF16 repair logits
    tokens: torch.Tensor                 # [1,7], actual choices
    predecessors: torch.Tensor           # [1,7], verified anchor then choices
    predictions: list[dict[str, torch.Tensor]]
    verified_prefix: tuple[int, ...]
