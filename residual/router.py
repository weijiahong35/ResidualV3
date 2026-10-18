from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualRouter(nn.Module):
    """V20 Router: Repair chooses expert identity; Risk controls correction gate."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        *,
        risk_input_dim: int | None = None,
        gate_init_bias: float = -1.5,
        balance_mode: str = "none",
        bias_update_rate: float = 0.0,
        load_ema: float = 0.95,
        bias_clip: float = 2.0,
        temperature: float = 1.0,
        routing_mode: str = "topk",
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.balance_mode = balance_mode
        self.bias_update_rate = float(bias_update_rate)
        self.load_ema = float(load_ema)
        self.bias_clip = float(bias_clip)
        self.temperature = float(temperature)
        self.routing_mode = routing_mode
        risk_input_dim = int(input_dim if risk_input_dim is None else risk_input_dim)

        self.repair_norm = nn.LayerNorm(input_dim)
        self.repair_hidden = nn.Linear(input_dim, input_dim)
        self.expert_head = nn.Linear(input_dim, num_experts)
        self.risk_norm = nn.LayerNorm(risk_input_dim)
        self.risk_hidden = nn.Linear(risk_input_dim, risk_input_dim)
        self.gate_head = nn.Linear(risk_input_dim, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, gate_init_bias)

        self.register_buffer("expert_bias", torch.zeros(num_experts), persistent=True)
        self.register_buffer(
            "load_ema_state", torch.full((num_experts,), 1.0 / max(num_experts, 1)), persistent=True
        )

    def set_routing_mode(self, mode: str, temperature: float | None = None) -> None:
        if mode not in {"soft", "st_topk", "topk"}:
            raise ValueError(f"Unknown routing mode {mode}")
        self.routing_mode = mode
        if temperature is not None:
            self.temperature = float(temperature)

    @torch.no_grad()
    def _update_loss_free_bias(self, selected: torch.Tensor) -> None:
        if self.balance_mode != "loss_free" or self.bias_update_rate <= 0:
            return
        load = selected.float().reshape(-1, self.num_experts).mean(dim=0)
        load = load / load.sum().clamp_min(1e-9)
        self.load_ema_state.mul_(self.load_ema).add_(load * (1 - self.load_ema))
        target = torch.full_like(load, 1.0 / self.num_experts)
        self.expert_bias.add_(self.bias_update_rate * (target - self.load_ema_state))
        if self.bias_clip > 0:
            self.expert_bias.clamp_(-self.bias_clip, self.bias_clip)
        self.expert_bias.sub_(self.expert_bias.mean())

    def forward(self, repair_x: torch.Tensor, risk_x: torch.Tensor | None = None):
        if risk_x is None:
            risk_x = repair_x
        rh = F.silu(self.repair_hidden(self.repair_norm(repair_x)))
        route_logits = self.expert_head(rh)
        if self.balance_mode == "loss_free":
            route_logits = route_logits + self.expert_bias.detach().to(route_logits.dtype)
        soft = F.softmax(route_logits / max(self.temperature, 1e-4), dim=-1)

        if self.routing_mode == "soft" or self.top_k <= 0 or self.top_k >= self.num_experts:
            probs = soft
            selected = probs > 0
        else:
            vals, ids = torch.topk(soft, self.top_k, dim=-1)
            hard = torch.zeros_like(soft).scatter(-1, ids, vals)
            hard = hard / hard.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            if self.routing_mode == "st_topk" and self.training:
                probs = hard.detach() - soft.detach() + soft
            else:
                probs = hard
            selected = hard > 0

        gh = F.silu(self.risk_hidden(self.risk_norm(risk_x)))
        gate = torch.sigmoid(self.gate_head(gh))
        if self.training:
            self._update_loss_free_bias(selected)
        # Keep old MoLE interface: row sum carries gate, expert proportions carry identity.
        effective = probs * gate
        return effective, route_logits, gate


def router_balance_regularizer(weights: torch.Tensor) -> torch.Tensor:
    p = weights.float().reshape(-1, weights.shape[-1]).clamp_min(0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    usage = p.mean(dim=0)
    uniform = torch.full_like(usage, 1.0 / usage.numel())
    return ((usage - uniform) ** 2).mean()


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logits.float(), dim=-1).square().mean()


def route_consistency_loss(current_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Stage3 weak KL anchor to the Stage2 routing partition."""
    teacher = F.softmax(teacher_logits.detach().float(), dim=-1)
    return F.kl_div(F.log_softmax(current_logits.float(), dim=-1), teacher, reduction="batchmean")
