from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PreviousRoundState:
    """Inference-facing historical state for hybrid routing.

    ``target_layers`` and ``drafter_layers`` are complete hidden states from the
    already-finished previous speculative block. Runtime additionally supplies
    the current block's Drafter Layer-1 and Layer-2 states ``D_t^1`` and
    ``D_t^2`` directly to Predictor; these current probes are not stored in this
    previous-state cache.
    """

    target_layers: torch.Tensor   # [A,Lt,H] or [1,Lt,H] for one runtime request
    drafter_layers: torch.Tensor  # [A,Ld,H] or [1,Ld,H]


class PreviousRoundStateCache:
    def __init__(self) -> None:
        self._state: PreviousRoundState | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, target_layers: torch.Tensor, drafter_layers: torch.Tensor) -> None:
        self._state = PreviousRoundState(target_layers.detach(), drafter_layers.detach())

    def get(self) -> PreviousRoundState | None:
        return self._state
