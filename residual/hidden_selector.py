from __future__ import annotations

import torch


def select_hidden_layer_indices(num_layers: int, requested_layers: int = 5) -> list[int]:
    """Select ordered, depth-uniform layer indices.

    If the model is shallower than requested, every layer is retained. For deeper
    models, indices span early/middle/late depth rather than taking only the tail.
    Absolute target/drafter layer numbers are intentionally not matched because
    models with different depths do not have semantically comparable absolute IDs.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if requested_layers <= 0:
        raise ValueError(f"requested_layers must be positive, got {requested_layers}")
    if num_layers <= requested_layers:
        return list(range(num_layers))
    idx = torch.linspace(0, num_layers - 1, steps=requested_layers).round().long().tolist()
    # round() can theoretically collide for unusual ratios; enforce order/uniqueness.
    idx = sorted(set(int(i) for i in idx))
    if len(idx) != requested_layers:
        unused = [i for i in range(num_layers) if i not in idx]
        for i in unused:
            idx.append(i)
            if len(idx) == requested_layers:
                break
        idx.sort()
    return idx
