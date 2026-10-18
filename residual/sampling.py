from __future__ import annotations

import torch


def sampling_warp_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    """Apply Qwen/vLLM-style temperature + top-k + top-p sampling warpers.

    This function is intentionally shared by training mismatch/acceptance targets
    and the reference runtime evaluator. The support mask is piecewise constant;
    gradients flow through logits that remain in the sampling support.
    """
    x = logits.float()
    if temperature <= 0:
        # A verifier generation_config with do_sample=False resolves to greedy.
        out = torch.full_like(x, -torch.inf)
        out.scatter_(-1, x.argmax(dim=-1, keepdim=True), 0.0)
        return out

    x = x / float(temperature)
    vocab = x.shape[-1]

    k = int(top_k)
    if k > 0 and k < vocab:
        kth = torch.topk(x, k, dim=-1).values[..., -1, None]
        x = x.masked_fill(x < kth, -torch.inf)

    p = float(top_p)
    if 0.0 < p < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > p
        # Keep the first token that crosses the nucleus threshold.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        x = torch.full_like(x, -torch.inf).scatter(-1, sorted_idx, sorted_logits)

    return x


def sampling_policy_probs(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    warped = sampling_warp_logits(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return torch.softmax(warped, dim=-1)
