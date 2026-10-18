"""Head-only full-verifier-vocabulary baselines on a frozen DFlash backbone.

DSpark delegates to this checkout's native MarkovHead and ConfidenceHead.
The DFlash2 selector matches the published inference score, while its unary
candidate restriction remains Top-K as required by the original source.
"""
from __future__ import annotations

import torch
from torch import nn

class MarkovHead(nn.Module):
    def __init__(self, *, verifier_vocab_size, draft_vocab_size, markov_rank, hidden_size, head_type):
        super().__init__()
        if head_type != 'vanilla':
            raise ValueError('this baseline implements vanilla Markov only')
        self.markov_w1 = nn.Embedding(verifier_vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, draft_vocab_size, bias=False)
    def prev_embeddings(self, ids):
        return self.markov_w1(ids.long())

class ConfidenceHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)
    def forward(self, features):
        return self.proj(features).squeeze(-1)


class DSparkHeadBaseline(nn.Module):
    def __init__(self, vocab_size: int = 151936, hidden_size: int = 4096,
                 rank: int = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.markov_head = MarkovHead(
            verifier_vocab_size=vocab_size, draft_vocab_size=vocab_size,
            markov_rank=rank, hidden_size=hidden_size, head_type='vanilla')
        self.confidence_head = ConfidenceHead(hidden_size + rank)
        # Zero bias reproduces frozen raw DFlash before head training.
        nn.init.zeros_(self.markov_head.markov_w2.weight)
        nn.init.zeros_(self.confidence_head.proj.weight)
        nn.init.zeros_(self.confidence_head.proj.bias)
        self.to(dtype=torch.bfloat16)

    def forward(self, unary: torch.Tensor, hidden: torch.Tensor,
                predecessor_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if unary.shape[-1] != self.vocab_size or hidden.shape[-1] != self.hidden_size:
            raise ValueError('full-vocabulary DSpark input shape mismatch')
        prev = self.markov_head.prev_embeddings(predecessor_ids)
        bias = self.markov_head.markov_w2(prev)
        scores = unary.to(torch.bfloat16) + bias
        conf_input = torch.cat([hidden.to(torch.bfloat16), prev], dim=-1)
        confidence = self.confidence_head(conf_input)
        return scores, confidence


