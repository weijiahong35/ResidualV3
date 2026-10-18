"""DFlash2 source-formula full-vocabulary candidate selector baseline."""
from __future__ import annotations
import torch
from torch import nn


class DFlash2CandidateSelector(nn.Module):
    """Source-formula selector with full-vocabulary codebooks and unary Top-K.

    This is the DFlash2 *head* only. Dynamic local convolutions belong inside
    each draft decoder layer and cannot be recreated from a frozen hidden cache.
    """

    def __init__(self, vocab_size: int = 151936, hidden_size: int = 4096,
                 rank: int = 256, top_k: int = 16) -> None:
        super().__init__()
        if not 1 <= top_k <= vocab_size:
            raise ValueError('top_k outside vocabulary')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.top_k = top_k
        self.predecessor_codebook = nn.Embedding(vocab_size, rank)
        self.successor_codebook = nn.Embedding(vocab_size, rank)
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        nn.init.normal_(self.predecessor_codebook.weight, std=.02)
        nn.init.zeros_(self.successor_codebook.weight)
        nn.init.normal_(self.hidden_projection.weight, std=.02)
        self.to(dtype=torch.bfloat16)

    def candidates(self, unary: torch.Tensor) -> torch.Tensor:
        if unary.shape[-1] != self.vocab_size:
            raise ValueError('DFlash2 unary logits must be full verifier vocabulary')
        return unary.topk(self.top_k, dim=-1).indices

    def score_candidates(self, unary: torch.Tensor, hidden: torch.Tensor,
                         predecessor_ids: torch.Tensor,
                         candidate_ids: torch.Tensor) -> torch.Tensor:
        if unary.shape[-1] != self.vocab_size or hidden.shape[-1] != self.hidden_size:
            raise ValueError('DFlash2 input shape mismatch')
        context = (self.predecessor_codebook(predecessor_ids.long())
                   * self.hidden_projection(hidden.to(torch.bfloat16)))
        successor = self.successor_codebook(candidate_ids.long())
        transition = (context.unsqueeze(-2) * successor).sum(-1)
        return unary.to(torch.bfloat16).gather(-1, candidate_ids.long()) + transition

    def inference(self, unary: torch.Tensor, hidden: torch.Tensor,
                  predecessor_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = self.candidates(unary)
        return candidates, self.score_candidates(unary, hidden, predecessor_ids, candidates)

    def training_candidates(self, unary: torch.Tensor,
                            target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Experimental Speculators objective: replace weakest candidate if needed."""
        candidates = self.candidates(unary)
        present = candidates.eq(target_ids.long().unsqueeze(-1))
        augmented = candidates.clone()
        augmented[..., -1] = torch.where(present.any(-1), augmented[..., -1], target_ids.long())
        label = augmented.eq(target_ids.long().unsqueeze(-1)).long().argmax(-1)
        return augmented, label
