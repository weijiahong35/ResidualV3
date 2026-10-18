"""BF16 head recurrence; FP32 probability boundary; actual selected-token feedback."""
import torch

POLICY = 'bf16-head-v2-fp32-prob-loss'

def sample_logits(logits, temperature, generator=None):
    if temperature < 0:
        raise ValueError('temperature must be nonnegative')
    if temperature == 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

def sample_block(head, raw, hidden, anchor, *, temperature=0., generator=None, sample_step=None):
    if raw.dtype != torch.bfloat16 or hidden.dtype != torch.bfloat16:
        raise TypeError('native raw/hidden must be BF16')
    if head is not None and any(p.dtype != torch.bfloat16 for p in head.parameters()):
        raise TypeError('head parameters must be BF16')
    choose = sample_step or (lambda scores, j: sample_logits(scores, temperature, generator))
    ids, rows, previous = [], [], []
    prev = anchor.long()
    selector = head is not None and hasattr(head, 'score_candidates')
    with torch.autocast(raw.device.type, enabled=False):
        if selector:
            # The candidate set depends only on raw unary logits; compute all positions once.
            candidates = head.candidates(raw)
            unary = raw.gather(-1, candidates)
            gates = head.hidden_projection(hidden)
            successors = head.successor_codebook(candidates)
        for j in range(raw.shape[1]):
            previous.append(prev)
            if head is None:
                scores = raw[:, j]
            elif selector:
                context = head.predecessor_codebook(prev) * gates[:, j]
                transition = (context[:, None, :] * successors[:, j]).sum(-1)
                local = unary[:, j] + transition
                scores = raw[:, j].new_full(raw[:, j].shape, float('-inf'))
                scores.scatter_(-1, candidates[:, j], local)
            else:
                scores, _ = head(raw[:, j], hidden[:, j], prev)
            if scores.dtype != torch.bfloat16:
                raise TypeError('sampling boundary must be BF16')
            prev = choose(scores, j).long()
            ids.append(prev)
            rows.append(scores)
    return torch.stack(ids, 1), torch.stack(rows, 1), torch.stack(previous, 1)
