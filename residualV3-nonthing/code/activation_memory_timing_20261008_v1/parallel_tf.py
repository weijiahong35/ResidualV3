"""Batch ONLY teacher-forced no-soft training; native sampling stays recursive."""
import torch
from runtime.precision import bf16_logit_update
from repair.model import project_candidates


def teacher_forced_logits(head, raw, hidden, basis, anchor, answers):
    if head.use_soft:
        raise ValueError('Soft recurrence cannot use this TF batching path')
    if answers.shape != raw.shape[:2]:
        raise ValueError('Expected one fixed teacher token per prediction slot')
    hp, qp, hn, ids, _, outrows = head.prepare(raw, hidden, basis)
    previous = torch.cat((anchor[:, None], answers[:, :-1]), 1)
    coeff, dose = head.predict(hp, head.feature_weight[previous], qp, None)
    direction = project_candidates(coeff, outrows)
    corrected, _ = bf16_logit_update(raw.gather(-1, ids), direction, dose, hn)
    return raw.scatter(-1, ids, corrected)
