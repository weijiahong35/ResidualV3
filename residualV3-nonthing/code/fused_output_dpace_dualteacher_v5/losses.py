"""Original prefix OUTPUT and official-style target-argmax CE + D-PACE.

Inputs remain BF16; per-position fused reductions and small prefix tensors FP32.
Targets are frozen; native decoding is not changed by this module.
"""
import torch
from fused_kernels import fused_tv_loss, fused_ce_loss


def output_loss(logits, teacher):
    overlap = (1.0 - fused_tv_loss(logits, teacher)).clamp(0.0, 1.0)
    return 1.0 - overlap.cumprod(-1).mean()


def dpace_loss(logits, teacher, alpha=0.5):
    ce = fused_ce_loss(logits, teacher)
    # All supplied positions are the 15 valid prediction slots; known anchor
    # slot 0 is already removed. Same valid-position denominator as upstream.
    with torch.no_grad():
        smooth = alpha + (1.0 - alpha) * torch.exp(-ce)
        prefix = smooth.cumprod(-1)
        weights = prefix.flip(-1).cumsum(-1).flip(-1)
    return (ce * weights).mean()


def objective(logits, teacher, arm):
    if arm == 'OUTPUT':
        return output_loss(logits, teacher)
    if arm == 'CE_DPACE':
        return dpace_loss(logits, teacher)
    if arm == 'CE_DPACE_OUTPUT':
        # Prespecified unscaled sum; these coefficients are not tuned optima.
        return dpace_loss(logits, teacher) + output_loss(logits, teacher)
    raise ValueError(arm)


def eager_objective(logits, teacher, arm):
    if arm == 'OUTPUT':
        p = teacher.float().softmax(-1)
        q = logits.float().log_softmax(-1).exp()
        return 1.0 - torch.minimum(p, q).sum(-1).clamp(0, 1).cumprod(-1).mean()
    ce = torch.nn.functional.cross_entropy(
        logits.float().flatten(0, 1), teacher.argmax(-1).flatten(),
        reduction='none').reshape(logits.shape[:2])
    with torch.no_grad():
        prefix = (0.5 + 0.5 * torch.exp(-ce)).cumprod(-1)
        weights = prefix.flip(-1).cumsum(-1).flip(-1)
    value = (ce * weights).mean()
    if arm == 'CE_DPACE_OUTPUT':
        value = value + eager_objective(logits, teacher, 'OUTPUT')
    return value
