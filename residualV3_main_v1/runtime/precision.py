"""Explicit BF16 model arithmetic; FP32 is reserved for stable reductions/losses.

This is a new numerical contract, not bitwise reproduction of historical runs.
Keep this module local to this experiment (no sibling-experiment imports).
"""
import torch

POLICY = 'bf16-model-basis-cache-v1'
DESCRIPTION = ('BF16 repair parameters, repair_basis, token_basis, projections and '
               'logit updates; explicit FP32 probability/reduction/loss boundaries; '
               'stock sampler/verifier unchanged; historical checkpoints cast at load')
DTYPE = torch.bfloat16


def require_bf16(**tensors):
    for name, tensor in tensors.items():
        if tensor.dtype != DTYPE:
            raise TypeError(f'{POLICY}: {name} must be BF16, got {tensor.dtype}; '
                            'convert once at the loading boundary')


def load_repair_bf16(module, state, basis, *, device):
    """Convert destinations once; never overwrite the source checkpoint."""
    module.to(device=device, dtype=DTYPE)
    module.load_state_dict(state, strict=True)
    return basis.detach().to(device=device, dtype=DTYPE).clone()


def build_token_basis(lm, basis):
    """BF16 inputs/output, no FP32 full-vocabulary temporary or retained copy."""
    require_bf16(lm=lm, repair_basis=basis)
    with torch.no_grad(), torch.autocast(lm.device.type, enabled=False):
        return (lm @ basis.T).detach()


def project_direction(coeff, token_basis):
    require_bf16(coeff=coeff, token_basis=token_basis)
    with torch.autocast(coeff.device.type, enabled=False):
        return coeff @ token_basis.T


def hidden_norm_bf16(hidden, eps):
    # Accumulate a reduction in FP32, return once to the model dtype.
    return hidden.float().norm(dim=-1).clamp_min(eps).to(DTYPE)


def bf16_logit_update(raw, direction, dose, hidden_norm, logit_scale=1.):
    """One rounding contract shared by recurrence, training and dose diagnosis."""
    require_bf16(raw=raw, direction=direction, dose=dose, hidden_norm=hidden_norm)
    delta = (dose * hidden_norm).unsqueeze(-1) * direction * float(logit_scale)
    return raw + delta, delta
