"""Experimental TF batching. Keep frozen feature encoders and native recursion unchanged."""
from contextlib import contextmanager
import types
import torch
from torch import nn
from parallel_tf import teacher_forced_logits


def position_linear(self, x):
    if x.ndim != 3:
        return torch.nn.functional.linear(x, self.weight, self.bias)
    positions = x.shape[1]
    y = torch.bmm(x.transpose(0, 1).contiguous(),
                  self.weight.T.unsqueeze(0).expand(positions, -1, -1))
    if self.bias is not None:
        y = y + self.bias
    return y.transpose(0, 1)


@contextmanager
def prediction_linears(head, mode):
    changed = []
    if mode == 'parallel_position_bmm':
        roots = [head.previous_proj, head.fusion, head.blocks, head.direction_head, head.dose_head]
        for root in roots:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    changed.append((module, module.forward))
                    module.forward = types.MethodType(position_linear, module)
    try:
        yield
    finally:
        for module, previous in changed:
            module.forward = previous


def parallel_logits(head, raw, h, basis, anchor, answers, mode):
    with prediction_linears(head, mode):
        return teacher_forced_logits(head, raw, h, basis, anchor, answers)
