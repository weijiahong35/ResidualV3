"""Inference-only epilogue, preserving the original PyTorch norm reduction.

A full Triton norm changed a BF16 boundary on real N40 data. Keep that
reduction and fuse only clamp/cast/division plus the scalar dose operations.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _finish(X, NORM, DIRECTION, DOSE, M: tl.constexpr, RANK: tl.constexpr,
            XS0: tl.constexpr, XS1: tl.constexpr, EPS: tl.constexpr,
            DOSE_MAX: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, B)
    x = tl.load(X + row * XS0 + cols * XS1, cols < RANK, other=0.).to(tl.float32)
    norm = tl.load(NORM + row)
    denominator = tl.maximum(norm, EPS).to(tl.bfloat16).to(tl.float32)
    direction = tl.div_rn(x, denominator).to(tl.bfloat16)
    tl.store(DIRECTION + row * RANK + cols, direction, cols < RANK)
    logit = tl.load(X + row * XS0 + RANK * XS1).to(tl.float32)
    sigmoid = tl.div_rn(1., 1. + libdevice.exp(-logit))
    dose = (DOSE_MAX * sigmoid).to(tl.bfloat16)
    tl.store(DOSE + row, dose)


@torch.no_grad()
def finish(output, rank, eps, dose_max):
    assert output.dtype == torch.bfloat16 and output.shape[-1] == rank + 1
    n = output.shape[0]
    direction = output.new_empty((n, rank))
    dose = output.new_empty((n,))
    norm = output[:,:rank].float().norm(dim=-1,keepdim=True)
    _finish[(n,)](output, norm, direction, dose, n, rank, *output.stride(), eps,
                  dose_max, triton.next_power_of_2(rank), num_warps=4,
                  enable_fp_fusion=False)
    return direction, dose
