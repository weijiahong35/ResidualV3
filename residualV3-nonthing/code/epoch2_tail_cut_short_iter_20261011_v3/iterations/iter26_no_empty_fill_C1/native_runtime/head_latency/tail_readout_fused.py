"""Frozen q45 MLP in one launch, retaining every BF16 rounding point.

The input normalization and the final sigmoid remain FP32. Both matrix
products use BF16 operands with FP32 MMA, as in the installed invariant
linear operator. This is inference-only and does not modify training.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _readout(X, MEAN, SCALE, W0, B0, W1, B1, O, M: tl.constexpr,
             XS0: tl.constexpr, XS1: tl.constexpr, BM: tl.constexpr = 16):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    kk = tl.arange(0, 64)
    hh = tl.arange(0, 32)
    x = tl.load(X + rows[:, None] * XS0 + kk[None, :] * XS1,
                (rows[:, None] < M) & (kk[None, :] < 45), other=0.)
    mean = tl.load(MEAN + kk, kk < 45, other=0.)
    scale = tl.load(SCALE + kk, kk < 45, other=1.)
    normalized = tl.div_rn(x - mean[None, :], scale[None, :])
    normalized = tl.where(kk[None, :] < 45, normalized, 0.).to(tl.bfloat16)
    w0 = tl.load(W0 + hh[None, :] * 45 + kk[:, None], kk[:, None] < 45, other=0.)
    a = tl.dot(normalized, w0).to(tl.bfloat16).to(tl.float32)
    bias0 = tl.load(B0 + hh).to(tl.float32)
    a = (a + bias0[None, :]).to(tl.bfloat16).to(tl.float32)
    h = tl.div_rn(a, 1. + libdevice.exp(-a)).to(tl.bfloat16)
    cols = tl.arange(0, 16)
    w1 = tl.load(W1 + cols[None, :] * 32 + hh[:, None], cols[None, :] < 2, other=0.)
    output = tl.dot(h, w1).to(tl.bfloat16).to(tl.float32)
    bias1 = tl.load(B1 + cols, cols < 2, other=0.).to(tl.float32)
    output = (output + bias1[None, :]).to(tl.bfloat16).to(tl.float32)
    probability = tl.div_rn(1., 1. + libdevice.exp(-output))
    tl.store(O + rows[:, None] * 2 + cols[None, :], probability,
             (rows[:, None] < M) & (cols[None, :] < 2))


@torch.no_grad()
def readout(head, q45):
    assert head._tail_branch == 'Q45_MLP' and q45.dtype == torch.float32
    net = head._tail_readouts['Q45_MLP']
    assert net[0].weight.is_contiguous() and net[2].weight.is_contiguous()
    out = q45.new_empty((q45.shape[0], 2))
    _readout[(triton.cdiv(q45.shape[0], 16),)](
        q45, head._tail_mean, head._tail_scale, net[0].weight, net[0].bias,
        net[2].weight, net[2].bias, out, q45.shape[0], *q45.stride(),
        num_warps=4, num_stages=3, enable_fp_fusion=False)
    return out
