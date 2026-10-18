"""Fuse SiLU then multiply with the original intervening BF16 rounding."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _silu_mul(U,O,N:tl.constexpr,I:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,BLOCK:tl.constexpr):
    t=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=t//I;col=t%I;valid=t<N*I
    a=tl.load(U+row*S0+col*S1,valid,other=0).to(tl.float32)
    g=tl.load(U+row*S0+(col+I)*S1,valid,other=0).to(tl.float32)
    # PyTorch CUDA opmath SiLU: x / (1 + exp(-x)), then BF16 output.
    # Preserve this cast before BF16 multiplication; do not contract stages.
    s=tl.div_rn(g,1.+libdevice.exp(-g)).to(tl.bfloat16)
    h=(a*s.to(tl.float32)).to(tl.bfloat16)
    tl.store(O+t,h,valid)


def silu_mul(up):
    assert up.ndim==2 and up.shape[-1]%2==0 and up.dtype==torch.bfloat16
    n=up.shape[0];inner=up.shape[1]//2
    if not up.is_cuda:
        a,g=up.chunk(2,-1)
        return F.silu(g)*a
    out=up.new_empty((n,inner))
    _silu_mul[(triton.cdiv(n*inner,256),)](up,out,n,inner,up.stride(0),up.stride(1),BLOCK=256,num_warps=4)
    return out


def block_forward(block,x):
    assert not block.training and block.dropout.p==0.
    return x+block.down(silu_mul(block.up(block.norm(x))))


def exhaustive_check():
    """All finite BF16 gates, three multipliers; no state cache written."""
    bits=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16)
    gates=bits.view(torch.bfloat16)
    gates=gates[torch.isfinite(gates)]
    rows=[]
    for multiplier in (1.,-1.,.3125):
        a=torch.full_like(gates,multiplier)
        up=torch.cat([a,gates]).reshape(1,-1)
        reference=F.silu(gates)*a
        actual=silu_mul(up).reshape(-1)
        finite=torch.isfinite(reference)
        rows.append(dict(multiplier=multiplier,finite_gate_count=gates.numel(),
            mismatches=int((reference.view(torch.int16)!=actual.view(torch.int16)).sum()),
            max_finite_abs=float((reference[finite].float()-actual[finite].float()).abs().max())))
    return rows
