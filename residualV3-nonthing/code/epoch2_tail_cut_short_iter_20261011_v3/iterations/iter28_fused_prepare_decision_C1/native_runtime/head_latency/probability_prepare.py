"""Avoid full FP32 cast/subtract intermediates before unchanged ATen summation.

BF16 max is exactly representable in FP32. Exp uses expf, matching the
original torch.exp pointwise path; denominator reduction remains in ATen.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _exp(X,MAX,O,V:tl.constexpr,L:tl.constexpr,S0:tl.constexpr,
         S1:tl.constexpr,S2:tl.constexpr,B:tl.constexpr=4096):
    row=tl.program_id(0);cols=tl.program_id(1)*B+tl.arange(0,B)
    x=tl.load(X+(row//L)*S0+(row%L)*S1+cols*S2,cols<V,other=0.).to(tl.float32)
    maximum=tl.load(MAX+row)
    result=libdevice.exp(x-maximum)
    tl.store(O+row*V+cols,result,cols<V)


@torch.no_grad()
def exp_logits(raw):
    assert raw.ndim==3 and raw.dtype==torch.bfloat16
    n,l,v=raw.shape
    maximum=torch.amax(raw,dim=-1,keepdim=True).float()
    out=torch.empty((n,l,v),device=raw.device,dtype=torch.float32)
    _exp[(n*l,triton.cdiv(v,4096))](raw,maximum,out,v,l,*raw.stride(),
         num_warps=4,enable_fp_fusion=False)
    return out
