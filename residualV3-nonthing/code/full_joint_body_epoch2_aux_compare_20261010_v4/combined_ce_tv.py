"""CE and TV share one gradient buffer; preserve each branch's BF16 rounding.

Forward deliberately calls the existing per-OP kernels and tile tuner so their
statistics and reductions are unchanged. Only backward buffer traffic is fused.
"""
import torch
import triton
import triton.language as tl
from fused_kernels import loss_forward_kernel,_OP_CE,_OP_TV,_N_STATS

@triton.jit
def combined_backward(raw,gold,grad,ce_go,tv_go,ce_stats,tv_stats,R:tl.constexpr,V:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);tile=tl.program_id(1)
    offsets=tile*BLOCK+tl.arange(0,BLOCK);valid=offsets<V
    x=tl.load(raw+row*V+offsets,valid,other=0).to(tl.float32)
    y=tl.load(gold+row*V+offsets,valid,other=0).to(tl.float32)
    ce_lse=tl.load(ce_stats+row)+tl.log(tl.load(ce_stats+R+row))
    tv_lse=tl.load(tv_stats+row)+tl.log(tl.load(tv_stats+R+row))
    target_lse=tl.load(tv_stats+2*R+row)+tl.log(tl.load(tv_stats+3*R+row))
    target_idx=tl.load(ce_stats+4*R+row).to(tl.int64)
    extra=tl.load(tv_stats+4*R+row)
    qc=tl.exp(x-ce_lse);qt=tl.exp(x-tv_lse);pt=tl.exp(y-target_lse)
    gc=tl.load(ce_go+row)*(qc-(offsets==target_idx).to(tl.float32))
    gt=tl.load(tv_go+row)*(-qt*(tl.where(qt<pt,1.,tl.where(qt==pt,.5,0.))-extra))
    # The legacy autograd graph creates two BF16 tensors and adds them in
    # BF16. Reproduce both intermediate roundings instead of a FP32 shortcut.
    gc=gc.to(tl.bfloat16).to(tl.float32);gt=gt.to(tl.bfloat16).to(tl.float32)
    tl.store(grad+row*V+offsets,gc+gt,valid)

class Combined(torch.autograd.Function):
    @staticmethod
    def forward(ctx,raw,gold):
        assert raw.dtype==gold.dtype==torch.bfloat16
        raw=raw.contiguous();gold=gold.contiguous();B,T,V=raw.shape;R=B*T
        ce=torch.empty((B,T),device=raw.device,dtype=torch.float32);tv=torch.empty_like(ce)
        cs=torch.empty((_N_STATS,R),device=raw.device,dtype=torch.float32);ts=torch.empty_like(cs)
        loss_forward_kernel[(R,)](raw,gold,ce,cs,R,V,OP=_OP_CE.value)
        loss_forward_kernel[(R,)](raw,gold,tv,ts,R,V,OP=_OP_TV.value)
        ctx.save_for_backward(raw,gold,cs,ts);ctx.set_materialize_grads(False)
        return ce,tv
    @staticmethod
    def backward(ctx,ce_go,tv_go):
        raw,gold,cs,ts=ctx.saved_tensors;R=raw.shape[0]*raw.shape[1];V=raw.shape[2]
        if ce_go is None:ce_go=torch.zeros(raw.shape[:2],device=raw.device,dtype=torch.float32)
        if tv_go is None:tv_go=torch.zeros(raw.shape[:2],device=raw.device,dtype=torch.float32)
        grad=torch.empty_like(raw)
        combined_backward[(R,triton.cdiv(V,8192))](raw,gold,grad,ce_go.contiguous(),tv_go.contiguous(),cs,ts,R,V,8192,num_warps=8,enable_fp_fusion=False)
        return grad,None

def objective(raw,gold,alpha=.5,return_auxiliary_labels=False):
    ce,tv=Combined.apply(raw,gold)
    with torch.no_grad():weights=(alpha+(1-alpha)*torch.exp(-ce)).cumprod(-1).flip(-1).cumsum(-1).flip(-1)
    overlap=(1-tv).clamp(0,1)
    loss=(ce*weights).mean()+(1-overlap.cumprod(-1).mean())
    if return_auxiliary_labels:
        # Reuse the original OUTPUT overlap; never normalize a TopK subset.
        return loss,overlap.detach()
    return loss
