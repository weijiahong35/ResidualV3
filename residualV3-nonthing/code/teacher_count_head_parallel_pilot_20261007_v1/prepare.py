"""Frozen-RAW no-soft features without a dense FP32 probability allocation."""
import types
import torch
import triton
import triton.language as tl
from fused_kernels import _online_stats
from runtime.precision import hidden_norm_bf16


@triton.jit
def lognorm_kernel(raw, result, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    m,z=_online_stats(raw+row*n_cols,n_cols,BLOCK_SIZE)
    tl.store(result+row,m+tl.log(z))


def streamed_prepare(self,raw,hidden,output_basis,*,q_override=None,ids_override=None):
    if self.use_soft or raw.requires_grad:
        raise ValueError('This optimization is only validated for frozen RAW/no-soft heads')
    ids=self.q_feature_ids(raw) if ids_override is None else ids_override
    if q_override is None:
        shape=raw.shape[:-1]; vocabulary=raw.shape[-1]
        norms=torch.empty(shape,device=raw.device,dtype=torch.float32)
        lognorm_kernel[(raw.numel()//vocabulary,)](raw,norms,vocabulary,BLOCK_SIZE=8192,num_warps=8)
        # Global full-vocabulary normalizer, NOT a renormalized TopK softmax.
        values=(raw.gather(-1,ids).float()-norms.unsqueeze(-1)).exp()
    else: values=q_override.gather(-1,ids)
    rows=self.feature_basis[ids]
    moment=(values.to(rows.dtype).unsqueeze(-2)@rows).squeeze(-2)
    del rows,values
    qp=self.q_proj(moment)
    # Frozen input-feature rows are no longer needed for no-soft recurrence.
    # Output rows are still saved by autograd for coefficient gradients.
    outrows=output_basis[ids]
    hp=self.hidden_proj(hidden) if self.use_hidden else None
    scale=hidden_norm_bf16(hidden,self.eps) if self.scale_mode=='hidden_norm' else torch.full(hidden.shape[:-1],self.fixed_scale,device=hidden.device,dtype=hidden.dtype)
    return hp,qp,scale,ids,None,outrows


def install(head):
    head.prepare=types.MethodType(streamed_prepare,head)
