"""Inference-only head kernels. Target ATen dispatch is never changed.

Reuse the installed vLLM fixed-order kernels with smaller M/N tiles. The K=64
loop and BF16 cast before a separate bias add are retained. Tile changes are
only accepted after bitwise checks against batch-invariant runtime kernels.
"""
import math
import types
import torch
from vllm.model_executor.layers.batch_invariant import matmul_kernel_persistent, bmm_kernel
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units

MODES = ('baseline', 'linear16_64', 'linear16_128', 'linear32_64', 'all16_64')

def config(mode):
    return dict(BLOCK_SIZE_M=32 if mode=='linear32_64' else 16,
                BLOCK_SIZE_N=128 if mode=='linear16_128' else 64,
                BLOCK_SIZE_K=64, num_stages=3, num_warps=4)

def linear(x, weight, bias, mode):
    assert x.dtype==weight.dtype==torch.bfloat16
    shape=x.shape[:-1];a=x.reshape(-1,x.shape[-1]);b=weight.T
    m,k=a.shape;n=b.shape[1];c=torch.empty((m,n),device=x.device,dtype=x.dtype)
    cfg=config(mode);sms=num_compute_units(x.device.index)
    grid=(min(sms,triton.cdiv(m,cfg['BLOCK_SIZE_M'])*triton.cdiv(n,cfg['BLOCK_SIZE_N'])),)
    matmul_kernel_persistent[grid](a,b,c,None,m,n,k,*a.stride(),*b.stride(),*c.stride(),
        GROUP_SIZE_M=8,NUM_SMS=sms,A_LARGE=False,B_LARGE=False,C_LARGE=False,HAS_BIAS=False,**cfg)
    # Installed linear_batch_invariant rounds the GEMM to BF16 before adding bias.
    if bias is not None:c=c+bias
    return c.reshape(*shape,n)

def bmm(a,b):
    assert a.ndim==b.ndim==3 and a.dtype==b.dtype==torch.bfloat16
    batch,m,k=a.shape;n=b.shape[-1]
    assert b.shape[:2]==(batch,k)
    c=torch.empty((batch,m,n),device=a.device,dtype=a.dtype)
    cfg=config('all16_64')
    # A skinny output column is also padded to 16, the minimum MMA width.
    if n==1:cfg['BLOCK_SIZE_M']=64;cfg['BLOCK_SIZE_N']=16
    grid=(batch,triton.cdiv(m,cfg['BLOCK_SIZE_M'])*triton.cdiv(n,cfg['BLOCK_SIZE_N']))
    bmm_kernel[grid](a,b,c,batch,m,n,k,*a.stride(),*b.stride(),*c.stride(),
        A_LARGE=False,B_LARGE=False,C_LARGE=False,**cfg)
    return c

def attach(head):
    assert not head.training and not head.use_soft
    if hasattr(head,'_small_tile_attached'):return
    for name,module in head.named_modules():
        if isinstance(module,torch.nn.Linear):
            original=module.forward
            def forward(self,x,original=original):
                mode=head._small_tile_mode
                if mode=='baseline':return original(x)
                assert not self.training and not torch.is_grad_enabled()
                return linear(x,self.weight,self.bias,mode)
            module.forward=types.MethodType(forward,module)
    original_part=head.distribution_part
    def distribution(self,probabilities,ids,*,rows=None):
        if self._small_tile_mode!='all16_64':return original_part(probabilities,ids,rows=rows)
        if rows is None:rows=self.feature_basis[ids]
        values=probabilities.gather(-1,ids).to(rows.dtype)
        prefix=values.shape[:-1];batch=math.prod(prefix)
        output=bmm(values.reshape(batch,1,-1),rows.reshape(batch,rows.shape[-2],rows.shape[-1]))
        return self.q_proj(output.reshape(*prefix,rows.shape[-1]))
    head.distribution_part=types.MethodType(distribution,head)
    head._small_tile_attached=True
    head._small_tile_mode='baseline'

def set_mode(head,mode):
    assert mode in MODES
    attach(head);head._small_tile_mode=mode
    import head_latency.optimized as opt
    from repair.model import project_candidates
    if mode=='all16_64':
        opt.project_candidates=lambda coeff,rows:bmm(rows,coeff.unsqueeze(-1)).squeeze(-1)
    else:opt.project_candidates=project_candidates
