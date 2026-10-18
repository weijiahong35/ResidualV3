"""Full-vocabulary TopK softmax features with a live, BF16 logit gradient."""
import types
import torch
import triton
import triton.language as tl
from fused_kernels import _online_stats
from runtime.precision import hidden_norm_bf16

@triton.jit
def lognorm(raw,out,V:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    m,z=_online_stats(raw+row*V,V,BLOCK)
    tl.store(out+row,m+tl.log(z))

@triton.jit
def background_grad(raw,norm,sums,out,V:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);tile=tl.program_id(1)
    v=tile*BLOCK+tl.arange(0,BLOCK);valid=v<V
    x=tl.load(raw+row*V+v,valid,other=0).to(tl.float32)
    result=-tl.exp(x-tl.load(norm+row))*tl.load(sums+row)
    tl.store(out+row*V+v,result,valid)

@triton.jit
def selected_grad(ids,q,g,sums,out,V:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);k=tl.arange(0,BLOCK);valid=k<K
    v=tl.load(ids+row*K+k,valid,other=0).to(tl.int64)
    p=tl.load(q+row*K+k,valid,other=0).to(tl.float32)
    incoming=tl.load(g+row*K+k,valid,other=0).to(tl.float32)
    # Overwrite selected entries with a single FP32-computed, BF16-rounded value.
    tl.store(out+row*V+v,p*(incoming-tl.load(sums+row)),valid)

class SelectedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx,raw,ids):
        assert raw.dtype==torch.bfloat16 and raw.is_contiguous()
        norms=torch.empty(raw.shape[:-1],device=raw.device,dtype=torch.float32)
        lognorm[(raw.numel()//raw.shape[-1],)](raw,norms,raw.shape[-1],8192,num_warps=8)
        q=(raw.gather(-1,ids).float()-norms[...,None]).exp()
        ctx.save_for_backward(raw,ids,norms,q)
        return q
    @staticmethod
    def backward(ctx,grad):
        raw,ids,norms,q=ctx.saved_tensors
        grad=grad.contiguous().float();sums=(grad*q).sum(-1).contiguous()
        out=torch.empty_like(raw);V=raw.shape[-1];K=ids.shape[-1];rows=norms.numel()
        background_grad[(rows,triton.cdiv(V,8192))](raw,norms,sums,out,V,8192,num_warps=8)
        selected_grad[(rows,)](ids,q,grad,sums,out,V,K,triton.next_power_of_2(K),num_warps=4)
        return out,None

def prepare(self,raw,hidden,output_basis,*,q_override=None,ids_override=None,omit_output_rows=False):
    assert not self.use_soft
    ids=self.q_feature_ids(raw) if ids_override is None else ids_override
    values=SelectedSoftmax.apply(raw.contiguous(),ids) if q_override is None else q_override.gather(-1,ids)
    from torch.utils.checkpoint import checkpoint
    def moment_for(q,indices):
        rows=self.feature_basis[indices]
        return (q.to(rows.dtype).unsqueeze(-2)@rows).squeeze(-2)
    # Recompute frozen gathered input rows instead of retaining another 6 GiB.
    moment=checkpoint(moment_for,values,ids,use_reentrant=False,preserve_rng_state=False)
    qp=self.q_proj(moment);del values
    hp=self.hidden_proj(hidden) if self.use_hidden else None
    hn=hidden_norm_bf16(hidden,self.eps)
    return hp,qp,hn,ids,None,None if omit_output_rows else output_basis[ids]

def install(head):head.prepare=types.MethodType(prepare,head)

def gate():
    torch.manual_seed(9531)
    x=torch.randn(4,15,151936,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    ids=x.detach().topk(256,-1).indices
    incoming=torch.randn(4,15,256,device='cuda',dtype=torch.float32)
    q=x.float().softmax(-1).gather(-1,ids);q.backward(incoming)
    reference=x.grad.clone();values=q.detach().clone();x.grad=None;del q
    q=SelectedSoftmax.apply(x,ids);q.backward(incoming)
    delta=(x.grad.float()-reference.float());den=reference.float().norm()
    result=dict(positions=60,vocab=151936,topk=256,value_max_abs=float((q.detach()-values).abs().max()),gradient_relative_l2=float(delta.norm()/den),gradient_cosine=float(torch.nn.functional.cosine_similarity(x.grad.float().flatten()[None],reference.float().flatten()[None])),gradient_bitwise_equal=torch.equal(x.grad,reference),unselected_gradient_nonzero=bool((x.grad!=0).sum()>ids.numel()),scope='Full-vocabulary normalization derivative, including non-TopK logits; random BF16 arithmetic test, not whole-model parity')
    result['passed']=result['value_max_abs']<2e-6 and result['gradient_relative_l2']<.002 and result['gradient_cosine']>.99999 and result['unselected_gradient_nonzero']
    assert result['passed'],result
    return result
