"""Fixed-weight inference epilogues, selective q normalization, and sampler finish.

Every arithmetic intermediate that originally rounded to BF16 still does so.
K reduction stays in the same 64-wide MMA order as vLLM's invariant GEMM.
Only head-local calls use these kernels; target/stock sampler modules are intact.
"""
import math,types,os
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from runtime.components import Rollout,normalized_rows
from runtime.precision import hidden_norm_bf16,bf16_logit_update
from head_latency.small_tile import bmm,set_mode as small_mode
from head_latency.predecessor import predict_encoded as base_predict
from head_latency.write_kernel import write_candidates

MODES=('baseline','combined','gather_q','gather_output','lean_combined')
FUSED_MODES=MODES[1:]

@triton.jit
def _gemm(A,W,O,B,R,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
          AS0:tl.constexpr,AS1:tl.constexpr,WS0:tl.constexpr,WS1:tl.constexpr,
          RS0:tl.constexpr,RS1:tl.constexpr,HAS_R:tl.constexpr,
          BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=64):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(K,BK)):
        ks=start*BK+kk
        a=tl.load(A+rows[:,None]*AS0+ks[None,:]*AS1,(rows[:,None]<M)&(ks[None,:]<K),other=0)
        w=tl.load(W+cols[None,:]*WS0+ks[:,None]*WS1,(cols[None,:]<N)&(ks[:,None]<K),other=0)
        acc=tl.dot(a,w,acc)
    # Separate BF16 rounding is essential: this is not an FP32 fused bias.
    y=acc.to(tl.bfloat16).to(tl.float32)
    bias=tl.load(B+cols,cols<N,other=0).to(tl.float32)
    y=(y+bias[None,:]).to(tl.bfloat16).to(tl.float32)
    if HAS_R:
        residual=tl.load(R+rows[:,None]*RS0+cols[None,:]*RS1,(rows[:,None]<M)&(cols[None,:]<N),other=0).to(tl.float32)
        y=(y+residual).to(tl.bfloat16).to(tl.float32)
    tl.store(O+rows[:,None]*N+cols[None,:],y,(rows[:,None]<M)&(cols[None,:]<N))

@triton.jit
def _up_silu(A,W,O,B,M:tl.constexpr,I:tl.constexpr,K:tl.constexpr,
             AS0:tl.constexpr,AS1:tl.constexpr,WS0:tl.constexpr,WS1:tl.constexpr,
             BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=64):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);aa=tl.full((BM,BN),0,tl.float32);gg=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(K,BK)):
        ks=start*BK+kk
        x=tl.load(A+rows[:,None]*AS0+ks[None,:]*AS1,(rows[:,None]<M)&(ks[None,:]<K),other=0)
        wa=tl.load(W+cols[None,:]*WS0+ks[:,None]*WS1,(cols[None,:]<I)&(ks[:,None]<K),other=0)
        wg=tl.load(W+(cols[None,:]+I)*WS0+ks[:,None]*WS1,(cols[None,:]<I)&(ks[:,None]<K),other=0)
        aa=tl.dot(x,wa,aa);gg=tl.dot(x,wg,gg)
    a=aa.to(tl.bfloat16).to(tl.float32);g=gg.to(tl.bfloat16).to(tl.float32)
    ba=tl.load(B+cols,cols<I,other=0).to(tl.float32);bg=tl.load(B+cols+I,cols<I,other=0).to(tl.float32)
    a=(a+ba[None,:]).to(tl.bfloat16).to(tl.float32)
    g=(g+bg[None,:]).to(tl.bfloat16).to(tl.float32)
    s=tl.div_rn(g,1.+libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    h=(a*s).to(tl.bfloat16)
    tl.store(O+rows[:,None]*I+cols[None,:],h,(rows[:,None]<M)&(cols[None,:]<I))

@triton.jit
def _correct(R,C,I,Z,RAW,D,H,N:tl.constexpr,Q:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
             RS0:tl.constexpr,RS1:tl.constexpr,RS2:tl.constexpr,
             CS0:tl.constexpr,CS1:tl.constexpr,IS0:tl.constexpr,IS1:tl.constexpr,
             ZS0:tl.constexpr,ZS1:tl.constexpr,RAW0:tl.constexpr,RAW1:tl.constexpr,
             DS:tl.constexpr,HS:tl.constexpr,DIRECT:tl.constexpr=False,BM:tl.constexpr=64,BN:tl.constexpr=16,BK:tl.constexpr=64):
    batch=tl.program_id(0);rows=tl.program_id(1)*BM+tl.arange(0,BM)
    cols=tl.arange(0,BN);kk=tl.arange(0,BK);acc=tl.full((BM,BN),0,tl.float32)
    if DIRECT:token_ids=tl.load(I+batch*IS0+rows*IS1,rows<Q,other=0)
    for start in range(tl.cdiv(K,BK)):
        ks=start*BK+kk
        if DIRECT:r=tl.load(R+token_ids[:,None]*RS1+ks[None,:]*RS2,(rows[:,None]<Q)&(ks[None,:]<K),other=0)
        else:r=tl.load(R+batch*RS0+rows[:,None]*RS1+ks[None,:]*RS2,(rows[:,None]<Q)&(ks[None,:]<K),other=0)
        c=tl.load(C+batch*CS0+ks[:,None]*CS1,(ks[:,None]<K)&(cols[None,:]==0),other=0)
        acc=tl.dot(r,c,acc)
    direction=tl.gather(acc,tl.full((BM,1),0,tl.int32),axis=1).reshape((BM,)).to(tl.bfloat16).to(tl.float32)
    dose=tl.load(D+batch*DS).to(tl.float32);hn=tl.load(H+batch*HS).to(tl.float32)
    scale=(dose*hn).to(tl.bfloat16).to(tl.float32)
    delta=(scale*direction).to(tl.bfloat16).to(tl.float32)
    ids=tl.load(I+batch*IS0+rows*IS1,rows<Q,other=0)
    raw=tl.load(RAW+batch*RAW0+ids*RAW1,rows<Q,other=0).to(tl.float32)
    repaired=(raw+delta).to(tl.bfloat16)
    tl.store(Z+batch*ZS0+ids*ZS1,repaired,rows<Q)

@triton.jit
def _sample_finish(MAX,IDS,OUT,N:tl.constexpr,BLOCKS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);offset=tl.arange(0,B)
    x=tl.load(MAX+row*BLOCKS+offset,offset<BLOCKS,other=float('-inf'))
    _,idx=tl.max(x,axis=0,return_indices=True)
    token=tl.load(IDS+row*BLOCKS+idx)
    tl.store(OUT+row,token)

def linear(x,w,b,residual=None):
    assert not torch.is_grad_enabled() and x.dtype==w.dtype==b.dtype==torch.bfloat16
    shape=x.shape[:-1];a=x.reshape(-1,x.shape[-1]);m,k=a.shape;n=w.shape[0]
    out=x.new_empty((m,n));r=residual.reshape(m,n) if residual is not None else None
    _gemm[(triton.cdiv(m,16),triton.cdiv(n,64))](a,w,out,b,r,m,n,k,*a.stride(),*w.stride(),
        *(r.stride() if r is not None else (0,0)),HAS_R=r is not None,num_warps=4,num_stages=3,enable_fp_fusion=False)
    return out.reshape(*shape,n)

def up_silu(x,module):
    m,k=x.shape;i=module.weight.shape[0]//2;out=x.new_empty((m,i))
    _up_silu[(triton.cdiv(m,16),triton.cdiv(i,64))](x,module.weight,out,module.bias,m,i,k,*x.stride(),*module.weight.stride(),
        num_warps=4,num_stages=3,enable_fp_fusion=False)
    return out

def correction_write(raw,rows,coeff,dose,hn,ids,z):
    n,q,k=rows.shape
    _correct[(n,triton.cdiv(q,64))](rows,coeff,ids,z,raw,dose,hn,n,q,k,raw.shape[-1],
        *rows.stride(),*coeff.stride(),*ids.stride(),*z.stride(),*raw.stride(),dose.stride(0),hn.stride(0),
        num_warps=4,num_stages=3,enable_fp_fusion=False)
    return z

def gather_correction(raw,basis,coeff,dose,hn,ids,z):
    n,q=ids.shape;k=coeff.shape[-1]
    _correct[(n,triton.cdiv(q,64))](basis,coeff,ids,z,raw,dose,hn,n,q,k,raw.shape[-1],
        0,*basis.stride(),*coeff.stride(),*ids.stride(),*z.stride(),*raw.stride(),dose.stride(0),hn.stride(0),DIRECT=True,
        num_warps=4,num_stages=3,enable_fp_fusion=False)
    return z

@triton.jit
def _gather_moment(P,IDS,BASIS,O,N:tl.constexpr,Q:tl.constexpr,D:tl.constexpr,
                   PS:tl.constexpr,ID0:tl.constexpr,BS0:tl.constexpr,BS1:tl.constexpr,
                   BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=64):
    row=tl.program_id(0);cols=tl.program_id(1)*BN+tl.arange(0,BN)
    mm=tl.arange(0,BM);kk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(Q,BK)):
        ks=start*BK+kk
        tokens=tl.load(IDS+row*ID0+ks,ks<Q,other=0)
        probs=tl.load(P+row*PS+ks[None,:],(mm[:,None]==0)&(ks[None,:]<Q),other=0)
        basis=tl.load(BASIS+tokens[:,None]*BS0+cols[None,:]*BS1,(ks[:,None]<Q)&(cols[None,:]<D),other=0)
        acc=tl.dot(probs,basis,acc)
    result=tl.gather(acc,tl.full((1,BN),0,tl.int32),axis=0).reshape((BN,)).to(tl.bfloat16)
    tl.store(O+row*D+cols,result,cols<D)

def gather_moment(values,ids,basis):
    prefix=values.shape[:-1];p=values.reshape(-1,values.shape[-1]);i=ids.reshape_as(p)
    n,q=p.shape;d=basis.shape[-1];out=p.new_empty((n,d))
    _gather_moment[(n,triton.cdiv(d,64))](p,i,basis,out,n,q,d,p.stride(0),i.stride(0),*basis.stride(),num_warps=4,num_stages=3,enable_fp_fusion=False)
    return out.reshape(*prefix,d)

def sample(logits,idx,temp,seeds,pos,*,apply_temperature,output_processed_logits=None,output_processed_logits_col=None,use_fp64=False):
    # Stage1 is the original stock kernel, with exactly the original launch/tile.
    from vllm.v1.worker.gpu.sample.gumbel import _gumbel_sample_kernel
    idx=idx.contiguous();pos=pos.contiguous()
    col=output_processed_logits_col.contiguous() if output_processed_logits_col is not None else None
    n,v=logits.shape;blocks=triton.cdiv(v,1024)
    ids=torch.empty((n,blocks),device=logits.device,dtype=torch.int64)
    maximum=torch.empty((n,blocks),device=logits.device,dtype=torch.float64 if use_fp64 else torch.float32)
    out=torch.empty((n,),device=logits.device,dtype=torch.int64)
    _gumbel_sample_kernel[(n,blocks)](ids,ids.stride(0),maximum,maximum.stride(0),output_processed_logits,
        output_processed_logits.stride(0) if output_processed_logits is not None else 0,col,
        logits,logits.stride(0),idx,seeds,pos,temp,v,BLOCK_SIZE=1024,APPLY_TEMPERATURE=apply_temperature,
        USE_FP64=use_fp64,PER_TOKEN_COL=col is not None and col.dim()>0)
    _sample_finish[(n,)](maximum,ids,out,n,blocks,triton.next_power_of_2(blocks),num_warps=4)
    return out

def selected_prepare(head,raw,hidden,basis):
    # Same complete-vocabulary max, exp and sum as invariant softmax. Move the
    # gather before division; only the downstream-used TopK probabilities exist.
    z=raw.float();ids=head.q_feature_ids(raw)
    exp=torch.exp(z-torch.amax(z,dim=-1,keepdim=True));denom=exp.sum(-1,keepdim=True)
    values=(exp.gather(-1,ids)/denom).to(torch.bfloat16)
    prefix=values.shape[:-1];batch=math.prod(prefix)
    direct_q=head._fusion_mode in ('gather_q','lean_combined')
    direct_output=head._fusion_mode in ('gather_output','lean_combined')
    rows=None if direct_q else head.feature_basis[ids]
    outrows=None if direct_output else basis[ids]
    if direct_q:moment=gather_moment(values,ids,head.feature_basis)
    else:moment=bmm(values.reshape(batch,1,-1),rows.reshape(batch,rows.shape[-2],rows.shape[-1])).reshape(*prefix,rows.shape[-1])
    qp=head.q_proj(moment)
    hp=head.hidden_proj(hidden) if head.use_hidden else None
    hn=hidden_norm_bf16(hidden,head.eps)
    return hp,qp,hn,ids,rows,outrows

def attach(head):
    assert not head.training and not head.use_soft and head.scale_mode=='hidden_norm'
    if hasattr(head,'_fusion_attached'):return
    for module in head.modules():
        if isinstance(module,torch.nn.Linear):
            old=module.forward
            def forward(self,x,old=old):
                if head._fusion_mode not in FUSED_MODES:return old(x)
                return linear(x,self.weight,self.bias)
            module.forward=types.MethodType(forward,module)
    old=head.prepare
    def prepare(self,raw,hidden,basis,*,q_override=None,ids_override=None):
        if self._fusion_mode not in FUSED_MODES:return old(raw,hidden,basis,q_override=q_override,ids_override=ids_override)
        if q_override is not None or ids_override is not None:
            return old(raw,hidden,basis,q_override=q_override,ids_override=ids_override)
        return selected_prepare(self,raw,hidden,basis)
    head.prepare=types.MethodType(prepare,head)
    head.register_buffer('_packed_output_weight',torch.cat([head.direction_head.weight,head.dose_head.weight],0).contiguous(),persistent=False)
    head.register_buffer('_packed_output_bias',torch.cat([head.direction_head.bias,head.dose_head.bias],0).contiguous(),persistent=False)
    head._fusion_mode='baseline';head._fusion_attached=True

def set_mode(head,mode):
    assert mode in MODES
    small_mode(head,'all16_64');attach(head);head._fusion_mode=mode

def predict(head,hp,hard,qp,*,previous=None):
    if head._fusion_mode not in FUSED_MODES:
        return base_predict(head,hp,hard,qp,fused_swiglu=True)
    parts={'hidden':hp,'hard_previous':hard,'q':qp}
    if previous is not None:
        from .input_assembly import assemble
        fused_input=assemble(head,hp,qp,previous)
    else:fused_input=torch.cat([parts[n] for n in head.branch_names],-1)
    x=head.fusion(fused_input)
    for block in head.blocks:
        h=up_silu(block.norm(x),block.up)
        x=linear(h,block.down.weight,block.down.bias,residual=x)
    latent=head.output_norm(x)
    output=linear(latent,head._packed_output_weight,head._packed_output_bias)
    if os.environ.get('TAIL_FUSED_FINAL','0')=='1':
        from .output_epilogue import finish
        return finish(output,head.basis_rank,head.eps,head.dose_max)
    direction=normalized_rows(output[:,:head.basis_rank],eps=head.eps)
    dose=(head.dose_max*torch.sigmoid(output[:,head.basis_rank].float())).to(latent.dtype)
    return direction,dose

def rollout(head,raw,hidden,weight,basis,anchor,choose,*,capture=True):
    hp,qp,hn,ids,_,outrows=head.prepare(raw,hidden,basis)
    buffer=raw.transpose(0,1).clone(memory_format=torch.contiguous_format)
    previous=anchor;scores=[];tokens=[];history=[]
    for j in range(hidden.shape[1]):
        fuse_input=os.environ.get('TAIL_FUSED_INPUT','0')=='1' and head._fusion_mode in FUSED_MODES
        hard=None if fuse_input else head._previous_tables[hidden.shape[0]][previous.long()]
        coeff,dose=predict(head,hp[:,j],hard,qp[:,j],previous=previous if fuse_input else None);z=buffer[j]
        if head._fusion_mode in FUSED_MODES:
            if head._fusion_mode in ('gather_output','lean_combined'):
                gather_correction(raw[:,j],basis,coeff,dose,hn[:,j],ids[:,j],z)
            else:correction_write(raw[:,j],outrows[:,j],coeff,dose,hn[:,j],ids[:,j],z)
        else:
            direction=bmm(outrows[:,j],coeff.unsqueeze(-1)).squeeze(-1)
            corrected,_=bf16_logit_update(raw[:,j].gather(-1,ids[:,j]),direction,dose,hn[:,j])
            write_candidates(z,ids[:,j],corrected)
        token=choose(z.detach(),j).long();tokens.append(token)
        if capture:scores.append(z);history.append(previous)
        previous=token
    return Rollout(torch.stack(scores,1) if capture else None,torch.stack(tokens,1),torch.stack(history,1) if capture else None,[],())
