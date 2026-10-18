"""Exact Top256 after sparse repair; parallel, order-preserving entropy.

Repair touches only candidate IDs. Cache the largest256 unmodified logits
before recurrence, then merge these with the repaired candidates. Entropy
retains the original8192-value reduction tree and sequential rescaling order.
"""
import math,os
import torch
import triton
import triton.language as tl


@triton.jit
def _unmodified(TOP, IDS, CAND, OUT, V:tl.constexpr, Q:tl.constexpr,
                C:tl.constexpr, B:tl.constexpr, QB:tl.constexpr, STEPS:tl.constexpr):
    row=tl.program_id(0)
    cols=tl.arange(0,B); q=tl.arange(0,QB)
    candidates=tl.sort(tl.load(CAND+row*Q+q,q<Q,other=2147483647))
    ids=tl.load(IDS+row*C+cols,cols<C,other=-1)
    left=tl.full((B,),0,tl.int32);right=tl.full((B,),Q,tl.int32)
    for _ in range(STEPS):
        mid=(left+right)//2
        value=tl.gather(candidates,tl.minimum(mid,QB-1),0)
        right=tl.where(value>=ids,mid,right)
        left=tl.where(value<ids,mid+1,left)
    found=(left<Q)&(tl.gather(candidates,tl.minimum(left,QB-1),0)==ids)
    values=tl.load(TOP+row*C+cols,cols<C,other=float('-inf')).to(tl.float32)
    values=tl.where(found,float('-inf'),values)
    ordered=tl.sort(values,descending=True)
    take=tl.arange(0,256)
    tl.store(OUT+row*256+take,tl.gather(ordered,take,0))


@triton.jit
def _chunk_max(X,M,A,AST:tl.constexpr,V:tl.constexpr,P:tl.constexpr,B:tl.constexpr=8192):
    row=tl.program_id(0);part=tl.program_id(1)
    if tl.load(A+row*AST)!=0:
        c=part*B+tl.arange(0,B)
        x=tl.load(X+row*V+c,c<V,other=float('-inf')).to(tl.float32)
        maximum=tl.max(x,0)
    else:maximum=0.
    tl.store(M+row*P+part,maximum)


@triton.jit
def _maximum(x,y):return tl.maximum(x,y)


@triton.jit
def _prefix_max(M,C,P:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);parts=tl.arange(0,B)
    maxima=tl.load(M+row*P+parts,parts<P,other=float('-inf'))
    cumulative=tl.associative_scan(maxima,0,_maximum)
    tl.store(C+row*P+parts,cumulative,parts<P)


@triton.jit
def _prefix_sampled_max(M,C,A,AST:tl.constexpr,P:tl.constexpr,SUBPARTS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);parts=tl.arange(0,B)
    subparts=parts[:,None]*8+tl.arange(0,8)[None,:]
    x=tl.load(M+row*SUBPARTS+subparts,subparts<SUBPARTS,other=float('-inf'))
    maxima=tl.max(x,1)
    maxima=tl.where(tl.load(A+row*AST)!=0,maxima,0.)
    cumulative=tl.associative_scan(maxima,0,_maximum)
    tl.store(C+row*P+parts,cumulative,parts<P)


@triton.jit
def _mass(X,C,MASS,A,AST:tl.constexpr,V:tl.constexpr,P:tl.constexpr,B:tl.constexpr=8192):
    row=tl.program_id(0);part=tl.program_id(1)
    if tl.load(A+row*AST)!=0:
        c=part*B+tl.arange(0,B);valid=c<V
        x=tl.load(X+row*V+c,valid,other=float('-inf')).to(tl.float32)
        maximum=tl.load(C+row*P+part)
        mass=tl.where(valid,tl.exp(x-maximum),0.)
        den=tl.sum(mass,0);weighted=tl.sum(mass*tl.where(valid,x,0.),0)
    else:den=0.;weighted=0.
    tl.store(MASS+(row*P+part)*2,den)
    tl.store(MASS+(row*P+part)*2+1,weighted)


@triton.jit
def _merge(X,UNMOD,CAND,TOP,S,C,MASS,A,AST:tl.constexpr,
           V:tl.constexpr,Q:tl.constexpr,Q0:tl.constexpr,Q1:tl.constexpr,
           N:tl.constexpr,P:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);cols=tl.arange(0,B)
    if tl.load(A+row*AST)!=0:
        unmod=tl.load(UNMOD+row*256+cols,cols<256,other=float('-inf'))
        cidx=tl.maximum(cols-256,0)
        ids=tl.load(CAND+row*Q0+cidx*Q1,(cols>=256)&(cidx<Q),other=0)
        repaired=tl.load(X+row*V+ids,(cols>=256)&(cidx<Q),other=float('-inf')).to(tl.float32)
        values=tl.where(cols<256,unmod,repaired)
        ordered=tl.sort(values,descending=True)
        take=tl.arange(0,256);top=tl.gather(ordered,take,0)
        maximum=float('-inf');denominator=0.;weighted=0.
        for part in range(P):
            next_max=tl.load(C+row*P+part)
            factor=tl.exp(maximum-next_max)
            denominator=denominator*factor+tl.load(MASS+(row*P+part)*2)
            weighted=weighted*factor+tl.load(MASS+(row*P+part)*2+1)
            maximum=next_max
        lse=maximum+tl.log(denominator)
        entropy=(lse-weighted/denominator)/tl.log(V*1.0)
    else:
        top=tl.full((256,),float('-inf'),tl.float32);lse=0.;entropy=0.
    tl.store(TOP+row*256+tl.arange(0,256),top)
    tl.store(S+row,lse);tl.store(S+N+row,entropy)


@torch.no_grad()
def prepare_topk(raw,ids):
    n,l,v=raw.shape;q=ids.shape[-1];c=256+q
    assert raw.dtype==torch.bfloat16 and c<=v
    z=raw.reshape(n*l,v);candidates=ids.reshape(n*l,q).contiguous()
    values,indices=z.topk(c,-1)
    out=torch.empty((n*l,256),device=raw.device,dtype=torch.float32)
    _unmodified[(n*l,)](values,indices,candidates,out,v,q,c,triton.next_power_of_2(c),triton.next_power_of_2(q),q.bit_length()+1,num_warps=4)
    return out.reshape(n,l,256).transpose(0,1).contiguous(),ids.transpose(0,1).contiguous()

def prepare(raw,ids):
    if os.environ.get('TAIL_SPARSE_PREP','topk')=='bounded':
        from .tail_bounded_cache import prepare as bounded_prepare
        return bounded_prepare(raw,ids)
    if os.environ.get('TAIL_SPARSE_PREP','topk')=='hist':
        from .tail_unmodified_hist import prepare as histogram_prepare
        return histogram_prepare(raw,ids)
    return prepare_topk(raw,ids)


@torch.no_grad()
def features(logits,active,unmodified,candidates,*,scan_group=None,sampled_maxima=None):
    z=logits.detach().contiguous();n,v=z.shape;q=candidates.shape[-1];p=triton.cdiv(v,8192)
    alloc=lambda shape:torch.empty(shape,device=z.device,dtype=torch.float32)
    maximum=alloc((n,p));cumulative=alloc((n,p));mass=alloc((n,p,2));top=alloc((n,256));stats=alloc((2,n))
    if sampled_maxima is None and scan_group is not None and n>=128 and os.environ.get('TAIL_SERIAL_EARLY_ENTROPY','0')=='1':
        # The first positions form a large batch of independent rows. Stock
        # entropy can then saturate the GPU without separate max/mass passes,
        # retaining exactly its original sequential FP32 reduction order.
        from .tail_fast import _entropy
        _entropy[(n,)](z,stats,active,active.stride(0),n,v,num_warps=8)
        _merge_only[(n,)](z,unmodified,candidates,top,active,active.stride(0),v,q,*candidates.stride(),triton.next_power_of_2(256+q),num_warps=8)
    else:
        if sampled_maxima is not None:
            assert sampled_maxima.shape==(n,triton.cdiv(v,1024)) and sampled_maxima.is_contiguous()
            _prefix_sampled_max[(n,)](sampled_maxima,cumulative,active,active.stride(0),p,
                sampled_maxima.shape[1],triton.next_power_of_2(p),num_warps=4)
        else:
            _chunk_max[(n,p)](z,maximum,active,active.stride(0),v,p,num_warps=8)
            _prefix_max[(n,)](maximum,cumulative,p,triton.next_power_of_2(p),num_warps=4)
        _mass[(n,p)](z,cumulative,mass,active,active.stride(0),v,p,num_warps=8)
        _merge[(n,)](z,unmodified,candidates,top,stats,cumulative,mass,active,active.stride(0),v,q,*candidates.stride(),n,p,triton.next_power_of_2(256+q),num_warps=8)
    log_top=top-stats[0,:,None];probabilities=log_top.exp()
    group=n if scan_group is None else scan_group
    if os.environ.get('TAIL_FUSED_SCAN','0')=='1' and group>1:
        # Match ATen's Sklansky tree and block carry, including its choice
        # of x threads from the ORIGINAL per-position row count. N=1 uses
        # deterministic CUB, so retain torch.cumsum for that case.
        logx=max(4,min(9,(9+8-(group-1).bit_length())//2))
        result=alloc((n,45))
        _scan_assemble[(n,)](probabilities,log_top,stats,result,n,logx,2*(1<<logx),num_warps=4)
        return result
    # torch.cumsum changes its reduction layout with the row count. Preserve
    # the original per-position row count when batching early q extraction.
    cumulative=probabilities.cumsum(-1) if scan_group is None else torch.cat(
        [probabilities[start:start+scan_group].cumsum(-1) for start in range(0,n,scan_group)],0)
    if os.environ.get('TAIL_FUSED_FEATURE_ASSEMBLY','0')=='1':
        result=alloc((n,45))
        _assemble[(n,)](probabilities,cumulative,log_top,stats,result,n,num_warps=4)
        return result
    four=torch.stack((probabilities[:,0],probabilities[:,1],probabilities[:,0]-probabilities[:,1],stats[1]),-1)
    cum=torch.stack([cumulative[:,i] for i in (0,1,3,7,15,31,63,127,255)],-1)
    return torch.cat((four,cum,log_top[:,:32].clamp_min(math.log(1e-30))),-1)


@triton.jit
def _merge_only(X,UNMOD,CAND,TOP,A,AST:tl.constexpr,V:tl.constexpr,Q:tl.constexpr,Q0:tl.constexpr,Q1:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);cols=tl.arange(0,B)
    if tl.load(A+row*AST)!=0:
        unmod=tl.load(UNMOD+row*256+cols,cols<256,other=float('-inf'))
        cidx=tl.maximum(cols-256,0)
        ids=tl.load(CAND+row*Q0+cidx*Q1,(cols>=256)&(cidx<Q),other=0)
        repaired=tl.load(X+row*V+ids,(cols>=256)&(cidx<Q),other=float('-inf')).to(tl.float32)
        ordered=tl.sort(tl.where(cols<256,unmod,repaired),descending=True)
        result=tl.gather(ordered,tl.arange(0,256),0)
    else:result=tl.full((256,),float('-inf'),tl.float32)
    tl.store(TOP+row*256+tl.arange(0,256),result)


@triton.jit
def _assemble(PROBS,CUM,LOGS,STATS,OUT,N:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,64)
    rank=tl.where(col<13,(1<<tl.maximum(col-4,0))-1,tl.maximum(col-13,0))
    cumulative=tl.load(CUM+row*256+rank,(col>=4)&(col<13),other=0.)
    logprob=tl.load(LOGS+row*256+rank,(col>=13)&(col<45),other=0.)
    p0=tl.load(PROBS+row*256);p1=tl.load(PROBS+row*256+1)
    entropy=tl.load(STATS+N+row)
    first=tl.where(col==0,p0,tl.where(col==1,p1,tl.where(col==2,p0-p1,entropy)))
    result=tl.where(col<4,first,tl.where(col<13,cumulative,tl.maximum(logprob,-69.07755278982137)))
    tl.store(OUT+row*45+col,result,col<45)


@triton.jit
def _scan_assemble(PROBS,LOGS,STATS,OUT,N:tl.constexpr,LOGX:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);idx=tl.arange(0,B);col=tl.arange(0,64)
    rank=(1<<tl.maximum(col-4,0))-1
    cumulative=tl.full((64,),0.,tl.float32);carry=0.
    for block in tl.static_range((256+B-1)//B):
        values=tl.load(PROBS+row*256+block*B+idx,block*B+idx<256,other=0.)
        values=tl.where(idx==0,values+carry,values)
        for m in tl.static_range(LOGX+1):
            span=1<<m
            source=(idx//(2*span))*(2*span)+span-1
            prefix=tl.gather(values,source,0)
            values=tl.where(idx%(2*span)>=span,values+prefix,values)
        carry=tl.sum(tl.where(idx==B-1,values,0.),0)
        selected=tl.gather(values,tl.minimum(tl.maximum(rank-block*B,0),B-1),0)
        cumulative=tl.where((rank>=block*B)&(rank<(block+1)*B),selected,cumulative)
    logprob=tl.load(LOGS+row*256+tl.maximum(col-13,0),(col>=13)&(col<45),other=0.)
    p0=tl.load(PROBS+row*256);p1=tl.load(PROBS+row*256+1)
    entropy=tl.load(STATS+N+row)
    first=tl.where(col==0,p0,tl.where(col==1,p1,tl.where(col==2,p0-p1,entropy)))
    result=tl.where(col<4,first,tl.where(col<13,cumulative,tl.maximum(logprob,-69.07755278982137)))
    tl.store(OUT+row*45+col,result,col<45)
