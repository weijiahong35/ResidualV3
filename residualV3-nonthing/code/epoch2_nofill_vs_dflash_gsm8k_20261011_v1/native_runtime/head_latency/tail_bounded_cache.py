"""Exact unmodified Top256, with a bounded radix fast path and device fallback.

Original repair candidates are the raw Top256. Every unmodified value is at
most the minimum candidate value. Count two BF16 radix groups below that
upper bound, subtract candidate multiplicities, and reconstruct values. If
this range contains fewer than256 unmodified values, run the existing exact
full-range cache only on those rows. No approximate threshold or host branch.
"""
import torch
import triton
import triton.language as tl
from .tail_q45_hist import _ordered,_coarse
from .tail_unmodified_hist import _boundary,_fine,_finish


@triton.jit
def _count(X,IDS,H,LOW,V:tl.constexpr,Q:tl.constexpr,P:tl.constexpr,L:tl.constexpr,USED:tl.constexpr,B:tl.constexpr=8192):
    row=tl.program_id(0);part=tl.program_id(1)
    if row % L >= USED:
        return
    minimum_id=tl.load(IDS+row*Q+Q-1)
    minimum=tl.load(X+row*V+minimum_id)
    low=tl.maximum((_ordered(minimum)>>8).to(tl.int32)-1,0)
    if part==0:tl.store(LOW+row,low)
    cols=part*B+tl.arange(0,B);valid=cols<V
    value=tl.load(X+row*V+cols,valid,other=float('-inf'))
    key=_ordered(value).to(tl.int32)-low*256
    histogram=tl.histogram(key,512,mask=valid&(key>=0)&(key<512))
    bins=tl.arange(0,512)
    tl.store(H+(row*P+part)*512+bins,histogram)


@triton.jit
def _resolve(X,IDS,H,LOW,FLAG,OUT,V:tl.constexpr,Q:tl.constexpr,QB:tl.constexpr,P:tl.constexpr,PB:tl.constexpr,L:tl.constexpr,USED:tl.constexpr):
    row=tl.program_id(0)
    if row % L >= USED:
        tl.store(FLAG+row,0)
        tl.store(OUT+row*256+tl.arange(0,256),float('-inf'))
        return
    parts=tl.arange(0,PB);bins=tl.arange(0,512);qs=tl.arange(0,QB)
    low=tl.load(LOW+row)
    hist=tl.load(H+(row*P+parts[:,None])*512+bins[None,:],parts[:,None]<P,other=0)
    ids=tl.load(IDS+row*Q+qs,qs<Q,other=0)
    values=tl.load(X+row*V+ids,qs<Q,other=float('-inf'))
    keys=_ordered(values).to(tl.int32)-low*256
    candidate_hist=tl.histogram(keys,512,mask=(qs<Q)&(keys>=0)&(keys<512))
    remaining=tl.sum(hist,0)-candidate_hist
    total=tl.sum(remaining,0)
    fallback=total<256
    tl.store(FLAG+row,fallback.to(tl.int32))
    if not fallback:
        cdf=tl.cumsum(tl.gather(remaining,511-bins,0),0)
        ranks=tl.arange(0,256)+1
        left=tl.full((256,),0,tl.int32);right=tl.full((256,),512,tl.int32)
        for _ in range(10):
            mid=(left+right)//2
            count=tl.gather(cdf,tl.minimum(mid,511),0)
            right=tl.where(count>=ranks,mid,right);left=tl.where(count>=ranks,left,mid+1)
        key=low*256+511-left
        bits=tl.where(key>=32768,key^32768,(~key)&65535).to(tl.uint16)
        result=bits.to(tl.bfloat16,bitcast=True).to(tl.float32)
        tl.store(OUT+row*256+tl.arange(0,256),result)


@triton.jit
def _fallback_boundary(A,H,CAND,META,OFF,P:tl.constexpr,PB:tl.constexpr,Q:tl.constexpr,QB:tl.constexpr):
    if tl.load(A+tl.program_id(0))!=0:
        _boundary(H,CAND,META,OFF,P,PB,Q,QB)


@triton.jit
def _fallback_fine(A,X,H,U,META,OFF,V:tl.constexpr,P:tl.constexpr,UB:tl.constexpr):
    if tl.load(A+tl.program_id(0))!=0:
        _fine(X,H,U,META,OFF,V,P,UB)


@triton.jit
def _fallback_finish(A,H,U,CAND,META,O,P:tl.constexpr,PB:tl.constexpr,Q:tl.constexpr,QB:tl.constexpr,UB:tl.constexpr,
                     USTEPS:tl.constexpr,QSTEPS:tl.constexpr):
    if tl.load(A+tl.program_id(0))!=0:
        _finish(H,U,CAND,META,O,P,PB,Q,QB,UB,USTEPS,QSTEPS)


@torch.no_grad()
def prepare(raw,ids):
    n,l,v=raw.shape;q=ids.shape[-1];rows=n*l;used=min(l,int(__import__('os').environ.get('TAIL_CACHE_POSITIONS',str(l))));p=triton.cdiv(v,8192);qb=triton.next_power_of_2(q);ub=triton.next_power_of_2(q+256)
    assert q==256 and raw.dtype==torch.bfloat16
    z=raw.reshape(rows,v);ci=ids.reshape(rows,q).contiguous()
    alloc=lambda shape,dtype:torch.empty(shape,device=z.device,dtype=dtype)
    limited=alloc((rows,p,512),torch.int32);low=alloc(rows,torch.int32);flag=alloc(rows,torch.int32);out=alloc((rows,256),torch.float32)
    _count[(rows,p)](z,ci,limited,low,v,q,p,l,used,num_warps=8)
    _resolve[(rows,)](z,ci,limited,low,flag,out,v,q,qb,p,triton.next_power_of_2(p),l,used,num_warps=8)
    # These launches retain a device fallback for every possible finite BF16
    # input distribution; inactive rows do not enter their expensive paths.
    candidates=z.gather(-1,ci).contiguous()
    h=alloc((rows,p,256),torch.int32);fine=alloc((rows,p,256),torch.int32);maximum=alloc((rows,p),torch.float32)
    meta=alloc((rows,3),torch.int32);off=alloc((rows,p),torch.int32);upper=alloc((rows,ub),torch.float32)
    _coarse[(rows,p)](z,h,maximum,flag,1,v,p,num_warps=8)
    _fallback_boundary[(rows,)](flag,h,candidates,meta,off,p,triton.next_power_of_2(p),q,qb,num_warps=8)
    _fallback_fine[(rows,p)](flag,z,fine,upper,meta,off,v,p,ub,num_warps=8)
    _fallback_finish[(rows,)](flag,fine,upper,candidates,meta,out,p,triton.next_power_of_2(p),q,qb,ub,ub.bit_length()+1,qb.bit_length()+1,num_warps=8)
    return out.reshape(n,l,256).transpose(0,1).contiguous(),ids.transpose(0,1).contiguous()
