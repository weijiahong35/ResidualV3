"""Cache unmodified Top256 by exact BF16 counts, subtracting repair support.

Coarse and fine histograms count raw logits; subtract candidate value counts.
Only <=Q+255 values above the fine boundary need sorting. Token tie order is
irrelevant here: original repair IDs remain unchanged, and q45 uses values.
"""
import torch
import triton
import triton.language as tl
from .tail_q45_hist import _coarse,_ordered


@triton.jit
def _boundary(H,CAND,META,OFF,P:tl.constexpr,PB:tl.constexpr,Q:tl.constexpr,QB:tl.constexpr):
    row=tl.program_id(0);parts=tl.arange(0,PB);cols=tl.arange(0,256);qs=tl.arange(0,QB)
    h=tl.load(H+(row*P+parts[:,None])*256+cols[None,:],parts[:,None]<P,other=0)
    candidates=tl.load(CAND+row*Q+qs,qs<Q,other=0)
    ch=tl.histogram((_ordered(candidates)>>8).to(tl.int32),256,mask=qs<Q)
    remaining=tl.sum(h,0)-tl.gather(ch,cols,0)
    cdf=tl.cumsum(tl.gather(remaining,255-cols,0),0)
    boundary=255-tl.min(tl.where(cdf>=256,cols,256),0)
    raw_upper_parts=tl.sum(tl.where(cols[None,:]>boundary,h,0),1)
    raw_upper=tl.sum(raw_upper_parts,0)
    upper=tl.sum(tl.where(cols>boundary,remaining,0),0)
    offset=tl.cumsum(raw_upper_parts,0)-raw_upper_parts
    tl.store(META+row*3,boundary);tl.store(META+row*3+1,raw_upper);tl.store(META+row*3+2,upper)
    tl.store(OFF+row*P+parts,offset,parts<P)


@triton.jit
def _fine(X,H,U,META,OFF,V:tl.constexpr,P:tl.constexpr,UB:tl.constexpr,B:tl.constexpr=8192):
    row=tl.program_id(0);part=tl.program_id(1);cols=part*B+tl.arange(0,B);valid=cols<V
    x=tl.load(X+row*V+cols,valid,other=float('-inf'))
    key=_ordered(x);boundary=tl.load(META+row*3)
    same=valid&((key>>8)==boundary);above=valid&((key>>8)>boundary)
    hist=tl.histogram((key&255).to(tl.int32),256,mask=same)
    take=tl.arange(0,256);tl.store(H+(row*P+part)*256+take,tl.gather(hist,take,0))
    dest=tl.load(OFF+row*P+part)+tl.cumsum(above.to(tl.int32),0)-1
    tl.store(U+row*UB+dest,x.to(tl.float32),above)


@triton.jit
def _bound_desc(values,query,SIZE:tl.constexpr,B:tl.constexpr,STEPS:tl.constexpr,STRICT:tl.constexpr):
    left=tl.full((B,),0,tl.int32);right=tl.full((B,),SIZE,tl.int32)
    for _ in range(STEPS):
        mid=(left+right)//2
        v=tl.gather(values,tl.minimum(mid,SIZE-1),0)
        above=(mid<SIZE)&(v>query if STRICT else v>=query)
        left=tl.where(above,mid+1,left);right=tl.where(above,right,mid)
    return left


@triton.jit
def _finish(H,U,CAND,META,O,P:tl.constexpr,PB:tl.constexpr,Q:tl.constexpr,QB:tl.constexpr,UB:tl.constexpr,
            USTEPS:tl.constexpr,QSTEPS:tl.constexpr):
    row=tl.program_id(0);parts=tl.arange(0,PB);cols=tl.arange(0,256);qs=tl.arange(0,QB);us=tl.arange(0,UB)
    boundary=tl.load(META+row*3);raw_upper=tl.load(META+row*3+1);upper=tl.load(META+row*3+2)
    candidates=tl.load(CAND+row*Q+qs,qs<Q,other=float('-inf'))
    candidate_keys=_ordered(candidates)
    same=(qs<Q)&((candidate_keys>>8)==boundary)
    ch=tl.histogram((candidate_keys&255).to(tl.int32),256,mask=same)
    h=tl.load(H+(row*P+parts[:,None])*256+cols[None,:],parts[:,None]<P,other=0)
    remaining=tl.sum(h,0)-tl.gather(ch,cols,0)
    cdf=tl.cumsum(tl.gather(remaining,255-cols,0),0)
    raw_values=tl.load(U+row*UB+us,us<raw_upper,other=float('-inf'))
    ordered=tl.sort(raw_values,descending=True)
    cordered=tl.sort(candidates.to(tl.float32),descending=True)
    end=_bound_desc(ordered,ordered,UB,UB,USTEPS,False)
    cstart=_bound_desc(cordered,ordered,QB,UB,QSTEPS,True)
    cend=_bound_desc(cordered,ordered,QB,UB,QSTEPS,False)
    keep=(us<raw_upper)&(us<end-(cend-cstart))
    dest=tl.cumsum(keep.to(tl.int32),0)-1
    tl.store(O+row*256+dest,ordered,keep)
    rank=cols-upper+1;left=tl.full((256,),0,tl.int32);right=tl.full((256,),256,tl.int32)
    for _ in range(9):
        mid=(left+right)//2;count=tl.gather(cdf,tl.minimum(mid,255),0)
        right=tl.where(count>=rank,mid,right);left=tl.where(count>=rank,left,mid+1)
    key=(boundary<<8)|(255-left)
    bits=tl.where(key>=32768,key^32768,(~key)&65535).to(tl.uint16)
    value=bits.to(tl.bfloat16,bitcast=True).to(tl.float32)
    tl.store(O+row*256+cols,value,cols>=upper)


@torch.no_grad()
def prepare(raw,ids):
    n,l,v=raw.shape;q=ids.shape[-1];rows=n*l;p=triton.cdiv(v,8192);ub=triton.next_power_of_2(q+256)
    z=raw.reshape(rows,v);candidates=z.gather(-1,ids.reshape(rows,q)).contiguous()
    alloc=lambda shape,dtype:torch.empty(shape,device=z.device,dtype=dtype)
    h=alloc((rows,p,256),torch.int32);fine=alloc((rows,p,256),torch.int32)
    maximum=alloc((rows,p),torch.float32);meta=alloc((rows,3),torch.int32);off=alloc((rows,p),torch.int32)
    upper=alloc((rows,ub),torch.float32);out=alloc((rows,256),torch.float32)
    active=torch.ones(rows,device=z.device,dtype=torch.int32)
    _coarse[(rows,p)](z,h,maximum,active,1,v,p,num_warps=8)
    _boundary[(rows,)](h,candidates,meta,off,p,triton.next_power_of_2(p),q,triton.next_power_of_2(q),num_warps=8)
    _fine[(rows,p)](z,fine,upper,meta,off,v,p,ub,num_warps=8)
    _finish[(rows,)](fine,upper,candidates,meta,out,p,triton.next_power_of_2(p),q,triton.next_power_of_2(q),ub,ub.bit_length()+1,triton.next_power_of_2(q).bit_length()+1,num_warps=8)
    return out.reshape(n,l,256).transpose(0,1).contiguous(),ids.transpose(0,1).contiguous()
