"""Exact BF16 Top256 values via two-level radix histograms, with device gating.

No full-vocabulary sort. At most255 values above the selected coarse bin
are compacted and sorted; the remaining values come from the fine histogram.
Entropy uses the original8192-wide sums and original running-max recurrence.
"""
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _maximum(a,b):
    return tl.maximum(a,b)

@triton.jit
def _ordered(x):
    bits=x.to(tl.uint16,bitcast=True).to(tl.uint32)
    return tl.where((bits&32768)!=0,(~bits)&65535,bits^32768)

@triton.jit
def _coarse(X,H,MAX,A,AST:tl.constexpr,V:tl.constexpr,P:tl.constexpr,B:tl.constexpr=8192):
    row=tl.program_id(0);part=tl.program_id(1);take=tl.arange(0,256)
    if tl.load(A+row*AST)!=0:
        c=part*B+tl.arange(0,B);valid=c<V
        x=tl.load(X+row*V+c,valid,other=float('-inf'))
        key=_ordered(x);hist=tl.histogram((key>>8).to(tl.int32),256,mask=valid)
        h=tl.gather(hist,take,0);mx=tl.max(x.to(tl.float32),0)
    else:
        h=tl.full((256,),0,tl.int32);mx=float('-inf')
    tl.store(H+(row*P+part)*256+take,h);tl.store(MAX+row*P+part,mx)

@triton.jit
def _coarse_finish(H,MAX,A,AST:tl.constexpr,META,OFF,CUMMAX,P:tl.constexpr,PB:tl.constexpr):
    row=tl.program_id(0);parts=tl.arange(0,PB);cols=tl.arange(0,256)
    if tl.load(A+row*AST)!=0:
        h=tl.load(H+(row*P+parts[:,None])*256+cols[None,:],parts[:,None]<P,other=0)
        total=tl.sum(h,0)
        descending=tl.gather(total,255-cols,0);cdf=tl.cumsum(descending,0)
        boundary=255-tl.min(tl.where(cdf>=256,cols,256),0)
        upper_parts=tl.sum(tl.where(cols[None,:]>boundary,h,0),1)
        upper=tl.sum(upper_parts,0);offset=tl.cumsum(upper_parts,0)-upper_parts
        maximum=tl.load(MAX+row*P+parts,parts<P,other=float('-inf'))
        cumulative_max=tl.associative_scan(maximum,0,_maximum)
    else:
        boundary=0;upper=0;offset=tl.full((PB,),0,tl.int32)
        cumulative_max=tl.full((PB,),0.,tl.float32)
    tl.store(META+row*2,boundary);tl.store(META+row*2+1,upper)
    tl.store(OFF+row*P+parts,offset,parts<P)
    tl.store(CUMMAX+row*P+parts,cumulative_max,parts<P)

@triton.jit
def _fine(X,H,U,MASS,A,AST:tl.constexpr,META,OFF,CUMMAX,V:tl.constexpr,P:tl.constexpr,B:tl.constexpr=8192):
    row=tl.program_id(0);part=tl.program_id(1);take=tl.arange(0,256)
    if tl.load(A+row*AST)!=0:
        c=part*B+tl.arange(0,B);valid=c<V
        x=tl.load(X+row*V+c,valid,other=float('-inf'));xf=x.to(tl.float32)
        key=_ordered(x);boundary=tl.load(META+row*2)
        same=valid&((key>>8)==boundary);above=valid&((key>>8)>boundary)
        hist=tl.histogram((key&255).to(tl.int32),256,mask=same)
        h=tl.gather(hist,take,0)
        destination=tl.load(OFF+row*P+part)+tl.cumsum(above.to(tl.int32),0)-1
        tl.store(U+row*256+destination,xf,above)
        maximum=tl.load(CUMMAX+row*P+part)
        mass=tl.where(valid,tl.exp(xf-maximum),0.)
        denominator=tl.sum(mass,0)
        weighted=tl.sum(mass*tl.where(valid,xf,0.),0)
    else:
        h=tl.full((256,),0,tl.int32);denominator=0.;weighted=0.
    tl.store(H+(row*P+part)*256+take,h)
    tl.store(MASS+(row*P+part)*2,denominator)
    tl.store(MASS+(row*P+part)*2+1,weighted)

@triton.jit
def _finish(H,U,MASS,A,AST:tl.constexpr,META,CUMMAX,TOP,S,R:tl.constexpr,V:tl.constexpr,P:tl.constexpr,PB:tl.constexpr):
    row=tl.program_id(0);parts=tl.arange(0,PB);cols=tl.arange(0,256)
    if tl.load(A+row*AST)!=0:
        h=tl.load(H+(row*P+parts[:,None])*256+cols[None,:],parts[:,None]<P,other=0)
        total=tl.sum(h,0);cdf=tl.cumsum(tl.gather(total,255-cols,0),0)
        upper=tl.load(META+row*2+1);boundary=tl.load(META+row*2)
        u=tl.load(U+row*256+cols,cols<upper,other=float('-inf'))
        ordered=tl.sort(u,descending=True)
        rank=cols-upper+1;left=tl.full((256,),0,tl.int32);right=tl.full((256,),256,tl.int32)
        for _ in range(9):
            middle=(left+right)//2
            count=tl.gather(cdf,tl.minimum(middle,255),0)
            found=count>=rank
            right=tl.where(found,middle,right);left=tl.where(found,left,middle+1)
        key=(boundary<<8)|(255-left)
        bits=tl.where(key>=32768,key^32768,(~key)&65535).to(tl.uint16)
        value=bits.to(tl.bfloat16,bitcast=True).to(tl.float32)
        top=tl.where(cols<upper,ordered,value)
        maximum=float('-inf');denominator=0.;weighted=0.
        for part in range(P):
            next_max=tl.load(CUMMAX+row*P+part)
            factor=tl.exp(maximum-next_max)
            denominator=denominator*factor+tl.load(MASS+(row*P+part)*2)
            weighted=weighted*factor+tl.load(MASS+(row*P+part)*2+1)
            maximum=next_max
        lse=maximum+tl.log(denominator)
        entropy=(lse-weighted/denominator)/tl.log(V*1.0)
    else:
        top=tl.full((256,),float('-inf'),tl.float32);lse=0.;entropy=0.
    tl.store(TOP+row*256+cols,top)
    tl.store(S+row,lse);tl.store(S+R+row,entropy)

@torch.no_grad()
def features(logits,active):
    z=logits.detach().contiguous();n,v=z.shape;p=triton.cdiv(v,8192)
    assert z.dtype==torch.bfloat16 and active.shape==(n,)
    alloc=lambda shape,dtype:torch.empty(shape,device=z.device,dtype=dtype)
    coarse=alloc((n,p,256),torch.int32);fine=alloc((n,p,256),torch.int32)
    maximum=alloc((n,p),torch.float32);cumulative_max=alloc((n,p),torch.float32)
    meta=alloc((n,2),torch.int32);offsets=alloc((n,p),torch.int32)
    upper=alloc((n,256),torch.float32);mass=alloc((n,p,2),torch.float32)
    top=alloc((n,256),torch.float32);stats=alloc((2,n),torch.float32)
    _coarse[(n,p)](z,coarse,maximum,active,active.stride(0),v,p,num_warps=8)
    _coarse_finish[(n,)](coarse,maximum,active,active.stride(0),meta,offsets,cumulative_max,p,triton.next_power_of_2(p),num_warps=8)
    _fine[(n,p)](z,fine,upper,mass,active,active.stride(0),meta,offsets,cumulative_max,v,p,num_warps=8)
    _finish[(n,)](fine,upper,mass,active,active.stride(0),meta,cumulative_max,top,stats,n,v,p,triton.next_power_of_2(p),num_warps=8)
    log_top=top-stats[0,:,None];probabilities=log_top.exp();cumulative=probabilities.cumsum(-1)
    four=torch.stack((probabilities[:,0],probabilities[:,1],probabilities[:,0]-probabilities[:,1],stats[1]),-1)
    cum=torch.stack([cumulative[:,i] for i in (0,1,3,7,15,31,63,127,255)],-1)
    return torch.cat((four,cum,log_top[:,:32].clamp_min(math.log(1e-30))),-1)
