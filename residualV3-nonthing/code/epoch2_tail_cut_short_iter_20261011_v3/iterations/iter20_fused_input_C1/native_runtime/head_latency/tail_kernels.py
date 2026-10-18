"""Per-tile branches omit weight loads and MMA after stopping. Active rows preserve K64/BF16. Kernel launch slots and tiny norms remain."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _gated_gemm(ACTIVE,COUNTS,AST:tl.constexpr,COUNT:tl.constexpr,A,W,O,B,R,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
          AS0:tl.constexpr,AS1:tl.constexpr,WS0:tl.constexpr,WS1:tl.constexpr,
          RS0:tl.constexpr,RS1:tl.constexpr,HAS_R:tl.constexpr,
          BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=64):
    rr=tl.program_id(0)*BM+tl.arange(0,BM)
    enabled=tl.max(tl.load(ACTIVE+rr*AST,rr<M,other=0))!=0
    if enabled:
        if COUNT:
            tl.atomic_add(COUNTS+0,1)
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
    else:
        cc=tl.program_id(1)*BN+tl.arange(0,BN)
        tl.store(O+rr[:,None]*N+cc[None,:],0,(rr[:,None]<M)&(cc[None,:]<N))

@triton.jit
def _gated_up_silu(ACTIVE,COUNTS,AST:tl.constexpr,COUNT:tl.constexpr,A,W,O,B,M:tl.constexpr,I:tl.constexpr,K:tl.constexpr,
             AS0:tl.constexpr,AS1:tl.constexpr,WS0:tl.constexpr,WS1:tl.constexpr,
             BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=64):
    rr=tl.program_id(0)*BM+tl.arange(0,BM)
    enabled=tl.max(tl.load(ACTIVE+rr*AST,rr<M,other=0))!=0
    if enabled:
        if COUNT:
            tl.atomic_add(COUNTS+1,1)
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
    else:
        cc=tl.program_id(1)*BN+tl.arange(0,BN)
        tl.store(O+rr[:,None]*I+cc[None,:],0,(rr[:,None]<M)&(cc[None,:]<I))

@triton.jit
def _gated_correct(ACTIVE,COUNTS,AST:tl.constexpr,COUNT:tl.constexpr,R,C,I,Z,RAW,D,H,N:tl.constexpr,Q:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
             RS0:tl.constexpr,RS1:tl.constexpr,RS2:tl.constexpr,
             CS0:tl.constexpr,CS1:tl.constexpr,IS0:tl.constexpr,IS1:tl.constexpr,
             ZS0:tl.constexpr,ZS1:tl.constexpr,RAW0:tl.constexpr,RAW1:tl.constexpr,
             DS:tl.constexpr,HS:tl.constexpr,DIRECT:tl.constexpr=False,BM:tl.constexpr=64,BN:tl.constexpr=16,BK:tl.constexpr=64):
    batch=tl.program_id(0)
    enabled=tl.load(ACTIVE+batch*AST)!=0
    if enabled:
        if COUNT:
            tl.atomic_add(COUNTS+2,1)
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

def linear(x,w,b,active,counts,count=False,residual=None):
    shape=x.shape[:-1];a=x.reshape(-1,x.shape[-1]);m,k=a.shape;n=w.shape[0]
    out=x.new_empty((m,n));r=residual.reshape(m,n) if residual is not None else None
    _gated_gemm[(triton.cdiv(m,16),triton.cdiv(n,64))](active,counts,active.stride(0),count,a,w,out,b,r,m,n,k,*a.stride(),*w.stride(),
        *(r.stride() if r is not None else (0,0)),HAS_R=r is not None,num_warps=4,num_stages=3,enable_fp_fusion=False)
    return out.reshape(*shape,n)

def up_silu(x,module,active,counts,count=False):
    m,k=x.shape;i=module.weight.shape[0]//2;out=x.new_empty((m,i))
    _gated_up_silu[(triton.cdiv(m,16),triton.cdiv(i,64))](active,counts,active.stride(0),count,x,module.weight,out,module.bias,m,i,k,*x.stride(),*module.weight.stride(),num_warps=4,num_stages=3,enable_fp_fusion=False)
    return out

def correct(raw,basis,coeff,dose,hn,ids,z,active,counts,count=False):
    n,q=ids.shape;k=coeff.shape[-1]
    _gated_correct[(n,triton.cdiv(q,64))](active,counts,active.stride(0),count,basis,coeff,ids,z,raw,dose,hn,n,q,k,raw.shape[-1],
        0,*basis.stride(),*coeff.stride(),*ids.stride(),*z.stride(),*raw.stride(),dose.stride(0),hn.stride(0),DIRECT=True,
        num_warps=4,num_stages=3,enable_fp_fusion=False)
    return z

@triton.jit
def _maxq_parts(Z,P,ACTIVE,N:tl.constexpr,V:tl.constexpr,BLOCKS:tl.constexpr,
                ZS:tl.constexpr,AST:tl.constexpr,B:tl.constexpr=2048):
    row=tl.program_id(0);part=tl.program_id(1);cols=part*B+tl.arange(0,B)
    enabled=tl.load(ACTIVE+row*AST)!=0
    if enabled:
        z=tl.load(Z+row*ZS+cols,cols<V,other=-float('inf')).to(tl.float32)
        mx=tl.max(z,0);ex=tl.exp(z-mx);denom=tl.sum(ex,0)
    else:
        mx=0.;denom=1.
    tl.store(P+row*(2*BLOCKS)+part,mx)
    tl.store(P+row*(2*BLOCKS)+BLOCKS+part,denom)

@triton.jit
def _risk_finish(P,ACTIVE,SURV,PATIENCE,MAXQ,CONFIG,TEMP,MAP,N:tl.constexpr,L:tl.constexpr,
                 BLOCKS:tl.constexpr,J:tl.constexpr,MINIMUM:tl.constexpr,PAT:tl.constexpr,FORCE:tl.constexpr,
                 BN:tl.constexpr,BB:tl.constexpr):
    rows=tl.arange(0,BN);blocks=tl.arange(0,BB)
    alive=tl.load(ACTIVE+rows*(L+2)+J,rows<N,other=0)
    prior=tl.load(SURV+rows*(L+1)+J,rows<N,other=1.)
    previous_count=tl.load(PATIENCE+rows*(L+1)+J,rows<N,other=0)
    mx=tl.load(P+rows[:,None]*(2*BLOCKS)+blocks[None,:],(rows[:,None]<N)&(blocks[None,:]<BLOCKS),other=-float('inf'))
    ss=tl.load(P+rows[:,None]*(2*BLOCKS)+BLOCKS+blocks[None,:],(rows[:,None]<N)&(blocks[None,:]<BLOCKS),other=0.)
    largest=tl.max(mx,1);denom=tl.sum(ss*tl.exp(mx-largest[:,None]),1)
    maxq=1./denom
    map=tl.load(MAP+rows,rows<N,other=0);temperature=tl.load(TEMP+map)
    t=tl.where(temperature==0.,0,1)
    w=tl.load(CONFIG+t*3);b=tl.load(CONFIG+t*3+1);threshold=tl.load(CONFIG+t*3+2)
    hazard=tl.sigmoid(w*maxq+b)
    survival=tl.where(alive!=0,prior*(1.-hazard),prior)
    high=((1.-survival)>=threshold)&(J+1>=MINIMUM)
    consecutive=tl.where(high,previous_count+1,0)
    next_alive=tl.load(ACTIVE+rows*(L+2)+J+1,rows<N,other=0)
    keep=next_alive&tl.cast(consecutive<PAT,tl.int32)
    if FORCE>0:keep=tl.full((BN,),int(J+2<FORCE),tl.int32)
    tl.store(SURV+rows*(L+1)+J+1,survival,rows<N)
    tl.store(PATIENCE+rows*(L+1)+J+1,consecutive,rows<N)
    tl.store(MAXQ+rows*L+J,maxq,rows<N)
    tl.store(ACTIVE+rows*(L+2)+J+2,keep,rows<N)


def signal(z,active,survival,patience,maxq,config,temp,mappings,j,*,minimum=8,pat=2,force=0):
    n,v=z.shape;length=maxq.shape[1];blocks=triton.cdiv(v,2048)
    parts=torch.empty((n,2*blocks),device=z.device,dtype=torch.float32)
    _maxq_parts[(n,blocks)](z,parts,active[:,j],n,v,blocks,z.stride(0),active.stride(0),num_warps=4)
    _risk_finish[(1,)](parts,active,survival,patience,maxq,config,temp,mappings,n,length,blocks,j,minimum,pat,force,
        triton.next_power_of_2(n),triton.next_power_of_2(blocks),num_warps=4)

@triton.jit
def _aggregate(A,CUTS,COUNTERS,N:tl.constexpr,L:tl.constexpr,B:tl.constexpr):
    rows=tl.arange(0,B);cut=tl.full((B,),0,tl.int32)
    for j in range(L):cut+=tl.load(A+rows*(L+2)+j,rows<N,other=0)
    tl.store(CUTS+rows,cut,rows<N)
    tl.atomic_add(COUNTERS+cut,1,rows<N)

def aggregate(active,counters,length):
    n=active.shape[0];cuts=torch.empty((n,),device=active.device,dtype=torch.int32)
    _aggregate[(1,)](active,cuts,counters,n,length,triton.next_power_of_2(n),num_warps=4)
    return cuts
