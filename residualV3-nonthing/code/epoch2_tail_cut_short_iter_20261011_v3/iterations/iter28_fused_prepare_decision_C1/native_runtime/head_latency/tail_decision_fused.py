"""Frozen Q45 readout and delayed stop-state update in one launch.

Retain the existing BF16 MMA/bias/SiLU rounding points. This only handles a
single position; the early batched prefix keeps its original implementation.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _readout_finish(X,MEAN,SCALE,W0,B0,W1,B1,ACTIVE,SURV,PATIENCE,MAXQ,SCORES,
                    THRESH,TEMP,MAP,M:tl.constexpr,L:tl.constexpr,J:tl.constexpr,
                    XS0:tl.constexpr,XS1:tl.constexpr,MINIMUM:tl.constexpr,
                    PAT:tl.constexpr,FORCE:tl.constexpr,BM:tl.constexpr=16):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    kk=tl.arange(0,64);hh=tl.arange(0,32)
    x=tl.load(X+rows[:,None]*XS0+kk[None,:]*XS1,
              (rows[:,None]<M)&(kk[None,:]<45),other=0.)
    mean=tl.load(MEAN+kk,kk<45,other=0.)
    scale=tl.load(SCALE+kk,kk<45,other=1.)
    normalized=tl.div_rn(x-mean[None,:],scale[None,:])
    normalized=tl.where(kk[None,:]<45,normalized,0.).to(tl.bfloat16)
    w0=tl.load(W0+hh[None,:]*45+kk[:,None],kk[:,None]<45,other=0.)
    a=tl.dot(normalized,w0).to(tl.bfloat16).to(tl.float32)
    bias0=tl.load(B0+hh).to(tl.float32)
    a=(a+bias0[None,:]).to(tl.bfloat16).to(tl.float32)
    h=tl.div_rn(a,1.+libdevice.exp(-a)).to(tl.bfloat16)
    cols=tl.arange(0,16)
    w1=tl.load(W1+cols[None,:]*32+hh[:,None],cols[None,:]<2,other=0.)
    output=tl.dot(h,w1).to(tl.bfloat16).to(tl.float32)
    bias1=tl.load(B1+cols,cols<2,other=0.).to(tl.float32)
    output=(output+bias1[None,:]).to(tl.bfloat16).to(tl.float32)
    probability=tl.div_rn(1.,1.+libdevice.exp(-output))

    alive=tl.load(ACTIVE+rows*(L+2)+J,rows<M,other=0)
    prior=tl.load(SURV+rows*(L+1)+J,rows<M,other=1.)
    old=tl.load(PATIENCE+rows*(L+1)+J,rows<M,other=0)
    mappings=tl.load(MAP+rows,rows<M,other=0)
    temperature=tl.load(TEMP+mappings)
    t=tl.where(temperature==0.,0,1)
    # One finite positive term plus zeros: exact selection, no changed sum
    # of confidences or reassociation of the survival product.
    confidence=tl.sum(tl.where(cols[None,:]==t[:,None],probability,0.),1)
    survival=tl.where(alive!=0,prior*confidence,prior)
    threshold=tl.load(THRESH+t)
    low=(survival<threshold)&(J+1>=MINIMUM)
    count=tl.where(low,old+1,0)
    next_alive=tl.load(ACTIVE+rows*(L+2)+J+1,rows<M,other=0)
    keep=next_alive&tl.cast(count<PAT,tl.int32)
    if FORCE>0:
        keep=tl.full((BM,),int(J+2<FORCE),tl.int32)
    tl.store(SURV+rows*(L+1)+J+1,survival,rows<M)
    tl.store(PATIENCE+rows*(L+1)+J+1,count,rows<M)
    maximum=tl.load(X+rows*XS0,rows<M,other=0.)
    tl.store(MAXQ+rows*L+J,maximum,rows<M)
    tl.store(SCORES+(rows[:,None]*L+J)*2+cols[None,:],probability,
             (rows[:,None]<M)&(cols[None,:]<2))
    tl.store(ACTIVE+rows*(L+2)+J+2,keep,rows<M)


@torch.no_grad()
def signal(head,q45,active,survival,patience,maxq,scores,temp,mappings,j,force):
    assert head._tail_branch=='Q45_MLP' and q45.dtype==torch.float32
    net=head._tail_readouts['Q45_MLP'];cfg=head._tail_config
    assert net[0].weight.is_contiguous() and net[2].weight.is_contiguous()
    _readout_finish[(triton.cdiv(q45.shape[0],16),)](
        q45,head._tail_mean,head._tail_scale,net[0].weight,net[0].bias,
        net[2].weight,net[2].bias,active,survival,patience,maxq,scores,
        head._tail_thresholds,temp,mappings,q45.shape[0],maxq.shape[1],j,
        *q45.stride(),cfg['minimum_repair'],cfg['patience'],force,
        num_warps=4,num_stages=3,enable_fp_fusion=False)
