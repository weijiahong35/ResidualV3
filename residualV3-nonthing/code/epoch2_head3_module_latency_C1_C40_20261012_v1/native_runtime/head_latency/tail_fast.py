"""Device-gated q45 extraction and sampling; kept prefixes retain stock math.

Top256 values are exact: each chunk contributes its own largest256 values.
Token IDs are deliberately absent from q45. The sampler uses vLLM's original
gumbel_block_argmax and its original1024-value tiles, seeds and positions.
Inactive suffix logits/q are never submitted to target verification.
"""
import math
import torch
import triton
import triton.language as tl
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax


@triton.jit
def _entropy(X, S, ACTIVE, AST:tl.constexpr, R:tl.constexpr, V:tl.constexpr,
             B:tl.constexpr=8192):
    row=tl.program_id(0).to(tl.int64)
    if tl.load(ACTIVE+row*AST)!=0:
        maximum=float('-inf'); denominator=0.; weighted=0.
        for start in range(0,V,B):
            offsets=start+tl.arange(0,B); valid=offsets<V
            x=tl.load(X+row*V+offsets,valid,other=float('-inf')).to(tl.float32)
            next_max=tl.maximum(maximum,tl.max(x,0))
            factor=tl.exp(maximum-next_max)
            mass=tl.where(valid,tl.exp(x-next_max),0.)
            denominator=denominator*factor+tl.sum(mass,0)
            weighted=weighted*factor+tl.sum(mass*tl.where(valid,x,0.),0)
            maximum=next_max
        lse=maximum+tl.log(denominator)
        entropy=(lse-weighted/denominator)/tl.log(V*1.0)
    else:
        lse=0.; entropy=0.
    tl.store(S+row,lse); tl.store(S+R+row,entropy)


@triton.jit
def _chunk_top(X, C, ACTIVE, AST:tl.constexpr, V:tl.constexpr, PARTS:tl.constexpr,
               B:tl.constexpr=8192):
    row=tl.program_id(0); part=tl.program_id(1)
    take=tl.arange(0,256)
    if tl.load(ACTIVE+row*AST)!=0:
        cols=part*B+tl.arange(0,B)
        x=tl.load(X+row*V+cols,cols<V,other=float('-inf')).to(tl.float32)
        ordered=tl.sort(x,descending=True)
        top=tl.gather(ordered,take,axis=0)
    else:
        top=tl.full((256,),float('-inf'),tl.float32)
    tl.store(C+(row*PARTS+part)*256+take,top)


@triton.jit
def _merge_top(C, O, ACTIVE, AST:tl.constexpr, PARTS:tl.constexpr, B:tl.constexpr):
    row=tl.program_id(0); take=tl.arange(0,256)
    if tl.load(ACTIVE+row*AST)!=0:
        cols=tl.arange(0,B)
        x=tl.load(C+row*PARTS*256+cols,cols<PARTS*256,other=float('-inf'))
        ordered=tl.sort(x,descending=True)
        top=tl.gather(ordered,take,axis=0)
    else:
        top=tl.full((256,),float('-inf'),tl.float32)
    tl.store(O+row*256+take,top)


@torch.no_grad()
def bitonic_features(logits,active):
    # Keep the old normalizer, exp and cumsum arithmetic, including FP32
    # rounding. Only TopK selection and the inactive row branches change.
    logits=logits.detach().contiguous();rows,vocab=logits.shape
    assert logits.dtype==torch.bfloat16 and active.shape==(rows,)
    parts=triton.cdiv(vocab,8192)
    stats=torch.empty((2,rows),device=logits.device,dtype=torch.float32)
    chunks=torch.empty((rows,parts,256),device=logits.device,dtype=torch.float32)
    top=torch.empty((rows,256),device=logits.device,dtype=torch.float32)
    _entropy[(rows,)](logits,stats,active,active.stride(0),rows,vocab,num_warps=8)
    _chunk_top[(rows,parts)](logits,chunks,active,active.stride(0),vocab,parts,num_warps=8)
    _merge_top[(rows,)](chunks,top,active,active.stride(0),parts,triton.next_power_of_2(parts*256),num_warps=8)
    log_top=top-stats[0,:,None];probabilities=log_top.exp();cumulative=probabilities.cumsum(-1)
    four=torch.stack((probabilities[:,0],probabilities[:,1],probabilities[:,0]-probabilities[:,1],stats[1]),-1)
    cum=torch.stack([cumulative[:,i] for i in (0,1,3,7,15,31,63,127,255)],-1)
    return torch.cat((four,cum,log_top[:,:32].clamp_min(math.log(1e-30))),-1)

# The bitonic prototype is preserved for audit; profiling rejected it as slower.
from .tail_q45_hist import features


@triton.jit
def _sample_parts(IDS,MAXIMUM,Q,QS:tl.constexpr,COL,Z,ZS:tl.constexpr,MAP,
                  SEEDS,POS,TEMP,ACTIVE,OVERRIDE,AST:tl.constexpr,V:tl.constexpr,
                  PARTS:tl.constexpr,APPLY:tl.constexpr,FP64:tl.constexpr,
                  PER_COL:tl.constexpr,B:tl.constexpr=1024):
    row=tl.program_id(0).to(tl.int64);part=tl.program_id(1)
    override=tl.load(OVERRIDE)!=0 if OVERRIDE is not None else False
    if (tl.load(ACTIVE+row*AST)!=0)|override:
        block=part*B+tl.arange(0,B);mask=block<V
        logits=tl.load(Z+row*ZS+block,mask,other=float('-inf')).to(tl.float32)
        value,index=gumbel_block_argmax(logits,block,mask,row,MAP,TEMP,SEEDS,POS,
             Q,QS,COL,V,APPLY_TEMPERATURE=APPLY,USE_FP64=FP64,PER_TOKEN_COL=PER_COL)
        token=part*B+index
    else:
        value=tl.full((),float('-inf'),tl.float64 if FP64 else tl.float32);token=0
    tl.store(IDS+row*PARTS+part,token)
    tl.store(MAXIMUM+row*PARTS+part,value)


@triton.jit
def _sample_finish(MAXIMUM,IDS,O,ACTIVE,OVERRIDE,AST:tl.constexpr,PARTS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    override=tl.load(OVERRIDE)!=0 if OVERRIDE is not None else False
    if (tl.load(ACTIVE+row*AST)!=0)|override:
        offsets=tl.arange(0,B)
        values=tl.load(MAXIMUM+row*PARTS+offsets,offsets<PARTS,other=float('-inf'))
        _,index=tl.max(values,axis=0,return_indices=True)
        token=tl.load(IDS+row*PARTS+index)
    else:
        token=tl.full((),0,tl.int64)
    tl.store(O+row,token)


def sample(logits,idx,temp,seeds,pos,*,active,apply_temperature,override=None,
           output_processed_logits=None,output_processed_logits_col=None,use_fp64=False):
    idx=idx.contiguous();pos=pos.contiguous()
    col=output_processed_logits_col.contiguous() if output_processed_logits_col is not None else None
    n,v=logits.shape;parts=triton.cdiv(v,1024)
    ids=torch.empty((n,parts),device=logits.device,dtype=torch.int64)
    mx=torch.empty((n,parts),device=logits.device,dtype=torch.float64 if use_fp64 else torch.float32)
    out=torch.empty((n,),device=logits.device,dtype=torch.int64)
    _sample_parts[(n,parts)](ids,mx,output_processed_logits,
         output_processed_logits.stride(0) if output_processed_logits is not None else 0,
         col,logits,logits.stride(0),idx,seeds,pos,temp,active,override,active.stride(0),v,parts,
         apply_temperature,use_fp64,col is not None and col.dim()>0,num_warps=4)
    _sample_finish[(n,)](mx,ids,out,active,override,active.stride(0),parts,triton.next_power_of_2(parts),num_warps=4)
    return out
