# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stock1024 Gumbel tiles also export the pre-temperature maximum.

Reuse vLLM's unchanged gumbel_block_argmax and seeds/positions. The auxiliary
maximum avoids a later vocabulary read; it never changes proposal scores.
"""
import torch
import triton
import triton.language as tl
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax
from .tail_fast import _sample_finish


@triton.jit
def _parts(IDS,MAXIMUM,RAWMAX,Q,QS:tl.constexpr,COL,Z,ZS:tl.constexpr,MAP,
           SEEDS,POS,TEMP,ACTIVE,OVERRIDE,AST:tl.constexpr,V:tl.constexpr,
           PARTS:tl.constexpr,APPLY:tl.constexpr,FP64:tl.constexpr,
           PER_COL:tl.constexpr,B:tl.constexpr=1024):
    row=tl.program_id(0).to(tl.int64);part=tl.program_id(1)
    override=tl.load(OVERRIDE)!=0 if OVERRIDE is not None else False
    enabled=True if ACTIVE is None else ((tl.load(ACTIVE+row*AST)!=0)|override)
    if enabled:
        block=part*B+tl.arange(0,B);mask=block<V
        logits=tl.load(Z+row*ZS+block,mask,other=float('-inf')).to(tl.float32)
        raw_max=tl.max(logits,0)
        value,index=gumbel_block_argmax(logits,block,mask,row,MAP,TEMP,SEEDS,POS,
             Q,QS,COL,V,APPLY_TEMPERATURE=APPLY,USE_FP64=FP64,PER_TOKEN_COL=PER_COL)
        token=part*B+index
    else:
        value=tl.full((),float('-inf'),tl.float64 if FP64 else tl.float32)
        token=0;raw_max=0.
    tl.store(IDS+row*PARTS+part,token)
    tl.store(MAXIMUM+row*PARTS+part,value)
    tl.store(RAWMAX+row*PARTS+part,raw_max)


@torch.no_grad()
def sample(logits,idx,temp,seeds,pos,*,raw_maximum,active=None,override=None,
           apply_temperature,output_processed_logits=None,
           output_processed_logits_col=None,use_fp64=False):
    from .structure_fusion import _sample_finish as dense_finish
    idx=idx.contiguous();pos=pos.contiguous()
    col=output_processed_logits_col.contiguous() if output_processed_logits_col is not None else None
    n,v=logits.shape;parts=triton.cdiv(v,1024)
    assert raw_maximum.shape==(n,parts) and raw_maximum.is_contiguous()
    ids=torch.empty((n,parts),device=logits.device,dtype=torch.int64)
    maximum=torch.empty((n,parts),device=logits.device,dtype=torch.float64 if use_fp64 else torch.float32)
    out=torch.empty((n,),device=logits.device,dtype=torch.int64)
    _parts[(n,parts)](ids,maximum,raw_maximum,output_processed_logits,
        output_processed_logits.stride(0) if output_processed_logits is not None else 0,
        col,logits,logits.stride(0),idx,seeds,pos,temp,active,override,
        active.stride(0) if active is not None else 0,v,parts,apply_temperature,
        use_fp64,col is not None and col.dim()>0,num_warps=4)
    if active is None:
        dense_finish[(n,)](maximum,ids,out,n,parts,triton.next_power_of_2(parts),num_warps=4)
    else:
        _sample_finish[(n,)](maximum,ids,out,active,override,active.stride(0),parts,
            triton.next_power_of_2(parts),num_warps=4)
    return out
