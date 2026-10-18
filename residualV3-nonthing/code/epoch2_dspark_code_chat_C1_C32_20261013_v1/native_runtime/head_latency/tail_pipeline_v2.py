"""One-position delayed prediction overlaps next causal repair on a second stream."""
import json,os
from pathlib import Path
import torch
from runtime.components import Rollout,normalized_rows
from . import tail_kernels as k

MODES=('FULL15','BODY_ONLY','SEQUENTIAL','PIPELINE','EXPLORATORY')


def prepare(head,config):
    head._tail_config=config
    head._tail_graph_stats={}
    head._tail_stream=torch.cuda.Stream()
    limit=int(os.environ.get('LOCAL_MAX_CONCURRENCY','0'))
    sizes=range(1,limit+1) if limit else (1,7,18,40)
    head._tail_events={n:([torch.cuda.Event() for _ in range(15)],[torch.cuda.Event() for _ in range(15)]) for n in sizes}
    head._tail_aggregate=torch.zeros(16,device='cuda',dtype=torch.int64)


def predict(head,hp,hard,qp,active,counts):
    count=os.environ.get('TAIL_INSTRUMENT','0')=='1'
    parts={'hidden':hp,'hard_previous':hard,'q':qp}
    fused_input=head.fusion[0](torch.cat([parts[n] for n in head.branch_names],-1))
    x=k.linear(fused_input,head.fusion[1].weight,head.fusion[1].bias,active,counts,count)
    for block in head.blocks:
        h=k.up_silu(block.norm(x),block.up,active,counts,count)
        x=k.linear(h,block.down.weight,block.down.bias,active,counts,count,residual=x)
    latent=head.output_norm(x)
    output=k.linear(latent,head._packed_output_weight,head._packed_output_bias,active,counts,count)
    direction=normalized_rows(output[:,:head.basis_rank],eps=head.eps)
    dose=(head.dose_max*torch.sigmoid(output[:,head.basis_rank].float())).to(latent.dtype)
    return direction,dose,latent


def rollout(head,raw,hidden,weight,basis,anchor,choose,*,capture=True,mode=None):
    from .structure_fusion import rollout as full
    mode=mode or os.environ.get('TAIL_MODE','FULL15')
    if mode=='FULL15':return full(head,raw,hidden,weight,basis,anchor,choose,capture=capture)
    assert mode in MODES and not head.use_soft and not head.training
    n,length,vocab=raw.shape
    config=head._tail_config
    hp,qp,hn,ids,_,_=head.prepare(raw,hidden,basis)
    buffer=raw.transpose(0,1).clone(memory_format=torch.contiguous_format)
    active=torch.ones((n,length+2),device=raw.device,dtype=torch.int32)
    if mode=='BODY_ONLY':active.zero_()
    survival=torch.ones((n,length+1),device=raw.device,dtype=torch.float32)
    patience=torch.zeros((n,length+1),device=raw.device,dtype=torch.int32)
    maxq=torch.empty((n,length),device=raw.device,dtype=torch.float32)
    scores_out=torch.empty((n,length,4 if head._tail_branch=='BOTH' else 2),device=raw.device,dtype=torch.float32)
    counts=torch.zeros(3,device=raw.device,dtype=torch.int32)
    previous=anchor;tokens=[];history=[];scores=[];signal_latents=[]
    main=torch.cuda.current_stream();side=head._tail_stream
    ready,done=head._tail_events[n]
    pipeline=mode in ('PIPELINE','EXPLORATORY')
    if pipeline:side.wait_stream(main)
    # Config and sampling temperatures are installed by native run_eager before rollout.
    temp=head._tail_temperatures;mappings=head._tail_mapping
    force=int(os.environ.get('TAIL_FORCE_CUTOFF','0'))
    for j in range(length):
        # Prediction of j-2 must be ready before deciding whether to repair j.
        if pipeline and j>=2:main.wait_event(done[j-2])
        hard=head._previous_tables[n][previous.long()]
        coeff,dose,latent=predict(head,hp[:,j],hard,qp[:,j],active[:,j],counts)
        signal_latents.append(latent) # keep main-stream storage alive through side-stream reads
        z=buffer[j]
        k.correct(raw[:,j],basis,coeff,dose,hn[:,j],ids[:,j],z,active[:,j],counts,
                  os.environ.get('TAIL_INSTRUMENT','0')=='1')
        if pipeline:
            ready[j].record(main)
            with torch.cuda.stream(side):
                side.wait_event(ready[j])
                from .trained_tail_v2 import signal
                signal(head,z,latent,active,survival,patience,maxq,scores_out,temp,mappings,j,force)
                done[j].record(side)
        else:
            from .trained_tail_v2 import signal
            signal(head,z,latent,active,survival,patience,maxq,scores_out,temp,mappings,j,force)
        if os.environ.get('TAIL_FAST_SAMPLE','1')=='1':
            token=choose(z.detach(),j,active=active[:,j]).long()
        else:
            token=choose(z.detach(),j).long()
        tokens.append(token)
        if capture:scores.append(z);history.append(previous)
        previous=token
    if pipeline:main.wait_event(done[length-1])
    cuts=k.aggregate(active,head._tail_aggregate,length)
    from .tail_graph_state import record_tail_state
    record_tail_state(head,n,(active,survival,maxq,counts,scores_out,cuts),capturing=torch.cuda.is_current_stream_capturing())
    return Rollout(torch.stack(scores,1) if capture else None,torch.stack(tokens,1),
                   torch.stack(history,1) if capture else None,[],())
