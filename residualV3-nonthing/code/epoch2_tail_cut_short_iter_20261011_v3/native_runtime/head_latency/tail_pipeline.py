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
    head._tail_repair_stream=torch.cuda.Stream(priority=-1)
    limit=int(os.environ.get('LOCAL_MAX_CONCURRENCY','0'))
    sizes=range(1,limit+1) if limit else (1,7,18,40)
    head._tail_events={n:([torch.cuda.Event() for _ in range(15)],[torch.cuda.Event() for _ in range(15)]) for n in sizes}
    head._tail_aggregate=torch.zeros(16,device='cuda',dtype=torch.int64)


def predict(head,hp,hard,qp,active,counts,*,known_active=False,previous=None):
    count=os.environ.get('TAIL_INSTRUMENT','0')=='1'
    if known_active and not count:
        from .structure_fusion import linear,up_silu
        linear_fn=lambda x,w,b,residual=None:linear(x,w,b,residual=residual)
        up_fn=up_silu
    else:
        linear_fn=lambda x,w,b,residual=None:k.linear(x,w,b,active,counts,count,residual=residual)
        up_fn=lambda x,module:k.up_silu(x,module,active,counts,count)
    parts={'hidden':hp,'hard_previous':hard,'q':qp}
    if previous is not None:
        from .input_assembly import assemble
        assembled=assemble(head,hp,qp,previous)
    else:assembled=torch.cat([parts[n] for n in head.branch_names],-1)
    fused_input=head.fusion[0](assembled)
    x=linear_fn(fused_input,head.fusion[1].weight,head.fusion[1].bias)
    for block in head.blocks:
        h=up_fn(block.norm(x),block.up)
        x=linear_fn(h,block.down.weight,block.down.bias,residual=x)
    latent=head.output_norm(x)
    output=linear_fn(latent,head._packed_output_weight,head._packed_output_bias)
    if os.environ.get('TAIL_FUSED_FINAL','0')=='1':
        from .output_epilogue import finish
        direction,dose=finish(output,head.basis_rank,head.eps,head.dose_max)
        return direction,dose,latent
    direction=normalized_rows(output[:,:head.basis_rank],eps=head.eps)
    dose=(head.dose_max*torch.sigmoid(output[:,head.basis_rank].float())).to(latent.dtype)
    return direction,dose,latent


def rollout(head,raw,hidden,weight,basis,anchor,choose,*,capture=True,mode=None):
    mode=mode or os.environ.get('TAIL_MODE','FULL15')
    if mode in ('PIPELINE','EXPLORATORY') and os.environ.get('TAIL_HIGH_PRIORITY_MAIN','0')=='1':
        parent=torch.cuda.current_stream();repair=head._tail_repair_stream
        repair.wait_stream(parent)
        with torch.cuda.stream(repair):
            result=_rollout(head,raw,hidden,weight,basis,anchor,choose,capture=capture,mode=mode)
        parent.wait_stream(repair)
        return result
    return _rollout(head,raw,hidden,weight,basis,anchor,choose,capture=capture,mode=mode)


def _rollout(head,raw,hidden,weight,basis,anchor,choose,*,capture=True,mode=None):
    from .structure_fusion import rollout as full
    mode=mode or os.environ.get('TAIL_MODE','FULL15')
    if mode=='FULL15':return full(head,raw,hidden,weight,basis,anchor,choose,capture=capture)
    assert mode in MODES and not head.use_soft and not head.training
    n,length,vocab=raw.shape
    config=head._tail_config
    hp,qp,hn,ids,_,_=head.prepare(raw,hidden,basis)
    buffer=raw.transpose(0,1).clone(memory_format=torch.contiguous_format)
    main=torch.cuda.current_stream();side=head._tail_stream
    pipeline=mode in ('PIPELINE','EXPLORATORY')
    reuse_sample_max=pipeline and os.environ.get('TAIL_REUSE_SAMPLE_MAX','0')=='1'
    head._tail_sampler_max=torch.empty((length,n,(vocab+1023)//1024),device=raw.device,dtype=torch.float32) if reuse_sample_max else None
    if pipeline:side.wait_stream(main)
    if os.environ.get('TAIL_SPARSE_Q45','1')=='1':
        from .tail_sparse_q45 import prepare as prepare_sparse
        if pipeline and os.environ.get('TAIL_ASYNC_CACHE','1')=='1':
            # Cache reads ORIGINAL raw, never the buffer undergoing repair.
            # It has no causal predecessor dependency and can overlap the
            # guaranteed-active repairs before the first stop decision.
            with torch.cuda.stream(side):
                head._tail_sparse_cache=prepare_sparse(raw,ids)
        else:
            used=length-2
            cached,ci=prepare_sparse(buffer[:used].reshape(1,used*n,vocab),ids.transpose(0,1)[:used].reshape(1,used*n,-1))
            head._tail_sparse_cache=(cached.reshape(used,n,256),ci.reshape(used,n,-1))
    active=torch.ones((n,length+2),device=raw.device,dtype=torch.int32)
    if mode=='BODY_ONLY':active.zero_()
    survival=torch.ones((n,length+1),device=raw.device,dtype=torch.float32)
    patience=torch.zeros((n,length+1),device=raw.device,dtype=torch.int32)
    maxq=torch.zeros((n,length),device=raw.device,dtype=torch.float32)
    scores_out=torch.zeros((n,length,4 if head._tail_branch=='BOTH' else 2),device=raw.device,dtype=torch.float32)
    counts=torch.zeros(3,device=raw.device,dtype=torch.int32)
    previous=anchor;tokens=[];history=[];scores=[];signal_latents=[]
    ready,done=head._tail_events[n]
    # Config and sampling temperatures are installed by native run_eager before rollout.
    temp=head._tail_temperatures;mappings=head._tail_mapping
    force=int(os.environ.get('TAIL_FORCE_CUTOFF','0'))
    earliest=config['minimum_repair']+config['patience']
    early_batch=earliest-1 if os.environ.get('TAIL_BATCH_EARLY','1')=='1' and (force==0 or force>=earliest) else 0
    if early_batch:
        early_batch=min(early_batch,int(os.environ.get('TAIL_EARLY_BATCH_SIZE',str(early_batch))))
        assert early_batch>0
    for j in range(length):
        # Prediction of j-2 must be ready before deciding whether to repair j.
        if pipeline and j>=max(2,force if force>0 else config['minimum_repair']+config['patience']):main.wait_event(done[j-2])
        fuse_input=os.environ.get('TAIL_FUSED_INPUT','0')=='1'
        hard=None if fuse_input else head._previous_tables[n][previous.long()]
        guaranteed=min(force,earliest) if force>0 else earliest
        known_active=(mode!='BODY_ONLY' and j<guaranteed and
                      os.environ.get('TAIL_DENSE_GUARANTEED','0')=='1' and
                      os.environ.get('TAIL_INSTRUMENT','0')!='1')
        coeff,dose,latent=predict(head,hp[:,j],hard,qp[:,j],active[:,j],counts,known_active=known_active,
                                previous=previous if fuse_input else None)
        signal_latents.append(latent) # keep main-stream storage alive through side-stream reads
        z=buffer[j]
        if known_active:
            from .structure_fusion import gather_correction
            gather_correction(raw[:,j],basis,coeff,dose,hn[:,j],ids[:,j],z)
        else:
            k.correct(raw[:,j],basis,coeff,dose,hn[:,j],ids[:,j],z,active[:,j],counts,
                      os.environ.get('TAIL_INSTRUMENT','0')=='1')
        if reuse_sample_max:
            sample_args=dict(active=active[:,j]) if not known_active and os.environ.get('TAIL_FAST_SAMPLE','1')=='1' else {}
            if j<length-2:sample_args['entropy_max']=head._tail_sampler_max[j]
            token=choose(z.detach(),j,**sample_args).long()
        if pipeline and j<length-2 and j>=early_batch-1:
            ready[j].record(main)
            with torch.cuda.stream(side):
                side.wait_event(ready[j])
                from .trained_tail import signal,signal_prefix
                if early_batch and j==early_batch-1:
                    signal_prefix(head,buffer[:early_batch],signal_latents,active,survival,patience,maxq,scores_out,temp,mappings,force)
                else:signal(head,z,latent,active,survival,patience,maxq,scores_out,temp,mappings,j,force)
                done[j].record(side)
        elif not pipeline and j<length-2:
            from .trained_tail import signal
            signal(head,z,latent,active,survival,patience,maxq,scores_out,temp,mappings,j,force)
        if not reuse_sample_max:
            if os.environ.get('TAIL_FAST_SAMPLE','1')=='1' and not known_active:
                token=choose(z.detach(),j,active=active[:,j]).long()
            else:
                token=choose(z.detach(),j).long()
        tokens.append(token)
        if capture:scores.append(z);history.append(previous)
        previous=token
    if pipeline:main.wait_event(done[length-3])
    cuts=k.aggregate(active,head._tail_aggregate,length)
    from .tail_graph_state import record_tail_state
    record_tail_state(head,n,(active,survival,maxq,counts,scores_out,cuts),capturing=torch.cuda.is_current_stream_capturing())
    return Rollout(torch.stack(scores,1) if capture else None,torch.stack(tokens,1),
                   torch.stack(history,1) if capture else None,[],())
