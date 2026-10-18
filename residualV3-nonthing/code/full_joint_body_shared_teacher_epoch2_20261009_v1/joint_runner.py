"""Bounded joint updates with stage timing, live feature gradients and OOM evidence."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback
import torch
from safetensors.torch import save_file
from runtime.utils import dump,now
from repair.model import rollout
from parallel_tf import teacher_forced_logits
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks
from joint_model import require_backbone_gradients,cpu_tree
from host_pipeline import HostPipeline,upload
from dp_teacher import install as install_provider
import runtime.vllm_hidden as hidden
from losses import dpace_loss,output_loss

def timed(fn):
    torch.cuda.synchronize();a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    wall=time.perf_counter();a.record();value=fn();b.record();b.synchronize()
    return value,dict(wall=time.perf_counter()-wall,cuda=a.elapsed_time(b)/1000)

def backbone(body,pack,seed,cap=1536):
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(seed)
    eligible=pack['eligible'];anchors=torch.sort(eligible[torch.randperm(len(eligible),device='cuda')[:cap]]).values
    assert 0 < len(anchors) <= cap
    ids,docs,positions=pack['ids'],pack['docs'],pack['positions']
    index=anchors[:,None]+torch.arange(16,device='cuda')[None]
    assert torch.equal(docs[0,index],docs[0,anchors,None].expand(-1,16))
    query=torch.full((1,len(anchors)*16),body.mask_token_id,device='cuda',dtype=torch.long);query[:,::16]=ids[:,anchors]
    noise=body.embed_tokens(query)
    context=checkpoint(lambda x:body.hidden_norm(body.fc(x)),pack['aux'],use_reentrant=False,preserve_rng_state=False)
    mask=body._create_attention_mask(docs,65536,anchors,torch.device('cuda'),sliding_window=body.sliding_window,sliding_window_non_causal=body.sliding_window_non_causal)
    synthetic=get_base_indices_for_anchored_blocks(positions[0,anchors],16)
    rotary=body.rotary_emb(context,torch.cat((positions,synthetic[None]),1))
    checkpoint_layers={int(v) for v in os.environ.get('DRAFT_CHECKPOINT_LAYERS','0,1,2,3,4').split(',') if v}
    for layer_index,layer in enumerate(body.layers):
        def forward(x,ctx,layer=layer):
            return layer(hidden_states=x,target_hidden=ctx,attention_mask=mask,position_ids=None,use_cache=False,position_embeddings=rotary)
        noise=checkpoint(forward,noise,context,use_reentrant=False,preserve_rng_state=False) if layer_index in checkpoint_layers else forward(noise,context)
    h=body.norm(noise).reshape(len(anchors),16,4096)[:,1:].contiguous()
    return h,pack['last'][index[:,:-1]].contiguous(),ids[0,index[:,1:]].contiguous(),ids[0,anchors],anchors

def reset(pair,opt,initial):
    opt.zero_grad()
    for optimizer in opt.optimizers:optimizer.state.clear()
    with torch.no_grad():
        for name,param,master in opt.pairs:master.copy_(initial[name]);param.copy_(master)
    gc.collect();torch.cuda.empty_cache()

def head_loss(pair,target,basis,leaf,th,answers,anchor,start,end,mode):
    h=leaf[start:end]
    def projection():
        raw=h@target.lm_head.weight.T
        with torch.no_grad():gold=th[start:end]@target.lm_head.weight.T
        return raw,gold
    (raw,gold),projection_time=timed(projection)
    def forward():
        if pair.head is None:return dpace_loss(raw,gold)
        if os.environ.get('HEAD_REPAIR_IMPLEMENTATION')=='optimized':
            from optimized_repair import logits as repair_logits
            from combined_ce_tv import objective as repaired_objective
            logits=repair_logits(pair.head,raw,h,basis,anchor[start:end],answers[start:end],recompute_projection=os.environ.get('RECOMPUTE_OUTPUT_ROWS','1')=='1')
            return dpace_loss(raw,gold)+repaired_objective(logits,gold)
        if mode=='sequential':logits=rollout(pair.head,raw,h,target.lm_head.weight,basis,anchor[start:end],lambda z,j:answers[start:end,j]).logits
        else:logits=teacher_forced_logits(pair.head,raw,h,basis,anchor[start:end],answers[start:end])
        return dpace_loss(raw,gold)+dpace_loss(logits,gold)+output_loss(logits,gold)
    loss,head_time=timed(forward)
    return loss,projection_time,head_time

def update(pair,target,basis,pack,seed,chunk,opt,check=False,step=True):
    if check and pair.head is not None and os.environ.get('HEAD_REPAIR_IMPLEMENTATION')=='optimized' and not os.environ.get('HEAD_PREVALIDATED_GATE'):
        from full_online_gate import run as run_gate
        run_gate(Path(os.environ['HEAD_FULL_GATE_PATH']),pair,target,basis,pack,seed)
    opt.zero_grad();torch.cuda.reset_peak_memory_stats()
    (h,th,answers,anchor,positions),body_time=timed(lambda:backbone(pair.body,pack,seed))
    assert h.requires_grad
    leaf=h.detach().requires_grad_(True)
    sums=dict(projection_wall=0.,projection_cuda=0.,head_forward_loss_wall=0.,head_forward_loss_cuda=0.,head_backward_wall=0.,head_backward_cuda=0.)
    loss_total=0.;count=len(h)
    for start in range(0,count,chunk):
        end=min(start+chunk,count)
        loss,projection,forward=head_loss(pair,target,basis,leaf,th,answers,anchor,start,end,'sequential')
        assert torch.isfinite(loss)
        _,backward=timed(lambda:(loss*((end-start)/count)).backward())
        opt.accumulate();loss_total+=float(loss.detach())*(end-start)/count
        for label,measurement in [('projection',projection),('head_forward_loss',forward),('head_backward',backward)]:
            for unit in ['wall','cuda']:sums[label+'_'+unit]+=measurement[unit]
        del loss
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    _,body_backward=timed(lambda:h.backward(leaf.grad))
    opt.accumulate()
    gradients=opt.grad_summary() if check else None
    if check:require_backbone_gradients(gradients,'dflash_body' if pair.head is None else 'dflash_repair_joint')
    if getattr(opt,'probe_capture',None) is not None:opt.probe_capture(opt)
    norm,clip=timed(lambda:torch.nn.utils.clip_grad_norm_(opt.master_parameters,1.,error_if_nonfinite=True))
    optimizer=dict(wall=0.,cuda=0.)
    if step:_,optimizer=timed(lambda:opt.step(0. if os.environ.get('TIMING_ONLY_NO_PARAMETER_UPDATE')=='1' else getattr(opt,'lr_scale',1.)))
    result=dict(loss=loss_total,grad_norm=float(norm),body_forward_wall=body_time['wall'],body_forward_cuda=body_time['cuda'],body_backward_wall=body_backward['wall'],body_backward_cuda=body_backward['cuda'],clip_wall=clip['wall'],optimizer_wall=optimizer['wall'],optimizer_cuda=optimizer['cuda'],gradients=gradients,head_chunk_anchors=chunk,anchors=count,anchor_sha256=hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),**sums)
    opt.zero_grad();return result

def save(out,pair,opt,step,p):
    out.mkdir(exist_ok=True)
    tensors={name:param.detach().cpu().clone().contiguous() for name,param,_ in opt.pairs}
    save_file(tensors,str(out/'joint_pilot_weights.safetensors'))
    if not p.get('save_optimizer_state',True):return
    state=dict(step=step,trainable=tensors,masters={name:master.detach().cpu() for name,_,master in opt.pairs},optimizers=[cpu_tree(o.state_dict()) for o in opt.optimizers],cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),protocol=p)
    torch.save(state,out/'joint_pilot_full_state.pt');del state,tensors

def one_pack(root,p,service,source,packs,target):
    e=json.loads((service/'endpoint.json').read_text());os.environ['THROUGHPUT_REQUESTS']='128'
    install_provider(root/'hidden_probe_requests.jsonl',e)
    provider=hidden.OnlineHiddenProvider(e['endpoint'],e['root'],p['target_model'],prefetch=16)
    pipe=HostPipeline(provider,source,packs[:1])
    try:
        indices,rows,host,_=pipe.get();device,timing=upload(host,target);del host;pipe.release();return device
    finally:
        pipe.close();provider.close();assert provider.requests==provider.deleted
        dump(root/'probe_hidden_cleanup.json',dict(requests=provider.requests,deleted=provider.deleted))

def feasibility(out,p,service,source,packs,target,pair,opt,basis,initial,budget):
    out.mkdir();torch.cuda.set_per_process_memory_fraction(budget*2**30/torch.cuda.get_device_properties(0).total_memory)
    pack=one_pack(out,p,service,source,packs,target);attempts=[];selected=None
    for chunk in [1536,512,256,128,64]:
        reset(pair,opt,initial)
        try:
            value=update(pair,target,basis,pack,p['experiment_seed']+200299,chunk,opt,check=True)
            attempts.append(dict(status='passed',chunk=chunk,**value));selected=chunk
            print(now(),'JOINT_MEMORY_PASS',chunk,value['peak_allocated_bytes']/2**30,flush=True)
        except torch.OutOfMemoryError as exc:
            attempts.append(dict(status='oom',chunk=chunk,error=str(exc),traceback=traceback.format_exc()))
            print(now(),'JOINT_MEMORY_OOM',chunk,flush=True)
        dump(out/'memory_probe.json',dict(attempts=attempts,selected=selected,budget_gib=budget,utc=now()))
        if selected is not None:break
    del pack;reset(pair,opt,initial)
    assert selected is not None,'No feasible joint chunk preserving65536tokens/1536anchors'
    return selected

def benchmark(out,p,service,source,packs,target,pair,opt,basis,initial,chunk):
    out.mkdir();reset(pair,opt,initial)
    e=json.loads((service/'endpoint.json').read_text());os.environ['THROUGHPUT_REQUESTS']='128'
    install_provider(out/'hidden_requests.jsonl',e)
    provider=hidden.OnlineHiddenProvider(e['endpoint'],e['root'],p['target_model'],prefetch=16)
    pipe=HostPipeline(provider,source,packs);records=[]
    try:
        for i in range(len(packs)):
            begin=time.perf_counter();indices,rows,host,host_time=pipe.get();wait=time.perf_counter()-begin
            pack,transfer=upload(host,target);del host;pipe.release()
            values=update(pair,target,basis,pack,p['experiment_seed']+200299+i,chunk,opt,check=i==0)
            record=dict(pack=i+1,measured=i>=3,total_update_seconds=time.perf_counter()-begin,hidden_visible_wait_seconds=wait,input_tokens=sum(len(row['input_ids']) for row in rows),examples=len(rows),**transfer,**host_time,**values,utc=now())
            records.append(record)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            dump(out.parent/'status.json',dict(status='joint_benchmarking',layout=out.name,latest=record,worker_pid=os.getpid(),utc=now()))
            print(now(),'JOINT_UPDATE',out.name,i+1,'/',len(packs),'seconds',round(record['total_update_seconds'],3),'wait',round(wait,3),'body_fwd',round(values['body_forward_wall'],3),'head_fwd',round(values['head_forward_loss_wall'],3),'head_bwd',round(values['head_backward_wall'],3),'body_bwd',round(values['body_backward_wall'],3),'opt',round(values['optimizer_wall'],3),flush=True)
            del pack
        measured=[v for v in records if v['measured']]
        fields=['total_update_seconds','hidden_visible_wait_seconds','h2d_wall_seconds','h2d_cuda_seconds','body_forward_wall','body_forward_cuda','projection_wall','projection_cuda','head_forward_loss_wall','head_forward_loss_cuda','head_backward_wall','head_backward_cuda','body_backward_wall','body_backward_cuda','optimizer_wall','optimizer_cuda','clip_wall']
        summary=dict(status='complete',measured_updates=len(measured),means={k:statistics.mean(v[k] for v in measured) for k in fields},head_chunk=chunk,anchors_per_update=1536,input_token_budget=65536,trainable='DFlash body and no-soft3 head; frozen target',peak_allocated_bytes=max(v['peak_allocated_bytes'] for v in records),records_count=len(records),utc=now())
        dump(out/'summary.json',summary)
        save(out,pair,opt,len(packs),p)
    finally:
        pipe.close();provider.close();assert provider.requests==provider.deleted
        dump(out/'hidden_cleanup.json',dict(requests=provider.requests,deleted=provider.deleted,no_persistent_hidden=True))
    return summary

def run(root,p,target,pair,opt,basis,initial):
    source=[json.loads(x) for x in Path(p['data_source']).read_text().splitlines()]
    import importlib.util
    path=Path(__file__).resolve().parent.parent/'teacher_count_head_parallel_pilot_20261007_v1/train.py'
    spec=importlib.util.spec_from_file_location('timing_pack_reference',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    packs=module.packs_for_epoch(source,p['experiment_seed'],2,65536)[:12]
    dump(root/'packs.json',dict(indices=packs,anchor_seeds=[p['experiment_seed']+200299+i for i in range(len(packs))],same_packs_both_arms=True))
    first_service=root/'teacher_dp2'
    selected=feasibility(root/'joint_memory',p,first_service,source,packs,target,pair,opt,basis,initial,p['learner_memory_budget_gib'])
    first=benchmark(root/'teacher2_learner1',p,first_service,source,packs,target,pair,opt,basis,initial,selected)
    dump(root/'switch_to_dp1.json',dict(reason='DP2 joint pilot and full state saved; request DP1 on same first teacher GPU',utc=now()))
    reset(pair,opt,initial)
    second_service=root/'teacher_dp1';deadline=time.monotonic()+1200
    while not (second_service/'endpoint.json').exists():
        assert time.monotonic()<deadline,'DP1 teacher startup timeout';time.sleep(2)
    second=benchmark(root/'teacher1_learner1',p,second_service,source,packs,target,pair,opt,basis,initial,selected)
    records=[]
    for name in ['teacher2_learner1','teacher1_learner1']:
        records.append([json.loads(x) for x in (root/name/'metrics.jsonl').read_text().splitlines()])
    matched=all(a['anchor_sha256']==b['anchor_sha256'] and a['input_tokens']==b['input_tokens'] and a['examples']==b['examples'] for a,b in zip(*records))
    assert matched,'Paired data/anchors differ'
    dump(root/'summary.json',dict(status='complete',teacher2_learner1=first,teacher1_learner1=second,paired_loss_chunk=selected,paired_data_anchors_match=matched,change_only='teacher DP size: TP1/DP2 versus TP1/DP1; fixed learner GPU, initialization, data, anchors, optimizer, objective and loss chunk',scope='24 joint pilot optimizer updates plus documented memory probes; not training-quality or native performance results',order=['teacher2_learner1','teacher1_learner1'],utc=now()))
