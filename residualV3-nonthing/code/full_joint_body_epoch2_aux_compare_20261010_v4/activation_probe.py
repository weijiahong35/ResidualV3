"""Same-state full1536 gradient gates, activation policies, reusable CUDA pool."""
import gc
import hashlib
import math
import os
from pathlib import Path
import statistics
import time
import torch
from runtime.utils import dump,now


def weight_hashes(opt):
    result={}
    for name,param,_ in opt.pairs:
        tensor=param.detach().cpu().contiguous().view(torch.uint8)
        result[name]=hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    return result


def reserve_workspace(path,target_gib):
    torch.cuda.synchronize();wanted=max(0,int(target_gib*2**30)-torch.cuda.memory_reserved())
    if wanted:
        free,_=torch.cuda.mem_get_info();assert free>wanted+2**30,'Not enough free memory for owned allocator pool'
        # Release the Tensor, retain its reusable CUDA allocator storage.
        # Subsequent forward/backward allocations can split/reuse this pool.
        arena=torch.empty(wanted,dtype=torch.uint8,device='cuda');del arena
    torch.cuda.synchronize()
    result=dict(target_reserved_gib=target_gib,allocated_gib=torch.cuda.memory_allocated()/2**30,
                reserved_gib=torch.cuda.memory_reserved()/2**30,free_gib=torch.cuda.mem_get_info()[0]/2**30,
                reusable_cuda_allocator_pool=True,live_dummy_tensor=False,utc=now())
    dump(path,result);print(now(),'CUDA_ALLOCATOR_POOL',result,flush=True)


def compare_gradients(left,right):
    totals={prefix:dict(norm=0.,diff=0.,equal=True,max_abs=0.) for prefix in ['body','head']}
    params={}
    assert left.keys()==right.keys()
    for name,a in left.items():
        b=right[name];equal=torch.equal(a,b);norm=diff=0.;maximum=0.
        aa=a.reshape(-1);bb=b.reshape(-1)
        for start in range(0,a.numel(),2**20):
            x=aa[start:start+2**20];y=bb[start:start+2**20]
            norm+=float((x*x).sum())
            if not equal:
                d=x-y;diff+=float((d*d).sum());maximum=max(maximum,float(d.abs().max()))
        prefix='head' if name.startswith('head.') else 'body';t=totals[prefix]
        t['norm']+=norm;t['diff']+=diff;t['equal']=t['equal'] and equal;t['max_abs']=max(t['max_abs'],maximum)
        params[name]=dict(bitwise_equal=equal,relative_l2=math.sqrt(diff/max(norm,1e-40)),max_absolute_error=maximum)
    return dict(groups={k:dict(bitwise_equal=v['equal'],relative_l2=math.sqrt(v['diff']/max(v['norm'],1e-40)),max_absolute_error=v['max_abs']) for k,v in totals.items()},per_parameter=params)


def choose(out,p,pair,target,basis,pack,seed,opt):
    from joint_runner import update
    cases=[('checkpoint5',[0,1,2,3,4],True),('cache_all',[],True)]
    if pair.head is not None:cases.append(('cache_all_and_output_rows',[],False))
    cases.extend([('checkpoint1',[4],True),('checkpoint2',[3,4],True)])
    reference=None;attempts=[];candidates=[]
    for name,layers,recompute in cases:
        # Partial checkpoint cases are fallback when full activation retention
        # is infeasible. Do not spend extra timing when both full modes passed.
        if name.startswith('checkpoint') and name!='checkpoint5' and any(c['name'].startswith('cache_all') for c in candidates):continue
        os.environ['DRAFT_CHECKPOINT_LAYERS']=','.join(map(str,layers));os.environ['RECOMPUTE_OUTPUT_ROWS']='1' if recompute else '0'
        capture={}
        opt.probe_capture=lambda state:capture.update({n:m.grad.detach().cpu().clone() for n,_,m in state.pairs if m.grad is not None})
        try:
            first=update(pair,target,basis,pack,seed,1536,opt,check=True)
            opt.probe_capture=None
            if reference is None:reference=dict(loss=first['loss'],grads=capture);comparison=dict(status='reference')
            else:
                comparison=compare_gradients(reference['grads'],capture);comparison['loss_absolute_error']=abs(first['loss']-reference['loss'])
                dump(out/(name+'_gradient_comparison.json'),comparison)
                assert comparison['loss_absolute_error']<2e-5
                assert max(v['relative_l2'] for v in comparison['groups'].values())<.002
                comparison['status']='passed'
            del capture;gc.collect()
            # One additional warmup, followed by three fixed-state timings.
            update(pair,target,basis,pack,seed,1536,opt)
            measured=[update(pair,target,basis,pack,seed,1536,opt) for _ in range(3)]
            fields=['body_forward_wall','projection_wall','head_forward_loss_wall','head_backward_wall','body_backward_wall','clip_wall','optimizer_wall']
            means={k:statistics.mean(v[k] for v in measured) for k in fields}
            value=dict(name=name,checkpoint_layers=layers,recompute_output_rows=recompute,status='passed',gradient_gate=comparison,
                       means=means,core_seconds=sum(means.values()),peak_allocated_gib=max(v['peak_allocated_bytes'] for v in measured)/2**30,
                       peak_reserved_gib=max(v['reserved_bytes'] for v in measured)/2**30,anchors=1536,head_chunk=1536)
            attempts.append(value);candidates.append(value)
            print(now(),'ACTIVATION_POLICY_PASS',name,value['core_seconds'],value['peak_allocated_gib'],flush=True)
        except torch.OutOfMemoryError as exc:
            opt.probe_capture=None;attempts.append(dict(name=name,status='oom',checkpoint_layers=layers,recompute_output_rows=recompute,error=str(exc)))
            print(now(),'ACTIVATION_POLICY_OOM',name,flush=True);del capture;opt.zero_grad();gc.collect();torch.cuda.empty_cache()
        finally:
            dump(out/'activation_probe_progress.json',dict(attempts=attempts,utc=now()))
    assert candidates
    selected=min(candidates,key=lambda v:v['core_seconds']);os.environ['DRAFT_CHECKPOINT_LAYERS']=','.join(map(str,selected['checkpoint_layers']));os.environ['RECOMPUTE_OUTPUT_ROWS']='1' if selected['recompute_output_rows'] else '0'
    dump(out/'activation_probe.json',dict(status='complete',selected=selected,attempts=attempts,scope='same online pack; full gradients before clip; LR0 optimizer executes but weights fixed',utc=now()))
    del reference;opt.probe_capture=None;opt.zero_grad();gc.collect()
    return selected
