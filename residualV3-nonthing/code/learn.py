"""Online nonthinking mismatch basis, bounded tuning, then scratch no-soft3 OUTPUT."""
import argparse
import gc
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path

FAMILY=Path(__file__).resolve().parents[1]
RUNTIME=FAMILY/'code/runtime_bundle'
sys.path.insert(0,str(RUNTIME))
import torch
from safetensors.torch import load_file, save_file
from repair.model import DualFeedbackRepair, rollout, output_loss
from repair.run import digest
from repair.recovery_state import save_checkpoint
from runtime.live_trainer import load_models, online_states, schedule_scale
from runtime.live_run import order_for_epoch
from runtime.master_optimizer import MasterAdamW
from runtime.precision import build_token_basis
from runtime.utils import dump,sha,now


def basis_fit(target,draft,train,p,out):
    destination=out/'basis';destination.mkdir(exist_ok=True)
    if (destination/'summary.json').exists():
        summary=json.loads((destination/'summary.json').read_text())
        assert summary['status']=='complete'
        assert sha(summary['path'])==summary['sha256']
        return Path(summary['path'])
    provider=target.hidden_provider
    recipe=p['basis_fit'];examples=train[:recipe['examples']]
    covariance=torch.zeros(4096,4096,device='cuda',dtype=torch.float32)
    projection=target.lm_head.weight.float()
    provider.begin_epoch(examples)
    vectors=0;states=0
    with torch.no_grad():
        for index,example in enumerate(examples):
            torch.manual_seed(recipe['seed']+index)
            h,th,answers,anchor=online_states(target,draft,example,recipe['anchors'],torch.device('cuda:0'))
            assert h.shape[1]==15 and th.shape==h.shape
            for start in range(0,len(h),recipe['chunk_anchors']):
                sl=slice(start,start+recipe['chunk_anchors'])
                raw=h[sl]@target.lm_head.weight.T;gold=th[sl]@target.lm_head.weight.T
                q=raw.float().softmax(-1);teacher=gold.float().softmax(-1)
                gradient=(teacher-q)@projection
                norm=gradient.norm(dim=-1,keepdim=True)
                directions=(gradient/norm.clamp_min(1e-12))[norm.squeeze(-1)>1e-12]
                covariance.add_(directions.T@directions)
                states+=h[sl].shape[0]*15;vectors+=len(directions)
                del raw,gold,q,teacher,gradient,norm,directions
            row=dict(index=index,anchors=len(h),states=states,vectors=vectors,anchor_sha256=digest(anchor),answer_sha256=digest(answers),utc=now())
            with (destination/'collection_metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            dump(destination/'status.json',dict(status='collecting',examples=index+1,total=len(examples),latest=row))
            print(now(),'BASIS',index+1,'/',len(examples),'anchors',len(h),flush=True)
            del h,th,answers,anchor
    assert vectors>=512
    covariance=(covariance/vectors).cpu()
    del projection
    gc.collect();torch.cuda.empty_cache()
    dump(destination/'status.json',dict(status='eigendecomposition',vectors=vectors,states=states))
    values,columns=torch.linalg.eigh(covariance)
    basis=columns[:,-512:].T.flip(0).contiguous()
    gram=float((basis@basis.T-torch.eye(512)).abs().max())
    assert gram<1e-4
    path=destination/'repair_basis_512.safetensors';save_file({'repair_basis':basis},str(path))
    dump(destination/'summary.json',dict(status='complete',path=str(path),sha256=sha(path),examples=len(examples),vectors=vectors,states=states,rank=512,recipe=recipe,gram_max_abs=gram,
        spectral_energy_512=float(values[-512:].sum()/values.clamp_min(0).sum()),training_only=True,no_hidden_or_logits_persisted=True,utc=now()))
    dump(destination/'status.json',dict(status='complete',utc=now()))
    return path


def head_backward(head,output_basis,target,h,th,answers,anchor,chunk,opt=None):
    count=len(h);loss_sum=0.
    for start in range(0,count,chunk):
        sl=slice(start,start+chunk)
        with torch.no_grad():
            raw=h[sl]@target.lm_head.weight.T;gold=th[sl]@target.lm_head.weight.T
        result=rollout(head,raw,h[sl],target.lm_head.weight,output_basis,anchor[sl],lambda z,j:answers[sl,j])
        assert torch.equal(result.predecessors,torch.cat((anchor[sl,None],answers[sl,:-1]),1))
        loss=output_loss(result.logits,gold)
        assert torch.isfinite(loss)
        scale=len(h[sl])/count
        (loss*scale).backward()
        if opt is not None:opt.accumulate()
        loss_sum+=float(loss.detach())*scale
        del raw,gold,result,loss
    return loss_sum


def tune(head,basis,target,draft,train,p,out):
    path=out/'training_tuning.json'
    if path.exists():return json.loads(path.read_text())
    provider=target.hidden_provider
    probe=sorted(train[:128],key=lambda row:len(row['input_ids']))
    examples=[probe[min(127,index*8)] for index in range(16)]
    provider.begin_epoch([examples[8]])
    torch.manual_seed(610009)
    h,th,answers,anchor=online_states(target,draft,examples[8],p['max_anchors'],torch.device('cuda:0'))
    head_rows=[]
    for chunk in p['head_chunk_candidates']:
        head.zero_grad(set_to_none=True)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.perf_counter()
        try:
            loss=head_backward(head,basis,target,h[:128],th[:128],answers[:128],anchor[:128],chunk)
            torch.cuda.synchronize()
            row=dict(chunk_anchors=chunk,seconds=time.perf_counter()-started,loss=loss,peak_allocated_bytes=torch.cuda.max_memory_allocated(),passed=True)
        except torch.OutOfMemoryError as exc:
            row=dict(chunk_anchors=chunk,passed=False,error=str(exc))
        head_rows.append(row);head.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        print(now(),'TUNE_HEAD',row,flush=True)
    passed=[row for row in head_rows if row['passed']]
    assert passed
    chunk=min(passed,key=lambda row:row['seconds'])['chunk_anchors']
    del h,th,answers,anchor
    prefetch_rows=[]
    for prefetch in p['hidden_prefetch_candidates']:
        provider.prefetch=prefetch;provider.begin_epoch(examples)
        torch.cuda.synchronize();started=time.perf_counter()
        for index,example in enumerate(examples):
            torch.manual_seed(610010+index)
            h,th,answers,anchor=online_states(target,draft,example,p['max_anchors'],torch.device('cuda:0'))
            head.zero_grad(set_to_none=True)
            head_backward(head,basis,target,h,th,answers,anchor,chunk)
            head.zero_grad(set_to_none=True)
            del h,th,answers,anchor
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        row=dict(prefetch=prefetch,examples=len(examples),seconds=seconds,examples_per_second=len(examples)/seconds)
        prefetch_rows.append(row);print(now(),'TUNE_PREFETCH',row,flush=True)
        dump(out/'tuning_progress.json',dict(status='tuning',head_rows=head_rows,prefetch_rows=prefetch_rows))
    winner=max(prefetch_rows,key=lambda row:row['examples_per_second'])
    result=dict(status='complete',head_chunk_anchors=chunk,hidden_prefetch=winner['prefetch'],head_rows=head_rows,prefetch_rows=prefetch_rows,
        optimizer_updates_during_tuning=0,max_anchors_unchanged=p['max_anchors'],scope='single bounded end-to-end pilot on 16 training prompts; throughput estimate, not native speed result',utc=now())
    dump(path,result);return result


def native_protocol(p,basis_path,checkpoint,out):
    p=dict(p);arm='N_no_soft_3layer_nonthinking'
    record=dict(path=str(checkpoint),sha256=sha(checkpoint),head_config_path=p['head_config_path'],head_config_sha256=sha(p['head_config_path']))
    p.update(basis_path=str(basis_path),basis_sha256=sha(basis_path),native_checkpoints={name:dict(record) for name in p['native_arms']},native_configs={name:{} for name in p['native_arms']},architecture_by_arm={name:p['architecture'] for name in p['native_arms']},
        training_completions={arm:sha(out/'training/summary.json')},training_summary=str(out/'training/summary.json'))
    p['asset_sha256']=dict(p['asset_sha256'],**{str(basis_path):sha(basis_path),str(checkpoint):sha(checkpoint)})
    dump(out/'native_protocol.json',p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();p=json.loads(args.protocol.read_text())
    for field in ['source_files','asset_sha256']:
        for path,expected in p[field].items():assert sha(path)==expected,path
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(p['training_memory_fraction'],0)
    train=[json.loads(line) for line in Path(p['data_source']).read_text().splitlines()]
    assert len(train)==20000
    target,draft=load_models(torch.device('cuda:0'),f"http://127.0.0.1:{p['port']}/v1",p['transfer_root'],p['initial_hidden_prefetch'])
    provider=target.hidden_provider
    total=20000*3;steps=0;head=None;opt=None;stop_requested=[]
    checkpoint_dir=out/'training/recovery_checkpoints'
    destination=out/'training';destination.mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM,lambda sig,frame:stop_requested.append(sig))
    signal.signal(signal.SIGINT,lambda sig,frame:stop_requested.append(sig))
    try:
        assert draft.block_size==16 and len(draft.layers)==5
        basis_path=basis_fit(target,draft,train,p,out)
        b=load_file(str(basis_path))['repair_basis'].to('cuda',dtype=torch.bfloat16)
        output_basis=build_token_basis(target.lm_head.weight,b)
        input_basis=build_token_basis(target.model.embed_tokens.weight,b)
        torch.manual_seed(p['experiment_seed'])
        head=DualFeedbackRepair(**p['architecture']).to('cuda',dtype=torch.bfloat16)
        head.bind_features(target.model.embed_tokens.weight,input_basis)
        initial={name:digest(value) for name,value in head.state_dict().items()}
        settings=tune(head,output_basis,target,draft,train,p,out)
        assert initial=={name:digest(value) for name,value in head.state_dict().items()},'Tuning must not change model weights'
        provider.prefetch=settings['hidden_prefetch']
        opt=MasterAdamW(head,list(head.parameters()),[],p)
        protocol_sha=sha(args.protocol)
        latest=checkpoint_dir/'resume_latest.pt'
        if latest.exists():
            state=torch.load(latest,map_location='cpu',weights_only=True)
            assert state['protocol_sha256']==protocol_sha and 0<=state['step']<=total
            head.load_state_dict(state['head'],strict=True)
            with torch.no_grad():
                for name,parameter,master in opt.pairs:
                    master.copy_(state['masters'][name]);assert torch.equal(parameter.cpu(),master.to(parameter.dtype).cpu())
            opt.optimizer.load_state_dict(state['optimizer']);steps=state['step']
            torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
            dump(destination/'resume_loaded.json',dict(step=steps,optimizer_and_master_restored=True,utc=now()))
            metrics=destination/'metrics.jsonl'
            if metrics.exists():
                all_rows=[json.loads(line) for line in metrics.read_text().splitlines()]
                beyond=[row for row in all_rows if row['step']>steps]
                if beyond:
                    (destination/f'interrupted_metrics_{time.time_ns()}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in beyond))
                    metrics.write_text(''.join(json.dumps(row)+'\n' for row in all_rows if row['step']<=steps))
        else:
            dump(destination/'initialization.json',dict(source='fresh random',no_old_repair_or_basis_loaded=True,parameter_count=sum(value.numel() for value in head.parameters()),tensor_sha256=initial,utc=now()))
            save_checkpoint(checkpoint_dir,head,opt,0,protocol_sha,'initial_before_first_update')
        for epoch in range(1,4):
            indices=order_for_epoch(20000,p['experiment_seed'],epoch)
            already=max(0,min(20000,steps-(epoch-1)*20000))
            indices=indices[already:]
            if not indices:continue
            provider.begin_epoch([train[index] for index in indices])
            for index in indices:
                if stop_requested:
                    save_checkpoint(checkpoint_dir,head,opt,steps,protocol_sha,'signal_before_next_update')
                    dump(destination/'status.json',dict(status='interrupted',step=steps,total=total,utc=now()))
                    raise SystemExit(75)
                started=time.perf_counter();next_step=steps+1
                torch.manual_seed(p['experiment_seed']+next_step+200000)
                dump(destination/'live_progress.json',dict(step=next_step,index=index,phase='online_states',epoch=epoch,utc=now()))
                h,th,answers,anchor=online_states(target,draft,train[index],p['max_anchors'],torch.device('cuda:0'))
                assert h.shape[1]==15
                if next_step==1:
                    from repair.validate import real_checks
                    with torch.no_grad():raw=h[:1]@target.lm_head.weight.T
                    effects,gradients=real_checks(head,raw,h[:1],target.lm_head.weight,output_basis,anchor[:1],lambda z,j:answers[:1,j])
                    dump(destination/'first_real_input_checks.json',dict(effects=effects,gradients=gradients,proposal_positions=15,teacher_token_alignment_exact=True))
                    del raw
                opt.zero_grad()
                for group in opt.param_groups:group['lr']=group['base_lr']*schedule_scale(next_step,total,p)
                loss=head_backward(head,output_basis,target,h,th,answers,anchor,settings['head_chunk_anchors'],opt)
                norm=float(torch.nn.utils.clip_grad_norm_(opt.master_parameters,p['clip_grad_norm'],error_if_nonfinite=True))
                assert math.isfinite(norm)
                opt.step();opt.zero_grad();steps=next_step
                row=dict(step=steps,total=total,epoch=epoch,index=index,anchors=len(h),positions=15,loss=loss,grad_norm=norm,
                    seconds=time.perf_counter()-started,anchor_sha256=digest(anchor),answer_sha256=digest(answers),
                    example_tokens=len(train[index]['input_ids']),hidden_requests=provider.requests,hidden_deleted=provider.deleted,hidden_prefetch=provider.prefetch,head_chunk_anchors=settings['head_chunk_anchors'],utc=now())
                with (destination/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                dump(destination/'status.json',dict(status='training',step=steps,total=total,latest=row))
                print(now(),'TRAIN',steps,'/',total,'epoch',epoch,'anchors',len(h),'loss',loss,'seconds',round(row['seconds'],3),flush=True)
                if steps==1 or steps%p['checkpoint_interval']==0 or stop_requested:
                    save_checkpoint(checkpoint_dir,head,opt,steps,protocol_sha,'signal' if stop_requested else 'periodic')
                del h,th,answers,anchor
            path=destination/f'epoch_{epoch:02d}.safetensors'
            assert not path.exists(),'Never overwrite a completed epoch weight'
            save_file({key:value.detach().cpu().contiguous().clone() for key,value in head.state_dict().items()},str(path))
            save_checkpoint(checkpoint_dir,head,opt,steps,protocol_sha,f'epoch_{epoch}_complete')
            shutil.copy2(checkpoint_dir/'resume_latest.pt',destination/f'epoch_{epoch:02d}_resume.pt')
            dump(destination/f'epoch_{epoch:02d}.json',dict(step=steps,weight=str(path),sha256=sha(path),full_state=str(destination/f'epoch_{epoch:02d}_resume.pt'),utc=now()))
        final=destination/'epoch_03.safetensors'
        dump(destination/'summary.json',dict(status='complete',steps=steps,total=total,epochs=3,train_count=20000,
            checkpoints={str(epoch):json.loads((destination/f'epoch_{epoch:02d}.json').read_text()) for epoch in range(1,4)},
            basis_path=str(basis_path),basis_sha256=sha(basis_path),settings=settings,utc=now()))
        dump(destination/'status.json',dict(status='complete',step=steps,total=total,utc=now()))
        native_protocol(p,basis_path,final,out)
    except BaseException as exc:
        if head is not None and opt is not None:
            save_checkpoint(checkpoint_dir,head,opt,steps,sha(args.protocol),'exception')
        dump(destination/'failure.json',dict(error=repr(exc),step=steps,utc=now()))
        raise
    finally:
        provider.close()
        remaining=list(provider.root.iterdir())
        dump(out/'hidden_cleanup.json',dict(requests=provider.requests,deleted=provider.deleted,files_remaining=[str(path) for path in remaining],utc=now()))
        assert provider.requests==provider.deleted and not remaining


if __name__=='__main__':main()
