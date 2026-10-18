"""Real online states, original weights copied, all probe updates discarded."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

CODE = Path(__file__).resolve().parent
BASE = CODE.parent
RUNTIME = BASE / 'runtime_bundle'
sys.path[:0] = [str(CODE), str(RUNTIME), str(RUNTIME/'src'), str(RUNTIME/'hs_connectors/src'),
               str(BASE/'packed20k_recovery_v1'), str(BASE/'packed20k_pipeline_v1')]
import torch
from safetensors.torch import load_file
from runtime.utils import dump, sha, now
from runtime.live_trainer import load_models
from runtime.precision import build_token_basis
from runtime.master_optimizer import MasterAdamW
from repair.model import DualFeedbackRepair, rollout
from repair.run import digest
from transport import install
from compute import sharded_states
from losses import objective, eager_objective
from parallel_tf import teacher_forced_logits
from prepare import install as install_prepare
from numeric_diagnostic import diagnose


def vector(head):
    return torch.cat([p.grad.float().flatten() if p.grad is not None else torch.zeros_like(p).float().flatten()
                      for p in head.parameters()])


def compare(a, b):
    return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),
                cosine=float(torch.nn.functional.cosine_similarity(a[None],b[None])),
                max_abs=float((a-b).abs().max()))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True)
    out=ap.parse_args().output; p=json.loads((out/'protocol.json').read_text())
    original=json.loads(Path(p['original_training_protocol']).read_text())
    provider=None
    def status(stage, **kwargs):
        dump(out/'status.json',dict(status='running',stage=stage,pid=os.getpid(),utc=now(),**kwargs))
        print(stage, kwargs,flush=True)
    def memory():
        return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    try:
        for path,expected in p['protected_sha256'].items(): assert sha(path)==expected,path
        torch.set_num_threads(4); torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(p['allocator_budget_bytes']/torch.cuda.get_device_properties(0).total_memory)
        # Full vocabulary primitive check before loading target/draft.
        status('fused_full_vocabulary_correctness')
        torch.manual_seed(20261007)
        gold=(torch.randn(2,15,151936,device='cuda')*2).bfloat16()
        raw=(gold.float()+torch.randn_like(gold.float())*.5).bfloat16()
        primitive=[]
        for arm in p['arms']:
            x=raw.detach().requires_grad_(); a=eager_objective(x,gold,arm); ga=torch.autograd.grad(a,x)[0].float().flatten()
            va=float(a); del a,x
            x=raw.detach().requires_grad_(); b=objective(x,gold,arm); gb=torch.autograd.grad(b,x)[0].float().flatten()
            item=dict(arm=arm,loss_abs=abs(float(b)-va),**compare(ga,gb))
            item['passed']=item['loss_abs']<2e-5 and item['relative_l2']<.02 and item['cosine']>.999
            primitive.append(item); print('PRIMITIVE',item,flush=True)
            del x,b,ga,gb
        dump(out/'primitive_checks.json',primitive)
        assert all(r['passed'] for r in primitive),'Fused primitive mismatch'
        del raw,gold; gc.collect(); torch.cuda.empty_cache()
        install(out/'hidden_connection_retries.jsonl')
        status('load_models')
        target,draft=load_models(torch.device('cuda:0'),f"http://127.0.0.1:{p['port']}/v1",p['transfer_root'],16)
        provider=target.hidden_provider
        state=torch.load(p['checkpoint'],map_location='cpu',weights_only=True)
        basis=load_file(original['basis_path'])['repair_basis'].to('cuda',dtype=torch.bfloat16)
        output_basis=build_token_basis(target.lm_head.weight,basis)
        input_basis=build_token_basis(target.model.embed_tokens.weight,basis)
        head=DualFeedbackRepair(**original['architecture']).to('cuda',dtype=torch.bfloat16)
        head.bind_features(target.model.embed_tokens.weight,input_basis)
        head.load_state_dict(state['head'],strict=True)
        opt=MasterAdamW(head,list(head.parameters()),[],original)
        spec=importlib.util.spec_from_file_location('old_train',BASE/'packed20k_split_v3/train.py')
        old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        data=[json.loads(line) for line in Path(original['data_source']).read_text().splitlines()]
        indices=old.packs_for_epoch(data,original['experiment_seed'],1,65536)[0]
        rows=[data[i] for i in indices]; del data
        status('online_hidden')
        provider.begin_epoch(rows); chunks=[provider(row) for row in rows]
        torch.manual_seed(original['experiment_seed']+state['step']+1+200000)
        h,th,answers,anchor,positions=sharded_states(target,draft,rows,chunks,65536,1536,shard_anchors=512)
        assert h.shape[:2]==(1536,15)
        del chunks; provider.close()
        dump(out/'hidden_cleanup.json',dict(requests=provider.requests,deleted=provider.deleted,
             files_remaining=[str(x) for x in provider.root.iterdir()],no_persistent_hidden_or_logits=True))
        assert provider.requests==provider.deleted and not list(provider.root.iterdir())
        dump(out/'sample.json',dict(documents=len(rows),input_tokens=sum(len(x['input_ids']) for x in rows),
             anchors=len(h),positions=15,anchor_sha256=digest(positions),source_step=state['step'],utc=now()))
        del rows
        # Use the same actual states, BF16 logits, head weights and TF predecessors.
        with torch.no_grad():
            raw=h[:32]@target.lm_head.weight.T; gold=th[:32]@target.lm_head.weight.T
        gradients=[]; parallel=[]
        original_prepare=head.prepare
        for arm in p['arms']:
            status('real_head_loss_gradient_check',arm=arm)
            head.prepare=original_prepare
            head.zero_grad(set_to_none=True)
            z=rollout(head,raw,h[:32],target.lm_head.weight,output_basis,anchor[:32],lambda z,j:answers[:32,j]).logits
            a=eager_objective(z,gold,arm); a.backward(); va=float(a.detach()); ga=vector(head)
            old_logits=z.detach(); del z,a
            install_prepare(head)
            head.zero_grad(set_to_none=True)
            z=rollout(head,raw,h[:32],target.lm_head.weight,output_basis,anchor[:32],lambda z,j:answers[:32,j]).logits
            b=objective(z,gold,arm); b.backward(); vb=float(b.detach()); gb=vector(head)
            item=dict(arm=arm,loss_abs=abs(vb-va),**compare(ga,gb))
            item['passed']=item['loss_abs']<2e-5 and item['relative_l2']<.02 and item['cosine']>.999
            gradients.append(item); del z,b,ga
            old_named={name:par.grad.detach().float().clone() for name,par in head.named_parameters() if par.grad is not None}
            head.zero_grad(set_to_none=True)
            z=teacher_forced_logits(head,raw,h[:32],output_basis,anchor[:32],answers[:32])
            c=objective(z,gold,arm); c.backward(); gcandidate=vector(head)
            item=dict(arm=arm,loss_abs=abs(float(c.detach())-vb),argmax_agreement=float((old_logits.argmax(-1)==z.argmax(-1)).float().mean()),
                      logits_bitwise_agreement=float((old_logits==z).float().mean()),**compare(gb,gcandidate))
            item['per_parameter']={name:compare(old_named[name].flatten(),par.grad.float().flatten())
                                   for name,par in head.named_parameters() if name in old_named}
            item['passed']=item['loss_abs']<2e-5 and item['relative_l2']<.02 and item['cosine']>.999 and item['argmax_agreement']==1.0
            parallel.append(item)
            print('HEAD',gradients[-1],'PARALLEL', {k:v for k,v in item.items() if k!='per_parameter'},flush=True)
            del z,c,gb,gcandidate,old_logits,old_named
        dump(out/'head_loss_gradient_checks.json',gradients)
        dump(out/'parallel_tf_diagnostics.json',parallel)
        assert all(r['passed'] for r in gradients),'Real-head fused loss gradient gate failed'
        parallel_ok=all(r['passed'] for r in parallel)
        if not parallel_ok:
            dump(out/'parallel_numeric_root_cause.json',diagnose(head,raw,h[:32],output_basis,anchor[:32],answers[:32],original['architecture']))
        del raw,gold; head.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        # Explicit full-pack temporary optimizer update and multiple warm repeats.
        reports=[]
        modes=['sequential']+(['parallel_tf'] if parallel_ok else [])
        for mode in modes:
            for chunk in [64,256,1536]:
                for repetition in range(3):
                    status('chunk_benchmark',mode=mode,chunk=chunk,repetition=repetition)
                    gc.collect(); torch.cuda.empty_cache(); head.load_state_dict(state['head'])
                    with torch.no_grad():
                        for name,parameter,master in opt.pairs: master.copy_(state['masters'][name])
                    opt.optimizer.load_state_dict(__import__('copy').deepcopy(state['optimizer'])); opt.zero_grad()
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
                    report=dict(mode=mode,chunk=chunk,repetition=repetition,status='running',stage='logits')
                    try:
                        loss_sum=0.0
                        for begin in range(0,len(h),chunk):
                            sl=slice(begin,begin+chunk)
                            with torch.no_grad(): raw=h[sl]@target.lm_head.weight.T; gold=th[sl]@target.lm_head.weight.T
                            report['stage']='head'
                            if mode=='sequential':
                                z=rollout(head,raw,h[sl],target.lm_head.weight,output_basis,anchor[sl],lambda z,j:answers[sl,j]).logits
                            else: z=teacher_forced_logits(head,raw,h[sl],output_basis,anchor[sl],answers[sl])
                            report['stage']='fused_output'; loss=objective(z,gold,'OUTPUT')
                            report['stage']='backward'; (loss*len(h[sl])/len(h)).backward(); opt.accumulate()
                            loss_sum+=float(loss.detach())*len(h[sl])/len(h)
                            del raw,gold,z,loss
                        report['stage']='adam'; norm=float(torch.nn.utils.clip_grad_norm_(opt.master_parameters,original['clip_grad_norm'],error_if_nonfinite=True)); opt.step()
                        torch.cuda.synchronize(); report.update(status='passed',loss=loss_sum,grad_norm=norm,updates=1)
                    except torch.OutOfMemoryError as error:
                        report.update(status='oom',error=str(error),updates=0)
                        for name in ['raw','gold','z','loss']:
                            if name in locals(): del locals()[name]
                    report.update(seconds=time.perf_counter()-started,**memory())
                    reports.append(report); dump(out/'chunk_benchmarks.json',reports); print('BENCH',report,flush=True)
                    opt.zero_grad()
                    if report['status']=='oom': break
        for path,expected in p['protected_sha256'].items(): assert sha(path)==expected,path
        full=[r for r in reports if r['chunk']==1536 and r['status']=='passed']
        chosen='parallel_tf' if parallel_ok and any(r['mode']=='parallel_tf' for r in full) else 'sequential'
        summary=dict(status='complete',fused_gate_passed=True,parallel_gate_passed=parallel_ok,
                     full1536_passed=bool(full),selected_mode=chosen,selected_chunk=1536 if full else None,
                     primitive_checks=primitive,head_checks=gradients,benchmarks=reports,
                     source_weights_unchanged=True,temporary_updates_discarded=True,hardware=p['hardware'],utc=now())
        dump(out/'summary.json',summary)
        dump(out/'status.json',dict(status='complete',stage='probe_complete',utc=now()))
    except BaseException as error:
        dump(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=now()))
        dump(out/'status.json',dict(status='failed',error=repr(error),utc=now())); raise
    finally:
        if provider is not None: provider.close()

if __name__=='__main__': main()
