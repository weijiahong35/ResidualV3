"""Compare identical online states and gradients; dry-run full1536 with next upload."""
import argparse, gc, importlib.util, json, os, sys, time, traceback, subprocess
from pathlib import Path
CODE=Path(__file__).resolve().parent
RUNTIME=CODE.parent/'runtime_bundle'
sys.path[:0]=[str(CODE),str(RUNTIME),str(RUNTIME/'src'),str(RUNTIME/'hs_connectors/src'),str(CODE.parent/'packed20k_recovery_v1')]
import torch
from safetensors.torch import load_file
from runtime.live_trainer import load_models
from runtime.utils import dump, now, sha
from runtime.precision import build_token_basis
from runtime.master_optimizer import MasterAdamW
from repair.model import DualFeedbackRepair, rollout
from prepare import install as install_prepare
from losses import objective
from pipeline import allocate_host, fill_host, upload
from compute import sharded_states
from dp_teacher import install
import train as learner


def vector(head):
    return torch.cat([p.grad.float().flatten() if p.grad is not None else
                      torch.zeros_like(p).float().flatten() for p in head.parameters()])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--service',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--auto-gpus',nargs='+',type=int)
    a=ap.parse_args();out=a.output;out.mkdir(exist_ok=False)
    provider=None
    def stage(name,**kw):
        print(now(),name,kw,flush=True)
        dump(out/'status.json',dict(status='running',stage=name,pid=os.getpid(),utc=now(),**kw))
    try:
        torch.set_num_threads(4)
        selected_gpu = None
        if a.auto_gpus:
            assert not torch.cuda.is_initialized()
            deadline=time.monotonic()+180
            while True:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                candidates=[]
                for line in raw.splitlines():
                    i,u,f,v=[x.strip() for x in line.split(',')]
                    if int(i) in a.auto_gpus and int(f)>=63500:
                        candidates.append((int(v),-int(f),int(i),u))
                if candidates:
                    _,_,selected_gpu,uuid=min(candidates)
                    break
                stage('waiting_validation_memory',minimum_free_mib=63500)
                assert time.monotonic()<deadline, 'No sufficiently free validation GPU'
                time.sleep(2)
        from arena import install as install_arena, small_gate, stats
        global arena_library, arena_allocator
        arena_allocator,arena_library=install_arena(68)
        stage('arena_small_gate', result=small_gate(arena_library))
        torch.cuda.set_device(0)
        if selected_gpu is not None:
            tasks=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            matching=[line.split(',')[0].strip() for line in tasks.splitlines() if line.split(',')[1].strip()==str(os.getpid())]
            assert uuid in matching, ('CUDA identity check',uuid,matching)
            stage('validation_device',physical_gpu=selected_gpu,uuid=uuid)
        torch.backends.cuda.matmul.allow_tf32=False
        p=json.loads((a.run/'training_arms/CE_DPACE_OUTPUT/training_protocol.json').read_text())
        ep=json.loads((a.service/'endpoint.json').read_text())
        state=torch.load(a.run/'training_arms/CE_DPACE_OUTPUT/training/recovery_checkpoints/resume_latest.pt',map_location='cpu',weights_only=True)
        checkpoint_step=int(state['step'])
        install(out/'hidden_requests.jsonl',dict(endpoint=ep['endpoint'],root=ep['root']))
        stage('load_models',source_checkpoint_step=checkpoint_step)
        target,draft=load_models(torch.device('cuda'),ep['endpoint'],ep['root'],16)
        provider=target.hidden_provider
        source=[json.loads(s) for s in Path(p['data_source']).read_text().splitlines()]
        indices=learner.packs_for_epoch(source,p['experiment_seed'],2,65536)[0]
        rows=[source[i] for i in indices];del source
        stage('online_hidden')
        provider.begin_epoch(rows);chunks=[provider(row) for row in rows]
        provider.close()
        stage('pinned_cpu_pack')
        host=allocate_host(rows,65536);it=iter(chunks)
        fill_host(host,rows,lambda row:next(it))
        assert torch.equal(host.aux[0,:host.offset],torch.cat([h[:,:-1].flatten(1) for h in chunks]))
        assert torch.equal(host.final,torch.cat([h[:,-1] for h in chunks]))
        basis=load_file(p['basis_path'])['repair_basis'].to('cuda',dtype=torch.bfloat16)
        head=DualFeedbackRepair(**p['architecture']).to('cuda',dtype=torch.bfloat16)
        head.load_state_dict(state['head'],strict=True)
        head.bind_features(target.model.embed_tokens.weight,build_token_basis(target.model.embed_tokens.weight,basis))
        output_basis=build_token_basis(target.lm_head.weight,basis);install_prepare(head)
        del state
        spec=importlib.util.spec_from_file_location('previous_compute',CODE.parent/'fused_output_dpace_dp2_v6/compute.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        stream=torch.cuda.default_stream()
        torch.manual_seed(p['experiment_seed']+200001)
        stage('reference_states')
        reference=old.sharded_states(target,draft,rows,chunks,65536,1536,shard_anchors=512)
        anchors=reference[-1].cpu().tolist()
        stage('async_states')
        uploaded=upload(host,torch.device('cuda'),stream,target,draft);uploaded.wait()
        candidate=sharded_states(target,draft,rows,uploaded,65536,1536,explicit_anchors=anchors,shard_anchors=512)
        torch.cuda.current_stream().synchronize()
        equal=[torch.equal(x,y) for x,y in zip(reference,candidate)]
        assert all(equal),equal
        checks=[]
        for arm in ['OUTPUT','CE_DPACE','CE_DPACE_OUTPUT']:
            stage('loss_gradient',arm=arm)
            gradients=[];losses=[]
            for h,th,answers,anchor,_ in [reference,candidate]:
                head.zero_grad(set_to_none=True)
                with torch.no_grad():raw=h[:32]@target.lm_head.weight.T;gold=th[:32]@target.lm_head.weight.T
                logits=rollout(head,raw,h[:32],target.lm_head.weight,output_basis,anchor[:32],lambda z,j:answers[:32,j]).logits
                loss=objective(logits,gold,arm);loss.backward()
                losses.append(float(loss.detach()));gradients.append(vector(head));del raw,gold,logits,loss
            check=dict(arm=arm,loss_bitwise_equal=losses[0]==losses[1],gradient_bitwise_equal=torch.equal(*gradients))
            assert all(v for k,v in check.items() if k!='arm'),check
            checks.append(check)
            dump(out/'partial_gate.json',dict(bitwise_states=equal,checks=checks,utc=now()))
        uploaded.release();del uploaded,reference,chunks
        head.zero_grad(set_to_none=True);gc.collect()
        # Next pack's 3GiB device copy remains resident during the full head graph.
        next_pack=upload(host,torch.device('cuda'),stream,target,draft)
        next_pack.ready.synchronize()
        opt=MasterAdamW(head,list(head.parameters()),[],p)
        learner.ACTIVE_ARM='CE_DPACE_OUTPUT';learner.ACTIVE_MODE='sequential'
        torch.cuda.current_stream().synchronize();None
        stage('full1536_backward_with_next_upload')
        start=time.perf_counter()
        h,th,answers,anchor,_=candidate
        value=learner.head_backward(head,output_basis,target,h,th,answers,anchor,1536,opt)
        torch.cuda.current_stream().synchronize();next_pack.ready.synchronize()
        full=dict(loss=value,seconds=time.perf_counter()-start,anchors=len(h),
                  peak_allocated=stats(arena_library)["peak"],peak_reserved=stats(arena_library)["capacity"])
        assert len(h)==1536 and all(torch.isfinite(m).all() for m in opt.master_parameters)
        assert all(m.grad is None or torch.isfinite(m.grad).all() for m in opt.master_parameters)
        full['gradient_norm']=float(torch.nn.utils.clip_grad_norm_(opt.master_parameters,p['clip_grad_norm'],error_if_nonfinite=True))
        opt.zero_grad();next_pack.release();del next_pack
        # Random anchor selection must also remain identical, not only supplied anchors.
        uploaded=upload(host,torch.device('cuda'),stream,target,draft);uploaded.wait()
        torch.manual_seed(p['experiment_seed']+200001)
        random_check=sharded_states(target,draft,rows,uploaded,65536,1536,shard_anchors=512)
        assert random_check[-1].cpu().tolist()==anchors
        assert all(torch.equal(x,y) for x,y in zip(candidate,random_check))
        torch.cuda.current_stream().synchronize();uploaded.release()
        summary=dict(status='passed',bitwise_states=equal,three_loss_gradient_checks=checks,
                     full1536_memory_passed=True,full1536=full,random_anchor_selection_unchanged=True,
                     source_checkpoint_step=checkpoint_step,no_optimizer_updates=True,validation_gpu=selected_gpu,
                     next_pack_gpu_components='frozen projected context and normalized last hidden',staging_allocation_guard=True,workspace_reservation_gib=68, fixed_live_allocation=True,upload_stream='default_fixed_arena',
                     ordered_peak_running=provider.ordered.peak_running,ordered_peak_pending=provider.ordered.peak_pending,
                     requests=provider.requests,deleted=provider.deleted,
                     input_tokens=host.offset,pinned_cpu_pack_bitwise_equal=True,
                     scope='one real 65536-token pack, all1536 state parity, 32-anchor three-objective gradient parity, full1536 backward with one next device pack',utc=now())
        dump(out/'summary.json',summary);dump(out/'status.json',summary);print(json.dumps(summary),flush=True)
    except BaseException as exc:
        failure=dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),utc=now())
        dump(out/'failure.json',failure);dump(out/'status.json',failure);raise
    finally:
        if provider is not None:
            provider.close()
            dump(out/'hidden_cleanup.json',dict(requests=provider.requests,deleted=provider.deleted,no_persistent_hidden=True,utc=now()))


if __name__=='__main__':main()
