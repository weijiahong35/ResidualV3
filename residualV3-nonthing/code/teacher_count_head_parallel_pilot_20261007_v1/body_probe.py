"""Frozen draft block-size probe; identical online hidden and sampled anchors."""
import argparse,gc,json,os,sys,time,traceback
from pathlib import Path
CODE=Path(__file__).resolve().parent;BASE=CODE.parent;RUNTIME=BASE/'runtime_bundle'
sys.path[:0]=[str(CODE),str(RUNTIME),str(RUNTIME/'src'),str(RUNTIME/'hs_connectors/src'),str(BASE/'packed20k_recovery_v1'),str(BASE/'packed20k_pipeline_v1')]
import torch
from safetensors.torch import load_file
from runtime.utils import dump,sha,now
from runtime.live_trainer import load_models
from runtime.precision import build_token_basis
from repair.model import DualFeedbackRepair,rollout
from repair.run import digest
from transport import install
from compute import sharded_states
from losses import objective
from prepare import install as install_prepare
from train import packs_for_epoch

def vector(head):return torch.cat([p.grad.float().flatten() if p.grad is not None else torch.zeros_like(p).float().flatten() for p in head.parameters()])
def metrics(x,y):
    x=x.double().flatten();y=y.double().flatten()
    return dict(relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-12)),cosine=float(torch.nn.functional.cosine_similarity(x,y,dim=0)),max_abs=float((x-y).abs().max()))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--service',type=Path,required=True);a=ap.parse_args()
    out=a.output;out.mkdir(exist_ok=False);p=json.loads((a.run/'training_arms/CE_DPACE/training_protocol.json').read_text());e=json.loads((a.service/'endpoint.json').read_text());provider=None
    def status(stage,**extra):dump(out/'status.json',dict(status='running',stage=stage,pid=os.getpid(),utc=now(),**extra));print(stage,extra,flush=True)
    try:
        torch.set_num_threads(4);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
        free,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(min(free-8*2**30,20*2**30)/total)
        install(out/'connection_retries.jsonl');status('load_models')
        target,draft=load_models(torch.device('cuda'),e['endpoint'],e['root'],16);provider=target.hidden_provider
        state=torch.load(a.run/'training_arms/CE_DPACE/training/recovery_checkpoints/resume_latest.pt',map_location='cpu',weights_only=True)
        checkpoint_step=int(state['step'])
        # Copy a committed checkpoint before this probe; never read a rotating pointer twice.
        train=[json.loads(x) for x in Path(p['data_source']).read_text().splitlines()];indices=packs_for_epoch(train,p['experiment_seed'],1,65536)[0];rows=[train[i] for i in indices];del train
        status('online_hidden');provider.begin_epoch(rows);chunks=[provider(row) for row in rows];provider.close()
        basis=load_file(p['basis_path'])['repair_basis'].to('cuda',dtype=torch.bfloat16)
        head=DualFeedbackRepair(**p['architecture']).to('cuda',dtype=torch.bfloat16);head.load_state_dict(state['head'],strict=True)
        input_basis=build_token_basis(target.model.embed_tokens.weight,basis);output_basis=build_token_basis(target.lm_head.weight,basis)
        head.bind_features(target.model.embed_tokens.weight,input_basis);install_prepare(head)
        del state;torch.manual_seed(p['experiment_seed']+610007)
        status('reference_512');h,th,answers,anchor,positions=sharded_states(target,draft,rows,chunks,65536,1536,shard_anchors=512)
        anchors=positions.cpu().tolist();results={};times={};gates={}
        for size in [512,1536,256]:
            measurements=[];candidate=None
            for repeat in range(4):
                if candidate is not None:del candidate;candidate=None
                status('body_benchmark',chunk=size,repeat=repeat);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                z=sharded_states(target,draft,rows,chunks,65536,1536,explicit_anchors=anchors,shard_anchors=size);torch.cuda.synchronize()
                measurements.append(dict(seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
                candidate=z
                del z
            hz,tz,az,bz,pz=candidate
            assert torch.equal(th,tz) and torch.equal(answers,az) and torch.equal(anchor,bz) and torch.equal(positions,pz)
            item=metrics(h,hz);item['hidden_bitwise_equal']=torch.equal(h,hz)
            # Check actual repair objectives and head gradients at 480 real positions.
            checks=[]
            for loss_arm in ['OUTPUT','CE_DPACE','CE_DPACE_OUTPUT']:
                vectors=[];losses=[];argmax=[]
                for source_hidden in [h,hz]:
                    head.zero_grad(set_to_none=True)
                    with torch.no_grad():raw=source_hidden[:32]@target.lm_head.weight.T;gold=th[:32]@target.lm_head.weight.T
                    logits=rollout(head,raw,source_hidden[:32],target.lm_head.weight,output_basis,anchor[:32],lambda z,j:answers[:32,j]).logits
                    loss=objective(logits,gold,loss_arm);loss.backward();vectors.append(vector(head));losses.append(float(loss.detach()));argmax.append(logits.detach().argmax(-1));del raw,gold,logits,loss
                check=dict(arm=loss_arm,loss_abs=abs(losses[0]-losses[1]),argmax_agreement=float((argmax[0]==argmax[1]).float().mean()),**metrics(vectors[0],vectors[1]))
                check['passed']=check['loss_abs']<2e-5 and check['relative_l2']<.02 and check['cosine']>.999 and check['argmax_agreement']==1.
                checks.append(check);print('HEAD_GRAD',size,check,flush=True)
            item.update(head_checks=checks,passed=item['relative_l2']<.02 and item['cosine']>.999 and all(c['passed'] for c in checks));gates[str(size)]=item;times[str(size)]=measurements
            dump(out/'partial_results.json',dict(gates=gates,times=times));del candidate,hz,tz,az,bz,pz;head.zero_grad(set_to_none=True)
        import statistics
        base=statistics.median(t['seconds'] for t in times['512'][1:]);whole=statistics.median(t['seconds'] for t in times['1536'][1:])
        chosen=1536 if gates['1536']['passed'] and whole<base*.9 else 512
        summary=dict(status='complete',gates=gates,times=times,selected_draft_shard_anchors=chosen,head_chunk_anchors=1536,
                     observed_body_speed_ratio=base/whole,same_1536_anchors=True,anchor_sha256=digest(positions),
                     source_checkpoint_step=checkpoint_step,
                     no_optimizer_updates=True,shared_gpu=True,scope='one real pack and 32-anchor head-gradient check; no universal bitwise proof',utc=now())
        dump(out/'summary.json',summary);dump(out/'status.json',dict(status='complete',selected_draft_shard_anchors=chosen,utc=now()))
    except BaseException as exc:
        dump(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),utc=now()))
        dump(out/'status.json',dict(status='failed',error=repr(exc),utc=now()));raise
    finally:
        if provider is not None:provider.close()
        if provider is not None:dump(out/'hidden_cleanup.json',dict(requests=provider.requests,deleted=provider.deleted,source_weights_unchanged=True,no_persistent_hidden_or_logits=True,utc=now()))
if __name__=='__main__':main()
