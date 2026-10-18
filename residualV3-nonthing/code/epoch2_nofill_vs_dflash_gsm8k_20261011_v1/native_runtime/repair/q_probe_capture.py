"""RAM-only labels from the actual target verifier, with independent alignment checks."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
from runtime.utils import dump,now

ROOT=Path(os.environ.get('Q_PROBE_ROOT','/tmp/q_probe_unused'))
RECORDS=[]
BLOCKS={}
TRACES={}
PROJECTED=None
LATEST_RAW=None
LATEST_POST=None
CONTRACTS=0


def features(raw):
    q=raw.float().softmax(-1)
    top,ids=q.topk(256,dim=-1)
    entropy=-(q*q.clamp_min(1e-30).log()).sum(-1)/math.log(q.shape[-1])
    stats=torch.stack([top[:,0],top[:,1],top[:,0]-top[:,1],entropy]+[top[:,:k].sum(-1) for k in (1,2,4,8,16,32,64,128,256)],-1)
    stats=torch.cat([stats,top[:,:32].clamp_min(1e-30).log()],-1)
    return stats


def microbench():
    """Same-shaped pre/post signal cost, excluding repair; never service TPS."""
    if LATEST_RAW is None:return dict(status='no_input')
    # Fixed weights isolate implementation cost, not fitted decision quality.
    out={}
    for origin,tensor in (('PRE',LATEST_RAW),('POST',LATEST_POST)):
        for length in (1,15):
            raw=tensor[:length].clone()
            weight=torch.zeros((45,1),device=raw.device)
            for kind in ('MAXQ','STATS_LINEAR'):
                def run():
                    if kind=='MAXQ':
                        z=raw.float()
                        return (z.amax(-1)-z.logsumexp(-1)).exp()
                    return (features(raw) @ weight).sigmoid()
                for _ in range(3):run()
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):value=run()
                graph.replay();torch.cuda.synchronize()
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):graph.replay()
                end.record();end.synchronize()
                out[f'{origin}_{kind}_{length}']=dict(signal_ms=start.elapsed_time(end)/30,positions=length,
                    scope='logits to q signal only; excludes head and routing/synchronization',output_finite=bool(value.isfinite().all()))
    return dict(status='complete',formal_speed_evidence=False,gpu_shared=os.environ['Q_PROBE_SHARED']=='1',timings=out)


def flush():
    global PROJECTED,LATEST_RAW,LATEST_POST
    from .q_pre_post_analysis import analyze
    if (ROOT/'RESULTS.json').exists():return
    dump(ROOT/'capture_status.json',dict(status='analyzing',states=len(RECORDS),utc=now()))
    result=microbench()
    analyze(ROOT,RECORDS,result)
    RECORDS.clear();PROJECTED=None;LATEST_RAW=None;LATEST_POST=None
    dump(ROOT/'capture_status.json',dict(status='complete',states_on_disk=0,utc=now()))


def install_capture():
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
    original=RejectionSampler.__call__
    def observed(self,logits,batch,draft_logits=None):
        global CONTRACTS,LATEST_RAW,LATEST_POST
        control=json.loads((ROOT/'capture_control.json').read_text()) if (ROOT/'capture_control.json').exists() else {}
        enabled=control.get('enabled',False)
        key=control.get('request_key','disabled')
        block=BLOCKS.get(key,0)
        wanted=enabled and batch.num_reqs==1 and batch.num_draft_tokens==15 and block<int(control['max_blocks'])
        gold=logits[:15].detach().clone() if wanted else None
        output=original(self,logits,batch,draft_logits)
        if control.get('trace_pass') and batch.num_reqs==1:
            count=int(output.num_sampled[0]);emitted=output.sampled_token_ids[0,:count].tolist()
            TRACES.setdefault(control['trace_pass'],[]).extend(emitted)
            dump(ROOT/f"trace_{control['trace_pass']}.json",dict(token_sha256=hashlib.sha256(json.dumps(TRACES[control['trace_pass']]).encode()).hexdigest(),tokens=len(TRACES[control['trace_pass']]),utc=now()))
        if control.get('action')=='reset':
            RECORDS.clear();BLOCKS.clear();CONTRACTS=0
            dump(ROOT/'capture_status.json',dict(status='reset',states=0,utc=now()))
            return output
        if not wanted and control.get('action')!='flush':return output
        cpu_before=torch.random.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state().clone()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
            if wanted:
                from runtime.native import MODEL
                inputs,processed,result,fp64=MODEL.diagnostic_latest
                raw,hidden,anchor,positions,idx,temp,seeds,step=inputs
                tokens,scores,previous=result
                assert raw.shape[:2]==(1,15) and not self.use_block_verification
                assert len(RECORDS)<23040,'capture bound exceeded'
                assert batch.cu_num_logits_np.tolist()==[0,16]
                scheduled=batch.input_ids[batch.logits_indices].long()
                assert torch.equal(scheduled[1:],tokens[0]),'actual verifier prefix differs from captured proposal'
                assert int(scheduled[0])==int(anchor[0]),'anchor alignment'
                assert torch.equal(previous[:,0],anchor) and torch.equal(previous[:,1:],tokens[:,:-1]),'actual repaired lag-1 alignment'
                index=int(batch.idx_mapping[0])
                assert torch.equal(draft_logits[index,:15].float(),scores[0].float()),'verifier q differs from actual repair scores'
                target_positions=batch.positions[batch.logits_indices]
                assert torch.equal(target_positions[1:],target_positions[:-1]+1),'target positions not consecutive'
                # Installed vLLM DFlash passes predicted position Q-2 to the sampler; +1 is verifier predecessor Q-1.
                assert torch.equal(positions.reshape(-1)+1,target_positions[:15]),'target/raw sampling-key alignment'
                raw=raw[0];scores=scores[0]
                q=raw.float().softmax(-1);p=gold.float().softmax(-1);r=scores.float().softmax(-1)
                q_overlap=torch.minimum(p,q).sum(-1);r_overlap=torch.minimum(p,r).sum(-1)
                mismatch=q.argmax(-1)!=p.argmax(-1);repaired_wrong=r.argmax(-1)!=p.argmax(-1)
                values=features(raw).cpu().tolist();post_values=features(scores).cpu().tolist()
                accepted=max(0,min(15,int(output.num_sampled[0])-1))
                labels=torch.stack([mismatch.float(),(1-q_overlap).clamp(0,1),(mismatch & ~repaired_wrong).float(),(~mismatch & repaired_wrong).float(),r_overlap-q_overlap,repaired_wrong.float(),(1-r_overlap).clamp(0,1)],-1).cpu().tolist()
                for j in range(15):
                    RECORDS.append(dict(features_pre=values[j],features_post=post_values[j],prompt=control['probe_index'],split=control['split'],T=control['T'],block=block,position=j,reached=j<=accepted,
                        argmax_mismatch=bool(labels[j][0]),tv_raw=labels[j][1],greedy_fixed=bool(labels[j][2]),
                        greedy_harmed=bool(labels[j][3]),overlap_gain=labels[j][4],repaired_argmax_mismatch=bool(labels[j][5]),tv_post=labels[j][6],
                        accepted_proposals=accepted,native_reject=(j==accepted and accepted<15),
                        accepted_suffix_after_current=max(accepted-j-1,0)))
                BLOCKS[key]=block+1;LATEST_RAW=raw.detach().clone();LATEST_POST=scores.detach().clone();CONTRACTS+=1
                dump(ROOT/'capture_status.json',dict(status='collecting',states=len(RECORDS),native_blocks=CONTRACTS,probe_index=control['probe_index'],T=control['T'],
                    target_prefix_and_verifier_logits_checked=True,actual_lag1_checked=True,proposal_length=15,states_on_disk=0,utc=now()))
            if control.get('action')=='flush':flush()
        assert torch.equal(cpu_before,torch.random.get_rng_state()) and torch.equal(cuda_before,torch.cuda.get_rng_state()),'diagnostic changed RNG state'
        return output
    RejectionSampler.__call__=observed
