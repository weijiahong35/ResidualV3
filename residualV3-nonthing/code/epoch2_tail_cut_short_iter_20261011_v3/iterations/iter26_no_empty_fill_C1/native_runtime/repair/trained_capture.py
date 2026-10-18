"""RAM-only native prefix observations; freeze thresholds before formal GSM8K test."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from runtime.utils import dump,now

ROOT=Path(os.environ.get('Q_PROBE_ROOT','/tmp/q_probe_unused'))
RECORDS=[]; BLOCKS={}; TRACES={}
BRANCHES=('Q45_MLP','Q45_REPAIR_LATENT_MLP')

def cutoff(values,threshold,minimum=8,patience=2):
    survival=1.; consecutive=0
    for j,p in enumerate(values):
        survival*=float(p)
        consecutive=consecutive+1 if survival<threshold and j+1>=minimum else 0
        if consecutive>=patience:
            return min(j+2,15)
    return 15

def opportunity(records,name,t,threshold):
    rows=[r for r in records if r['T']==t]
    cuts=[cutoff(r['scores'][name],threshold) for r in rows]
    full=sum(1+r['accepted'] for r in rows)
    retained=sum(1+min(r['accepted'],c) for r,c in zip(rows,cuts))
    return dict(blocks=len(rows),threshold=threshold,full_prefix_mean=full/max(len(rows),1),
        retained_prefix_mean=retained/max(len(rows),1),retention=retained/max(full,1),
        mean_cutoff=sum(cuts)/max(len(cuts),1),cutoff_counts={str(j):cuts.count(j) for j in range(1,16)},
        shortened_fraction=sum(c<15 for c in cuts)/max(len(cuts),1),
        limitation='Fixed full-repair native paths; adaptive-rollout AL/TPS must be measured separately.')

def auc(records,name,t):
    y=[]; score=[]
    for r in records:
        if r['T']!=t:continue
        for j in range(min(r['accepted']+1,15)):
            y.append(j==r['accepted']);score.append(1-r['scores'][name][j])
    y=np.asarray(y,dtype=bool);score=np.asarray(score)
    pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:return dict(auroc=None,positive=pos,negative=neg)
    order=np.argsort(score,kind='stable');ranks=np.empty(len(score),dtype=float)
    start=0
    while start<len(score):
        end=start+1
        while end<len(score) and score[order[end]]==score[order[start]]:end+=1
        ranks[order[start:end]]=(start+1+end)/2
        start=end
    return dict(auroc=float((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg)),positive=pos,negative=neg,
        label='realized native first rejection among reached positions; ranking only')

def flush():
    p=json.loads((ROOT.parent/'native_protocol.json').read_text())
    calibration=[r for r in RECORDS if r['split']=='calibration']
    holdout=[r for r in RECORDS if r['split']=='holdout']
    assert calibration and holdout
    grid=[0.,.0001,.0003,.001,.003,.01,.02,.03,.05,.08,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99]
    policies={}; diagnostics={}
    for name in BRANCHES:
        thresholds=[]; diagnostics[name]={}
        for t in (0,1):
            candidates=[opportunity(calibration,name,t,x) for x in grid]
            valid=[c for c in candidates if c['retention']>=.995]
            chosen=min(valid,key=lambda c:(c['mean_cutoff'],-c['retention'],c['threshold']))
            thresholds.append(chosen['threshold'])
            diagnostics[name][str(t)]=dict(calibration=chosen,holdout=opportunity(holdout,name,t,chosen['threshold']),
                calibration_auroc=auc(calibration,name,t),holdout_auroc=auc(holdout,name,t),threshold_grid=candidates)
        policies[name]=dict(minimum_repair=8,patience=2,thresholds=thresholds)
    policy=dict(format='epoch2_frozen_readout_prefix_policy_v1',policies=policies,
        calibration_policy=dict(minimum_repair=8,patience=2,thresholds=[0.,0.]),
        auxiliary_checkpoint_sha256=p['auxiliary_checkpoint_sha256'],checkpoint_sha256=p['native_checkpoints']['FULL15']['sha256'],
        threshold_selection='Minimize repaired prefix length subject to >=99.5% full-path prefix retention on calibration only',
        prediction_delay_positions=1,calibration_source='openai/gsm8k main/train, disjoint by normalized question from all formal test prompts',
        heldout_scope='Native score readout check only; no additional predictor training',utc=now())
    dump(ROOT/'predictor.json',policy)
    dump(ROOT/'RESULTS.json',dict(status='complete',blocks=len(RECORDS),diagnostics=diagnostics,
        no_hidden_logits_or_per_state_feature_cache_saved=True,policy=policy,
        limitation='MLPs trained on teacher-forced proxies; native threshold calibration does not remove path shift.',utc=now()))
    (ROOT/'RESULTS_CN.txt').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\n')
    RECORDS.clear()
    dump(ROOT/'capture_status.json',dict(status='complete',states_on_disk=0,utc=now()))

def install_capture():
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
    original=RejectionSampler.__call__
    def observed(self,logits,batch,draft_logits=None):
        control=json.loads((ROOT/'capture_control.json').read_text()) if (ROOT/'capture_control.json').exists() else {}
        key=control.get('request_key','disabled');block=BLOCKS.get(key,0)
        wanted=control.get('enabled',False) and batch.num_reqs==1 and batch.num_draft_tokens==15 and block<int(control['max_blocks'])
        output=original(self,logits,batch,draft_logits)
        if control.get('trace_pass') and batch.num_reqs==1:
            emitted=output.sampled_token_ids[0,:int(output.num_sampled[0])].tolist()
            TRACES.setdefault(control['trace_pass'],[]).extend(emitted)
            dump(ROOT/f"trace_{control['trace_pass']}.json",dict(token_sha256=hashlib.sha256(json.dumps(TRACES[control['trace_pass']]).encode()).hexdigest(),tokens=len(TRACES[control['trace_pass']]),utc=now()))
        if control.get('action')=='reset':
            RECORDS.clear();BLOCKS.clear()
            dump(ROOT/'capture_status.json',dict(status='reset',states=0,utc=now()))
        elif wanted:
            from runtime.native import MODEL
            inputs,processed,result,fp64=MODEL.diagnostic_latest
            ids,scores,previous=result;raw,hidden,anchor,positions,idx,temp,seeds,step=inputs
            assert ids.shape==(1,15) and batch.cu_num_logits_np.tolist()==[0,16]
            scheduled=batch.input_ids[batch.logits_indices].long()
            assert torch.equal(scheduled[1:],ids[0]) and int(scheduled[0])==int(anchor[0])
            assert torch.equal(previous[:,0],anchor) and torch.equal(previous[:,1:],ids[:,:-1])
            actual=draft_logits[int(batch.idx_mapping[0]),:15]
            assert torch.equal(actual,processed[idx.long(),step.long()])
            from head_latency.tail_graph_state import bind_tail_state
            state=bind_tail_state(MODEL.head,1)
            assert int(state[-1][0])==15, ('calibration unexpectedly stopped',state[-1].tolist())
            assert state[-2].shape==(1,15,4), ('calibration score shape',state[-2].shape,MODEL.head._tail_branch)
            prediction=state[-2][0].detach().cpu().tolist();t=control['T']
            accepted=max(0,min(15,int(output.num_sampled[0])-1))
            RECORDS.append(dict(prompt=control['probe_index'],split=control['split'],T=t,block=block,accepted=accepted,
                scores={name:[row[2*k+t] for row in prediction] for k,name in enumerate(BRANCHES)}))
            BLOCKS[key]=block+1
            dump(ROOT/'capture_status.json',dict(status='collecting',native_blocks=len(RECORDS),native_prefix_q_alignment_checked=True,states_on_disk=0,utc=now()))
        elif control.get('action')=='flush':flush()
        return output
    RejectionSampler.__call__=observed
