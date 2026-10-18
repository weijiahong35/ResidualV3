"""Independent original recurrence with forced masks from optimized decisions."""
import json,os
from pathlib import Path
import torch
from runtime.components import Rollout
from runtime.precision import bf16_logit_update
from repair.model import project_candidates


def reference(head,raw,hidden,weight,basis,anchor,choose,cuts):
    # All15 preparation retains the same operation shapes as original FULL15.
    hp,qp,hn,ids,_,outrows=head.prepare(raw,hidden,basis)
    previous=anchor;scores=[];tokens=[];history=[]
    for j in range(raw.shape[1]):
        coeff,dose=head.predict(hp[:,j],head.feature_weight[previous.long()],qp[:,j],None)
        direction=project_candidates(coeff,outrows[:,j])
        corrected,_=bf16_logit_update(raw[:,j].gather(-1,ids[:,j]),direction,dose,hn[:,j])
        z=raw[:,j].clone();z.scatter_(-1,ids[:,j],corrected)
        z=torch.where((j<cuts)[:,None],z,raw[:,j])
        token=choose(z,j).long();scores.append(z);tokens.append(token);history.append(previous);previous=token
    return Rollout(torch.stack(scores,1),torch.stack(tokens,1),torch.stack(history,1),[],())


def verify_real_block(model,n):
    from head_latency.optimized import verify_real_block as original_verify
    if model.head is None:return
    mode=os.environ.get('TAIL_MODE','FULL15')
    if mode=='FULL15':return original_verify(model,n)
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    from .small_tile import set_mode
    inputs,q,result,fp64=model.integrated_inputs[n]
    raw,hidden,anchor,positions,idx,temp,seeds,step=inputs;length=raw.shape[1]
    cuts=model.head._tail_last_stats[-1].clone()
    # Verify the predictor input itself, not just logits under its chosen mask.
    # A numerically changed q45 could silently choose a different prefix while
    # every forced-mask repair check still passes.
    if os.environ.get('TAIL_SPARSE_Q45','1')=='1':
        from .q45_features import features as stock_features
        from .tail_sparse_q45 import features as cached_features
        active=model.head._tail_last_stats[0]
        cache,ci=model.head._tail_sparse_cache
        evidence=[]
        for j in range(length-2):
            z=result[1][:,j].contiguous();expected,_=stock_features(z)
            sampled=getattr(model.head,'_tail_sampler_max',None)
            actual=cached_features(z,active[:,j],cache[j],ci[j],sampled_maxima=sampled[j] if sampled is not None else None)
            mask=active[:,j].bool()
            equal=torch.equal(actual[mask],expected[mask]);assert equal,('real q45 differs',n,j)
            evidence.append(dict(position=j+1,active_rows=int(mask.sum()),bitwise_equal=equal))
        with (Path(os.environ['LOCAL_TRIAL'])/'post_repair_q45_parity.jsonl').open('a') as f:
            f.write(json.dumps(dict(batch_size=n,cache=os.environ.get('TAIL_SPARSE_PREP'),checks=evidence))+'\n')
    # Same-variant eager check has already verified Graph decisions/logits.
    refq=torch.empty_like(q);mappings=idx.reshape(n,length).T.contiguous()
    pos=(positions.reshape(n,length).T+1).contiguous();cols=step.reshape(n,length).T.contiguous()
    def choose(z,j):return gumbel_sample(z,mappings[j],temp,seeds,pos[j],apply_temperature=True,output_processed_logits=refq,output_processed_logits_col=cols[j],use_fp64=fp64)
    prior_small=model.head._small_tile_mode;prior_fusion=model.head._fusion_mode
    try:
        set_mode(model.head,'baseline');model.head._fusion_mode='baseline'
        ref=reference(model.head,raw,hidden,model.lm_head.weight,model.token_basis,anchor,choose,cuts)
    finally:
        set_mode(model.head,prior_small);model.head._fusion_mode=prior_fusion
    ids,scores,previous=result
    mask=torch.arange(length,device=raw.device)[None,:]>=cuts[:,None]
    kept=~mask;flat=kept.reshape(-1)
    checks=dict(logits_bitwise_equal=torch.equal(scores,ref.logits),sampled_ids_bitwise_equal=torch.equal(ids[kept],ref.tokens[kept]),
                actual_predecessors_bitwise_equal=torch.equal(previous[kept],ref.predecessors[kept]),
                verifier_q_bitwise_equal=torch.equal(q[idx[flat].long(),step[flat].long()],refq[idx[flat].long(),step[flat].long()]))
    checks['raw_suffix_logits_bitwise_equal']=torch.equal(scores[mask],raw[mask])
    row=dict(batch_size=n,mode=mode,temperature=float(temp[idx.long()].max()),cutoffs=cuts.tolist(),**checks)
    with (Path(os.environ['LOCAL_TRIAL'])/'head_implementation_parity.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    assert all(checks.values()),row
