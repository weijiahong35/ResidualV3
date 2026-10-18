"""Fixed-weight inference controls: original, prior storage, unique stores."""
import json
import os
from pathlib import Path
import torch
from repair.model import rollout as original_rollout, project_candidates
from runtime.components import Rollout
from runtime.precision import bf16_logit_update
from .write_kernel import write_candidates
from .predecessor import prepare_lookup, predict_encoded

VARIANTS = ('prev_lookup_exact', 'prev_lookup_swiglu', 'small_tile')


def prepare_variant(head, variant):
    assert variant in VARIANTS and not head.use_soft and not head.training
    if variant in VARIANTS:
        limit = int(os.environ.get('LOCAL_MAX_CONCURRENCY', '0'))
        sizes = tuple(range(1, limit+1)) if limit else (1, 7, 40)
        prepare_lookup(head, exact=True, batch_sizes=sizes)
    if variant == 'small_tile':
        from .small_tile import set_mode
        set_mode(head, os.environ['HEAD_SMALL_TILE_MODE'])
        from .structure_fusion import set_mode as fusion_mode
        fusion_mode(head,os.environ.get('HEAD_STRUCTURE_MODE','baseline'))
    if head.feature_weight.is_cuda:
        length = int(os.environ.get('NATIVE_PROPOSAL_LENGTH', '15'))
        z = head.feature_weight.new_zeros((1, head.vocab_size))
        ids = torch.arange(head.q_topk, device=z.device, dtype=torch.int64).reshape(1,1,-1).expand(1,length,-1).contiguous()
        values = head.feature_weight.new_zeros((1,head.q_topk))
        write_candidates(z, ids[:,0], values)


def optimized_rollout(head, raw, hidden, weight, basis, anchor, choose, *, capture=True,
                      soft_overrides=None, variant='triton_write'):
    if variant == 'small_tile' and getattr(head,'_fusion_mode','baseline')!='baseline':
        from .structure_fusion import rollout as fused_rollout
        return fused_rollout(head,raw,hidden,weight,basis,anchor,choose,capture=capture)
    if variant == 'baseline':
        return original_rollout(head,raw,hidden,weight,basis,anchor,choose,capture=capture,soft_overrides=soft_overrides)
    assert variant in VARIANTS and not head.use_soft and soft_overrides is None
    hp,qp,hn,ids,_,outrows = head.prepare(raw,hidden,basis)
    buffer = raw.transpose(0,1).clone(memory_format=torch.contiguous_format)
    previous = anchor
    scores,tokens,history = [],[],[]
    for j in range(hidden.shape[1]):
        if variant == 'triton_write':
            coeff,dose = head.predict(hp[:,j] if hp is not None else None,
                head.feature_weight[previous.long()],qp[:,j],None)
        else:
            table = head._previous_tables[hidden.shape[0]]
            hard = table[previous.long()]
            coeff,dose = predict_encoded(head, hp[:,j] if hp is not None else None, hard, qp[:,j], fused_swiglu=variant in ('prev_lookup_swiglu','small_tile'))
        direction = project_candidates(coeff,outrows[:,j])
        corrected,_ = bf16_logit_update(raw[:,j].gather(-1,ids[:,j]),direction,dose,hn[:,j])
        z = buffer[j]
        write_candidates(z,ids[:,j],corrected)
        token = choose(z.detach(),j).long()
        tokens.append(token)
        if capture:
            scores.append(z);history.append(previous)
        previous = token
    return Rollout(torch.stack(scores,1) if capture else None,torch.stack(tokens,1),
                   torch.stack(history,1) if capture else None,[],())


def verify_real_block(model,n):
    variant = os.environ.get('HEAD_LATENCY_OPT_VARIANT','baseline')
    if variant == 'baseline' or model.head is None:
        return
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    inputs,q,result,fp64 = model.integrated_inputs[n]
    raw,hidden,anchor,positions,idx,temp,seeds,step = inputs
    length = raw.shape[1]
    reference_q = torch.empty_like(q)
    mappings = idx.reshape(n,length).T.contiguous()
    pos = (positions.reshape(n,length).T+1).contiguous()
    columns = step.reshape(n,length).T.contiguous()
    def choose(z,j):
        return gumbel_sample(z,mappings[j],temp,seeds,pos[j],apply_temperature=True,
            output_processed_logits=reference_q,output_processed_logits_col=columns[j],use_fp64=fp64)
    with torch.no_grad():
        previous_mode = getattr(model.head, '_small_tile_mode', 'baseline')
        previous_fusion = getattr(model.head,'_fusion_mode','baseline')
        model.head._fusion_mode='baseline'
        if variant == 'small_tile':
            from .small_tile import set_mode
            set_mode(model.head, 'baseline')
        try:
            reference = original_rollout(model.head,raw,hidden,model.lm_head.weight,model.token_basis,anchor,choose,capture=True)
        finally:
            if variant == 'small_tile':set_mode(model.head,previous_mode)
            model.head._fusion_mode=previous_fusion
    ids,scores,previous = result
    checks = dict(logits_bitwise_equal=torch.equal(scores,reference.logits),
        sampled_ids_bitwise_equal=torch.equal(ids,reference.tokens),
        actual_predecessors_bitwise_equal=torch.equal(previous,reference.predecessors),
        verifier_q_bitwise_equal=torch.equal(q[idx.long(),step.long()],reference_q[idx.long(),step.long()]))
    evidence = dict(variant=variant,structure_mode=getattr(model.head,'_fusion_mode','baseline'),batch_size=n,temperature=float(temp[idx.long()].max()),**checks)
    with (Path(os.environ['LOCAL_TRIAL'])/'head_implementation_parity.jsonl').open('a') as stream:
        stream.write(json.dumps(evidence)+'\n')
    assert all(checks.values()),evidence


def install():
    variant = os.environ.get('HEAD_LATENCY_OPT_VARIANT','baseline')
    assert variant in (*VARIANTS, 'baseline')
    if variant == 'baseline':
        return
    from repair.model import DualFeedbackRepair
    original_bind = DualFeedbackRepair.bind_features
    def bind(self,weight,basis):
        original_bind(self,weight,basis)
        prepare_variant(self,variant)
    DualFeedbackRepair.bind_features = bind
    return lambda *args,**kwargs: optimized_rollout(*args,**kwargs,variant=variant)
