"""Branch interventions, absence checks, recurrence and eager/Graph equivalence."""
import argparse,importlib.util
from pathlib import Path
from unittest.mock import patch
import torch
from .model import DualFeedbackRepair,rollout,output_loss
from .protocol import read
from runtime.utils import dump,sha,ROOT


def changed_prob(q,ids):
    selected=q.gather(-1,ids)
    return q.clone().scatter(-1,ids,.9*selected+.1*selected.roll(1,-1))


@torch.no_grad()
def input_effects(head,raw,hidden,previous,basis,soft_logits,soft_ids):
    q=raw.float().softmax(-1);r=soft_logits.float().softmax(-1)
    hp,qp,hn,ids,_,_=head.prepare(raw,hidden,basis,q_override=q)
    sp=head.distribution_part(r,soft_ids) if head.use_soft else None
    def correction(h=hp,p=previous,c=qp,s=sp):return head.correction(h,p,c,s,hn,basis,ids)
    delta,coeff,dose=correction()
    interventions={
        'current_q':correction(c=head.distribution_part(changed_prob(q,ids),ids)),
        'hard_previous':correction(p=previous.roll(1,-1)),
        'hidden':correction(h=head.hidden_proj(hidden.roll(1,-1)) if head.use_hidden else None),
        'soft_previous':correction(s=head.distribution_part(changed_prob(r,soft_ids),soft_ids) if head.use_soft else None)}
    changes={k:float((delta.float()-v[0].float()).abs().max()) for k,v in interventions.items()}
    enabled={'current_q':True,'hard_previous':True,'hidden':head.use_hidden,'soft_previous':head.use_soft}
    for k,v in changes.items():assert (0<v<float('inf')) if enabled[k] else v==0,(k,v,enabled)
    return dict(passed=True,delta_max_abs=changes,enabled=enabled,residual_u_fixed=True,output_basis_fixed=True,
                amplitude_factor_fixed=True,current_candidate_ids_fixed=True,soft_candidate_ids_fixed=True,
                q_direction_max_abs=float((coeff.float()-interventions['current_q'][1].float()).abs().max()),
                q_dose_max_abs=float((dose.float()-interventions['current_q'][2].float()).abs().max()),
                removed_hidden_module_absent=head.hidden_proj is None,scale_mode=head.scale_mode)


def input_gradients(head,raw,hidden,previous,basis,soft_logits,soft_ids):
    head.zero_grad(set_to_none=True)
    q=raw.float().softmax(-1).detach().requires_grad_(True)
    r=soft_logits.float().softmax(-1).detach().requires_grad_(True)
    h=hidden.detach().requires_grad_(True);prev=previous.detach().requires_grad_(True)
    hp,qp,hn,ids,_,_=head.prepare(raw.detach(),h,basis,q_override=q)
    sp=head.distribution_part(r,soft_ids) if head.use_soft else None
    delta,_,_=head.correction(hp,prev,qp,sp,hn.detach(),basis,ids)
    (delta.float()*torch.linspace(-1,1,delta.shape[-1],device=delta.device)).sum().backward()
    inputs={k:float(v.grad.float().abs().sum()) if v.grad is not None else 0. for k,v in (('current_q',q),('soft_previous',r),('hidden',h),('hard_previous',prev))}
    enabled={'current_q':True,'hard_previous':True,'hidden':head.use_hidden,'soft_previous':head.use_soft}
    for k,v in inputs.items():assert (0<v<float('inf')) if enabled[k] else v==0,(k,v)
    branches={n:float(v.grad.float().abs().sum()) for n,v in head.named_parameters() if n.startswith(('q_proj.','dose_head.'))}
    fusion={k:float(head.fusion[1].weight.grad[:,i*head.width:(i+1)*head.width].float().abs().sum()) for i,k in enumerate(head.branch_names)}
    assert all(0<v<float('inf') for v in [*branches.values(),*fusion.values()])
    head.zero_grad(set_to_none=True)
    return dict(passed=True,input_gradient_l1=inputs,distribution_and_dose_parameter_gradients=branches,fusion_column_gradients=fusion,amplitude_factor_detached=True)


def real_checks(head,raw,hidden,weight,basis,anchor,choose):
    with torch.no_grad():
        ref=rollout(head,raw,hidden,weight,basis,anchor,choose)
        args=(head,raw[:,1],hidden[:,1],head.feature_weight[ref.predecessors[:,1]],basis,ref.logits[:,0],head.q_feature_ids(raw[:,0]))
        effects=input_effects(*args)
    return effects,input_gradients(*args)


def check(device,p):
    from .run import fresh_head
    torch.set_num_threads(4);torch.manual_seed(71)
    with patch.object(DualFeedbackRepair,'load_state_dict',side_effect=AssertionError('checkpoint forbidden')):
        a=fresh_head(p);b=fresh_head(p);c=fresh_head(p,smoke=True)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert not torch.equal(a.q_proj[1].weight,c.q_proj[1].weight)
    counts=dict(parameter_count=sum(v.numel() for v in a.parameters()),dose_parameters=sum(v.numel() for v in a.dose_head.parameters()))
    depth=p['architecture']['depth'];assert len(a.blocks)==depth
    assert len({x.up.weight.data_ptr() for x in a.blocks})==depth
    assert not any(any(w in n for w in ('risk','scale','confidence','position','prefix','successor','dose_grid')) for n in a.state_dict())
    del a,b,c
    hidden=torch.randn(2,7,32,device=device,dtype=torch.bfloat16)
    weight=torch.randn(97,32,device=device,dtype=torch.bfloat16);embedding=torch.randn_like(weight)
    b=torch.randn(12,32,device=device,dtype=torch.bfloat16)
    basis=(weight@b.T).detach();ebasis=(embedding@b.T).detach()
    raw=hidden@weight.T;anchor=torch.tensor([95,96],device=device);answers=torch.randint(97,(2,7),device=device)
    options={k:p['architecture'][k] for k in ('depth','use_hidden','use_soft','scale_mode','fixed_scale')}
    # Exercise fixed-scale logic at a small numerical fixture's natural scale.
    if options['scale_mode']=='fixed':options['fixed_scale']=5.5
    torch.manual_seed(82)
    head=DualFeedbackRepair(hidden_size=32,width=16,vocab_size=97,basis_rank=12,q_topk=16,**options).to(device,dtype=torch.bfloat16)
    head.bind_features(embedding,ebasis)
    assert 'feature_weight' not in head.state_dict() and 'feature_basis' not in head.state_dict()
    def fn():return rollout(head,raw,hidden,weight,basis,anchor,lambda z,j:answers[:,j])
    with patch.object(head,'q_feature_ids',wraps=head.q_feature_ids) as select,patch.object(head,'distribution_part',wraps=head.distribution_part) as encode:
        result=fn();assert select.call_count==1
        assert encode.call_count==(7 if head.use_soft else 1),encode.call_count
    prev=torch.cat([anchor[:,None],answers[:,:-1]],1);assert torch.equal(result.predecessors,prev)
    effects,gradients=real_checks(head,raw,hidden,weight,basis,anchor,lambda z,j:answers[:,j])
    hp,qp,hn,ids,rows,_=head.prepare(raw,hidden,basis)
    if head.scale_mode=='fixed':assert torch.equal(hn,torch.full_like(hn,head.fixed_scale))
    if not head.use_hidden:assert head.hidden_proj is None and hp is None
    soft=head.q_proj(head.feature_basis[anchor]) if head.use_soft else None;expected=[]
    for j in range(7):
        coeff,dose=head.predict(hp[:,j] if hp is not None else None,head.feature_weight[prev[:,j]],qp[:,j],soft)
        direction=(basis[None].float()*coeff[:,None].float()).sum(-1).to(raw.dtype)
        full=raw[:,j]+(dose*hn[:,j])[:,None]*direction
        z=raw[:,j].scatter(-1,ids[:,j],full.gather(-1,ids[:,j]));expected.append(z)
        if head.use_soft and j<6:
            probabilities=result.logits[:,j].float().softmax(-1).gather(-1,ids[:,j]).to(rows.dtype)
            moment=(probabilities.unsqueeze(-1).float()*rows[:,j].float()).sum(-2).to(rows.dtype)
            soft=head.q_proj(moment)
    torch.testing.assert_close(result.logits,torch.stack(expected,1),atol=.125,rtol=.02)
    support=torch.zeros_like(raw,dtype=torch.bool).scatter(-1,ids,True)
    assert torch.equal(result.logits[~support],raw[~support])
    outside_ids=(~support).float().argmax(-1)
    outside=rollout(head,raw,hidden,weight,basis,anchor,lambda z,j:outside_ids[:,j])
    assert torch.equal(outside.predecessors[:,1:],outside_ids[:,:-1])
    changed=answers.clone();changed[:,0]=(changed[:,0]+1)%97
    other=rollout(head,raw,hidden,weight,basis,anchor,lambda z,j:changed[:,j])
    assert torch.equal(result.logits[:,0],other.logits[:,0]) and not torch.equal(result.logits[:,1],other.logits[:,1])
    for tensor in (hidden,raw):
        saved=tensor.clone();tensor[:,5]=tensor[:,5].roll(1,-1)
        assert torch.equal(result.logits[:,:5],fn().logits[:,:5]),'future information leaked'
        tensor.copy_(saved)
    prevprob=result.logits[:,0].detach().float().softmax(-1)
    softother=rollout(head,raw,hidden,weight,basis,anchor,lambda z,j:answers[:,j],soft_overrides={1:(changed_prob(prevprob,ids[:,0]),ids[:,0])})
    assert torch.equal(result.logits[:,0],softother.logits[:,0])
    assert (not torch.equal(result.logits[:,1],softother.logits[:,1])) if head.use_soft else torch.equal(result.logits,softother.logits)
    head.zero_grad(set_to_none=True);loss=output_loss(fn().logits,torch.randn_like(raw));loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters())
    head.zero_grad(set_to_none=True)
    rg=raw.detach().requires_grad_(True)
    rr=rollout(head,rg,hidden,weight,basis,anchor,lambda z,j:answers[:,j])
    g=torch.autograd.grad(rr.logits[:,1].float().sum(),rg)[0][:,0]
    assert torch.isfinite(g).all()
    assert (g.abs().sum()>0) if head.use_soft else (g.abs().sum()==0)
    if device=='cuda':
        head.eval().requires_grad_(False)
        for temp in (0,1):
            noise=-torch.log(-torch.log(torch.rand(raw.shape,device=device).clamp(1e-6,1-1e-6))) if temp else torch.zeros_like(raw).float()
            def generated():return rollout(head,raw,hidden,weight,basis,anchor,lambda z,j:(z.float()+noise[:,j]).argmax(-1))
            with torch.no_grad():
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):generated()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):captured=generated()
                for _ in range(2):
                    raw.copy_(raw.roll(1,-1));anchor.add_(1).remainder_(97)
                    graph.replay();torch.cuda.synchronize();eager=generated()
                    assert torch.equal(captured.logits,eager.logits) and torch.equal(captured.tokens,eager.tokens) and torch.equal(captured.predecessors,eager.predecessors)
                del graph,captured
    return dict(**counts,input_effects=effects,input_gradients=gradients,blocks=depth,seed_reproducible=True,smoke_seed_distinct=True,
        topk_selected_once=True,outside_candidate_logits_unchanged=True,outside_candidate_feedback_supported=True,
        no_future_input=True,independent_output_reference=True,all_parameters_receive_output_gradients=True,
        soft_recurrence_differentiable=head.use_soft,removed_soft_computation_absent=not head.use_soft,graph_changed_inputs_exact=device=='cuda')


def baseline_parity():
    """A is exactly the original EMBEDDING math, including parameter gradients."""
    spec=importlib.util.spec_from_file_location('original_model',ROOT.parent/'repair/model.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    cfg=dict(hidden_size=32,width=16,vocab_size=97,basis_rank=12,q_topk=16)
    torch.manual_seed(82);a=DualFeedbackRepair(**cfg).to(dtype=torch.bfloat16)
    torch.manual_seed(82);b=original.DualFeedbackRepair(**cfg).to(dtype=torch.bfloat16)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    torch.manual_seed(21)
    h=torch.randn(2,7,32,dtype=torch.bfloat16);w=torch.randn(97,32,dtype=torch.bfloat16);e=torch.randn_like(w)
    basis=torch.randn(97,12,dtype=torch.bfloat16);ebasis=torch.randn_like(basis)
    anchor=torch.tensor([1,2]);tokens=torch.randint(97,(2,7));gold=torch.randn(2,7,97,dtype=torch.bfloat16)
    for head in (a,b):head.bind_features(e,ebasis)
    x=rollout(a,h@w.T,h,w,basis,anchor,lambda z,j:tokens[:,j])
    y=original.rollout(b,h@w.T,h,w,basis,anchor,lambda z,j:tokens[:,j])
    assert torch.equal(x.logits,y.logits) and torch.equal(x.predecessors,y.predecessors)
    output_loss(x.logits,gold).backward();original.output_loss(y.logits,gold).backward()
    assert all(torch.equal(v.grad,dict(b.named_parameters())[n].grad) for n,v in a.named_parameters())
    return dict(passed=True,initialization_exact=True,logits_exact=True,parameter_gradients_exact=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=ap.parse_args()
    p=read(a.protocol);result=check(a.device,p)
    dump(a.output,dict(status='passed',device=a.device,protocol_sha256=sha(a.protocol),**result));print(result,flush=True)
if __name__=='__main__':main()
