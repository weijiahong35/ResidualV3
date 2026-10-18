"""Bounded actual-state trace + FP32 control for the rejected TF batching."""
import torch
from repair.model import DualFeedbackRepair,rollout
from parallel_tf import teacher_forced_logits


def diagnose(head,raw,hidden,output_basis,anchor,answers,architecture):
    wanted=['previous_proj.0','previous_proj.1','fusion.0','fusion.1',
            *[f'blocks.{i}.{s}' for i in range(architecture['depth']) for s in ['norm','up','down']],
            'output_norm','direction_head','dose_head']
    traces={};hooks=[]
    for name,module in head.named_modules():
        if name in wanted:
            def hook(module,args,value,name=name): traces.setdefault(name,[]).append(value.detach().clone())
            hooks.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            rollout(head,raw,hidden,head.feature_weight,output_basis,anchor,lambda z,j:answers[:,j])
            seq={k:torch.stack(v,1) for k,v in traces.items()};traces.clear()
            teacher_forced_logits(head,raw,hidden,output_basis,anchor,answers)
            batched={k:v[0] for k,v in traces.items()}
    finally:
        for hook in hooks:hook.remove()
    layers=[]
    for name in wanted:
        a,b=seq[name],batched[name]
        layers.append(dict(module=name,bitwise_agreement=float((a==b).float().mean()),
                           relative_l2=float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12)),
                           max_abs=float((a.float()-b.float()).abs().max())))
    with torch.no_grad():
        hp,qp,_,_,_,_=head.prepare(raw,hidden,output_basis)
        prev=head.feature_weight[torch.cat((anchor[:,None],answers[:,:-1]),1)]
    control=DualFeedbackRepair(**architecture).to(raw.device,dtype=torch.float32)
    control.load_state_dict(head.state_dict())
    hp,qp,prev=hp.detach().float(),qp.detach().float(),prev.detach().float()
    gen=torch.Generator(device=raw.device).manual_seed(919191)
    up=torch.randn((*hp.shape[:2],architecture['basis_rank']),device=raw.device,generator=gen)
    dose_up=torch.randn(hp.shape[:2],device=raw.device,generator=gen)
    def run(parallel):
        control.zero_grad(set_to_none=True)
        if parallel: a,b=control.predict(hp,prev,qp,None)
        else:
            parts=[control.predict(hp[:,j],prev[:,j],qp[:,j],None) for j in range(hp.shape[1])]
            a=torch.stack([x[0] for x in parts],1); b=torch.stack([x[1] for x in parts],1)
        ((a*up).sum()+(b*dose_up).sum()).backward()
        g=torch.cat([p.grad.flatten() for p in control.parameters() if p.grad is not None])
        return a.detach(),b.detach(),g
    a,da,ga=run(False); b,db,gb=run(True)
    fp32=dict(direction_relative_l2=float((a-b).norm()/a.norm()),
              gradient_relative_l2=float((ga-gb).norm()/ga.norm()),
              gradient_cosine=float(torch.nn.functional.cosine_similarity(ga[None],gb[None])))
    return dict(actual_state_layer_trace=layers,
        first_different_module=next((r['module'] for r in layers if r['bitwise_agreement']<1.0),None),
        fp32_prediction_network_control=fp32,
        scope='FP32 is a diagnostic control only; not applied to production model or loss training',
        interpretation='changed GEMM batching and BF16 accumulation change numerical results; native predecessor recursion is not parallelized')
