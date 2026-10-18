"""Validate the inherited no-soft head at 15 positions before freezing sources."""
import argparse
import importlib.util
import json
import runpy
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runtime_bundle'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    out=ap.parse_args().output.resolve()
    import torch
    from repair.model import DualFeedbackRepair,rollout,output_loss
    from runtime.utils import dump
    torch.set_num_threads(1)
    old=HERE.parents[1]/'residualV3_main_v1/research/nosoft_depth_10k_recovery_v1/runtime_bundle/repair/model.py'
    spec=importlib.util.spec_from_file_location('original_nosoft3',old)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    cfg=dict(hidden_size=32,width=16,vocab_size=97,basis_rank=12,depth=3,q_topk=16,use_hidden=True,use_soft=False,scale_mode='hidden_norm',fixed_scale=None)
    torch.manual_seed(610006);head=DualFeedbackRepair(**cfg).to(dtype=torch.bfloat16)
    torch.manual_seed(610006);reference=original.DualFeedbackRepair(**cfg).to(dtype=torch.bfloat16)
    assert all(torch.equal(value,reference.state_dict()[name]) for name,value in head.state_dict().items())
    h=torch.randn(2,15,32,dtype=torch.bfloat16);w=torch.randn(97,32,dtype=torch.bfloat16)
    e=torch.randn_like(w);basis=torch.randn(97,12,dtype=torch.bfloat16);input_basis=torch.randn_like(basis)
    raw=h@w.T;gold=torch.randn_like(raw);anchor=torch.tensor([5,9]);answers=torch.randint(0,97,(2,15))
    for model in [head,reference]:model.bind_features(e,input_basis)
    actual=rollout(head,raw,h,w,basis,anchor,lambda z,j:answers[:,j])
    expected=original.rollout(reference,raw,h,w,basis,anchor,lambda z,j:answers[:,j])
    assert torch.equal(actual.logits,expected.logits)
    assert torch.equal(actual.predecessors,torch.cat((anchor[:,None],answers[:,:-1]),1))
    output_loss(actual.logits,gold).backward();original.output_loss(expected.logits,gold).backward()
    assert all(torch.equal(value.grad,dict(reference.named_parameters())[name].grad) for name,value in head.named_parameters())
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in head.parameters())
    ids=head.q_feature_ids(raw)
    support=torch.zeros_like(raw,dtype=torch.bool).scatter(-1,ids,True)
    assert torch.equal(actual.logits[~support],raw[~support])
    changed=answers.clone();changed[:,0]=(changed[:,0]+1)%97
    alternative=rollout(head,raw,h,w,basis,anchor,lambda z,j:changed[:,j])
    assert torch.equal(actual.logits[:,0],alternative.logits[:,0]) and not torch.equal(actual.logits[:,1],alternative.logits[:,1])
    fullcfg=json.loads((out/'head_config.json').read_text())
    parameter_count=sum(value.numel() for value in DualFeedbackRepair(**fullcfg).parameters())
    namespace=runpy.run_path(str(HERE/'evaluate.py'),run_name='request_contract_validation')
    p=json.loads((out/'prepared_protocol.json').read_text())
    sample=json.loads(Path(p['panel_paths']['final_C1']).read_text().splitlines()[0])
    request=namespace['_thinking_request']('local',sample,0,0,2096)
    assert request['messages']==sample['messages'] and request['chat_template_kwargs']=={'enable_thinking':False}
    result=dict(status='passed',positions=15,block_size=16,no_soft3_parameter_count=parameter_count,
        old_nosoft_head_initialization_exact=True,old_nosoft_logits_exact_at15_positions=True,
        old_nosoft_parameter_gradients_exact=True,hard_lag1_and_first_anchor_exact=True,
        outside_top256_logits_unchanged=True,nonthinking_request_contract=True,
        scope='CPU head/15-position recurrence and request checks; not complete backbone/native bitwise proof')
    dump(out/'cpu_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
