"""Same-layout BF16 gradient gates; no long training, native changes or caches."""
import argparse,json,os,sys,time,gc
from pathlib import Path
C=Path(__file__).resolve().parent;sys.path[:0]=[str(C),str(C.parent/'runtime_bundle'),str(C.parent/'runtime_bundle/src'),str(C.parent/'runtime_bundle/hs_connectors/src')]
import torch
from runtime.utils import dump,now
from losses import dpace_loss,output_loss
from combined_ce_tv import objective as combined
from optimized_repair import logits as optimized
from repair.model import DualFeedbackRepair,rollout
from runtime.vllm_hidden import load_target_components
from runtime.precision import build_token_basis
from safetensors.torch import load_file
import joint_features

def compare(left,right):
    l=left.float().flatten();r=right.float().flatten();d=l-r
    den=float(l.norm());return dict(relative_l2=float(d.norm())/max(den,1e-20),max_absolute_error=float(d.abs().max()),bitwise_equal=torch.equal(left,right),cosine=float(torch.nn.functional.cosine_similarity(l[None],r[None],dim=1)) if den else 1.)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(8507)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=json.loads((root/'baseline/protocol.json').read_text());result={}
    raw=torch.randn(8,15,151936,device='cuda',dtype=torch.bfloat16).requires_grad_();gold=(raw.detach()+torch.randn_like(raw)*.25).contiguous()
    old=dpace_loss(raw,gold)+output_loss(raw,gold);g,=torch.autograd.grad(old,raw)
    new=combined(raw,gold);ng,=torch.autograd.grad(new,raw)
    gate=dict(loss_absolute_error=float((old-new).abs()),gradients=compare(g,ng));result['combined_ce_tv']=gate
    dump(root/'numeric_gate_partial.json',result);print(now(),'COMBINED_LOSS_GATE',gate,flush=True)
    assert gate['loss_absolute_error']<2e-6 and gate['gradients']['relative_l2']<.002
    del raw,gold,old,new,g,ng;gc.collect();torch.cuda.empty_cache()
    endpoint=json.loads((root/'teacher/endpoint.json').read_text())
    target=load_target_components(p['target_model'],torch.device('cuda'),endpoint['endpoint'],endpoint['root']);target.hidden_provider.close()
    b=load_file(p['basis_path'])['repair_basis'].to('cuda',dtype=torch.bfloat16)
    basis=build_token_basis(target.lm_head.weight,b);feature_basis=build_token_basis(target.model.embed_tokens.weight,b);del b
    torch.manual_seed(42);head=DualFeedbackRepair(**json.loads(Path(p['head_config_path']).read_text())).to('cuda',dtype=torch.bfloat16)
    head.bind_features(target.model.embed_tokens.weight,feature_basis);joint_features.install(head)
    hidden=torch.randn(16,15,4096,device='cuda',dtype=torch.bfloat16).contiguous();raw=(hidden@target.lm_head.weight.T).detach();gold=(raw+torch.randn_like(raw)*.25).contiguous()
    answers=gold.argmax(-1);anchor=torch.randint(151936,(16,),device='cuda')
    result['variants']={};reference=None
    for mode in ['baseline','scatter_only','recompute_candidates','all_fused']:
        head.zero_grad(set_to_none=True);x=raw.detach().clone().requires_grad_();h=hidden.detach().clone().requires_grad_()
        if mode=='baseline':z=rollout(head,x,h,target.lm_head.weight,basis,anchor,lambda _,j:answers[:,j]).logits
        else:z=optimized(head,x,h,basis,anchor,answers,recompute_projection=mode!='scatter_only')
        value=dpace_loss(x,gold)+(combined(z,gold) if mode=='all_fused' else dpace_loss(z,gold)+output_loss(z,gold));value.backward()
        values={name:param.grad.detach().cpu() for name,param in head.named_parameters() if param.grad is not None}
        sample=dict(loss=float(value.detach()),logits=z.detach().cpu(),raw_gradient=x.grad.detach().cpu(),hidden_gradient=h.grad.detach().cpu(),head_gradients=values)
        if reference is None:reference=sample
        else:
            per={n:compare(reference['head_gradients'][n],v) for n,v in values.items()}
            left=torch.cat([v.float().flatten() for v in reference['head_gradients'].values()]);right=torch.cat([v.float().flatten() for v in values.values()])
            gate=dict(loss_absolute_error=abs(sample['loss']-reference['loss']),logits_bitwise_equal=torch.equal(reference['logits'],sample['logits']),argmax_agreement=float((reference['logits'].argmax(-1)==sample['logits'].argmax(-1)).float().mean()),raw_gradient=compare(reference['raw_gradient'],sample['raw_gradient']),hidden_gradient=compare(reference['hidden_gradient'],sample['hidden_gradient']),head_gradient=compare(left,right),per_parameter=per)
            result['variants'][mode]=gate;dump(root/'numeric_gate_partial.json',result)
            print(now(),'REPAIR_GATE',mode,'loss',gate['loss_absolute_error'],'logits bitwise',gate['logits_bitwise_equal'],'head relative',gate['head_gradient']['relative_l2'],'raw relative',gate['raw_gradient']['relative_l2'],'hidden relative',gate['hidden_gradient']['relative_l2'],flush=True)
            assert gate['loss_absolute_error']<2e-5 and gate['argmax_agreement']==1.
            assert max(gate[k]['relative_l2'] for k in ['head_gradient','raw_gradient','hidden_gradient'])<.002
        del x,h,z,value,sample;gc.collect();torch.cuda.empty_cache()
    result.update(status='passed',anchors=16,positions=15,full_vocabulary=151936,unchanged_sequential_network_shapes=True,scope='small synthetic gate using real frozen embedding/LM-head/basis; full1536 online-state gate still required',utc=now())
    dump(root/'numeric_gate.json',result)

if __name__=='__main__':main()
