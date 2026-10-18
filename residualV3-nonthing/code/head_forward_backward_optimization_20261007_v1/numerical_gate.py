"""Bounded synthetic gradient gate: combined loss against eager reference."""
import argparse,json,sys
from pathlib import Path
import torch
from losses import dpace_loss,output_loss,eager_objective

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(99107);torch.cuda.set_device(0)
    raw=torch.randn(2,15,151936,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    repaired=(raw.detach()+.1*torch.randn_like(raw)).requires_grad_(True)
    teacher=torch.randn_like(raw)
    teacher.scatter_(-1,torch.randint(151936,(2,15,1),device='cuda'),8.)
    value=dpace_loss(raw,teacher)+dpace_loss(repaired,teacher)+output_loss(repaired,teacher)
    grads=torch.autograd.grad(value,(raw,repaired));refs=(raw.detach().clone().requires_grad_(True),repaired.detach().clone().requires_grad_(True))
    reference=eager_objective(refs[0],teacher,'CE_DPACE')+eager_objective(refs[1],teacher,'CE_DPACE_OUTPUT')
    reference_grads=torch.autograd.grad(reference,refs)
    metrics=[]
    for g,h in zip(grads,reference_grads):
        diff=(g.float()-h.float());relative=float(diff.norm()/h.float().norm().clamp_min(1e-20));cos=float(torch.nn.functional.cosine_similarity(g.float().flatten(),h.float().flatten(),dim=0))
        metrics.append(dict(relative_gradient_l2=relative,cosine=cos,max_absolute_gradient_error=float(diff.abs().max())))
        assert relative<.002 and cos>.99999
    result=dict(status='passed',full_vocabulary=151936,positions=15,anchors=2,body_raw_loss='CE with detached D-PACE prefix weights',head_repaired_loss='CE with detached D-PACE prefix weights plus OUTPUT',body_weight=1.,head_ce_dpace_weight=1.,head_output_weight=1.,value=float(value),eager_value=float(reference),absolute_value_error=float((value-reference).abs()),gradients=metrics,bf16_large_tensors=True,fp32_reductions=True,scope='loss and logit gradient gate, not full-backbone bitwise proof')
    assert result['absolute_value_error']<2e-5
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
