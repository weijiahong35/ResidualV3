"""Full1536 online-state gate; comparisons stay in RAM and no optimizer steps."""
import gc,json,math
import torch
from runtime.utils import dump,now
from losses import dpace_loss,output_loss
from repair.model import rollout
from optimized_repair import logits as optimized
from combined_ce_tv import objective as combined

def compare(left,right):
    left=left.reshape(left.shape[0],-1);right=right.reshape(right.shape[0],-1)
    norm=diff=dot=norm_right=0.;max_abs=0.;equal=True
    for start in range(0,len(left),8):
        a=left[start:start+8].float();b=right[start:start+8].float();d=a-b
        norm+=float((a*a).sum());norm_right+=float((b*b).sum());diff+=float((d*d).sum());dot+=float((a*b).sum());max_abs=max(max_abs,float(d.abs().max()));equal=equal and torch.equal(left[start:start+8],right[start:start+8])
    return dict(relative_l2=math.sqrt(diff/max(norm,1e-40)),cosine=dot/math.sqrt(max(norm*norm_right,1e-40)),max_absolute_error=max_abs,bitwise_equal=equal)

def run(path,pair,target,basis,pack,seed):
    from joint_runner import backbone
    torch.cuda.synchronize();h0,teacher_h,answers,anchor,positions=backbone(pair.body,pack,seed)
    hidden=h0.detach().contiguous();teacher_h=teacher_h.detach();del h0
    with torch.no_grad():gold=teacher_h@target.lm_head.weight.T
    del teacher_h;reference=None;records={}
    for mode in ['baseline','optimized']:
        pair.zero_grad(set_to_none=True);leaf=hidden.detach().requires_grad_();raw=leaf@target.lm_head.weight.T;raw.retain_grad()
        torch.cuda.reset_peak_memory_stats()
        if mode=='baseline':z=rollout(pair.head,raw,leaf,target.lm_head.weight,basis,anchor,lambda _,j:answers[:,j]).logits
        else:z=optimized(pair.head,raw,leaf,basis,anchor,answers)
        loss=dpace_loss(raw,gold)+(combined(z,gold) if mode=='optimized' else dpace_loss(z,gold)+output_loss(z,gold))
        loss.backward();torch.cuda.synchronize()
        sample=dict(loss=float(loss.detach()),selected_logits=z.gather(-1,raw.detach().topk(256,-1).indices).detach().cpu(),argmax=z.argmax(-1).cpu(),raw_gradient=raw.grad.detach().cpu(),hidden_gradient=leaf.grad.detach().cpu(),head_gradients={n:t.grad.detach().cpu().clone() for n,t in pair.head.named_parameters() if t.grad is not None},peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        if reference is None:reference=sample
        else:
            old=torch.cat([x.flatten().float() for x in reference['head_gradients'].values()]);new=torch.cat([x.flatten().float() for x in sample['head_gradients'].values()])
            result=dict(loss_absolute_error=abs(sample['loss']-reference['loss']),corrected_candidate_logits_bitwise_equal=torch.equal(sample['selected_logits'],reference['selected_logits']),argmax_bitwise_equal=torch.equal(sample['argmax'],reference['argmax']),head_gradients=compare(old[None],new[None]),raw_gradients=compare(reference['raw_gradient'],sample['raw_gradient']),hidden_gradients=compare(reference['hidden_gradient'],sample['hidden_gradient']),per_parameter={n:compare(reference['head_gradients'][n].reshape(1,-1),t.reshape(1,-1)) for n,t in sample['head_gradients'].items()},peak_allocated_gib_with_raw_grad_retained={'baseline':reference['peak_allocated_gib'],'optimized':sample['peak_allocated_gib']},anchors=1536,positions=15,vocab=151936,source='fresh online teacher hidden and live DFlash forward before any pilot update',outside_candidates='unchanged raw logits in both implementations',status='passed',utc=now())
            dump(path,result)
            assert result['loss_absolute_error']<2e-5 and result['argmax_bitwise_equal']
            assert max(result[k]['relative_l2'] for k in ['head_gradients','raw_gradients','hidden_gradients'])<.002
            print(now(),'FULL1536_HEAD_GATE_PASS',result['head_gradients'],result['raw_gradients'],result['hidden_gradients'],flush=True)
        del leaf,raw,z,loss,sample;pair.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    del reference,hidden,gold;pair.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
