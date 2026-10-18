"""CPU contracts: dtype, actual non-Top1 feedback, candidate support, and gradients."""
import torch
from Baseline_sampling import sample_block, sample_logits
from Baseline_models import DFlash2CandidateSelector as HEAD_CLASS
FAMILY='dflash2'

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    model=HEAD_CLASS(31,12,4,8) if FAMILY=='dflash2' else HEAD_CLASS(31,12,4)
    for p in model.parameters():torch.nn.init.normal_(p,std=.2)
    assert all(p.dtype==torch.bfloat16 for p in model.parameters())
    seen=[]
    for m in model.modules():
        if isinstance(m,torch.nn.Linear):
            m.register_forward_hook(lambda m,a,y:seen.append((a[0].dtype,m.weight.dtype,y.dtype)))
    raw=torch.randn(2,7,31).bfloat16();hidden=torch.randn(2,7,12).bfloat16();anchor=torch.tensor([2,5])
    forced=[]
    def force(scores,j):
        picked=scores.topk(2,-1).indices[:,1];forced.append(picked)
        return picked
    ids,scores,prev=sample_block(model,raw,hidden,anchor,sample_step=force)
    assert torch.equal(prev[:,0],anchor) and torch.equal(prev[:,1:],ids[:,:-1])
    assert torch.equal(ids,torch.stack(forced,1)) and bool(ids.ne(scores.argmax(-1)).all())
    assert all(a==b==c==torch.bfloat16 for a,b,c in seen)
    for j in range(7):
        if FAMILY=='dflash2':
            candidates,ref=model.inference(raw[:,j],hidden[:,j],prev[:,j])
            assert torch.equal(scores[:,j].gather(-1,candidates),ref)
            assert bool(torch.isfinite(scores[:,j]).sum(-1).eq(8).all())
        else:
            ref,_=model(raw[:,j],hidden[:,j],prev[:,j]);assert torch.equal(ref,scores[:,j])
    for t in (0.,1.,.7):
        gen=torch.Generator().manual_seed(17)
        chosen,z,previous=sample_block(model,raw,hidden,anchor,temperature=t,generator=gen)
        assert torch.equal(previous[:,1:],chosen[:,:-1])
        assert bool(torch.isfinite(z.gather(-1,chosen[:,:,None])).all())
        if t==0:assert torch.equal(chosen,z.argmax(-1))
        else:
            q=torch.softmax(z.float()/t,-1)
            assert q.dtype==torch.float32 and torch.allclose(q.sum(-1),torch.ones(2,7))
    # Loss normalization remains FP32; model forward and gradients remain BF16.
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    if FAMILY=='dflash2':
        cand,label=model.training_candidates(raw[:,0],raw[:,0].argmin(-1))
        y=model.score_candidates(raw[:,0],hidden[:,0],anchor,cand)
        loss=torch.nn.functional.cross_entropy(y.float(),label)
    else:
        y,c=model(raw[:,0],hidden[:,0],anchor)
        loss=torch.nn.functional.cross_entropy(y.float(),raw[:,0].argmin(-1))+c.float().square().mean()
    loss.backward();assert torch.isfinite(loss)
    assert all(p.grad is None or p.grad.dtype==torch.bfloat16 and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    print(FAMILY,'PASS: BF16 heads, FP32 probabilities/loss, forced non-Top1 feedback, T0/T1, support, gradients')

if __name__=='__main__':main()
