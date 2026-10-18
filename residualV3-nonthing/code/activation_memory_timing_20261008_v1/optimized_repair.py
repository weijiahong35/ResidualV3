"""Keep all 15 BF16 network calls; change only candidate storage/scatter."""
import torch
from runtime.precision import bf16_logit_update
from repair.model import project_candidates

class FrozenProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,coeff,ids,basis):
        assert not basis.requires_grad
        ctx.save_for_backward(ids,basis)
        return project_candidates(coeff,basis[ids])
    @staticmethod
    def backward(ctx,grad):
        ids,basis=ctx.saved_tensors
        rows=basis[ids]
        result=(rows.transpose(-1,-2)@grad.unsqueeze(-1)).squeeze(-1)
        return result,None,None

def logits(head,raw,hidden,basis,anchor,answers,*,recompute_projection=True):
    assert not head.use_soft and head.scale_mode=='hidden_norm'
    hp,qp,hn,ids,_,outrows=head.prepare(raw,hidden,basis,omit_output_rows=recompute_projection)
    selected=raw.gather(-1,ids);previous=anchor;corrections=[]
    for j in range(hidden.shape[1]):
        coeff,dose=head.predict(hp[:,j] if hp is not None else None,head.feature_weight[previous.long()],qp[:,j],None)
        direction=FrozenProjection.apply(coeff,ids[:,j],basis) if recompute_projection else project_candidates(coeff,outrows[:,j])
        corrected,_=bf16_logit_update(selected[:,j],direction,dose,hn[:,j])
        corrections.append(corrected)
        previous=answers[:,j].long()
    # Same BF16 corrected entries, and unchanged logits outside candidates.
    # Avoid 15 scatter results, the [N,T,V] stack, and SliceBackward zeros.
    return raw.scatter(-1,ids,torch.stack(corrections,1))
