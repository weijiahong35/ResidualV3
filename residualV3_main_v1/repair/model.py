"""Four-input hard/soft lag-1 repair. Input tables differ; output LM geometry does not."""
import math
import torch
from torch import nn
from runtime.components import SwiGLUBlock
from runtime.components import normalized_rows
from runtime.precision import hidden_norm_bf16,bf16_logit_update
from runtime.components import Rollout

ARMS=('LMHEAD','EMBEDDING')

class DualFeedbackRepair(nn.Module):
    def __init__(self,hidden_size=4096,width=256,vocab_size=151936,basis_rank=512,depth=3,q_topk=256,dose_max=.8,dose_init=.01):
        super().__init__()
        assert depth==3 and 0<dose_init<dose_max
        self.hidden_size,self.width,self.vocab_size,self.basis_rank=hidden_size,width,vocab_size,basis_rank
        self.q_topk=min(q_topk,vocab_size);self.eps=1e-6;self.dose_max=dose_max
        self.hidden_proj=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,width))
        self.previous_proj=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,width))
        # Shared distribution encoder: input semantics distinguished by fusion columns.
        self.q_proj=nn.Sequential(nn.LayerNorm(basis_rank),nn.Linear(basis_rank,width))
        self.fusion=nn.Sequential(nn.LayerNorm(4*width),nn.Linear(4*width,width))
        self.blocks=nn.ModuleList([SwiGLUBlock(width,expansion=3) for _ in range(depth)])
        self.output_norm=nn.LayerNorm(width)
        self.direction_head=nn.Linear(width,basis_rank)
        self.dose_head=nn.Linear(width,1)
        nn.init.normal_(self.direction_head.weight,std=.01);nn.init.zeros_(self.direction_head.bias)
        nn.init.normal_(self.dose_head.weight,std=.01)
        nn.init.constant_(self.dose_head.bias,math.log(dose_init/(dose_max-dose_init)))

    def bind_features(self,weight,basis):
        assert weight.shape==(self.vocab_size,self.hidden_size) and basis.shape==(self.vocab_size,self.basis_rank)
        assert not weight.requires_grad and not basis.requires_grad
        # Frozen pretrained tensors are NOT repair parameters or saved checkpoint tensors.
        self.register_buffer('feature_weight',weight.detach(),persistent=False)
        self.register_buffer('feature_basis',basis.detach(),persistent=False)

    def q_feature_ids(self,raw):return raw.topk(self.q_topk,dim=-1).indices

    def distribution_part(self,probabilities,ids,*,rows=None):
        if rows is None:rows=self.feature_basis[ids]
        values=probabilities.gather(-1,ids).to(rows.dtype)
        return self.q_proj((values.unsqueeze(-2)@rows).squeeze(-2))

    def prepare(self,raw,hidden,output_basis,*,q_override=None,ids_override=None):
        q=raw.float().softmax(-1) if q_override is None else q_override
        ids=self.q_feature_ids(raw) if ids_override is None else ids_override
        rows=self.feature_basis[ids]
        # Same vocabulary IDs for both geometries. LMHEAD shares the very same gathered rows.
        outrows=rows if self.feature_basis.data_ptr()==output_basis.data_ptr() else output_basis[ids]
        return self.hidden_proj(hidden),self.distribution_part(q,ids,rows=rows),hidden_norm_bf16(hidden,self.eps),ids,rows,outrows

    def predict(self,hp,previous_row,qp,soft_part):
        hard=self.previous_proj(normalized_rows(previous_row,eps=self.eps))
        x=self.fusion(torch.cat((hp,hard,qp,soft_part),-1))
        for block in self.blocks:x=block(x)
        latent=self.output_norm(x)
        direction=normalized_rows(self.direction_head(latent),eps=self.eps)
        dose=(self.dose_max*torch.sigmoid(self.dose_head(latent).squeeze(-1).float())).to(latent.dtype)
        return direction,dose

    def correction(self,hp,previous_row,qp,soft_part,hn,output_basis,ids):
        coeff,dose=self.predict(hp,previous_row,qp,soft_part)
        selected=(dose*hn).unsqueeze(-1)*project_candidates(coeff,output_basis[ids])
        full=selected.new_zeros((*selected.shape[:-1],self.vocab_size)).scatter(-1,ids,selected)
        return full,coeff,dose


def project_candidates(coeff,rows):return (rows@coeff.unsqueeze(-1)).squeeze(-1)


def rollout(head,raw,hidden,weight,basis,anchor,choose,*,capture=True,soft_overrides=None):
    # weight is the OUTPUT LM-head in BOTH arms. Frozen input tables live on head.
    hp,qp,hn,ids,rows,outrows=head.prepare(raw,hidden,basis)
    previous=anchor
    # Known anchor is a one-hot distribution; exact moment, without a full V allocation.
    soft=head.q_proj(head.feature_basis[anchor.long()])
    scores,tokens,history=[],[],[]
    for j in range(hidden.shape[1]):
        # Test-only independent intervention; runtime always follows the recurrence below.
        if soft_overrides is not None and j in soft_overrides:
            probabilities,soft_ids=soft_overrides[j]
            soft=head.distribution_part(probabilities,soft_ids)
        coeff,dose=head.predict(hp[:,j],head.feature_weight[previous.long()],qp[:,j],soft)
        direction=project_candidates(coeff,outrows[:,j])
        corrected,_=bf16_logit_update(raw[:,j].gather(-1,ids[:,j]),direction,dose,hn[:,j])
        z=raw[:,j].scatter(-1,ids[:,j],corrected)
        token=choose(z.detach(),j).long()
        tokens.append(token)
        if capture:scores.append(z);history.append(previous)
        previous=token
        if j+1<hidden.shape[1]:
            # Fixed feature temperature 1, independent of sampling temperature/seed.
            # Differentiable seven-step recurrence; no target probabilities enter this path.
            soft=head.distribution_part(z.float().softmax(-1),ids[:,j],rows=rows[:,j])
    return Rollout(torch.stack(scores,1) if capture else None,torch.stack(tokens,1),
                   torch.stack(history,1) if capture else None,[],())


def output_loss(z,gold):
    p=gold.float().softmax(-1);r=z.float().log_softmax(-1).exp()
    return 1-torch.minimum(p,r).sum(-1).clamp(0,1).cumprod(-1).mean()
