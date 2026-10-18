import math
import torch
from torch import nn

class MaskEmbedding(nn.Module):
    def __init__(self,table,mask_id,mask):
        super().__init__();self.register_buffer('table',table.detach(),persistent=False)
        self.mask_token_id=mask_id;self.mask_embedding=nn.Parameter(mask.detach().reshape(-1).clone())
    def forward(self,ids):
        return torch.where((ids==self.mask_token_id)[...,None],self.mask_embedding,self.table[ids])

class TrainablePair(nn.Module):
    def __init__(self,body,head):
        super().__init__();self.body=body;self.head=head

class MasterOptim:
    """FP32 master weights/moments; BF16 forward. Body Muon, heads AdamW."""
    def __init__(self,model,p):
        self.pairs=[];matrix=[];other_body=[];heads=[]
        for name,param in model.named_parameters():
            if not param.requires_grad:continue
            master=nn.Parameter(param.detach().float().clone());self.pairs.append((name,param,master))
            if name.startswith('head.'):heads.append(master)
            elif param.ndim==2 and min(param.shape)>1:matrix.append(master)
            else:other_body.append(master)
        self.optimizers=[]
        if matrix:self.optimizers.append(torch.optim.Muon(matrix,lr=p['body_muon_lr'],momentum=p['muon_momentum'],weight_decay=p['muon_weight_decay'],ns_steps=p['muon_ns_steps'],adjust_lr_fn=p['muon_adjust_lr_fn']))
        groups=[]
        if other_body:groups.append(dict(params=other_body,lr=p['body_adam_lr']))
        if heads:groups.append(dict(params=heads,lr=p['head_lr']))
        if groups:self.optimizers.append(torch.optim.AdamW(groups,weight_decay=p['weight_decay'],eps=1e-8,fused=True))
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:group['base_lr']=group['lr']
        self.master_parameters=[master for _,_,master in self.pairs]
    def zero_grad(self):
        for optimizer in self.optimizers:optimizer.zero_grad(set_to_none=True)
        for _,param,_ in self.pairs:param.grad=None
    def accumulate(self):
        for _,param,master in self.pairs:
            if param.grad is None:continue
            grad=param.grad.detach().float()
            if master.grad is None:master.grad=grad
            else:master.grad.add_(grad)
            param.grad=None
    @torch.no_grad()
    def step(self,scale):
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:group['lr']=group['base_lr']*scale
            optimizer.step()
        for _,param,master in self.pairs:param.copy_(master)
    def grad_summary(self):
        result={}
        for name,_,master in self.pairs:
            if master.grad is None:continue
            value=float(master.grad.float().norm())
            if not math.isfinite(value):raise FloatingPointError(name)
            prefix='head' if name.startswith('head.') else 'body'
            result[prefix]=result.get(prefix,0.)+value
            if prefix=='body':
                component='.'.join(name.split('.')[:3]) if name.startswith('body.layers.') else '.'.join(name.split('.')[:2])
                result[component]=result.get(component,0.)+value
        return result

def require_backbone_gradients(gradients,arm):
    for name in ['body','body.fc',*[f'body.layers.{index}' for index in range(5)]]:
        assert gradients.get(name,0)>0,f'No live gradient for {name}'
    if arm!='dflash_body':assert gradients.get('head',0)>0,'Head is frozen or disconnected'

def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {key:cpu_tree(item) for key,item in value.items()}
    if isinstance(value,list):return [cpu_tree(item) for item in value]
    if isinstance(value,tuple):return tuple(cpu_tree(item) for item in value)
    return value
