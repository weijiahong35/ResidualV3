"""FP32 optimizer master weights; model weights and forward remain BF16.

Tiny continuation updates must accumulate in FP32 rather than disappear when
an AdamW update is rounded separately into a BF16 parameter each step.
"""
import torch


class MasterAdamW:
    def __init__(self,head,stage1,dose,protocol):
        groups=[];self.pairs=[];names={id(p):name for name,p in head.named_parameters()}
        for parameters,lr in ((stage1,protocol['stage1_lr']),(dose,protocol['dose_lr'])):
            masters=[]
            for parameter in parameters:
                master=torch.nn.Parameter(parameter.detach().float().clone())
                masters.append(master);self.pairs.append((names[id(parameter)],parameter,master))
            groups.append(dict(params=masters,lr=lr,base_lr=lr))
        self.optimizer=torch.optim.AdamW(groups,weight_decay=protocol['weight_decay'],eps=1e-8)
        self.param_groups=self.optimizer.param_groups
        self.master_parameters=[master for _,_,master in self.pairs]

    def zero_grad(self,set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for _,parameter,_ in self.pairs:parameter.grad=None

    def accumulate(self):
        for _,parameter,master in self.pairs:
            if parameter.grad is None:continue
            grad=parameter.grad.detach().float()
            if master.grad is None:master.grad=grad.clone()
            else:master.grad.add_(grad)
            parameter.grad=None

    def gradient_sums(self):
        result={}
        for prefix in ('direction_head.','dose_head.','direction_confidence_head.'):
            amount=sum(float(master.grad.abs().sum()) for name,_,master in self.pairs
                       if name.startswith(prefix) and master.grad is not None)
            if not 0<amount<float('inf'):raise RuntimeError(f'no finite accumulated gradient for {prefix}')
            result[prefix.rstrip('.')]=amount
        return result

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        for _,parameter,master in self.pairs:parameter.copy_(master)
