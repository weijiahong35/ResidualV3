"""Two detached online readouts; transparent restore of the original q45 state."""
import copy
import torch
from torch import nn
from q45_predictor import Auxiliary as LegacyAuxiliary

class Readout(nn.Module):
    def __init__(self, dimension, device):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, 32), nn.SiLU(), nn.Linear(32, 2)).to(device)
        # Nonzero first layer; zero last layer, equal initial predictions.
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x):
        # BF16 matrix products; FP32 parameters, normalization, BCE and Adam.
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            return self.net(x.detach()).float()

class TwoReadouts:
    def __init__(self, device='cuda', lr=.003):
        self.device = device; self.lr = lr
        devices = [torch.cuda.current_device()] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(20261010)
            self.model = nn.ModuleDict({
                'Q45_MLP': Readout(45, device),
                'Q45_REPAIR_LATENT_MLP': Readout(301, device),
            })
        self.optimizers = {k: torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=.001)
                           for k, m in self.model.items()}
        self.mean = torch.zeros(301, device=device)
        self.scale = torch.ones(301, device=device)
        self.initialized = False; self.steps = 0

    def train(self, x, labels):
        x = x.detach().float().reshape(-1, 301)
        labels = labels.detach().float().reshape(-1, 2)
        if not self.initialized:
            self.mean.copy_(x.mean(0)); self.scale.copy_(x.std(0, unbiased=False).clamp_min(.01))
            self.initialized = True
        z = ((x-self.mean)/self.scale).detach()
        result = {}; losses = []
        for name, model in self.model.items():
            optimizer = self.optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            predictions = model(z[:, :45] if name == 'Q45_MLP' else z)
            per_output = torch.nn.functional.binary_cross_entropy_with_logits(predictions, labels, reduction='none').mean(0)
            loss = per_output.mean(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
            result[name] = dict(loss=float(loss.detach()), T0_loss=float(per_output[0].detach()),
                               T1_loss=float(per_output[1].detach()), gradient_norm=float(norm))
        self.steps += 1
        return dict(loss=sum(losses)/len(losses), steps=self.steps, branches=result)

    def state_dict(self):
        return dict(format='two_detached_online_MLPs_v1', model=self.model.state_dict(),
            optimizers={k:o.state_dict() for k,o in self.optimizers.items()}, mean=self.mean,
            scale=self.scale, initialized=self.initialized, steps=self.steps,
            feature_count=301, q_feature_count=45, repair_latent_dimension=256,
            input='post_repair_q_plus_output_norm_latent', output_labels=['T0_argmax_agreement','T1_overlap'],
            parameter_counts={k:sum(p.numel() for p in m.parameters()) for k,m in self.model.items()})

    def load_state_dict(self, s):
        assert s['format']=='two_detached_online_MLPs_v1' and s['feature_count']==301
        self.model.load_state_dict(s['model'])
        for k, o in self.optimizers.items(): o.load_state_dict(s['optimizers'][k])
        self.mean.copy_(s['mean']); self.scale.copy_(s['scale'])
        self.steps=s['steps']; self.initialized=s['initialized']

class CompatibleAuxiliary:
    """Restore v2 bitwise first; initialize only the requested new branches next."""
    def __init__(self, device='cuda', lr=.003):
        self.device=device; self.lr=lr; self.impl=LegacyAuxiliary(device,lr)
    def activate(self):
        if isinstance(self.impl,TwoReadouts): return False
        self.impl=TwoReadouts(self.device,self.lr)
        return True
    @property
    def model(self): return self.impl.model
    @property
    def steps(self): return self.impl.steps
    def train(self,x,labels): return self.impl.train(x,labels)
    def state_dict(self): return self.impl.state_dict()
    def load_state_dict(self,s):
        if s.get('format')=='two_detached_online_MLPs_v1':
            if not isinstance(self.impl,TwoReadouts):self.impl=TwoReadouts(self.device,self.lr)
        elif isinstance(self.impl,TwoReadouts): self.impl=LegacyAuxiliary(self.device,self.lr)
        self.impl.load_state_dict(s)

def install_capture(head):
    # Observe the already-computed normalized representation after all 3 blocks.
    head._aux_latents=[]
    head._aux_capture_active=False
    def observe(_module,_args,value):
        if head._aux_capture_active:
            head._aux_latents.append(value.detach())
    head._aux_capture_handle=head.output_norm.register_forward_hook(observe)

