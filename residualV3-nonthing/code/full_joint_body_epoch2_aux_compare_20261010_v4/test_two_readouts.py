"""CPU preflight: main-gradient/RNG isolation and exact next-update resume."""
import copy
import json
import torch
from auxiliary_comparison import CompatibleAuxiliary, TwoReadouts
from full_checkpoint import digest_tree

torch.set_num_threads(2)
torch.manual_seed(43)
x=torch.randn(40,301,requires_grad=True)
y=torch.stack(((x[:,0]>0).float(),x[:,2].sigmoid()),-1)
cpu_rng=torch.get_rng_state().clone()
a=CompatibleAuxiliary('cpu');legacy=copy.deepcopy(a.state_dict())
a.load_state_dict(legacy)
assert digest_tree(a.state_dict())==digest_tree(legacy)
a.activate();assert torch.equal(cpu_rng,torch.get_rng_state())
main=(x.square().mean());main.backward();before=x.grad.clone()
a.train(x,y);assert torch.equal(before,x.grad)
assert torch.equal(cpu_rng,torch.get_rng_state())
state=copy.deepcopy(a.state_dict())
b=CompatibleAuxiliary('cpu');b.load_state_dict(state)
assert digest_tree(b.state_dict())==digest_tree(state)
a.train(x,y);b.train(x,y)
assert digest_tree(a.state_dict())==digest_tree(b.state_dict())
counts=a.state_dict()['parameter_counts']
assert counts==dict(Q45_MLP=1538,Q45_REPAIR_LATENT_MLP=9730)
print(json.dumps(dict(status='passed',parameter_counts=counts,
    legacy_state_exactly_restored=True,main_input_gradient_bitwise_unchanged=True,
    cpu_rng_unchanged=True,auxiliary_next_update_bitwise_equal_after_restore=True)))
