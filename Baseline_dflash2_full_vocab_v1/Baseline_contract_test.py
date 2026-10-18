#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from Baseline_models import DFlash2CandidateSelector
m=DFlash2CandidateSelector(31,8,4,5)
x=torch.randn(2,31).bfloat16()
h=torch.randn(2,8)
p=torch.tensor([3,5])
ids,scores=m.inference(x,h,p)
assert ids.shape==scores.shape==(2,5)
assert torch.allclose(scores,x.gather(1,ids))
t=x.argmin(-1)
assert not ids.eq(t[:,None]).any()
a,label=m.training_candidates(x,t)
assert torch.equal(a.gather(1,label[:,None]).squeeze(1),t)
assert torch.equal(a[:,-1],t)
print('DFlash2 full-vocab selector contracts PASS')
