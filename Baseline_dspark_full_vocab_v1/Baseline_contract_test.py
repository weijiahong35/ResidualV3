#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from Baseline_models import DSparkHeadBaseline
m=DSparkHeadBaseline(31,8,4)
x=torch.randn(2,31).bfloat16()
y,c=m(x,torch.randn(2,8),torch.tensor([3,5]))
assert y.shape==(2,31) and c.shape==(2,) and torch.equal(y,x)
print('DSpark full-vocab head contracts PASS')
