"""Regression for valid scalar cancellation versus disconnected/nonfinite gradients."""
import unittest
import torch
from .gradient_audit import audit_gradients
from runtime.master_optimizer import MasterAdamW

class GradientAuditTest(unittest.TestCase):
    def test_bf16_bias_cancellation_in_fp32_accumulation(self):
        head=torch.nn.Linear(1,1,dtype=torch.bfloat16)
        protocol=dict(stage1_lr=3e-4,dose_lr=3e-4,weight_decay=.01)
        opt=MasterAdamW(head,list(head.parameters()),[],protocol)
        opt.zero_grad()
        for x, sign in ((1.,1.),(-1.,-1.)):
            (head(torch.tensor([[x]],dtype=torch.bfloat16)).float().sum()*sign).backward()
            opt.accumulate()
        grads={name:float(master.grad.abs().sum()) for name,_,master in opt.pairs}
        self.assertEqual(grads['bias'],0.)
        self.assertGreater(grads['weight'],0.)
        result=audit_gradients(grads,dict(head.named_parameters()),{'q':0.})
        self.assertEqual(result['zero_gradient_parameters'],['bias'])
        self.assertEqual(result['zero_gradient_fusion_branches'],['q'])
        opt.step()
        self.assertTrue(all(torch.isfinite(v).all() for v in head.parameters()))
    def test_missing_gradient_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'connectivity'):
            audit_gradients({'weight':1.},['weight','bias'],{'q':1.})
    def test_nonfinite_rejected(self):
        for invalid in (float('nan'),float('inf'),-1.):
            for parameter in (True,False):
                with self.subTest(invalid=invalid,parameter=parameter):
                    with self.assertRaisesRegex(RuntimeError,'invalid'):
                        audit_gradients({'weight':invalid if parameter else 1.},['weight'],
                                        {'q':1. if parameter else invalid})

if __name__=='__main__':unittest.main()
