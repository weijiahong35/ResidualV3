"""A checkpoint must preserve the next Adam update, including hidden FP32 precision."""
import tempfile
from pathlib import Path
import torch
from runtime.master_optimizer import MasterAdamW
from .recovery_state import save_checkpoint,load_checkpoint


def main():
    torch.set_num_threads(1);torch.manual_seed(16)
    p=dict(stage1_lr=.0003,dose_lr=.0003,weight_decay=.01)
    def create():
        head=torch.nn.Linear(4,3).to(dtype=torch.bfloat16)
        return head,MasterAdamW(head,list(head.parameters()),[],p)
    a,ao=create();x=torch.randn(5,4,dtype=torch.bfloat16)
    def step(head,opt):
        opt.zero_grad();head(x).float().square().sum().backward();opt.accumulate();opt.step()
    with tempfile.TemporaryDirectory() as folder:
        for _ in range(3):step(a,ao)
        save_checkpoint(folder,a,ao,20003,'fixture','test')
        b,bo=create();n,_=load_checkpoint(folder,b,bo,'fixture');assert n==20003
        step(a,ao);step(b,bo)
        assert all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items())
        assert all(torch.equal(am,bm) for (_,_,am),(_,_,bm) in zip(ao.pairs,bo.pairs))
        for av,bv in zip(ao.optimizer.state.values(),bo.optimizer.state.values()):
            assert av.keys()==bv.keys()
            assert all(torch.equal(av[k],bv[k]) for k in av)
        save_checkpoint(folder,a,ao,20004,'fixture','test_rotation')
        Path(folder,'resume_latest.pt').write_bytes(b'interrupted transfer')
        c,co=create();n,_=load_checkpoint(folder,c,co,'fixture');assert n==20003
    print('PASS: next model/master/Adam update exact after restore; previous checkpoint recovers corrupted latest')

if __name__=='__main__':main()
