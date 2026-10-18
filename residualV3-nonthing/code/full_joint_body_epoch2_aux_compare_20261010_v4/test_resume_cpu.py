"""Meaningful tiny-state CPU tests: epoch transition, paired commit, next update."""
import json
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE))
from full_checkpoint import atomic,commit,digest_tensor,digest_tree,restore,save
from joint_model import MasterOptim,TrainablePair
from training_progress import lr_scale,resume_spec


def tiny_pair():
    body=nn.Sequential(nn.Linear(4,4),nn.LayerNorm(4)).to(dtype=torch.bfloat16)
    head=nn.Linear(4,2).to(dtype=torch.bfloat16)
    return TrainablePair(body,head)


def update(pair,opt,scale):
    opt.zero_grad()
    x=torch.randn(3,4).to(dtype=torch.bfloat16)
    loss=pair.head(pair.body(x)).float().square().mean()
    loss.backward();opt.accumulate();opt.step(scale)
    opt.zero_grad()


def main():
    out=Path(sys.argv[1]);out.parent.mkdir(exist_ok=True)
    protocol=dict(training_epoch=2,global_step_offset=15847,pack_count=15848,
        initial_checkpoint_proof=dict(protocol_sha256='old',order_sha256='old-order',plan_sha256='old-plan'),
        body_muon_lr=.001,body_adam_lr=.0002,head_lr=.0003,muon_momentum=.95,
        muon_weight_decay=.1,weight_decay=.01,muon_ns_steps=5,muon_adjust_lr_fn='match_rms_adamw')
    proof=dict(protocol_sha256='new',order_sha256='new-order',plan_sha256='new-plan')
    old=dict(cursor=15847)
    initial=dict(cursor=0,checkpoint=old,resume_kind='previous_epoch')
    assert resume_spec(initial,protocol,proof)==(old,protocol['initial_checkpoint_proof'],15847)
    current=dict(cursor=17,epoch=2,global_step=15864)
    assert resume_spec(dict(cursor=17,checkpoint=current,resume_kind='same_epoch'),protocol,proof)==(current,proof,17)
    try:resume_spec(dict(cursor=16,checkpoint=current,resume_kind='same_epoch'),protocol,proof)
    except AssertionError:pass
    else:raise AssertionError('Misaligned cursor was accepted')
    assert lr_scale(1,15848,.01)>0 and lr_scale(159,15848,.01)==1 and lr_scale(15848,15848,.01)==0
    torch.manual_seed(120);np.random.seed(120);random.seed(120)
    cuda_state=[torch.get_rng_state().clone()]
    def cuda_get():return cuda_state[0].clone()
    def cuda_set(v):cuda_state[0]=v.clone()
    with tempfile.TemporaryDirectory() as directory,patch.object(torch.cuda,'get_rng_state',cuda_get),patch.object(torch.cuda,'set_rng_state',cuda_set):
        root=Path(directory);atomic(root/'protocol.json',protocol)
        for slot in [0,1]:
            (root/'checkpoints'/f'slot_{slot}').mkdir(parents=True)
            for arm in ['joint','body']:
                with (root/'checkpoints'/f'slot_{slot}'/f'{arm}.state.bin').open('wb') as f:f.truncate(2**20)
        pair=tiny_pair();opt=MasterOptim(pair,protocol)
        update(pair,opt,lr_scale(1,15848,.01));a=save(root,'joint',0,1,opt,proof)
        assert a['epoch']==2 and a['global_step']==15848
        b=save(root,'body',0,1,opt,proof);pointer=commit(root,{'joint':a,'body':b},1)
        assert pointer['global_step']==15848
        baseline_hashes={n:digest_tensor(v) for n,v,_ in opt.pairs}
        optimizer_hash=digest_tree([v.state_dict() for v in opt.optimizers])
        update(pair,opt,lr_scale(2,15848,.01))
        expected={n:digest_tensor(v) for n,v,_ in opt.pairs}
        expected_master={n:digest_tensor(m) for n,_,m in opt.pairs}
        expected_optimizers=digest_tree([v.state_dict() for v in opt.optimizers])
        fresh=tiny_pair();fresh_opt=MasterOptim(fresh,protocol)
        result=restore(a,fresh_opt,proof)
        assert result['global_step']==15848
        assert {n:digest_tensor(v) for n,v,_ in fresh_opt.pairs}==baseline_hashes
        assert digest_tree([v.state_dict() for v in fresh_opt.optimizers])==optimizer_hash
        update(fresh,fresh_opt,lr_scale(2,15848,.01))
        assert {n:digest_tensor(v) for n,v,_ in fresh_opt.pairs}==expected
        assert {n:digest_tensor(m) for n,_,m in fresh_opt.pairs}==expected_master
        assert digest_tree([v.state_dict() for v in fresh_opt.optimizers])==expected_optimizers
        save(root,'joint',1,2,fresh_opt,proof)
        assert json.loads((root/'latest_checkpoint.json').read_text())==pointer
    atomic(out,dict(status='passed',cpu_fresh_restore_and_next_update_bitwise_equal=True,
        master_muon_adam_and_rng_preserved=True,epoch2_nonzero_lr_restart=True,
        epoch_and_global_step_accounting=True,uncommitted_slot_never_advances_pointer=True,
        note='CUDA RNG mocked for CPU-only test; actual full CUDA state gate runs after first update'))
    print('CPU resume and next-update tests passed')


if __name__=='__main__':main()
