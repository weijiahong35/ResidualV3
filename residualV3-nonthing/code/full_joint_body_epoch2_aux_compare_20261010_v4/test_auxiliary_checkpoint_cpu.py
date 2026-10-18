"""Full checkpoint includes the separate predictor optimizer and normalization."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import torch
from full_checkpoint import atomic,digest_tree,restore,save
from joint_model import MasterOptim
from auxiliary_comparison import CompatibleAuxiliary as Auxiliary
from test_resume_cpu import tiny_pair,update
from training_progress import resume_spec


def main():
    destination=Path(sys.argv[1])
    p=dict(training_epoch=2,global_step_offset=15847,pack_count=15854,
        initial_checkpoint_proof=dict(protocol_sha256='old',order_sha256='order',plan_sha256='plan'),
        body_muon_lr=.001,body_adam_lr=.0002,head_lr=.0003,muon_momentum=.95,
        muon_weight_decay=.1,weight_decay=.01,muon_ns_steps=5,muon_adjust_lr_fn='match_rms_adamw')
    proof=dict(protocol_sha256='new',order_sha256='order',plan_sha256='plan')
    checkpoint=dict(cursor=400,epoch=2,global_step=16247)
    assert resume_spec(dict(checkpoint=checkpoint,cursor=400,resume_kind='augmentation'),p,proof)==(checkpoint,p['initial_checkpoint_proof'],400)
    with tempfile.TemporaryDirectory() as name,patch.object(torch.cuda,'get_rng_state',torch.get_rng_state),patch.object(torch.cuda,'set_rng_state',torch.set_rng_state):
        root=Path(name);atomic(root/'protocol.json',p)
        (root/'checkpoints/slot_0').mkdir(parents=True)
        with(root/'checkpoints/slot_0/joint.state.bin').open('wb') as stream:stream.truncate(2**20)
        pair=tiny_pair();opt=MasterOptim(pair,p);opt.auxiliary=Auxiliary('cpu');opt.auxiliary_enabled=True
        opt.auxiliary.activate()
        x=torch.randn(32,301);y=torch.rand(32,2)
        update(pair,opt,1.);opt.auxiliary.train(x,y)
        descriptor=save(root,'joint',0,401,opt,proof)
        fresh=tiny_pair();other=MasterOptim(fresh,p);other.auxiliary=Auxiliary('cpu');other.auxiliary_enabled=True
        result=restore(descriptor,other,proof)
        assert result['auxiliary_state_restored']
        assert digest_tree(opt.auxiliary.state_dict())==digest_tree(other.auxiliary.state_dict())
        opt.auxiliary.train(x,y);other.auxiliary.train(x,y)
        assert digest_tree(opt.auxiliary.state_dict())==digest_tree(other.auxiliary.state_dict())
    atomic(destination,dict(status='passed',predictor_parameters_optimizer_normalization_and_steps_restored=True,
        auxiliary_next_update_bitwise_equal=True,same_epoch_augmentation_cursor_correct=True,
        main_optimizer_not_extended_with_predictor=True,note='CPU complete-state gate; real CUDA gate after first resumed update'))
    print('Auxiliary full-checkpoint CPU gate passed')


if __name__=='__main__':main()
