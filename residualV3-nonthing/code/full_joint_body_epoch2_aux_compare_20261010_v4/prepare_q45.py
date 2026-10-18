"""Continue the already-running epoch, with an isolated predictor and same LR."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
CODE=Path(__file__).resolve().parent
sys.path[:0]=[str(CODE),str(CODE.parent/'runtime_bundle')]
from full_checkpoint import atomic,prefix_sha,reserve
from runtime.utils import now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    old=a.previous.resolve();root=a.root.resolve()
    assert json.loads((old/'status.json').read_text())['status']=='paused'
    initial=json.loads((old/'latest_checkpoint.json').read_text())
    assert initial['status']=='committed' and initial['epoch']==2
    assert initial['joint_and_body_same_data_cursor']
    p=json.loads((old/'protocol.json').read_text());old_proof=json.loads((old/'identity_proof.json').read_text())
    assert hashlib.sha256((old/'protocol.json').read_bytes()).hexdigest()==old_proof['protocol_sha256']
    for arm,d in initial['artifacts'].items():
        assert all(d[k]==v for k,v in old_proof.items())
        assert prefix_sha(d['path'],d['logical_bytes'])==d['sha256']
        print(now(),'SOURCE_CHECKPOINT_VERIFIED',arm,'cursor',initial['cursor'],flush=True)
    sources=json.loads((old/'source_sha.json').read_text())
    assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in sources.items())
    for file in CODE.glob('*.py'):sources[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    p.update(job_root=str(root),variant='epoch2_post_repair_q45_auxiliary',
        initial_checkpoint_pointer=str(root/'initial_checkpoint.json'),initial_checkpoint_proof=old_proof,
        source_epoch_root=str(old),initial_epoch_cursor=initial['cursor'],
        initialization='Full same-epoch main checkpoint; no optimizer/RNG/LR reset; predictor starts fresh',
        auxiliary_predictor=dict(input='post_repair_unit_temperature_q_45_statistics',
            structure='two separate 45->1 linear readouts plus sigmoid; 92 trainable parameters',
            label_T0='teacher-prefix target argmax agreement',label_T1='teacher-prefix analytical overlap',
            lr=.003,weight_decay=.001,clip_norm=1.,feature_normalization='first training pack mean/std, then fixed',
            detached_inputs_and_labels=True,optimizer='independent AdamW',main_gradient_clipping_unchanged=True,
            max_compute_overhead_fraction=.05,original_joint_runner=str(CODE.parent/'full_joint_body_shared_teacher_epoch2_20261009_v1/joint_runner.py'),
            branch_start_epoch_cursor=initial['cursor'],no_native_stopping_enabled=True),utc=now())
    atomic(root/'protocol.json',p);atomic(root/'initial_checkpoint.json',initial)
    shutil.copy2(old/'teacher_input_protocol.json',root/'teacher_input_protocol.json')
    proof=dict(protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
        order_sha256=old_proof['order_sha256'],plan_sha256=old_proof['plan_sha256'])
    atomic(root/'identity_proof.json',proof);atomic(root/'source_sha.json',sources)
    atomic(root/'lineage.json',dict(previous_root=str(old),inherited_cursor=initial['cursor'],
        inherited_global_step=initial['global_step'],main_loss_model_data_anchors_lr_optimizer_rng_unchanged=True,
        predictor_initialized_at_switch=True,previous_original_evidence_preserved=True,
        initial_checkpoint_paths={k:v['path'] for k,v in initial['artifacts'].items()},utc=now()))
    reserve(root,p['checkpoint_slot_capacity_gib'])
    atomic(root/'gates/launch_preflight.json',dict(status='passed',old_sources_unchanged=True,
        old_main_full_checkpoints_verified=True,live_gradient_and_speed_gate_before_first_update=True,
        previous_epoch_not_restarted=True,utc=now()))
    atomic(root/'status.json',dict(status='prepared',training_epoch=2,cursor=initial['cursor'],
        total_packs=p['pack_count'],global_step=initial['global_step'],utc=now()))
    control=root/'control.sh'
    control.write_text('#!/usr/bin/env bash\nexec /data/weijiahong/miniconda3/envs/dflash/bin/python -B '+
        str(CODE/'control.py')+' "${1:-status}" --root '+str(root)+'\n')
    control.chmod(0o755)
    print(now(),'Q45_CONTINUATION_PREPARED',initial['cursor'],flush=True)


if __name__=='__main__':main()
