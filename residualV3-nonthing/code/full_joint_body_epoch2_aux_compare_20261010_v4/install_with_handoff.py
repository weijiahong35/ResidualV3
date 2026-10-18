"""Checkpoint-safe transition: retain teacher, bridge the two owned trainer cards."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE.parent/'runtime_bundle'))
from full_checkpoint import atomic,reserve
from runtime.utils import now,healthy
PYTHON='/data/weijiahong/miniconda3/envs/dflash/bin/python'

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except OSError:return False

def verified_pid(pid, needle):
    assert Path(f'/proc/{pid}').stat().st_uid==os.getuid()
    assert needle in Path(f'/proc/{pid}/cmdline').read_text(),(pid,needle)
    assert alive(pid)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--previous',type=Path,required=True)
    ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    old=a.previous.resolve();root=a.root.resolve()
    previous=json.loads((old/'current_attempt.json').read_text())
    trial=Path(previous['path']);teacher=Path(previous['teacher'])
    workers=json.loads((trial/'launch.json').read_text())['worker_pids']
    outer=json.loads((old/'controller_pid.json').read_text())['pid']
    teacher_controller=json.loads((teacher/'controller_launch.json').read_text())['pid']
    old_train_controller=json.loads((trial/'controller_launch.json').read_text())['pid']
    for pid,needle in [(outer,'adopted_teacher_controller.py'),(teacher_controller,'teacher_service.py'),
                       (old_train_controller,'train_controller.py')]:verified_pid(pid,needle)
    e=json.loads((teacher/'endpoint.json').read_text())
    port=int(e['endpoint'].split(':')[2].split('/')[0]);assert healthy(port)
    assert json.loads((root/'gates/checkpoint_cpu_gate.json').read_text())['status']=='passed'
    sources=json.loads((old/'source_sha.json').read_text())
    for path,h in sources.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    for arm in ['joint','body']:
        gate=json.loads((trial/arm/'full_state_roundtrip_gate.json').read_text())
        assert gate['status']=='passed'
    keepers={};parked=False;new_controller=None
    atomic(root/'status.json',dict(status='preparing_owned_card_handoff',old_job=str(old),utc=now()))
    try:
        for arm in ['joint','body']:
            verified_pid(workers[arm],'multiprocessing.spawn')
            assert int(Path(f'/proc/{workers[arm]}/stat').read_text().rsplit(')',1)[1].split()[1])==old_train_controller
            directory=root/'handoff'/arm;directory.mkdir(parents=True)
            uuid=json.loads((old/'protocol.json').read_text())['fixed_learner_uuid'][arm]
            cmd=[PYTHON,'-B',str(CODE/'gpu_handoff.py'),'--root',str(directory),'--uuid',uuid,'--old-worker',str(workers[arm])]
            with(directory/'keeper.log').open('x') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            keepers[arm]=dict(root=str(directory),pid=proc.pid,uuid=uuid)
        deadline=time.monotonic()+90
        while not all((Path(k['root'])/'ready.json').exists() for k in keepers.values()):
            assert all(alive(k['pid']) for k in keepers.values()),'Owned-card keeper failed'
            assert time.monotonic()<deadline,'Owned-card keeper startup timeout';time.sleep(.2)
        atomic(root/'handoff/keepers.json',keepers)
        # Park only verified CPU supervisors. Their existing CUDA children remain alive.
        os.kill(outer,signal.SIGSTOP);os.kill(teacher_controller,signal.SIGSTOP);parked=True
        atomic(old/'pause.request.json',dict(reason='user_requested_two_detached_online_readouts',
                                            new_root=str(root),requested_unix_time=time.time()))
        atomic(root/'status.json',dict(status='waiting_paired_checkpoint',teacher_gpu_retained=True,
            learner_gpu_keepers=keepers,utc=now()))
        deadline=time.monotonic()+1200
        while True:
            # The old trainer supervisor can request teacher cleanup once; protect
            # the adopted service until that verified supervisor has exited.
            (teacher/'stop.json').unlink(missing_ok=True)
            complete=trial/'broker_complete.json'
            if complete.exists() and json.loads(complete.read_text())['status']=='pause' and all(not alive(pid) for pid in workers.values()):break
            assert healthy(port),'Existing teacher lost health during checkpoint'
            assert all(alive(k['pid']) for k in keepers.values()),'Keeper failed during checkpoint'
            assert time.monotonic()<deadline,'Paired checkpoint timeout';time.sleep(.2)
        initial=json.loads((old/'latest_checkpoint.json').read_text())
        assert initial['joint_and_body_same_data_cursor'] and initial['epoch']==2
        assert initial['cursor']==json.loads(complete.read_text())['cursor']
        oldproof=json.loads((old/'identity_proof.json').read_text())
        p=json.loads((old/'protocol.json').read_text())
        p.update(job_root=str(root),variant='epoch2_two_detached_online_MLP_readouts',
            initial_checkpoint_pointer=str(root/'initial_checkpoint.json'),initial_checkpoint_proof=oldproof,
            initial_epoch_cursor=initial['cursor'],source_epoch_root=str(old),
            adopted_teacher_root=str(teacher),stop_teacher_on_exit=False,
            fixed_hardware=previous['hardware'],fixed_learner_gpus={k:previous['hardware'][k] for k in workers},
            fixed_learner_uuid={k:p['fixed_learner_uuid'][k] for k in workers},
            handoff_keepers=keepers,owned_keeper_pids=[k['pid'] for k in keepers.values()],
            auxiliary_predictor=dict(p['auxiliary_predictor'],
                structure='Q45_MLP 45->32->2 and Q45_REPAIR_LATENT_MLP 301->32->2; SiLU',
                input='post_repair_q45; optional third-block output_norm representation256',
                feature_normalization='shared q45 first new pack mean/std; then fixed',
                branch_start_epoch_cursor=initial['cursor'],parameter_counts=dict(Q45_MLP=1538,Q45_REPAIR_LATENT_MLP=9730),
                gradients='both inputs and labels detached; independent optimizer and clipping per readout',
                only_requested_two_branches_train=True),utc=now())
        atomic(root/'protocol.json',p);atomic(root/'initial_checkpoint.json',initial)
        proof=dict(protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
                   order_sha256=oldproof['order_sha256'],plan_sha256=oldproof['plan_sha256'])
        atomic(root/'identity_proof.json',proof)
        for path in CODE.glob('*.py'):sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        atomic(root/'source_sha.json',sources)
        atomic(root/'lineage.json',dict(previous_root=str(old),inherited_cursor=initial['cursor'],
            inherited_global_step=initial['global_step'],main_optimizer_rng_lr_data_order_unchanged=True,
            original_q45_checkpoint_preserved=True,only_two_new_readouts_initialized=True,
            teacher_server_pid=e['server_pid'],teacher_service_retained=True,utc=now()))
        for directory in ['runs','final']:(root/directory).mkdir(exist_ok=True)
        reserve(root,p['checkpoint_slot_capacity_gib'])
        atomic(root/'gates/launch_preflight.json',dict(status='passed',original_frozen_sources_unchanged=True,
            paired_initial_checkpoint_committed=True,teacher_retained=True,live_gradient_speed_gate_required=True,utc=now()))
        control=root/'control.sh'
        control.write_text('#!/usr/bin/env bash\nexec '+PYTHON+' -B '+str(CODE/'control.py')+' "${1:-status}" --root '+str(root)+'\n')
        control.chmod(0o755)
        atomic(old/'status.json',dict(status='paused',cursor=initial['cursor'],continuation_root=str(root),
            reason='complete checkpoint committed; teacher retained for continuation',utc=now()))
        atomic(old/'CONTINUED_IN.json',dict(root=str(root),cursor=initial['cursor'],teacher_retained=True,utc=now()))
        cmd=[PYTHON,'-B',str(CODE/'adopted_teacher_controller.py'),'--root',str(root)]
        with(root/'service.log').open('x') as log:
            new_controller=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        atomic(root/'service_launch.json',dict(pid=new_controller.pid,command=cmd,utc=now()))
        deadline=time.monotonic()+240
        while alive(old_train_controller):
            (teacher/'stop.json').unlink(missing_ok=True)
            assert new_controller.poll() is None,'Replacement controller failed'
            assert healthy(port),'Retained teacher failed'
            assert time.monotonic()<deadline,'Old trainer supervisor cleanup timeout';time.sleep(.2)
        (teacher/'stop.json').unlink(missing_ok=True)
        os.kill(teacher_controller,signal.SIGCONT)
        # Old outer is a parked CPU-only wrapper after both CUDA learners exited.
        # Its finally block would stop the now-adopted teacher, so retire this
        # verified wrapper only; the committed full state and all evidence remain.
        verified_pid(outer,'adopted_teacher_controller.py');os.kill(outer,signal.SIGKILL)
        atomic(root/'handoff/complete.json',dict(status='complete',cursor=initial['cursor'],
            teacher_server_pid_unchanged=e['server_pid'],teacher_controller_resumed=True,
            old_cuda_learners_exited_after_committed_checkpoint=True,
            old_cpu_outer_wrapper_retired=True,foreign_processes_untouched=True,utc=now()))
        print(now(),'HANDOFF_COMPLETE',initial['cursor'],'teacher retained; replacement trainers own cards',flush=True)
    except BaseException as exc:
        atomic(root/'handoff_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),utc=now()))
        # Do not release guards or resume an old supervisor while a replacement
        # could be active. Leave all committed states and ownership evidence.
        if not parked:
            for k in keepers.values():atomic(Path(k['root'])/'stop.json',dict(reason='pre-checkpoint preparation failed'))
        raise

if __name__=='__main__':main()
