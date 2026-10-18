"""One-epoch three-GPU job; bounded retry from the last paired checkpoint."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE.parent/'runtime_bundle'))
from device import assignment
from full_checkpoint import atomic,reserve
from runtime.utils import now
PYTHON='/data/weijiahong/miniconda3/envs/dflash/bin/python'
TEACHER=CODE.parent/'activation_memory_timing_20261008_v1/teacher_service.py'


def inventory():
    values=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    processes={}
    for line in apps.splitlines():
        fields=[x.strip() for x in line.split(',')]
        if len(fields)==3:processes.setdefault(fields[0],[]).append(int(fields[1]))
    result={}
    for line in values.splitlines():
        i,u,f,v=[x.strip() for x in line.split(',')]
        result[int(i)]=dict(uuid=u,free_mib=int(f),utilization=int(v),pids=processes.get(u,[]))
    return result


def choose(root):
    values=inventory();eligible=[i for i,v in values.items() if i not in [1,2] and not v['pids'] and v['free_mib']>=78000]
    if len(eligible)<3:
        atomic(root/'status.json',dict(status='waiting_three_cards',inventory=values,excluded_gpus=[1,2],epochs=1,utc=now()));return None
    preferred=[3,4,5,0,6,7];chosen=[i for i in preferred if i in eligible][:3]
    return dict(teacher=chosen[0],joint=chosen[1],body=chosen[2],initial_inventory=values)


def link(path,target):
    temporary=path.with_name(path.name+'.partial');temporary.unlink(missing_ok=True)
    temporary.symlink_to(target);temporary.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);root=ap.parse_args().root.resolve()
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=json.loads((root/'protocol.json').read_text());assert p['epochs']==1 and p['teacher_dp_size']==1
    for path,digest in json.loads((root/'source_sha.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    reserve(root,p['checkpoint_slot_capacity_gib'])
    atomic(root/'controller_pid.json',dict(pid=os.getpid(),source=str(__file__),utc=now()))
    def pause_signal(sig,frame):atomic(root/'pause.request.json',dict(reason='controller_signal',signal=sig,utc=now()))
    signal.signal(signal.SIGTERM,pause_signal);signal.signal(signal.SIGINT,pause_signal)
    if (root/'resume.request.json').exists():
        (root/'resume.request.json').unlink()
        (root/'pause.request.json').unlink(missing_ok=True)
    attempts=0;failures=0
    while True:
        if(root/'pause.request.json').exists():
            atomic(root/'status.json',dict(status='paused',checkpoint_available=(root/'latest_checkpoint.json').exists(),epochs=1,utc=now()))
            while not(root/'resume.request.json').exists():time.sleep(3)
            (root/'resume.request.json').unlink();(root/'pause.request.json').unlink();failures=0
        latest=json.loads((root/'latest_checkpoint.json').read_text()) if(root/'latest_checkpoint.json').exists() else None
        if latest and latest['cursor']==p['pack_count'] and all((root/'final'/f'{arm}_epoch_01.safetensors').exists() for arm in ['joint','body']):
            atomic(root/'status.json',dict(status='complete',epochs=1,cursor=p['pack_count'],training_rows=p['training_rows'],validation=False,evaluation=False,utc=now()));return
        hardware=choose(root)
        if hardware is None:time.sleep(5);continue
        runs=root/'runs';teachers=root/'teachers';runs.mkdir(exist_ok=True);teachers.mkdir(exist_ok=True)
        indices=[int(v.name.split('_')[-1]) for v in runs.glob('attempt_*')];number=max(indices,default=0)+1
        trial=runs/f'attempt_{number:04d}';teacher=teachers/f'attempt_{number:04d}';trial.mkdir();teacher.mkdir()
        for arm in ['joint','body']:(trial/arm).mkdir()
        (trial/'protocol.json').symlink_to(root/'protocol.json');(trial/'teacher').symlink_to(teacher,target_is_directory=True)
        atomic(trial/'driver_ready.json',dict(source_files=json.loads((root/'source_sha.json').read_text())))
        identities={role:assignment(hardware[role]) for role in ['teacher','joint','body']}
        atomic(trial/'assignment.json',dict(learner_gpus={a:hardware[a] for a in ['joint','body']},learner_uuid={a:identities[a]['uuid'] for a in ['joint','body']}))
        atomic(trial/'start_cursor.json',dict(cursor=latest['cursor'] if latest else 0,checkpoint=latest))
        atomic(root/'current_attempt.json',dict(path=str(trial),teacher=str(teacher),hardware=hardware,identities=identities,utc=now()))
        link(root/'joint_training.log',trial/'joint/training.log');link(root/'dflash_training.log',trial/'body/training.log')
        link(root/'teacher.log',teacher/'attempt_01/vllm.log');link(root/'training.log',trial/'controller.log')
        command=[PYTHON,'-B',str(TEACHER),'--output',str(teacher),'--protocol',str(root/'teacher_input_protocol.json'),
                 '--exclude-gpus',*map(str,sorted(set(range(8))-{hardware['teacher']})),'--dp-size','1','--kv-gib','48']
        server=learner=None
        try:
            with(teacher/'controller.log').open('x') as log:server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            atomic(teacher/'controller_launch.json',dict(pid=server.pid,command=command))
            atomic(root/'status.json',dict(status='starting_teacher',hardware=hardware,cursor=latest['cursor'] if latest else 0,epochs=1,utc=now()))
            deadline=time.monotonic()+1000
            while not(teacher/'endpoint.json').exists():
                assert server.poll() is None,'Teacher startup failed';assert time.monotonic()<deadline,'Teacher startup timeout';time.sleep(2)
            command=[PYTHON,'-B',str(CODE/'train_controller.py'),'--root',str(trial)]
            with(trial/'controller.log').open('x') as log:learner=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            atomic(trial/'controller_launch.json',dict(pid=learner.pid,command=command))
            learner_code=learner.wait()
            assert learner_code==0,f'Learner controller failed with {learner_code}'
            assert server.wait(timeout=200)==0,'Teacher shutdown failed'
            result=json.loads((trial/'status.json').read_text())['status'];failures=0
            if result=='complete':
                atomic(root/'summary.json',dict(status='complete',epochs=1,training_rows=p['training_rows'],input_tokens=p['input_tokens'],updates_per_arm=p['pack_count'],teacher_dp=1,teacher_tp=1,arms=['dflash','dflash_joint_nosoft3'],head_loss='CE_DPACE_OUTPUT',validation=False,evaluation=False,utc=now()))
                atomic(root/'status.json',dict(status='complete',epochs=1,training_rows=p['training_rows'],cursor=p['pack_count'],utc=now()));return
            assert result=='paused'
        except BaseException as exc:
            failures+=1;atomic(trial/'outer_failure.json',dict(error=repr(exc),utc=now()))
            atomic(root/'status.json',dict(status='retrying_from_committed_checkpoint',error=repr(exc),consecutive_failures=failures,epochs=1,utc=now()))
            # Repeated failures need inspection rather than an unbounded error loop.
            if failures>=3:atomic(root/'pause.request.json',dict(reason='three_consecutive_failures',error=repr(exc),utc=now()))
        finally:
            if server is not None and server.poll() is None:
                atomic(teacher/'stop.json',dict(reason='attempt ended'));server.wait(timeout=200)
        attempts+=1;time.sleep(2)


if __name__=='__main__':main()
