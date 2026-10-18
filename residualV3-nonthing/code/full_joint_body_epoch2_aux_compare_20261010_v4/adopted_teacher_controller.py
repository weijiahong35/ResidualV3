"""Continue from a paired checkpoint while retaining the existing teacher GPU."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,sys
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE.parent/'runtime_bundle'))
from full_checkpoint import atomic,reserve
from runtime.utils import now,healthy
from outer_controller import link
PYTHON='/data/weijiahong/miniconda3/envs/dflash/bin/python'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    root=ap.parse_args().root.resolve()
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=json.loads((root/'protocol.json').read_text())
    for path,digest in json.loads((root/'source_sha.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    assert json.loads((root/'gates/checkpoint_cpu_gate.json').read_text())['status']=='passed'
    reserve(root,p['checkpoint_slot_capacity_gib'])
    atomic(root/'controller_pid.json',dict(pid=os.getpid(),source=str(__file__),utc=now()))
    def request_pause(sig,frame):atomic(root/'pause.request.json',dict(reason='controller_signal',signal=sig))
    signal.signal(signal.SIGTERM,request_pause);signal.signal(signal.SIGINT,request_pause)
    (root/'resume.request.json').unlink(missing_ok=True)
    (root/'pause.request.json').unlink(missing_ok=True)
    teacher=Path(p['adopted_teacher_root'])
    number=0;failures=0
    while True:
        if (root/'pause.request.json').exists():
            atomic(root/'status.json',dict(status='paused',checkpoint_available=True,utc=now()))
            while not(root/'resume.request.json').exists():time.sleep(3)
            (root/'resume.request.json').unlink();(root/'pause.request.json').unlink();failures=0
        e=json.loads((teacher/'endpoint.json').read_text())
        assert healthy(int(e['endpoint'].split(':')[2].split('/')[0])),'Adopted teacher is not healthy'
        latest=json.loads((root/'latest_checkpoint.json').read_text()) if(root/'latest_checkpoint.json').exists() else None
        if latest and latest['cursor']==p['pack_count']:
            atomic(root/'status.json',dict(status='complete',cursor=latest['cursor'],training_epoch=2,utc=now()))
            atomic(teacher/'stop.json',dict(reason='Adopted epoch2 comparison complete',utc=now()))
            return
        initial=latest or json.loads((root/'initial_checkpoint.json').read_text())
        indices=[int(x.name.split('_')[-1]) for x in (root/'runs').glob('attempt_*')]
        number=max(indices,default=0)+1
        trial=root/'runs'/f'attempt_{number:04d}';trial.mkdir()
        for arm in ['joint','body']:(trial/arm).mkdir()
        (trial/'protocol.json').symlink_to(root/'protocol.json');(trial/'teacher').symlink_to(teacher,target_is_directory=True)
        atomic(trial/'driver_ready.json',dict(source_files=json.loads((root/'source_sha.json').read_text())))
        atomic(trial/'assignment.json',dict(learner_gpus=p['fixed_learner_gpus'],learner_uuid=p['fixed_learner_uuid']))
        atomic(trial/'start_cursor.json',dict(cursor=initial['cursor'],checkpoint=initial,
            resume_kind='same_epoch' if latest else 'augmentation'))
        atomic(root/'current_attempt.json',dict(path=str(trial),teacher=str(teacher),
            hardware=p['fixed_hardware'],teacher_restarted=False,utc=now()))
        link(root/'joint_training.log',trial/'joint/training.log');link(root/'dflash_training.log',trial/'body/training.log')
        link(root/'teacher.log',teacher/'attempt_01/vllm.log');link(root/'training.log',trial/'controller.log')
        cmd=[PYTHON,'-B',str(CODE/'train_controller.py'),'--root',str(trial)]
        with(trial/'controller.log').open('x') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        atomic(trial/'controller_launch.json',dict(pid=proc.pid,command=cmd))
        atomic(root/'status.json',dict(status='starting_replacement_learners',cursor=initial['cursor'],
            teacher_restarted=False,hardware=p['fixed_hardware'],utc=now()))
        print(now(),'ADOPTED_TEACHER_CONTINUATION',initial['cursor'],flush=True)
        rc=proc.wait()
        status=json.loads((trial/'status.json').read_text())
        if rc==0 and status['status']=='complete':continue
        if rc==0 and status['status']=='paused':continue
        failures+=1
        atomic(root/'status.json',dict(status='retrying_from_committed_checkpoint',error=status,
            teacher_restarted=False,failures=failures,utc=now()))
        if failures>=3:atomic(root/'pause.request.json',dict(reason='three_failures_need_inspection'))
        time.sleep(2)

if __name__=='__main__':main()
