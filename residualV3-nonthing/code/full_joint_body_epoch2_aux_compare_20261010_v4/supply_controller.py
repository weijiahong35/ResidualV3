"""Bounded DP1/DP2 supply timing; independent real learners; exclude GPU1/2."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

PYTHON='/data/weijiahong/miniconda3/envs/dflash/bin/python'


def dump(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def stop_teacher(path):
    dump(path/'stop.json',dict(reason='bounded supply comparison finished or failed'))
    deadline=time.monotonic()+180
    while not(path/'cleanup.json').exists() and time.monotonic()<deadline:time.sleep(1)
    assert (path/'cleanup.json').exists(),'Owned teacher cleanup timeout'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;c=Path(__file__).resolve().parent
    active_teacher=r/'teacher_dp1'
    try:
        gate=json.loads((r/'transport_lifetime_gate.json').read_text());assert gate['status']=='passed'
        previous=json.loads(Path(gate['prior_full1536_online_gradient_gate']).read_text())
        assert previous['loss_absolute_error']==0 and all(previous[k]['bitwise_equal'] for k in ['head_gradients','raw_gradients','hidden_gradients'])
        frozen=json.loads((r/'source_sha.json').read_text())
        for path,digest in frozen.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        for variant,teacher in [('dp1_fast_pack','teacher_dp1'),('dp2_fast_pack','teacher_dp2')]:
            active_teacher=r/teacher
            if teacher=='teacher_dp2':
                cmd=[PYTHON,'-B',str(c/'teacher_service.py'),'--output',str(active_teacher),'--protocol',str(r/'teacher_input_protocol.json'),
                     '--exclude-gpus','1','2','4','5','6','7','--dp-size','2','--kv-gib','48']
                with (active_teacher/'controller.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                dump(r/'teacher_dp2_controller_launch.json',dict(pid=p.pid,command=cmd))
            dump(r/'status.json',dict(status='benchmarking',variant=variant,teacher_dp_size=1 if teacher=='teacher_dp1' else 2,excluded_gpus=[1,2],full_data_queue='paused'))
            cmd=[PYTHON,'-B',str(c/'controller.py'),'--root',str(r/variant)]
            with (r/variant/'controller.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            dump(r/variant/'controller_launch.json',dict(pid=p.pid,command=cmd));assert p.wait()==0,variant
            stop_teacher(active_teacher)
        dump(r/'status.json',dict(status='complete',excluded_gpus=[1,2],full_data_queue='paused'))
    except BaseException as exc:
        dump(r/'status.json',dict(status='failed',error=repr(exc),excluded_gpus=[1,2],full_data_queue='paused'));raise
    finally:
        if (active_teacher/'launch.json').exists() and not(active_teacher/'cleanup.json').exists():stop_teacher(active_teacher)


if __name__=='__main__':main()
