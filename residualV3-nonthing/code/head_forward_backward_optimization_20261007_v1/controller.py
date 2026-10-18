"""Spawn two independent real learners and supervise one online hidden producer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
CODE=Path(__file__).resolve().parent
sys.path[:0]=[str(CODE),str(CODE.parent/'runtime_bundle')]
from runtime.utils import dump,now

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    import torch.multiprocessing as mp
    from learner_worker import entry
    ctx=mp.get_context('spawn');queues={arm:ctx.Queue(maxsize=2) for arm in ['joint','body']};acks=ctx.Queue()
    children={};monitor_stop=threading.Event()
    def monitor():
        while not monitor_stop.is_set():
            try:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                tasks=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
                with (r/'gpu_monitor.jsonl').open('a') as f:f.write(json.dumps(dict(utc=now(),gpu_rows=rows.splitlines(),process_rows=tasks.splitlines()))+'\n')
            except Exception as exc:dump(r/'monitor_error.json',dict(error=repr(exc)))
            monitor_stop.wait(5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        for arm in queues:
            proc=ctx.Process(target=entry,args=(str(r),arm,queues[arm],acks));proc.start();children[arm]=proc
        dump(r/'launch.json',dict(controller_pid=os.getpid(),worker_pids={arm:p.pid for arm,p in children.items()},utc=now()))
        deadline=time.monotonic()+900
        while not (r/'driver_ready.json').exists() or not all((r/arm/'model_ready.json').exists() for arm in children):
            assert time.monotonic()<deadline,'Startup timeout'
            for arm,proc in children.items():assert proc.is_alive(),f'{arm} exited {proc.exitcode}'
            time.sleep(1)
        from shared_broker import run_broker
        run_broker(r,queues,acks,children)
        for arm,proc in children.items():
            proc.join(timeout=300);assert proc.exitcode==0,(arm,proc.exitcode)
        dump(r/'status.json',dict(status='complete',utc=now()))
    except BaseException as exc:
        dump(r/'controller_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),utc=now()))
        dump(r/'status.json',dict(status='failed',error=repr(exc),utc=now()));raise
    finally:
        for proc in children.values():
            if proc.is_alive():proc.terminate();proc.join(timeout=30)
        p=json.loads((r/'protocol.json').read_text())
        if p.get('stop_teacher_on_exit',True) or (r/'controller_failure.json').exists():
            dump(r/'teacher/stop.json',dict(reason='Shared-hidden bounded pilot finished or failed',utc=now()))
            deadline=time.monotonic()+180
            while not (r/'teacher/cleanup.json').exists() and time.monotonic()<deadline:time.sleep(1)
        monitor_stop.set();thread.join(timeout=10)
        dump(r/'cleanup.json',dict(learners_stopped=all(not p.is_alive() for p in children.values()),teacher_cleaned=(r/'teacher/cleanup.json').exists(),foreign_processes_untouched=True,utc=now()))

if __name__=='__main__':main()
