"""Bounded paired pilot; monitor and stop only the study's teacher services."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone

CODE=Path(__file__).resolve().parent
PY='/data/weijiahong/miniconda3/envs/dflash/bin/python'
def now():return datetime.now(timezone.utc).isoformat()
def dump(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def smi(args):return subprocess.check_output(['nvidia-smi',*args],text=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    worker=None
    try:
        for dp in [1,2]:
            service=r/f'service_dp{dp}';assert (service/'endpoint.json').exists()
            assert json.loads((service/'status.json').read_text())['status']!='failed'
        inventory=[]
        apps=smi(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])
        busy={line.split(',')[0].strip() for line in apps.splitlines()}
        for line in smi(['--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits']).splitlines():
            idx,uuid,free=[x.strip() for x in line.split(',')];inventory.append(dict(index=int(idx),uuid=uuid,free=int(free)))
        candidates=[g for g in inventory if g['index'] not in [0,1,2,3] and g['uuid'] not in busy and g['free']>=75000]
        candidates.sort(key=lambda g:(g['index']!=6,g['index']));assert candidates,'No empty learner GPU with sufficient memory'
        gpu=candidates[0]
        resolved=dict(learner=gpu,teacher_dp1=json.loads((r/'service_dp1/endpoint.json').read_text()),teacher_dp2=json.loads((r/'service_dp2/endpoint.json').read_text()),utc=now())
        dump(r/'resolved_resources.json',resolved)
        dump(r/'source_freeze.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(CODE.iterdir()) if p.is_file() and p.suffix in ['.py','.so','.c']})
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True',TRITON_CACHE_DIR=str(r/'triton_cache'),TORCHINDUCTOR_CACHE_DIR=str(r/'inductor_cache'))
        cmd=[PY,'-B',str(CODE/'study_worker.py'),'--root',str(r)]
        with (r/'training_benchmark.log').open('x') as log:
            worker=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(r/'worker_launch.json',dict(pid=worker.pid,command=cmd,gpu=gpu,utc=now()))
        dump(r/'status.json',dict(status='starting_worker',controller_pid=os.getpid(),worker_pid=worker.pid,utc=now()))
        while worker.poll() is None:
            sample=dict(utc=now(),gpu=smi(['--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits']),processes=smi(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']))
            with (r/'gpu_monitor.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
            time.sleep(5)
        assert worker.returncode==0,f'Worker failed exit {worker.returncode}'
        profiles=[]
        for dp in [1,2]:profiles.extend(json.loads((r/f'dp{dp}/summary.json').read_text())['profiles'])
        gate=json.loads((r/'dp1/full1536_head_probe/summary.json').read_text())
        dump(r/'summary.json',dict(status='complete',profiles=profiles,full1536_head_probe=gate,scope='Head-only training pilots, frozen target/draft; GPU timings exclude native inference conclusions',utc=now()))
        with (r/'timing_comparison.csv').open('x',newline='') as f:
            keys=['teacher_dp_size','phase','head_mode','mean_seconds','mean_hidden_wait_seconds','mean_draft_seconds','mean_head_seconds','input_tokens_per_second']
            w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(profiles)
        dump(r/'status.json',dict(status='complete',controller_pid=os.getpid(),worker_pid=worker.pid,utc=now()))
    except BaseException as exc:
        dump(r/'status.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),utc=now()));raise
    finally:
        if worker is not None and worker.poll() is None:
            # A controller error does not kill a running training update.
            dump(r/'controller_error_worker_alive.json',dict(pid=worker.pid,utc=now()))
        else:
            for dp in [1,2]:dump(r/f'service_dp{dp}/stop.json',dict(reason='bounded paired pilot finished',utc=now()))

if __name__=='__main__':main()
