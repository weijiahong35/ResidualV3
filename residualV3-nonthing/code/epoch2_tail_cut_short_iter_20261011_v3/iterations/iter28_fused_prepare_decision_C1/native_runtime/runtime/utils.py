#!/usr/bin/env python3
"""Native T0/T1 generation: real proposal-prefix verification and simultaneous timing.
Fixed full-vocabulary LM-head checkpoint. Independent local repair experiment.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.request
import numpy as np
from .http import _http_json, _http_text, _metrics_delta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FAMILY = 'k256-cost-diagnostic'
POLICY = 'bf16-local-repair-v1-fp32-reductions'
EVAL = Path('/data/weijiahong/miniconda3/envs/llm_dflash_eval_v026')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def freeze(p,x):
    if p.exists() and json.loads(p.read_text()) != x: raise RuntimeError(f'protocol changed: {p}')
    dump(p,x)

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def resources(gpu):
    lines=subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    uuid,free=lines[0].strip(),int(lines[1])
    tasks=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return free,[int(x.split(',')[1]) for x in tasks.splitlines() if x.split(',')[0].strip()==uuid]

def members(pgid):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            f=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(f[2])==pgid and f[0]!='Z':result.append(int(p.name))
        except (OSError,ValueError,IndexError):pass
    return result

def stop(proc):
    for sig,seconds in [(signal.SIGTERM,15),(signal.SIGKILL,10)]:
        try:os.killpg(proc.pid,sig)
        except ProcessLookupError:pass
        end=time.monotonic()+seconds
        while members(proc.pid) and time.monotonic()<end:time.sleep(.2)
        if not members(proc.pid):break
    proc.wait(timeout=2)
    if members(proc.pid):raise RuntimeError('owned server group survived cleanup')

def healthy(port):
    try:return urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2).status==200
    except Exception:return False

def environment():
    env=os.environ.copy();env.pop('LD_PRELOAD',None)
    for key in list(env):
        if key.startswith(('DIAG_', 'ONLINE_', 'LOCAL_', 'STATIC_', 'REPAIR_')) or key=='NATIVE_FULL_GRAPH':
            env.pop(key)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_ENABLE_CUDA_COMPATIBILITY='1',
        VLLM_CUDA_COMPATIBILITY_PATH=str(EVAL/'cuda-compat'),LD_LIBRARY_PATH=f'{EVAL}/cuda-compat:{EVAL}/lib',
        PYTHONPATH=f'{ROOT}:{ROOT}/src:{ROOT}/hs_connectors/src',PYTHONDONTWRITEBYTECODE='1')
    return env

