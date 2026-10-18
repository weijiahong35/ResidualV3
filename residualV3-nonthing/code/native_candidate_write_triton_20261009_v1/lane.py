"""Two implementation controls on one GPU; smoke then formal, preserved logs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

C=Path(__file__).resolve().parent
sys.path.insert(0,str(C/'native_runtime'))
from runtime.utils import dump,sha,now,stop
PYTHON='/data/weijiahong/miniconda3/envs/dflash/bin/python'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--concurrency',type=int,choices=[1,40],required=True)
    a=ap.parse_args();root=a.root.resolve()
    p=json.loads((root/'native_protocol.json').read_text())
    for name,digest in p['source_files'].items():assert sha(name)==digest,name
    gpu=p['gpu_by_concurrency'][f'C{a.concurrency}'];proc=None
    def interrupted(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for smoke in (True,False):
            label='smoke' if smoke else 'formal';output=root/('formal_smoke' if smoke else 'formal')
            command=[PYTHON,'-B',str(C/'native_evaluate.py'),'--training',str(root),'--gpu',str(gpu),
                '--concurrency',str(a.concurrency),'--output',str(output),'--tokens',str(32 if smoke else 2096),
                '--prompts',str(a.concurrency if smoke else p['prompts_per_concurrency'][f'C{a.concurrency}']),'--rounds','1']
            if smoke:command.append('--smoke')
            dump(root/'status.json',dict(status='native_evaluation',phase=label,gpu=gpu,utc=now()))
            with (root/f'single_card_C{a.concurrency}_{label}.log').open('x') as stream:
                proc=subprocess.Popen(command,env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'},
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                if (output/'status.json').exists():
                    dump(root/'status.json',dict(status='native_evaluation',phase=label,gpu=gpu,
                        details=json.loads((output/'status.json').read_text()),utc=now()))
                time.sleep(3)
            assert proc.returncode==0,f'{label} failed with {proc.returncode}'
            proc=None
        assert json.loads((root/'formal/summary.json').read_text())['status']=='complete'
        dump(root/'status.json',dict(status='complete',gpu=gpu,utc=now()))
    except BaseException as exc:
        dump(root/'status.json',dict(status='failed',error=repr(exc),utc=now()));raise
    finally:
        if proc is not None and proc.poll() is None:stop(proc)


if __name__=='__main__':main()
