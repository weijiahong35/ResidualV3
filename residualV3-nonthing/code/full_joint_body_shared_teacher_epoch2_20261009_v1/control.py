"""File-based checkpoint/pause/resume; never signals unrelated processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from full_checkpoint import atomic

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['status','checkpoint','pause','resume']);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    if a.action=='status':
        print((root/'status.json').read_text())
        if(root/'latest_checkpoint.json').exists():
            d=json.loads((root/'latest_checkpoint.json').read_text());print('Latest committed cursor:',d['cursor'],'slot:',d['slot'])
        return
    atomic(root/f'{a.action}.request.json',dict(action=a.action,requested_unix_time=time.time(),source='user_control'))
    if a.action=='resume':
        alive=False
        if(root/'controller_pid.json').exists():
            try:
                pid=json.loads((root/'controller_pid.json').read_text())['pid']
                os.kill(pid,0)
                alive=str(Path(__file__).with_name('outer_controller.py')) in Path(f'/proc/{pid}/cmdline').read_text()
            except ProcessLookupError:pass
        if not alive:
            command=['/data/weijiahong/miniconda3/envs/dflash/bin/python','-B',str(Path(__file__).with_name('outer_controller.py')),'--root',str(root)]
            with(root/'controller.log').open('a') as log:subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    print('Requested:',a.action,'; checkpoint/pause waits for both learners to finish the produced packs.')

if __name__=='__main__':main()
