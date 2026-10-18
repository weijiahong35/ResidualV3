"""Same-GPU baseline/optimized real update pilot; preserve all final weights."""
import argparse,json,subprocess,time,hashlib
from pathlib import Path

def dump(p,v):p.write_text(json.dumps(v,indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;c=Path(__file__).resolve().parent
    try:
        assert json.loads((r/'numeric_gate.json').read_text())['status']=='passed'
        for variant in ['baseline','optimized']:
            dump(r/'status.json',dict(status='benchmarking',variant=variant,full_data_queue='paused',excluded_gpus=[1,2]))
            cmd=['/data/weijiahong/miniconda3/envs/dflash/bin/python','-B',str(c/'controller.py'),'--root',str(r/variant)]
            with (r/variant/'controller.log').open('x') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            dump(r/variant/'controller_launch.json',dict(pid=p.pid,command=cmd));assert p.wait()==0,variant
        dump(r/'status.json',dict(status='complete',full_data_queue='paused',excluded_gpus=[1,2]))
    except BaseException as exc:
        dump(r/'status.json',dict(status='failed',error=repr(exc),full_data_queue='paused'));raise
    finally:
        dump(r/'teacher/stop.json',dict(reason='bounded head optimization pilot finished or failed'))
        end=time.monotonic()+180
        while not(r/'teacher/cleanup.json').exists() and time.monotonic()<end:time.sleep(1)
if __name__=='__main__':main()
