"""Sequential A/B on the same GPUs and online teacher, with fresh learners."""
import argparse,json,os,subprocess,time
from pathlib import Path

def dump(p,v):p.write_text(json.dumps(v,indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;c=Path(__file__).resolve().parent
    try:
        for variant in ['baseline','transport_optimized']:
            dump(r/'status.json',dict(status='benchmarking',variant=variant,full_data_queue='paused'))
            cmd=['/data/weijiahong/miniconda3/envs/dflash/bin/python','-B',str(c/'controller.py'),'--root',str(r/variant)]
            with (r/variant/'controller.log').open('x') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            dump(r/variant/'controller_launch.json',dict(pid=proc.pid,command=cmd));status=proc.wait();assert status==0,(variant,status)
        dump(r/'status.json',dict(status='complete',full_data_queue='paused'))
    except BaseException as exc:
        dump(r/'status.json',dict(status='failed',error=repr(exc),full_data_queue='paused'));raise
    finally:
        dump(r/'teacher/stop.json',dict(reason='paired bounded timing complete or failed'))
        deadline=time.monotonic()+180
        while not(r/'teacher/cleanup.json').exists() and time.monotonic()<deadline:time.sleep(1)
if __name__=='__main__':main()
