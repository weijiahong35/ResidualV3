import argparse,json,os,signal,subprocess,fcntl
from pathlib import Path
from runtime.utils import ROOT,dump,sha,now,stop,resources

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--training',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(exist_ok=False)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    training=a.training.resolve();summary=json.loads((ROOT/'provenance/original_run/training/summary.json').read_text())
    assert summary['status']=='complete' and summary['steps']==384
    free,pids=resources(4);assert free>=77000 and not pids,(free,pids)
    py=str(Path('/data/weijiahong/miniconda3/envs/dflash/bin/python'));env=os.environ.copy();env['PYTHONPATH']=f'{ROOT}:{ROOT}/src:{ROOT}/hs_connectors/src';env['PYTHONUNBUFFERED']='1'
    dump(out/'launch.json',dict(pid=os.getpid(),gpu=4,concurrency=40,training=str(training),training_summary_sha256=sha(ROOT/'provenance/original_run/training/summary.json'),utc=now()))
    child=None;stage='starting';completed=False
    def interrupted(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
      for smoke in (True,False):
        stage='C40_smoke' if smoke else 'C40'
        cmd=[py,'-B','-m','evaluation_c40.evaluate','--training',str(training),'--gpu','4','--concurrency','40','--output',str(out/stage),'--tokens','128' if smoke else '4096','--prompts','300','--rounds','2']
        if smoke:cmd.append('--smoke')
        dump(out/(stage+'_command.json'),dict(command=cmd,utc=now()))
        with (out/(stage+'.log')).open('x') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        dump(out/'status.json',dict(status='running',stage=stage,pid=child.pid,log=str(out/(stage+'.log')),utc=now()))
        if child.wait():raise RuntimeError(f'{stage} failed; see {stage}.log')
      stage='report'
      subprocess.run([py,'-B','-m','evaluation_c40.report','--output',str(out)],cwd=ROOT,env=env,check=True)
      completed=True
    except BaseException as exc:
      dump(out/'status.json',dict(status='failed',stage=stage,error=repr(exc),utc=now()));raise
    finally:
      if child is not None and child.poll() is None:stop(child)
      if completed:dump(out/'status.json',dict(status='complete',utc=now(),results=str(out/'RESULTS.md')))
if __name__=='__main__':main()
