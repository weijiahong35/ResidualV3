"""Monitor one frozen short-test iteration, with explicit live logs and stages."""
import json,os,sys,fcntl,subprocess,time
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'native_runtime'))
from runtime.utils import resources,dump,now,stop
PYTHON='/data/weijiahong/miniconda3/envs/dflash/bin/python'


def main():
    root=Path(sys.argv[1]);c=int(sys.argv[2]);gpu=int(sys.argv[3]);p=json.loads((root/'native_protocol.json').read_text())
    assert p['short_development_only'] and p['gpu_by_concurrency'][f'C{c}']==gpu
    lease=(root.parents[2]/f'.native_gpu_{gpu}.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX)
    while True:
        free,pids=resources(gpu)
        if not pids and free>=78000:break
        dump(root/'status.json',dict(status='waiting_empty_gpu',gpu=gpu,free_mib=free,pids=pids,utc=now()))
        time.sleep(3)
    stages=[('smoke',['Q45_CUT'],max(c,2),64),('short',['FULL15','Q45_CUT'],p['prompts_per_concurrency'][f'C{c}'],256)]
    for phase,arms,count,tokens in stages:
        # Whole user directory, decimal GB: preserve headroom for compilation
        # artifacts rather than allocating beyond the user's storage limit.
        budget=int(p.get('user_disk_limit_bytes',600000000000))
        while True:
            used=int(subprocess.check_output(['du','-x','-B1','-s','/data/weijiahong'],text=True).split()[0])
            if used+4*1024**3<=budget:break
            dump(root/'status.json',dict(status='waiting_disk_headroom',allocated_bytes=used,
                budget_bytes=budget,protected_checkpoints_untouched=True,utc=now()))
            time.sleep(30)
        out=root/f'C{c}'/('short_smoke' if phase=='smoke' else 'short')
        cmd=[PYTHON,'-B',str(CODE/'native_evaluate.py'),'--training',str(root/f'C{c}'),'--gpu',str(gpu),
             '--concurrency',str(c),'--output',str(out),'--prompts',str(count),'--tokens',str(tokens),'--rounds','1','--arms',*arms]
        if phase=='smoke':cmd+=['--smoke']
        with (root/f'single_card_C{c}_{phase}.log').open('a') as log:
            proc=subprocess.Popen(cmd,cwd=CODE,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(root/'status.json',dict(status='running',phase=phase,gpu=gpu,pid=proc.pid,command=cmd,utc=now()))
        try:
            while proc.poll() is None:
                detail=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
                dump(root/'status.json',dict(status='running',phase=phase,gpu=gpu,pid=proc.pid,detail=detail,utc=now()))
                time.sleep(3)
            if proc.returncode:
                dump(root/'status.json',dict(status='failed',phase=phase,exit_code=proc.returncode,utc=now()))
                raise RuntimeError(f'{phase} failed, inspect immediate log')
        finally:
            if proc.poll() is None:stop(proc)
    result=json.loads((root/f'C{c}/short/summary.json').read_text());assert result['status']=='complete'
    baseline={w['T']:w for w in result['windows'] if w['arm']=='FULL15'}
    comparisons=[]
    for w in result['windows']:
        if w['arm']!='Q45_CUT':continue
        b=baseline[w['T']]
        comparisons.append(dict(T=w['T'],acceptance_delta_percent=(w['native_metrics']['acceptance_length']/b['native_metrics']['acceptance_length']-1)*100,
                tps_delta_percent=(w['tokens_per_second']/b['tokens_per_second']-1)*100,isolation=w['isolation_observed']))
    dump(root/'SHORT_COMPARISON.json',comparisons)
    dump(root/'status.json',dict(status='complete',short_only=True,comparisons=comparisons,utc=now()))
    print('SHORT_COMPLETE',comparisons,flush=True)


if __name__=='__main__':main()
