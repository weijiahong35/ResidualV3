"""Serial C1/C40 on one empty GPU; preserve every smoke and formal attempt."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

C=Path(__file__).resolve().parent
sys.path.insert(0,str(C/'native_runtime'))
from runtime.utils import dump,sha,now,stop
PYTHON='/data/weijiahong/miniconda3/envs/dflash/bin/python'


def report(root):
    rows=[]
    for concurrency in (1,40):
        evidence=json.loads((root/f'C{concurrency}/formal/summary.json').read_text())
        assert evidence['status']=='complete' and len(evidence['windows'])==4
        for temperature in (0,1):
            windows={w['arm']:w for w in evidence['windows'] if w['T']==temperature}
            base,fast=windows['BASE_HEAD'],windows['FAST_HEAD']
            rows.append(dict(concurrency=concurrency,temperature=temperature,
                baseline_AL=base['native_metrics']['acceptance_length'], optimized_AL=fast['native_metrics']['acceptance_length'],
                baseline_TPS=base['tokens_per_second'],optimized_TPS=fast['tokens_per_second'],
                TPS_gain_percent=100*(fast['tokens_per_second']/base['tokens_per_second']-1),
                performance_contaminated=any(w['performance_contaminated'] for w in windows.values())))
    pilot=json.loads((root/'pilot_summary.json').read_text())
    dump(root/'summary.json',dict(status='complete',selected=pilot['selected'],rows=rows,
        scope='same fixed joint body/head weights, block16; one formal round GSM8K nonthinking; head-only synthetic timing separated from native timing',utc=now()))
    lines=['同一联合训练权重：修复头实现优化对照','仅改变计算实现；实际lag-1、损失、权重、block16和stock sampler保持原样。',
        'C1=200/C40=1319，T0/T1，max2096；当前GSM8K开发面板。',
        '设置 | 原实现AL | 优化AL | 原TPS | 优化TPS | TPS变化 | 污染']
    for x in rows:
        lines.append(f"C{x['concurrency']} T{x['temperature']} | {x['baseline_AL']:.6f} | {x['optimized_AL']:.6f} | {x['baseline_TPS']:.3f} | {x['optimized_TPS']:.3f} | {x['TPS_gain_percent']:+.2f}% | {x['performance_contaminated']}")
    lines.extend(['','先通过同状态logits/q/采样/前驱bitwise校验再进入正式窗口。','synthetic microbenchmark只用于选择实现，不能当作端到端收益。单正式轮，非独立最终测试。'])
    (root/'RESULTS_CN.txt').write_text('\n'.join(lines)+'\n')


def run_pilot(root,gpu):
    from device import assignment,bind
    from runtime.utils import environment,EVAL,resources,members
    identity=assignment(gpu)
    env=bind(environment(),identity)
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_DETERMINISTIC='1',PYTHONHASHSEED='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    env.update(TRITON_CACHE_DIR=str(root/'compiled_cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(root/'compiled_cache/inductor'))
    command=[str(EVAL/'bin/python'),'-B',str(C/'pilot.py'),'--root',str(root)]
    proc=None;done=threading.Event();samples=[];errors=[]
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(sig,frame):raise KeyboardInterrupt(str(sig))
    for sig in previous_handlers:signal.signal(sig,interrupted)
    def watch():
        while not done.wait(5):
            try:
                free,pids=resources(gpu)
                samples.append(dict(utc=now(),free_mib=free,foreign_pids=sorted(set(pids)-set(members(proc.pid)))))
            except Exception as exc:errors.append(repr(exc))
    try:
        dump(root/'status.json',dict(status='head_microbenchmark',gpu=gpu,utc=now()))
        with (root/'pilot.log').open('x') as log:
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        watcher=threading.Thread(target=watch,daemon=True);watcher.start()
        assert proc.wait()==0, 'head microbenchmark failed; do not launch formal'
    finally:
        done.set()
        if proc:
            if proc.poll() is None:stop(proc)
            watcher.join(timeout=8)
        dump(root/'pilot_isolation.json',dict(samples=samples,errors=errors,performance_contaminated=bool(errors) or any(x['foreign_pids'] for x in samples)))
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
    value=json.loads((root/'pilot_summary.json').read_text())
    assert value['status']=='complete'
    return value['selected']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();root=args.root.resolve()
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=json.loads((root/'native_protocol.json').read_text())
    for filename,digest in p['source_files'].items():assert sha(filename)==digest,filename
    for key,path in p['panel_paths'].items():assert sha(path)==p['panel_sha256'][key]
    assert json.loads((Path(p['long_training'])/'status.json').read_text())['status']=='complete'
    assert json.loads((root/'cpu_equivalence.json').read_text())['status']=='passed'
    dump(root/'controller_pid.json',dict(pid=os.getpid(),source=str(Path(__file__).resolve()),utc=now()))
    try:
        selected=run_pilot(root,args.gpu)
    except BaseException as exc:
        dump(root/'status.json',dict(status='failed',phase='head_microbenchmark',error=repr(exc),utc=now()));raise
    if selected is None:
        dump(root/'status.json',dict(status='complete_no_adoption',reason='No strictly equivalent variant with >=3% speedup in every microbenchmark case',utc=now()))
        (root/'RESULTS_CN.txt').write_text('完整形状计算优化预实验已结束：没有实现同时通过bitwise一致性和四种设置各至少3%的提速门槛。未替换原头，未进行新大训练。\n详情见pilot_summary.json；microbenchmark不等于native性能。\n')
        return
    p['optimization_by_arm']={'BASE_HEAD':'baseline','FAST_HEAD':selected['variant']}
    dump(root/'native_protocol.json',p)
    processes={};mutex=threading.Lock();failures={};states={};interrupted=threading.Event()
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    dump(root/'controller_pid.json',dict(pid=os.getpid(),source=str(Path(__file__).resolve()),utc=now()))
    def update(stage,state):
        with mutex:
            states[stage]=state
            dump(root/'status.json',dict(status='native_evaluation',lanes=dict(states),acceptance_only=False,utc=now()))
    def signal_stop(sig,frame):
        interrupted.set();raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    def lane(concurrency):
        stage=f'C{concurrency}';gpu=p['gpu_by_concurrency'][stage]
        try:
            for smoke in (True,False):
                label='smoke' if smoke else 'formal'
                command=[PYTHON,'-B',str(C/'native_evaluate.py'),'--training',str(root),'--gpu',str(gpu),
                    '--concurrency',str(concurrency),'--output',str(root/stage/('formal_smoke' if smoke else 'formal')),
                    '--tokens',str(32 if smoke else 2096),'--prompts',str(concurrency if smoke else p['prompts_per_concurrency'][stage]),'--rounds','1']
                if smoke:command.append('--smoke')
                update(stage,dict(status='starting',phase=label,gpu=gpu))
                with (root/f'single_card_{stage}_{label}.log').open('x') as stream:
                    proc=subprocess.Popen(command,cwd=C,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                with mutex:
                    processes[stage]=proc
                    with (root/'process_history.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,phase=label,pid=proc.pid,command=command,utc=now()))+'\n')
                while proc.poll() is None and not interrupted.wait(3):
                    status=root/stage/('formal_smoke' if smoke else 'formal')/'status.json'
                    if status.exists():update(stage,dict(status=json.loads(status.read_text()),phase=label,gpu=gpu))
                assert not interrupted.is_set(),'controller interrupted'
                assert proc.returncode==0,f'{stage} {label} exited {proc.returncode}'
                with mutex:processes.pop(stage,None)
            update(stage,dict(status='complete',gpu=gpu))
        except BaseException as exc:
            with mutex:failures[stage]=dict(error=repr(exc),traceback=traceback.format_exc())
            update(stage,dict(status='failed',gpu=gpu,error=repr(exc)))
    def sequential_lanes():
        for concurrency in (1,40):
            if interrupted.is_set() or failures:break
            lane(concurrency)
    threads=[threading.Thread(target=sequential_lanes)]
    try:
        for thread in threads:thread.start()
        while any(thread.is_alive() for thread in threads):
            for thread in threads:thread.join(timeout=.5)
        assert not failures,failures
        report(root)
        dump(root/'status.json',dict(status='complete',acceptance_only=False,utc=now()))
    except BaseException as exc:
        dump(root/'status.json',dict(status='failed',error=repr(exc),lane_failures=failures,utc=now()));raise
    finally:
        interrupted.set()
        with mutex:active=list(processes.values())
        for proc in active:
            if proc.poll() is None:stop(proc)
        for thread in threads:
            if thread.is_alive():thread.join(timeout=30)


if __name__=='__main__':
    main()
