"""Bounded kernel pilot, then independent C1/C40 same-weight native controls."""
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
from device import assignment,bind
from runtime.utils import dump,sha,now,environment,EVAL,resources,stop
BASE=C.parent.parent
PYTHON='/data/weijiahong/miniconda3/envs/dflash/bin/python'


def descendants(pid):
    parents={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            parents[int(path.name)]=int(fields[1])
        except (OSError,ValueError,IndexError):pass
    owned={pid}
    while True:
        more={child for child,parent in parents.items() if parent in owned}-owned
        if not more:return owned
        owned.update(more)


def acquire(preferred):
    for gpu in [preferred,*[i for i in range(8) if i!=preferred]]:
        free,pids=resources(gpu)
        if pids or free<78000:continue
        lease=(BASE/'results'/f'.native_gpu_{gpu}.lock').open('a')
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lease.close();continue
        free,pids=resources(gpu)
        if pids or free<78000:lease.close();continue
        return gpu,lease
    return None,None


def report(root):
    p=json.loads((root/'native_protocol.json').read_text());rows=[];verified_artifacts=0;output_audit=[]
    for c,count in ((1,200),(40,1319)):
        evidence=json.loads((root/f'C{c}/formal/summary.json').read_text())
        assert evidence['status']=='complete' and len(evidence['windows'])==4
        by={(w['arm'],w['T']):w for w in evidence['windows']}
        assert set(by)=={(arm,t) for arm in ('BASE_HEAD','FAST_HEAD') for t in (0,1)}
        texts={}
        for arm in ('BASE_HEAD','FAST_HEAD'):
            trial=root/f'C{c}/formal/round_1/{arm}';complete=json.loads((trial/'complete.json').read_text())
            assert complete['status']=='complete'
            for name,digest in complete['artifacts'].items():
                assert sha(trial/name)==digest,(c,arm,name);verified_artifacts+=1
            for t in (0,1):
                requests=[json.loads(line) for line in (trial/f'T{t}_requests.jsonl').read_text().splitlines()]
                assert len(requests)==count and {x['index'] for x in requests}==set(range(count))
                assert all(x['temperature']==t and x['seed']==42+x['index'] and 'error' not in x for x in requests)
                w=by[arm,t];assert w['requests']==count and not w['smoke'] and not w['diagnostic_only']
                for name in ('execution_counters','target_execution_counters'):
                    counters=w[name];assert counters['captures']==counters['fallbacks']==counters['eager']==0 and counters['replays']>0
                if t==0:texts[arm]={x['index']:x['response']['choices'][0]['message']['content'] for x in requests}
        output_audit.append(dict(C=c,requests=count,T0_text_differences=sum(texts['BASE_HEAD'][i]!=texts['FAST_HEAD'][i] for i in range(count))))
        for t in (0,1):
            base,fast=by['BASE_HEAD',t],by['FAST_HEAD',t]
            rows.append(dict(C=c,T=t,base_AL=base['native_metrics']['acceptance_length'],fast_AL=fast['native_metrics']['acceptance_length'],
                base_TPS=base['tokens_per_second'],fast_TPS=fast['tokens_per_second'],
                TPS_gain_percent=100*(fast['tokens_per_second']/base['tokens_per_second']-1),
                performance_contaminated=base['performance_contaminated'] or fast['performance_contaminated']))
    for name,digest in p['source_files'].items():assert sha(name)==digest,name
    pilot=json.loads((root/'pilot_summary.json').read_text());assert pilot['selected']['variant']=='triton_write'
    dump(root/'summary.json',dict(status='complete',rows=rows,pilot_selected=pilot['selected'],T0_output_audit=output_audit,
        verified_artifact_hashes=verified_artifacts,head_weight_sha256=p['native_checkpoints']['BASE_HEAD']['sha256'],
        limitations='One formal round, development GSM8K panel. Implementation parity is on identical state; cross-run target batching/numerical paths need not be bitwise identical.',utc=now()))
    lines=['Top256候选专用回写内核：同权重实现优化（已核验）',
        '只改变回写内核及私有logits存储；BF16候选计算、三层no-soft头、全词表stock采样、实际lag-1、verifier读取值保持相同。',
        '没有训练；现有权重和冻结源码未改。新代码独立目录。',
        '微测人工输入、固定正式权重；不是native接受长度或端到端TPS。',
        '微测同时保留旧storage方案作控制，以分离少复制和替换scatter_的作用。',
        'C1=200/C40=1319，GSM8K nonthinking，block16=anchor+15proposal，max2096，T0/T1，一轮。',
        '设置 | 原AL | 优化AL | 原TPS | 优化TPS | TPS变化 | 污染']
    for x in rows:lines.append(f"C{x['C']} T{x['T']} | {x['base_AL']:.6f} | {x['fast_AL']:.6f} | {x['base_TPS']:.3f} | {x['fast_TPS']:.3f} | {x['TPS_gain_percent']:+.2f}% | {x['performance_contaminated']}")
    lines+=['','相同状态的完整logits、实际token、lag-1前驱、verifier q、Graph输入更新与复原一致性见pilot_summary.json；真实native一致性见各FAST_HEAD/head_implementation_parity.jsonl。']
    for x in output_audit:lines.append(f"C{x['C']}：{x['requests']}条T0最终文本有{x['T0_text_differences']}条跨实现不同；这是文本一致性检查，不是准确率。")
    lines+=['C40跨运行批调度/target浮点路径可以变化；同状态head bitwise证明不扩大为完整target逐状态证明。',f'核验{verified_artifacts}正式产物SHA及8个正式窗口。']
    (root/'RESULTS_CN.txt').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);root=ap.parse_args().root.resolve()
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=json.loads((root/'native_protocol.json').read_text())
    for name,digest in p['source_files'].items():assert sha(name)==digest,name
    assert json.loads((root/'cpu_equivalence.json').read_text())['status']=='passed'
    assert json.loads((Path(p['long_training'])/'status.json').read_text())['status']=='complete'
    active={};leases={};samples={};states={};mutex=threading.Lock();monitor_stop=threading.Event()
    def interrupted(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def monitor():
        while not monitor_stop.wait(5):
            with mutex:jobs=list(active.items())
            for label,(proc,gpu) in jobs:
                try:
                    free,pids=resources(gpu)
                    sample=dict(utc=now(),free_mib=free,foreign_pids=sorted(set(pids)-descendants(proc.pid)))
                except Exception as exc:sample=dict(utc=now(),monitor_error=repr(exc))
                with mutex:samples.setdefault(label,[]).append(sample)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    def launch(label,gpu,lease,command,destination,log):
        identity=assignment(gpu);env=bind(environment(),identity)
        env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_DETERMINISTIC='1',PYTHONHASHSEED='0',
            TRITON_CACHE_DIR=str(root/'triton_cache'))
        with log.open('x') as stream:
            proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        with mutex:active[label]=(proc,gpu)
        leases[label]=lease;states[label]=dict(status='running',gpu=gpu,pid=proc.pid)
        dump(destination/f'launch_{label}.json',dict(pid=proc.pid,command=command,assignment=identity,utc=now()))
        print(now(),'LAUNCHED',label,gpu,proc.pid,flush=True)
        return proc
    def finished(label):
        with mutex:proc,gpu=active.pop(label)
        leases.pop(label).close()
        with mutex:values=samples.get(label,[])
        dump(root/f'isolation_{label}.json',dict(samples=values,performance_contaminated=any(x.get('foreign_pids') or 'monitor_error' in x for x in values)))
        assert proc.returncode==0,f'{label} failed {proc.returncode}'
        states[label]=dict(status='complete',gpu=gpu,exit_code=0)
    dump(root/'controller_pid.json',dict(pid=os.getpid(),utc=now()))
    try:
        gpu,lease=acquire(7)
        while gpu is None:
            dump(root/'status.json',dict(status='waiting_empty_gpu',phase='pilot',utc=now()));time.sleep(10);gpu,lease=acquire(7)
        pilot=launch('PILOT',gpu,lease,[str(EVAL/'bin/python'),'-B',str(C/'pilot.py'),'--root',str(root)],root,root/'pilot.log')
        dump(root/'status.json',dict(status='head_microbenchmark',gpu=gpu,pid=pilot.pid,utc=now()))
        while pilot.poll() is None:time.sleep(3)
        finished('PILOT')
        result=json.loads((root/'pilot_summary.json').read_text())
        if result['selected'] is None:
            dump(root/'status.json',dict(status='complete_no_adoption',reason='No triton_write variant with exact checks and >=3% speedup in all4 full-head settings',utc=now()))
            (root/'RESULTS_CN.txt').write_text('专用候选回写预实验完成，未达到同时逐位一致且四个整头设置均至少3%提速的门槛；未启动正式测试，未改原权重。\n见pilot_summary.json。\n')
            return
        pending=[('C1',5,1),('C40',2,40)]
        while pending or active:
            for label,preferred,c in list(pending):
                gpu,lease=acquire(preferred)
                if gpu is None:states[label]=dict(status='waiting_empty_gpu');continue
                destination=root/label;destination.mkdir()
                lp=dict(p);lp['gpu_by_concurrency']={f'C{c}':gpu}
                dump(destination/'native_protocol.json',lp)
                launch(label,gpu,lease,[PYTHON,'-B',str(C/'lane.py'),'--root',str(destination),'--concurrency',str(c)],
                    destination,destination/'controller.log')
                pending.remove((label,preferred,c))
            with mutex:jobs=list(active.items())
            for label,(proc,gpu) in jobs:
                if proc.poll() is not None:finished(label)
            dump(root/'status.json',dict(status='native_evaluation' if pending or active else 'verifying',jobs=states,utc=now()))
            if pending or active:time.sleep(3)
        report(root)
        dump(root/'status.json',dict(status='complete',jobs=states,formal_windows=8,utc=now()))
    except BaseException as exc:
        dump(root/'status.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),jobs=states,utc=now()));raise
    finally:
        monitor_stop.set();watcher.join(timeout=8)
        with mutex:remaining=list(active.values())
        for proc,gpu in remaining:
            if proc.poll() is None:stop(proc)
        for lease in leases.values():lease.close()


if __name__=='__main__':main()
