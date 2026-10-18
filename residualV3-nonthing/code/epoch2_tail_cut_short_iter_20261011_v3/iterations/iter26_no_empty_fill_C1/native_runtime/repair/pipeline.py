"""Owned GPU4 hidden server and GPU5 scratch training, then C1 native."""
import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from .protocol import read
from .isolation import inspect_owned
from runtime.utils import ROOT,EVAL,dump,sha,now,resources,environment,healthy,stop,members
from device import assignment,bind

PACKAGE='repair'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke-only',action='store_true');ap.add_argument('--training-only',action='store_true');a=ap.parse_args()
    out=a.output.resolve();protocol=out/'preregistration.json';p=read(protocol);gpu=p['physical_gpu']
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'launch.json').exists(),'preserve old launch'
    gate=json.loads((out/'cpu_validation.json').read_text())
    assert gate['status']=='passed' and gate['protocol_sha256']==sha(protocol)
    identity=assignment(gpu);free,pids=resources(gpu)
    assert free>=77000 and not pids,(gpu,free,pids)
    cfg=p['hidden_servers']['0'];transfer=Path(cfg['transfer_root'])
    assert not transfer.exists(),'preserve existing transfer root'
    with socket.socket() as sock:sock.bind(('127.0.0.1',cfg['port']))
    py=str(Path('/data/weijiahong/miniconda3/envs/dflash/bin/python'))
    training=os.environ.copy();training.pop('LD_PRELOAD',None)
    training.update(PYTHONPATH=f'{ROOT}:{ROOT}/src:{ROOT}/hs_connectors/src',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                    TORCHINDUCTOR_CACHE_DIR=p['training_cache']['inductor'],TRITON_CACHE_DIR=p['training_cache']['triton'])
    bind(training,identity)
    server_identity=assignment(p['server_gpu'])
    server_free,server_pids=resources(p['server_gpu']);assert server_free>=77000 and not server_pids,(server_free,server_pids)
    serving=bind(environment(),server_identity)
    serving.update(PYTHONUNBUFFERED='1',TRITON_CACHE_DIR=str(out/'eval_triton'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    procs=[];stage='preflight';finished=False
    def status(state,**kw):
        dump(out/'pipeline_status.json',dict(status=state,stage=stage,mode=p['mode'],gpu=gpu,utc=now(),**kw))
    def spawn(cmd,name,env):
        dump(out/f'{name}_command.json',dict(command=cmd,CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],utc=now()))
        with (out/f'{name}.log').open('x') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        procs.append(proc)
        with (out/'process_history.jsonl').open('a') as history:
            history.write(json.dumps(dict(pid=proc.pid,name=name,command=cmd,utc=now()))+'\n')
        status('running',pid=proc.pid,log=str(out/f'{name}.log'));return proc
    def isolated():
        event=inspect_owned((gpu,p['server_gpu']),procs)
        if event['exited_during_query'] or event['vanished_owned_roots']:
            with (out/'isolation_exit_events.jsonl').open('a') as f:f.write(json.dumps(dict(utc=now(),**event))+'\n')
    def run(cmd,name,env):
        proc=spawn(cmd,name,env)
        while proc.poll() is None:
            isolated()
            for other in procs:
                if other is server and other.poll() is not None:raise RuntimeError('hidden server died')
            time.sleep(5)
        if proc.returncode:raise RuntimeError(f'{name} failed; see {name}.log')
    def interrupted(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    dump(out/'launch.json',dict(pid=os.getpid(),identity=identity,server_identity=server_identity,initial_free_mib=free,protocol_sha256=sha(protocol),utc=now()))
    server=None
    try:
        stage='cuda_math_gate'
        run([py,'-B','-m',PACKAGE+'.validate','--protocol',str(protocol),'--output',str(out/'cuda_validation.json'),'--device','cuda'],'cuda_validation',training)
        stage='prepare_initialization_native'
        run([py,'-B','-m',PACKAGE+'.prepare_native','--output',str(out),'--initialization-smoke'],'prepare_initialization_native',training)
        stage='initialization_native_smoke'
        # Evaluator owns server groups and its isolation checks. No trained weights.
        proc=spawn([py,'-B','-m',PACKAGE+'.evaluate','--training',str(out/'initialization_native'),'--gpu',str(gpu),'--concurrency','1','--output',str(out/'initialization_native/C1_smoke'),'--tokens','32','--prompts','1','--rounds','1','--arms','EMBEDDING','--smoke'],stage,training)
        if proc.wait():raise RuntimeError('initialized-head native smoke failed')
        stage='online_target_startup';isolated();transfer.mkdir()
        cmd=[str(EVAL/'bin/python'),'-B',str(ROOT/'scripts/launch_vllm.py'),str(Path('/data/weijiahong/huggingface/models/Qwen3-8B')),
             '--target-layer-ids','2','10','18','26','34','--hidden-states-path',str(transfer),'--',
             '--host','127.0.0.1','--port',str(cfg['port']),'--tensor-parallel-size','1','--data-parallel-size','1',
             '--dtype','bfloat16','--max-num-seqs',str(p['hidden_prefetch']),'--max-model-len','9216','--max-num-batched-tokens',str(p['hidden_max_batched_tokens']),
             '--gpu-memory-utilization',str(cfg['memory_utilization']),'--no-enable-prefix-caching','--enforce-eager','--disable-uvicorn-access-log']
        server=spawn(cmd,f'vllm_gpu{p["server_gpu"]}',dict(serving,VLLM_USE_V2_MODEL_RUNNER='0'));deadline=time.monotonic()+600
        while not healthy(cfg['port']):
            if server.poll() is not None:raise RuntimeError('hidden server startup failed')
            if time.monotonic()>deadline:raise TimeoutError('hidden server startup')
            time.sleep(1)
        stage='online_hidden_parity'
        run([py,'-B','-m','runtime.check_vllm_hidden','--gpu','0','--protocol',str(protocol),'--output',str(out/'hidden_parity.json')],'hidden_parity',training)
        stage='online_causality'
        run([py,'-B','-m',PACKAGE+'.causality','--protocol',str(protocol),'--output',str(out/'online_causality.json')],'online_causality',training)
        stage='training_smoke'
        run([py,'-B','-m',PACKAGE+'.run','--protocol',str(protocol),'--output',str(out/'smoke'),'--smoke'],'smoke',training)
        smoke_result=json.loads((out/'smoke/summary.json').read_text())
        assert smoke_result['status']=='complete' and smoke_result['steps']==2
        for arm in ('EMBEDDING',):
            assert json.loads((out/'smoke'/arm/'real_input_effects.json').read_text())['passed']
            assert json.loads((out/'smoke'/arm/'real_input_gradients.json').read_text())['passed']
        native_root=out/'initialization_native/C1_smoke/round_1'
        features=json.loads((out/'smoke/feature_sources.json').read_text())
        for arm in ('EMBEDDING',):
            native_features=json.loads((native_root/arm/'model_contract.json').read_text())['input_feature_signature']
            assert native_features==features[arm],('native/training feature table mismatch',arm)
        dump(out/'feature_table_parity.json',dict(passed=True,arms=['EMBEDDING']))
        if a.smoke_only:
            finished=True
            return
        stage=p['mode']
        run([py,'-B','-m',PACKAGE+'.run','--protocol',str(protocol),'--output',str(out/p['mode'])],p['mode'],training)
        result=json.loads((out/p['mode']/'summary.json').read_text());assert result['status']=='complete'
        assert json.loads((out/'training/fresh_reset_verified.json').read_text())['passed']
        # Native quality is assessed online only. Release hidden server first.
        stop(server);server=None
        stage='prepare_native'
        run([py,'-B','-m',PACKAGE+'.prepare_native','--output',str(out)],'prepare_native',training)
        if a.training_only:
            finished=True
            return
        for smoke in (True,False):
            stage='C1_smoke' if smoke else 'C1_formal'
            destination=out/'native'/('C1_smoke' if smoke else 'C1')
            cmd=[py,'-B','-m',PACKAGE+'.evaluate','--training',str(out/'native'),'--gpu',str(gpu),'--concurrency','1','--output',str(destination),'--tokens','32' if smoke else '4096','--prompts','1' if smoke else '20','--rounds','1']
            if smoke:cmd.append('--smoke')
            # Native evaluator owns its own server sessions and isolation checks.
            proc=spawn(cmd,stage,training)
            if proc.wait():raise RuntimeError(f'{stage} failed')
        stage='report'
        run([py,'-B','-m',PACKAGE+'.report','--output',str(out)],'report',training)
        finished=True
    except BaseException as exc:
        status('failed',error=repr(exc));raise
    finally:
        for proc in reversed(procs):
            if proc.poll() is None or members(proc.pid):stop(proc)
        leftovers=[]
        if transfer.exists():
            leftovers=list(transfer.iterdir())
            for f in leftovers:
                assert f.is_file(),'unexpected transfer content'
                f.unlink()
            transfer.rmdir()
        dump(out/'cleanup.json',dict(owned_processes_stopped=True,temporary_transfer_removed=True,
             remaining_before_cleanup=[str(f) for f in leftovers],utc=now()))
        if finished:status('complete',summary=str(out/('smoke/summary.json' if a.smoke_only else 'training/summary.json' if a.training_only else 'native/C1/summary.json')))

if __name__=='__main__':main()
