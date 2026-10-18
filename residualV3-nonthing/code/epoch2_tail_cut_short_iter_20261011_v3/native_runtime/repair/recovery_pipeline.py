"""Independent recovery: extra GPU processes are telemetry, never a stop condition."""
import argparse, fcntl, json, os, signal, socket, subprocess, time
from pathlib import Path
from .protocol import read
from runtime.utils import ROOT, EVAL, dump, sha, now, resources, environment, healthy, stop
from device import assignment, bind


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--training-only',action='store_true')
    out=ap.parse_args().output.resolve();protocol=out/'preregistration.json';p=read(protocol);gpu=p['physical_gpu']
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'launch.json').exists()
    gate=json.loads((out/'cpu_validation.json').read_text());assert gate['status']=='passed' and gate['protocol_sha256']==sha(protocol)
    identity=assignment(gpu);cfg=p['hidden_servers']['0'];transfer=Path(cfg['transfer_root']);assert not transfer.exists()
    with socket.socket() as sock:sock.bind(('127.0.0.1',cfg['port']))
    py='/data/weijiahong/miniconda3/envs/dflash/bin/python'
    training=os.environ.copy();training.pop('LD_PRELOAD',None)
    training.update(PYTHONPATH=f'{ROOT}:{ROOT}/src:{ROOT}/hs_connectors/src',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                    TORCHINDUCTOR_CACHE_DIR=p['training_cache']['inductor'],TRITON_CACHE_DIR=p['training_cache']['triton'])
    bind(training,identity);serving=bind(environment(),identity)
    serving.update(PYTHONUNBUFFERED='1',TRITON_CACHE_DIR=str(out/'eval_triton'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    procs=[];stage='waiting_for_memory';finished=False
    def status(state,**kw):dump(out/'pipeline_status.json',dict(status=state,stage=stage,gpu=gpu,utc=now(),**kw))
    def spawn(cmd,name,env):
        dump(out/f'{name}_command.json',dict(command=cmd,CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],utc=now()))
        with (out/f'{name}.log').open('x') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        procs.append(proc)
        with (out/'process_history.jsonl').open('a') as f:f.write(json.dumps(dict(pid=proc.pid,name=name,command=cmd,utc=now()))+'\n')
        status('running',pid=proc.pid,log=str(out/f'{name}.log'));return proc
    def observe():
        free,pids=resources(gpu)
        with (out/'gpu_occupancy.jsonl').open('a') as f:f.write(json.dumps(dict(utc=now(),free_mib=free,active_pids=pids,action='record_only'))+'\n')
    def interrupted(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    dump(out/'launch.json',dict(pid=os.getpid(),identity=identity,protocol_sha256=sha(protocol),utc=now()))
    try:
        # Wait only for sufficient startup memory; process ownership is not a training gate.
        while resources(gpu)[0]<int(81920*cfg['memory_utilization'])+12000:
            status('waiting_for_memory');observe();time.sleep(15)
        transfer.mkdir();stage='online_target_startup'
        cmd=[str(EVAL/'bin/python'),'-B',str(ROOT/'scripts/launch_vllm.py'),'/data/weijiahong/huggingface/models/Qwen3-8B',
             '--target-layer-ids','2','10','18','26','34','--hidden-states-path',str(transfer),'--',
             '--host','127.0.0.1','--port',str(cfg['port']),'--tensor-parallel-size','1','--data-parallel-size','1',
             '--dtype','bfloat16','--max-num-seqs',str(p['hidden_prefetch']),'--max-model-len','9216',
             '--max-num-batched-tokens',str(p['hidden_max_batched_tokens']),'--gpu-memory-utilization',str(cfg['memory_utilization']),
             '--no-enable-prefix-caching','--enforce-eager','--disable-uvicorn-access-log']
        server=spawn(cmd,f'vllm_gpu{gpu}',dict(serving,VLLM_USE_V2_MODEL_RUNNER='0'));deadline=time.monotonic()+600
        while not healthy(cfg['port']):
            if server.poll() is not None:raise RuntimeError('hidden server startup failed')
            if time.monotonic()>deadline:raise TimeoutError('hidden server startup')
            time.sleep(2)
        stage='online_hidden_parity'
        proc=spawn([py,'-B','-m','runtime.check_vllm_hidden','--gpu','0','--protocol',str(protocol),'--output',str(out/'hidden_parity.json')],'hidden_parity',training)
        while proc.poll() is None:observe();time.sleep(5)
        if proc.returncode:raise RuntimeError('hidden parity failed')
        stage='recovery_training'
        proc=spawn([py,'-B','-m','repair.run','--protocol',str(protocol),'--output',str(out/'training')],'training',training)
        while proc.poll() is None:
            observe()
            if server.poll() is not None:
                os.kill(proc.pid,signal.SIGTERM)
                raise RuntimeError('teacher service exited')
            time.sleep(5)
        if proc.returncode:raise RuntimeError(f'training exited {proc.returncode}; complete recovery checkpoint retained')
        summary=json.loads((out/'training/summary.json').read_text());assert summary['status']=='complete' and summary['steps']==30000
        stop(server);stage='prepare_native'
        proc=spawn([py,'-B','-m','repair.prepare_native','--output',str(out)],'prepare_native',training)
        if proc.wait():raise RuntimeError('native checkpoint preparation failed')
        finished=True;status('complete')
    except BaseException as exc:status('failed',error=repr(exc));raise
    finally:
        for proc in reversed(procs):
            if proc.poll() is None:
                try:os.kill(proc.pid,signal.SIGTERM);proc.wait(timeout=60)
                except ProcessLookupError:pass
                except subprocess.TimeoutExpired:stop(proc)
        if transfer.exists():
            for f in transfer.iterdir():assert f.is_file();f.unlink()
            transfer.rmdir()
        dump(out/'cleanup.json',dict(finished=finished,owned_processes_stopped=True,utc=now()))

if __name__=='__main__':main()
