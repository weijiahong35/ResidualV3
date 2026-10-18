"""An additional online teacher replica; manages only its own process group."""
import argparse,json,os,signal,socket,subprocess,sys,time,traceback,fcntl,itertools
from pathlib import Path
CODE=Path(__file__).resolve().parent;RUNTIME=CODE.parent/'runtime_bundle';sys.path.insert(0,str(RUNTIME))
from runtime.utils import dump,now,sha,environment,healthy,stop,EVAL
from device import assignment,bind

def inventory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True);tasks={}
    for row in pids.splitlines():
        u,p,m=[x.strip() for x in row.split(',')];tasks.setdefault(u,[]).append(dict(pid=int(p),used_mib=int(m) if m.isdigit() else None))
    result={}
    for row in rows.splitlines():
        i,u,f,v=[x.strip() for x in row.split(',')];result[int(i)]=dict(uuid=u,free_mib=int(f),util=int(v),host_processes=tasks.get(u,[]))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--exclude-gpus',nargs='+',type=int,required=True);ap.add_argument('--dp-size',type=int,choices=[1,2],required=True);ap.add_argument('--kv-gib',type=int,default=12);a=ap.parse_args()
    out=a.output.resolve();lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(out/'launch.json').exists()
    p=json.loads(a.protocol.read_text());sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    transfer=Path('/dev/shm')/f'residualV3_throughput_teacher_{os.getpid()}';transfer.mkdir(exist_ok=False)
    dump(out/'launch.json',dict(controller_pid=os.getpid(),source_sha256=sha(__file__),port=port,dp_size=a.dp_size,tp_size=1,transfer_root=str(transfer),excluded_gpus=a.exclude_gpus,utc=now()))
    server=None;cooldown={};attempt=0
    def interrupt(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        while not(out/'stop.json').exists():
            devices=inventory();candidates=[i for i,d in devices.items() if i not in a.exclude_gpus and d['free_mib']>=75000 and time.monotonic()>=cooldown.get(i,0)]
            if len(candidates)<a.dp_size:
                dump(out/'status.json',dict(status='waiting_memory',inventory=devices,utc=now()));time.sleep(5);continue
            selected=sorted(candidates,key=lambda i:(bool(devices[i]['host_processes']),devices[i]['util'],-devices[i]['free_mib'],i))[:a.dp_size]
            identity=[assignment(i) for i in selected];gpu=selected
            attempt+=1;trial=out/f'attempt_{attempt:02d}';trial.mkdir();env=environment();env.update(CUDA_VISIBLE_DEVICES=','.join(d['uuid'] for d in identity),CUDA_DEVICE_ORDER='PCI_BUS_ID')
            env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',VLLM_USE_V2_MODEL_RUNNER='0',
                       TRITON_CACHE_DIR=str(out.parent/'teacher_triton'),PYTORCH_ALLOC_CONF='expandable_segments:True')
            cmd=[str(EVAL/'bin/python'),'-B',str(RUNTIME/'scripts/launch_vllm.py'),p['target_model'],'--target-layer-ids','2','10','18','26','34',
                 '--hidden-states-path',str(transfer),'--','--host','127.0.0.1','--port',str(port),'--tensor-parallel-size','1','--data-parallel-size',str(a.dp_size),'--data-parallel-size-local',str(a.dp_size),'--data-parallel-backend','mp','--dtype','bfloat16',
                 '--max-num-seqs','64','--max-model-len','4097','--max-num-batched-tokens','65536','--gpu-memory-utilization','.92',
                 '--kv-cache-memory-bytes',str(a.kv_gib*2**30),'--enable-prefix-caching','--enforce-eager','--disable-uvicorn-access-log']
            with (trial/'vllm.log').open('x') as log:server=subprocess.Popen(cmd,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            dump(trial/'launch.json',dict(pid=server.pid,command=cmd,hardware=identity,initial_inventory=devices,utc=now()))
            deadline=time.monotonic()+900
            while server.poll() is None and not healthy(port) and time.monotonic()<deadline and not(out/'stop.json').exists():
                dump(out/'status.json',dict(status='starting',gpu=gpu,identity=identity,server_pid=server.pid,attempt=attempt,utc=now()));time.sleep(3)
            if healthy(port):
                dump(out/'endpoint.json',dict(endpoint=f'http://127.0.0.1:{port}/v1',root=str(transfer),gpu=gpu,identity=identity,server_pid=server.pid,attempt=attempt,
                                              same_model=p['target_model'],dp_size=a.dp_size,tp_size=1,dtype='bfloat16',target_layers=[2,10,18,26,34,36],utc=now()))
                while server.poll() is None and not(out/'stop.json').exists():
                    dump(out/'status.json',dict(status='ready',gpu=gpu,identity=identity,server_pid=server.pid,attempt=attempt,utc=now()))
                    with (out/'gpu_monitor.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=now(),devices=inventory()))+'\n')
                    time.sleep(5)
            stop(server);server=None
            files=list(transfer.iterdir())
            for path in files:
                assert path.is_file();path.unlink()
            dump(trial/'cleanup.json',dict(owned_process_stopped=True,hidden_files_removed=len(files),foreign_processes_untouched=True,utc=now()))
            if (out/'stop.json').exists():break
            tail=(trial/'vllm.log').read_text(errors='replace')[-40000:]
            if not any(word in tail.lower() for word in ['out of memory','free memory','no available memory']):raise RuntimeError(f'second teacher failed: {trial}')
            for i in selected:cooldown[i]=time.monotonic()+120
            dump(trial/'retry.json',dict(actual_memory_error=True,reassign=True,utc=now()))
        dump(out/'status.json',dict(status='stopped',utc=now()))
    except BaseException as exc:
        dump(out/'status.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),utc=now()));raise
    finally:
        if server is not None:stop(server)
        for path in transfer.iterdir():
            assert path.is_file();path.unlink()
        transfer.rmdir();dump(out/'cleanup.json',dict(owned_processes_stopped=True,foreign_processes_untouched=True,utc=now()))

if __name__=='__main__':main()
