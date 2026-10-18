"""Resume the validated 1536-anchor experiment with dynamic GPU assignment."""
import argparse, fcntl, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
CODE=Path(__file__).resolve().parent
RUNTIME=CODE.parent/'runtime_bundle'
sys.path.insert(0,str(RUNTIME))
from runtime.utils import dump,sha,now,environment,healthy,stop,EVAL
from device import assignment,bind

def inventory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    tasks=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    occupied={}
    for line in tasks.splitlines():
        uuid,pid,mem=[x.strip() for x in line.split(',')]
        occupied.setdefault(uuid,[]).append(dict(pid=int(pid),used_mib=int(mem) if mem.isdigit() else None))
    result={}
    for line in rows.splitlines():
        i,u,t,f,v=[x.strip() for x in line.split(',')]
        result[int(i)]=dict(uuid=u,total_mib=int(t),free_mib=int(f),utilization=int(v),host_processes=occupied.get(u,[]))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--recovery',type=Path,required=True);parser.add_argument('--extra-service',type=Path,required=True)
    a=parser.parse_args();out=a.run.resolve();rec=a.recovery.resolve();extra=a.extra_service.resolve()
    locks=[]
    for path in [out/'run.lock',rec/'run.lock']:
        handle=path.open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
    assert not (rec/'launch.json').exists(),'duplicate recovery launch'
    gate=json.loads((out/'summary.json').read_text());assert gate['fused_gate_passed'] and gate['full1536_passed']
    aligned=json.loads((rec/'teacher_alignment.json').read_text())
    assert aligned['status']=='passed' and aligned['ordered_concurrent_requests_passed']
    assert aligned['endpoint_fallbacks']==0,'Require both replicas healthy at initial integration'
    assert gate['selected_mode']=='sequential'
    source=Path(json.loads((out/'comparison_plan.json').read_text())['output_baseline'])
    original=json.loads((source/'training_protocol.json').read_text())
    # These validated numerical functions are byte-for-byte identical to v2.
    numerics=['fused_kernels.py','losses.py','prepare.py','parallel_tf.py','upstream_fused.py']
    assert all(sha(CODE/n)==sha(CODE.parent/'fused_output_dpace_v2'/n) for n in numerics)
    source_sha={str(p):sha(p) for p in CODE.glob('*.py')}
    base=json.loads((out/'training_arms/CE_DPACE/training_protocol.json').read_text())
    transfer=Path(base['transfer_root']);assert not healthy(base['port'])
    transfer.mkdir(exist_ok=False)
    dump(rec/'launch.json',dict(controller_pid=os.getpid(),utc=now(),validated_preflight_reused=True,
                              numerical_source_unchanged=True,head_chunk_anchors=1536,all_gpu_indices_considered=list(range(8)),teacher_replicas=2,extra_service=str(extra)))
    dump(out/'ACTIVE_RUN.json',dict(controller_status=str(rec/'controller_status.json'),controller_log=str(rec/'controller.log'),
                                  recovery_root=str(rec),utc=now()))
    server=worker=None;cooldown={};attempt=0
    # Account for observed foreign-memory peaks, not just an idle instant.
    foreign_peak={}
    previous=out/'recovery_20261007_v3'
    if (previous/'attempt_01_CE_DPACE/assignment.json').exists():
        initial=json.loads((previous/'attempt_01_CE_DPACE/assignment.json').read_text())['hardware']['inventory']
        foreign_ids={p['pid'] for d in initial.values() for p in d['host_processes']}
        for line in (previous/'gpu_monitor.jsonl').read_text().splitlines():
            for i,d in json.loads(line)['devices'].items():
                memory=sum(p['used_mib'] or 0 for p in d['host_processes'] if p['pid'] in foreign_ids)
                foreign_peak[int(i)]=max(foreign_peak.get(int(i),0),memory)
    def state(status,**extra):dump(rec/'controller_status.json',dict(status=status,controller_pid=os.getpid(),utc=now(),**extra))
    def interrupt(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def cleanup_owned():
        nonlocal worker,server
        for proc in [worker,server]:
            if proc is not None:stop(proc)
        worker=server=None
        files=list(transfer.iterdir())
        for path in files:
            assert path.is_file(),path
            path.unlink()
        return len(files)
    try:
        for arm in ['CE_DPACE','CE_DPACE_OUTPUT']:
            armout=out/'training_arms'/arm;armout.mkdir(exist_ok=True,parents=True)
            if (armout/'training/summary.json').exists():
                assert json.loads((armout/'training/summary.json').read_text())['status']=='complete';continue
            protocol_path=armout/'training_protocol.json'
            if not protocol_path.exists():
                ap=dict(original,output=str(armout),port=base['port'],transfer_root=str(transfer),loss_arm=arm,
                        head_mode='sequential',head_chunk_anchors=1536,allocator_budget_bytes=base['allocator_budget_bytes'])
                dump(protocol_path,ap)
            while True:
                devices=inventory();clock=time.monotonic()
                extra_endpoint=json.loads((extra/'endpoint.json').read_text())
                extra_port=int(extra_endpoint['endpoint'].split(':')[-1].split('/')[0])
                if not healthy(extra_port):
                    state('waiting_second_teacher',extra_service=str(extra));time.sleep(5);continue
                extra_gpu=extra_endpoint['gpu']
                learners=[i for i,d in devices.items() if i!=extra_gpu and d['free_mib']>=60000 and clock>=cooldown.get(i,0)]
                pairs=[(i,j) for i in learners for j,d in devices.items() if j not in [i,extra_gpu] and d['free_mib']>=43000 and clock>=cooldown.get(j,0)]
                if not pairs:
                    state('waiting_training_memory',arm=arm,minimum_learner_free_mib=60000,minimum_teacher_free_mib=43000,inventory=devices)
                    time.sleep(5);continue
                learner,teacher=min(pairs,key=lambda pair:(pair[0]!=7,pair[1]!=5,bool(devices[pair[0]]['host_processes']),foreign_peak.get(pair[0],0),devices[pair[0]]['utilization'],
                                                          -devices[pair[0]]['free_mib'],bool(devices[pair[1]]['host_processes']),foreign_peak.get(pair[1],0),
                                                          devices[pair[1]]['utilization'],-devices[pair[1]]['free_mib'],pair))
                hardware=dict(learner=assignment(learner),teacher=assignment(teacher),second_teacher=extra_endpoint['identity'],inventory=devices)
                attempt+=1;attemptout=rec/f'attempt_{attempt:02d}_{arm}';attemptout.mkdir()
                dump(attemptout/'assignment.json',dict(hardware=hardware,utc=now()))
                env=bind(environment(),hardware['teacher'])
                env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
                           TRITON_CACHE_DIR=str(out/'teacher_triton'),VLLM_USE_V2_MODEL_RUNNER='0')
                cmd=[str(EVAL/'bin/python'),'-B',str(RUNTIME/'scripts/launch_vllm.py'),original['target_model'],
                     '--target-layer-ids','2','10','18','26','34','--hidden-states-path',str(transfer),'--',
                     '--host','127.0.0.1','--port',str(base['port']),'--tensor-parallel-size','1','--dtype','bfloat16',
                     '--max-num-seqs','16','--max-model-len','4097','--max-num-batched-tokens','32768',
                     '--gpu-memory-utilization','.43','--kv-cache-memory-bytes',str(12*2**30),
                     '--enable-prefix-caching','--enforce-eager','--disable-uvicorn-access-log']
                with (attemptout/'teacher_vllm.log').open('x') as log:
                    server=subprocess.Popen(cmd,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                dump(attemptout/'teacher_launch.json',dict(pid=server.pid,command=cmd,hardware=hardware['teacher'],utc=now()))
                state('teacher_startup',arm=arm,attempt=attempt,hardware=hardware,teacher_pid=server.pid)
                deadline=time.monotonic()+900
                while not healthy(base['port']) and server.poll() is None and time.monotonic()<deadline:time.sleep(3)
                if not healthy(base['port']):
                    detail=(attemptout/'teacher_vllm.log').read_text(errors='replace')[-20000:]
                    if not any(word in detail.lower() for word in ['out of memory','free memory','no available memory','memory profiling']):
                        raise RuntimeError(f'teacher failed: {attemptout}')
                    cleanup_owned();cooldown[teacher]=time.monotonic()+600
                    dump(attemptout/'memory_retry.json',dict(stage='teacher_startup',retry=True,utc=now()));continue
                # Recheck secondary placement after primary startup, avoiding a startup race.
                extra_endpoint=json.loads((extra/'endpoint.json').read_text())
                if extra_endpoint['gpu'] in [learner,teacher] or not healthy(extra_port):
                    cleanup_owned();time.sleep(3);continue
                replicas=[dict(endpoint=f"http://127.0.0.1:{base['port']}/v1",root=str(transfer)),
                          dict(endpoint=extra_endpoint['endpoint'],root=extra_endpoint['root'])]
                env=bind(os.environ.copy(),hardware['learner']);env.pop('LD_PRELOAD',None)
                execution=attemptout/'execution_protocol.json'
                dump(execution,dict(source_sha256=source_sha,training_protocol_sha256=sha(protocol_path),hardware=hardware,
                                    preflight=str(out/'summary.json'),preflight_sha256=sha(out/'summary.json'),teacher_replicas=replicas,
                                    training_math_unchanged=True,total_prefetch_requests=32,per_replica_requests=16))
                env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                           OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTORCH_ALLOC_CONF='expandable_segments:True',
                           PYTHONPATH=f'{RUNTIME}:{RUNTIME}/src:{RUNTIME}/hs_connectors/src',
                           TORCHINDUCTOR_CACHE_DIR=str(out/'training_inductor'),TRITON_CACHE_DIR=str(out/'training_triton'),
                           PACKED_OPTIMIZATION_PROTOCOL=str(execution),PACKED_RECOVERY_ATTEMPT=str(attemptout),
                           PACKED_TEACHER_REPLICAS=json.dumps(replicas))
                cmd=['/data/weijiahong/miniconda3/envs/dflash/bin/python','-B',str(CODE/'train.py'),'--protocol',str(protocol_path)]
                with (attemptout/'training.log').open('x') as log:
                    worker=subprocess.Popen(cmd,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                dump(attemptout/'worker_launch.json',dict(pid=worker.pid,command=cmd,hardware=hardware['learner'],utc=now()))
                dump(out/'ACTIVE_RUN.json',dict(controller_status=str(rec/'controller_status.json'),controller_log=str(rec/'controller.log'),
                                              training_log=str(attemptout/'training.log'),training_status=str(armout/'training/status.json'),
                                              recovery_root=str(rec),arm=arm,hardware=hardware,utc=now()))
                while worker.poll() is None:
                    state('training',arm=arm,attempt=attempt,worker_pid=worker.pid,teacher_pid=server.pid,hardware=hardware,
                          training_log=str(attemptout/'training.log'),training_status=str(armout/'training/status.json'))
                    with (rec/'gpu_monitor.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=now(),attempt=attempt,devices=inventory()))+'\n')
                    time.sleep(5)
                result=worker.returncode
                removed=cleanup_owned()
                dump(attemptout/'cleanup.json',dict(owned_processes_stopped=True,transient_hidden_files_removed=removed,foreign_processes_untouched=True,utc=now()))
                if result==0:break
                failure_path=attemptout/'failure.json'
                failure=json.loads(failure_path.read_text()) if failure_path.exists() else {}
                tail=(attemptout/'training.log').read_text(errors='replace')[-20000:]
                teacher_tail=(attemptout/'teacher_vllm.log').read_text(errors='replace')[-40000:]
                teacher_oom='CUDA out of memory' in teacher_tail
                if not (failure.get('oom') or 'CUDA out of memory' in tail or teacher_oom):
                    raise RuntimeError(f'{arm} failed beyond memory retry scope: {attemptout}')
                # Resume the last committed checkpoint. Replayed steps carry an attempt id.
                failed_gpu=teacher if teacher_oom else learner
                cooldown[failed_gpu]=time.monotonic()+600
                foreign_peak[failed_gpu]=max(foreign_peak.get(failed_gpu,0),50000)
                dump(attemptout/'memory_retry.json',dict(retry=True,failed_gpu=failed_gpu,failure_stage='teacher' if teacher_oom else 'learner',head_chunk_anchors=1536,
                     checkpoint=str(armout/'training/recovery_checkpoints/resume_latest.pt'),
                     replay_from_last_committed_checkpoint=True,utc=now()))
                state('reassigning_after_oom',arm=arm,failed_gpu=failed_gpu)
        dump(extra/'stop.json',dict(reason='both loss groups trained; release secondary before native evaluation',utc=now()))
        deadline=time.monotonic()+120
        while healthy(extra_port) and time.monotonic()<deadline:time.sleep(3)
        assert not healthy(extra_port),'secondary shutdown pending'
        roots={'OUTPUT':source,'CE_DPACE':out/'training_arms/CE_DPACE','CE_DPACE_OUTPUT':out/'training_arms/CE_DPACE_OUTPUT'}
        old_native=json.loads((source/'native_protocol.json').read_text());checkpoints={}
        for arm,root in roots.items():
            weight=root/'training/epoch_03.safetensors'
            assert json.loads((root/'training/summary.json').read_text())['status']=='complete'
            checkpoints[arm]=dict(path=str(weight),sha256=sha(weight),head_config_path=original['head_config_path'],head_config_sha256=sha(original['head_config_path']))
        checkpoints['RAW']=dict(checkpoints['OUTPUT'])
        while True:
            devices=inventory();candidates=[i for i,d in devices.items() if d['free_mib']>=61440]
            if candidates:break
            state('waiting_evaluation_memory',minimum_free_mib=61440,inventory=devices);time.sleep(5)
        gpu=min(candidates,key=lambda i:(bool(devices[i]['host_processes']),devices[i]['utilization'],-devices[i]['free_mib']))
        native=dict(old_native,gpu=gpu,uuid=devices[gpu]['uuid'],native_arms=['RAW','OUTPUT','CE_DPACE','CE_DPACE_OUTPUT'],native_checkpoints=checkpoints,
                    native_configs={arm:{} for arm in checkpoints},architecture_by_arm={arm:original['architecture'] for arm in checkpoints},
                    source_files={**old_native['source_files'],**source_sha},
                    training_completions={arm:sha(root/'training/summary.json') for arm,root in roots.items()},
                    comparison='two teacher replicas; three losses, same 20k x3 packed anchor/data/initialization budget; reused completed OUTPUT, new fused/full-block accumulation',
                    asset_sha256={**old_native['asset_sha256'],**{row['path']:row['sha256'] for row in checkpoints.values()}})
        assert not (out/'native_protocol.json').exists();dump(out/'native_protocol.json',native)
        for concurrency,prompts in [(1,200),(40,2000)]:
            for smoke in [True,False]:
                phase=f'C{concurrency}_'+('smoke' if smoke else 'formal')
                destination=out/f'C{concurrency}'/('formal_smoke' if smoke else 'formal')
                cmd=['/data/weijiahong/miniconda3/envs/dflash/bin/python','-B',str(CODE/'evaluate.py'),'--training',str(out),'--gpu',str(gpu),
                     '--concurrency',str(concurrency),'--output',str(destination),'--tokens',str(32 if smoke else 2096),
                     '--prompts',str(concurrency if smoke else prompts),'--rounds','1']
                if smoke:cmd.append('--smoke')
                with (out/f'single_card_{phase}.log').open('x') as log:
                    worker=subprocess.Popen(cmd,env=os.environ.copy(),cwd=CODE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                while worker.poll() is None:
                    state('evaluating',stage=phase,worker_pid=worker.pid,gpu=gpu);time.sleep(5)
                assert worker.returncode==0,f'{phase} failed'
        subprocess.run(['/data/weijiahong/miniconda3/envs/dflash/bin/python','-B',str(CODE/'report.py'),'--output',str(out)],check=True,cwd=CODE)
        state('complete',results=str(out/'RESULTS_CN.txt'))
    except BaseException as error:
        state('failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        removed=cleanup_owned();transfer.rmdir()
        dump(extra/'stop.json',dict(reason='own dual-teacher training controller ended',utc=now()))
        dump(rec/'cleanup.json',dict(owned_processes_stopped=True,transient_hidden_files_removed=removed,foreign_processes_untouched=True,utc=now()))

if __name__=='__main__':main()
