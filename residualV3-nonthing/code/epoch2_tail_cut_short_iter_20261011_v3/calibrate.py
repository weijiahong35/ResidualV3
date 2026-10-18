"""Bounded paired pre/post-q diagnostic; no training of existing draft/repair weights."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

C=Path(__file__).resolve().parent
RUNTIME=C/'native_runtime'
sys.path.insert(0,str(RUNTIME))
from runtime.utils import dump,now,sha,resources,stop,healthy,EVAL,environment
from runtime.http import _http_json
from device import assignment,bind


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root.resolve();p=json.loads((root/'preregistration.json').read_text())
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'RESULTS.json').exists(),'do not rerun completed pilot'
    for path,digest in p['source_files'].items():assert sha(path)==digest,path
    ck=p['checkpoint']
    for path,digest in ((ck['path'],ck['sha256']),(ck['head_config_path'],ck['head_config_sha256']),(p['basis_path'],p['basis_sha256']),(p['panel_path'],p['probe_panel_sha256'])):
        assert sha(path)==digest,path
    for f,digest in p['body_checksums'].items():assert sha(Path(p['body_model'])/f)==digest,f
    gpu=p['physical_gpu'];identity=assignment(gpu);assert identity['uuid']==p['gpu_uuid']
    lease=(root.parent/f'.native_gpu_{gpu}.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    free,pids=resources(gpu)
    if free<p['minimum_free_mib']:raise RuntimeError(('insufficient memory',free,pids))
    trial=root/'native';trial.mkdir(exist_ok=True)
    for f in ('execution_counters.bin','target_execution_counters.bin'):(trial/f).write_bytes(bytes(32))
    (trial/'graph_captures.jsonl').write_text('')
    dump(root/'capture_control.json',dict(enabled=False))
    dump(root/'controller_pid.json',dict(pid=os.getpid(),gpu=gpu,identity=identity,utc=now()))
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env=bind(environment(),identity)
    env.update(VLLM_BATCH_INVARIANT='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_USE_AOT_COMPILE='1',PYTHONHASHSEED='0',VLLM_USE_V2_MODEL_RUNNER='1',
       INPUT_ABLATION_CONFIG='{}',RUNTIME_VARIANT='optimized',HIDDEN_ARM='EMBEDDING',ORIGINAL_RUNTIME_ARM='EMBEDDING',ABLATION_LABEL='Q_PRE_POST_TAIL_SIGNAL',NATIVE_PROPOSAL_LENGTH='15',
       HEAD_STRUCTURE_MODE='lean_combined',HEAD_LATENCY_OPT_VARIANT='small_tile',HEAD_SMALL_TILE_MODE='all16_64',TOKEN_FEATURE_SOURCE='EMBEDDING',D1_BASIS_PATH=p['basis_path'],D1_BASIS_SHA256=p['basis_sha256'],
       HIDDEN_HEAD_CONFIG=ck['head_config_path'],HIDDEN_HEAD_CONFIG_SHA256=ck['head_config_sha256'],RUNTIME_REAL_PROFILE='0',LOCAL_ARM='dense_combined',LOCAL_TRIAL=str(trial),
       LOCAL_CONTRACT_CHECKS='1',LOCAL_MAX_CONCURRENCY='1',NATIVE_FULL_GRAPH='1',REPAIR_CHECKPOINT=ck['path'],STATIC_M_PATH=p['m_ids_path'],STATIC_M_SHA256=p['m_ids_sha256'],STATIC_M_SIZE=str(p['m_size']),
       XDG_CACHE_HOME=str(Path(p['shared_eval_cache'])/'xdg'),TRITON_CACHE_DIR=str(Path(p['shared_eval_cache'])/'triton'),Q_PROBE_ROOT=str(root),Q_PROBE_SHARED='1' if pids else '0')
    env.update(TAIL_MODE='PIPELINE',TAIL_AUX_BRANCH='BOTH',TAIL_PREDICTOR=str(root/'initial_policy.json'),
               TAIL_AUX_CHECKPOINT=p['auxiliary_checkpoint'],TARGET_PREFIX_MODE='FULL')
    command=[str(EVAL/'bin/python'),'-B','-m','repair.q_probe_native','serve',p['target_model'],'--host','127.0.0.1','--port',str(port),'--device-ids','0','--tensor-parallel-size','1',
       '--dtype','bfloat16','--generation-config','vllm','--seed','0','--max-model-len','4096','--max-num-seqs','1','--max-num-batched-tokens','8192',
       '--gpu-memory-utilization',str(p['requested_gpu_memory_utilization']),'--kv-cache-memory-bytes',str(p['kv_cache_gib']*(1<<30)),
       '--no-enable-prefix-caching','--no-async-scheduling','--compilation-config',json.dumps(dict(inductor_compile_config=dict(combo_kernels=False,benchmark_combo_kernel=False),cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=[16])),
       '--speculative-config',json.dumps(dict(method='dflash',model=p['body_model'],num_speculative_tokens=15,enforce_eager=False,draft_sample_method='probabilistic'))]
    dump(root/'launch.json',dict(command=command,environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','REPAIR_GPU_UUID','VLLM_BATCH_INVARIANT','LOCAL_ARM','HEAD_STRUCTURE_MODE','HEAD_SMALL_TILE_MODE','Q_PROBE_SHARED')},initial_free_mib=free,initial_host_pids=pids,utc=now()))
    def interrupted(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    proc=None
    try:
        with (trial/'vllm.log').open('x') as log:proc=subprocess.Popen(command,cwd=RUNTIME,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(root/'status.json',dict(status='starting_native',pid=proc.pid,gpu=gpu,utc=now()))
        for _ in range(600):
            if proc.poll() is not None:raise RuntimeError(('server exit',proc.returncode))
            if healthy(port):break
            time.sleep(1)
        else:raise TimeoutError('server startup')
        endpoint=f'http://127.0.0.1:{port}';model=_http_json(endpoint+'/v1/models')['data'][0]['id']
        rows=[json.loads(v) for v in Path(p['panel_path']).read_text().splitlines()]
        def request(row,t,tokens,key,enabled,trace=None,action=None):
            control=dict(enabled=enabled,request_key=key,probe_index=row['probe_index'],split=row['probe_split'],T=t,max_blocks=p['maximum_blocks_per_prompt'])
            if trace:control['trace_pass']=trace
            if action:control['action']=action
            dump(root/'capture_control.json',control)
            return _http_json(endpoint+'/v1/completions',dict(model=model,prompt=row['prompt_token_ids'],temperature=t,top_p=1.,top_k=-1,max_tokens=tokens,seed=430004+row['probe_index']),timeout=1200)
        # Disabled warmup, then exact emitted-token trace witnesses at T0 and T1.
        for t in (0,1):request(rows[0],t,32,f'warmup_T{t}',False)
        smoke={}
        for t in (0,1):
            baseline=request(rows[0],t,64,f'base_T{t}',False,f'baseline_T{t}')
            observed=request(rows[0],t,64,f'observed_T{t}',True,f'observed_T{t}')
            b=json.loads((root/f'trace_baseline_T{t}.json').read_text());o=json.loads((root/f'trace_observed_T{t}.json').read_text())
            same=b['token_sha256']==o['token_sha256'] and b['tokens']==o['tokens'] and baseline['choices'][0]['text']==observed['choices'][0]['text'] and baseline['usage']==observed['usage']
            assert same,('observer changed native generation',t)
            smoke[str(t)]=dict(emitted_token_trace_exact=True,text_exact=True,usage_exact=True,trace_tokens=b['tokens'])
        # Smoke records are excluded from classification by a separate RAM reset.
        request(rows[0],0,32,'reset',False,action='reset')
        reset=json.loads((root/'capture_status.json').read_text());assert reset['status']=='reset' and reset['states']==0
        dump(root/'SMOKE_GATE.json',dict(status='passed',T=smoke,source_sha256=sha(root/'preregistration.json'),utc=now()))
        for i,row in enumerate(rows):
            for t in (0,1):
                result=request(row,t,p['capture_tokens'],f"prompt_{row['probe_index']}_T{t}",True)
                text=result['choices'][0]['text']
                entry=dict(probe_index=row['probe_index'],split=row['probe_split'],T=t,usage=result['usage'],text_sha256=hashlib.sha256(text.encode()).hexdigest(),utc=now())
                with (root/'requests.jsonl').open('a') as stream:stream.write(json.dumps(entry)+'\n')
                dump(root/'status.json',dict(status='collecting',completed_requests=i*2+t+1,total_requests=len(rows)*2,prompts_complete=i+(t==1),gpu=gpu,utc=now()))
            if (i+1)%8==0:print(now(),'PROMPTS',i+1,'/',len(rows),flush=True)
        dump(root/'status.json',dict(status='fitting_and_analyzing',completed_requests=len(rows)*2,utc=now()))
        request(rows[0],0,32,'flush',False,action='flush')
        final=json.loads((root/'RESULTS.json').read_text());assert final['status']=='complete'
        for path,digest in p['source_files'].items():assert sha(path)==digest,path
        free,end_pids=resources(gpu)
        dump(root/'COMPLETE.json',dict(status='complete',metrics_sha256=sha(root/'RESULTS.json'),native_smoke_sha256=sha(root/'SMOKE_GATE.json'),preregistration_sha256=sha(root/'preregistration.json'),initial_foreign_pids=pids,final_host_pids=end_pids,utc=now()))
        dump(root/'status.json',dict(status='complete',completed_requests=len(rows)*2,utc=now()))
        print(now(),'COMPLETE',root/'RESULTS_CN.txt',flush=True)
    except BaseException as exc:
        dump(root/'status.json',dict(status='failed',error=repr(exc),utc=now()));raise
    finally:
        if proc:stop(proc)

if __name__=='__main__':main()
