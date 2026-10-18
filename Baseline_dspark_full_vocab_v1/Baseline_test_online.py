#!/usr/bin/env python3
"""Native T0/T1 generation: real proposal-prefix verification and simultaneous timing.
This is a head baseline on DFlash, not the complete external method.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.request
import numpy as np
from Baseline_http import _http_json, _http_text, _metrics_delta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FAMILY = HERE.name.split('_')[1]
POLICY = 'bf16-head-v2-fp32-prob-loss'
EVAL = ROOT.parent/'miniconda3/envs/llm_dflash_eval_v026'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def freeze(p,x):
    if p.exists() and json.loads(p.read_text()) != x: raise RuntimeError(f'protocol changed: {p}')
    dump(p,x)

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def resources(gpu):
    lines=subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    uuid,free=lines[0].strip(),int(lines[1])
    tasks=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return free,[int(x.split(',')[1]) for x in tasks.splitlines() if x.split(',')[0].strip()==uuid]

def members(pgid):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            f=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(f[2])==pgid and f[0]!='Z':result.append(int(p.name))
        except (OSError,ValueError,IndexError):pass
    return result

def stop(proc):
    for sig,seconds in [(signal.SIGTERM,15),(signal.SIGKILL,10)]:
        try:os.killpg(proc.pid,sig)
        except ProcessLookupError:pass
        end=time.monotonic()+seconds
        while members(proc.pid) and time.monotonic()<end:time.sleep(.2)
        if not members(proc.pid):break
    proc.wait(timeout=2)
    if members(proc.pid):raise RuntimeError('owned server group survived cleanup')

def healthy(port):
    try:return urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2).status==200
    except Exception:return False

def environment():
    env=os.environ.copy();env.pop('LD_PRELOAD',None)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_ENABLE_CUDA_COMPATIBILITY='1',
        VLLM_CUDA_COMPATIBILITY_PATH=str(EVAL/'cuda-compat'),LD_LIBRARY_PATH=f'{EVAL}/cuda-compat:{EVAL}/lib',
        PYTHONPATH=f'{HERE}:{ROOT}/src:{ROOT}/hs_connectors/src',PYTHONDONTWRITEBYTECODE='1')
    return env

def request(model,prompt,index,temp,length):
    return dict(model=model,messages=prompt['messages'],max_tokens=length,stream=False,temperature=temp,
        seed=42+index,top_p=1.,top_k=-1,frequency_penalty=0.,presence_penalty=0.,repetition_penalty=1.)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--panel',type=Path,default=ROOT/'eval_results/residualV3_full_vocab_unified_lag1_v3_seed42/native_prompts.jsonl')
    ap.add_argument('--target',type=Path,default=ROOT.parent/'huggingface/models/Qwen3-8B')
    ap.add_argument('--body',type=Path,default=HERE/'body_only_draft')
    ap.add_argument('--port',type=int,default=18571 if FAMILY=='dflash2' else 18572)
    ap.add_argument('--top-k',type=int,default=16,help='DFlash2 candidate K; unrelated to sampling top_k')
    ap.add_argument('--smoke',action='store_true',help='two prompts per temperature, 32 tokens, contract checks; not runtime evidence')
    ap.add_argument('--allow-shared-gpu',action='store_true')
    ap.add_argument('--convert-legacy-fp32',action='store_true',help='explicit in-memory conversion of historical FP32 head; original files unchanged')
    args=ap.parse_args()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:raise RuntimeError('use physical --gpu without CUDA_VISIBLE_DEVICES')
    if not args.smoke and 'smoke' in str(args.checkpoint):raise RuntimeError('smoke-trained weights are only allowed with --smoke')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from safetensors import safe_open
    with safe_open(str(args.checkpoint),framework='pt',device='cpu') as f:
        keys=list(f.keys());dtypes={k:f.get_slice(k).get_dtype() for k in keys}
    head_keys=[k for k in keys if k.startswith('head.')] if any(k.startswith('head.') for k in keys) else keys
    if not args.convert_legacy_fp32 and any(dtypes[k]!='BF16' for k in head_keys):
        raise RuntimeError('FP32 checkpoint requires explicit --convert-legacy-fp32 or BF16 training checkpoint')
    prompts=[json.loads(s) for s in args.panel.read_text().splitlines() if s.strip()]
    if len(prompts)<2:raise ValueError('panel must contain at least two prompts')
    if args.smoke:prompts=prompts[:2]
    length=32 if args.smoke else 128
    mode='smoke' if args.smoke else 'runtime'
    protocol=dict(family=FAMILY,precision_policy=POLICY,checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha(args.checkpoint),
        checkpoint_dtypes=dtypes,convert_legacy_fp32=args.convert_legacy_fp32,
        body=str(args.body.resolve()),body_sha256=sha(args.body/'model.safetensors'),
        body_config_sha256=sha(args.body/'config.json'),mask_embedding_sha256=sha(args.body/'mask_embedding.pt'),
        target=str(args.target.resolve()),target_config_sha256=sha(args.target/'config.json'),
        body_overlay=any(k.startswith('body.') for k in keys),raw_uses_same_body_as_head=True,
        panel=str(args.panel.resolve()),panel_sha256=sha(args.panel),open_diagnostic_panel=True,
        top_k=args.top_k,temperatures=[0,1],seed='42+prompt index',engine_seed=0,
        eager=True,concurrency=1,proposal_tokens=7,fp32_boundaries=['probability normalization','loss and norm accumulation'],
        physical_gpu=args.gpu,allow_shared_gpu=args.allow_shared_gpu,
        source_sha256={p.name:sha(p) for p in sorted(HERE.glob('Baseline_*.py'))})
    freeze(out/'protocol.json',protocol)
    if not args.smoke:
        smoke=json.loads((out/'smoke_complete.json').read_text())
        if smoke['protocol_sha256']!=sha(out/'protocol.json'):raise RuntimeError('smoke protocol differs')
    orders=[['raw','head']] if args.smoke else [['raw','head'],['head','raw']]
    windows=[]
    def interrupted(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for rnd,arms in enumerate(orders,1):
            for arm in arms:
                trial=out/mode/f'round_{rnd}'/arm
                if (trial/'complete.json').exists():
                    windows.extend(json.loads((trial/'windows.json').read_text()));continue
                if trial.exists():raise RuntimeError(f'incomplete trial preserved; archive before retry: {trial}')
                free,pids=resources(args.gpu)
                if free<45000 or (pids and not args.allow_shared_gpu):raise RuntimeError(f'GPU unavailable: {free}MiB, PIDs={pids}')
                with socket.socket() as sock:
                    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',args.port))
                trial.mkdir(parents=True)
                env=environment();env.update(BASELINE_CHECKPOINT=str(args.checkpoint.resolve()),BASELINE_ARM=arm,
                    BASELINE_TRIAL=str(trial),BASELINE_TOP_K=str(args.top_k),BASELINE_CONTRACT_CHECKS='1' if args.smoke else '0',
                    XDG_CACHE_HOME=str(out/'cache'),TRITON_CACHE_DIR=str(out/'triton_cache'))
                spec=dict(method='dflash',model=str(args.body.resolve()),num_speculative_tokens=7,enforce_eager=True,draft_sample_method='probabilistic')
                cmd=[str(EVAL/'bin/python'),'-B',str(HERE/'Baseline_native_launch.py'),'serve',str(args.target),
                    '--host','127.0.0.1','--port',str(args.port),'--tensor-parallel-size','1','--device-ids',str(args.gpu),
                    '--dtype','bfloat16','--generation-config','vllm','--seed','0','--max-model-len','4096',
                    '--max-num-seqs','1','--max-num-batched-tokens','8192','--gpu-memory-utilization','.45',
                    '--no-enable-prefix-caching','--no-async-scheduling','--enforce-eager','--speculative-config',json.dumps(spec)]
                dump(trial/'command.json',dict(command=cmd,existing_pids=pids,free_mib=free))
                dump(out/'status.json',dict(status='starting',mode=mode,round=rnd,arm=arm,utc=now()))
                print(now(),'START',mode,rnd,arm,flush=True)
                server=None;done=threading.Event();errors=[];samples=[];monitor=None;trial_windows=[]
                def inspect_gpu():
                    avail,ids=resources(args.gpu);foreign=sorted(set(ids)-set(members(server.pid)))
                    samples.append(dict(utc=now(),free_mib=avail,foreign_pids=foreign))
                    if avail<6000 or (foreign and not args.allow_shared_gpu):raise RuntimeError(f'GPU memory/isolation guard: {samples[-1]}')
                def watch():
                    while not done.wait(5):
                        try:inspect_gpu()
                        except Exception as exc:
                            errors.append(str(exc))
                            try:os.killpg(server.pid,signal.SIGTERM)
                            except ProcessLookupError:pass
                            return
                try:
                    with (trial/'vllm.log').open('w') as log:
                        server=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        monitor=threading.Thread(target=watch,daemon=True);monitor.start()
                        for _ in range(360):
                            if errors:raise RuntimeError(errors)
                            if server.poll() is not None:raise RuntimeError(f'server failed: {trial}/vllm.log')
                            if healthy(args.port):break
                            time.sleep(1)
                        else:raise TimeoutError('server health timeout')
                        endpoint=f'http://127.0.0.1:{args.port}';model=_http_json(endpoint+'/v1/models')['data'][0]['id']
                        for temp in (0,1):
                            for i in range(1 if args.smoke else 4):
                                _http_json(endpoint+'/v1/chat/completions',request(model,prompts[i%len(prompts)],i,temp,length),timeout=600)
                            inspect_gpu();before=_http_text(endpoint+'/metrics');rows=[]
                            dump(out/'status.json',dict(status='measuring',mode=mode,round=rnd,arm=arm,temperature=temp,utc=now()))
                            begin=time.perf_counter_ns();begin_wall=time.time_ns()
                            for i,prompt in enumerate(prompts):
                                if errors:raise RuntimeError(errors)
                                start=time.perf_counter_ns();wall=time.time_ns()
                                response=_http_json(endpoint+'/v1/chat/completions',request(model,prompt,i,temp,length),timeout=600)
                                end=time.perf_counter_ns()
                                rows.append(dict(index=i,temperature=temp,seed=42+i,start_unix_ns=wall,end_unix_ns=time.time_ns(),elapsed_seconds=(end-start)/1e9,response=response))
                            end=time.perf_counter_ns();end_wall=time.time_ns()
                            after=_http_text(endpoint+'/metrics');inspect_gpu()
                            if errors:raise RuntimeError(errors)
                            metrics=_metrics_delta(before,after)
                            if metrics['num_drafts']<=0:raise RuntimeError('no native speculative counters observed')
                            positions=[metrics[f'acceptance_at_pos_{j}'] for j in range(7)]
                            if abs(1+sum(positions)-metrics['acceptance_length'])>1e-8:raise RuntimeError('position counters disagree')
                            tokens=sum(x['response']['usage']['completion_tokens'] for x in rows);duration=(end-begin)/1e9
                            delays=[x['elapsed_seconds'] for x in rows]
                            record=dict(round=rnd,arm=arm,temperature=temp,requests=len(rows),completion_tokens=tokens,
                                wall_seconds=duration,start_unix_ns=begin_wall,end_unix_ns=end_wall,output_tokens_per_second=tokens/duration,
                                latency_seconds=dict(mean=float(np.mean(delays)),p50=float(np.quantile(delays,.5)),p95=float(np.quantile(delays,.95))),
                                native_metrics=metrics,S_j_over_D=positions,ttft=None,tpot=None,
                                diagnostic_smoke=args.smoke,isolation_observed=not any(x['foreign_pids'] for x in samples),
                                foreign_pids_observed=sorted({pid for x in samples for pid in x['foreign_pids']}))
                            dump(trial/f'T{temp}_window.json',record)
                            (trial/f'T{temp}_requests.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
                            trial_windows.append(record);windows.append(record)
                            print(now(),arm,'T',temp,'A',metrics['acceptance_length'],'tokens/s',tokens/duration,flush=True)
                finally:
                    done.set()
                    if monitor:monitor.join(timeout=8)
                    if server:stop(server)
                    dump(trial/'isolation.json',dict(samples=samples,errors=errors))
                if args.smoke:
                    contracts=[json.loads(x) for x in (trial/'sampling_contract.jsonl').read_text().splitlines()]
                    if not contracts or not any(x['temperature']>0 and x['non_top1']>0 for x in contracts):raise RuntimeError('no non-Top1 stochastic feedback witness')
                    dump(trial/'contract_summary.json',dict(passed=True,steps=len(contracts),non_top1_T1=sum(x['non_top1'] for x in contracts if x['temperature']>0)))
                dump(trial/'windows.json',trial_windows);dump(trial/'complete.json',dict(status='complete',utc=now()))
        dump(out/f'{mode}_summary.json',dict(status='complete',windows=windows,scope='head-only transplant; actual native verification; paired raw uses identical body',diagnostic_smoke=args.smoke))
        dump(out/f'{mode}_complete.json',dict(status='complete',protocol_sha256=sha(out/'protocol.json'),utc=now()))
        lines=['# '+FAMILY+' native '+mode,'','Smoke timings include contract checks and are not performance evidence.' if args.smoke else 'Acceptance and end-to-end throughput measured in the same windows. No per-block observer.','', '| Round | Arm | T | A (includes bonus) | Output tokens/s | Isolated |','|---|---|---|---|---|---|']
        for w in windows:lines.append(f"| {w['round']} | {w['arm']} | {w['temperature']} | {w['native_metrics']['acceptance_length']:.6f} | {w['output_tokens_per_second']:.3f} | {w['isolation_observed']} |")
        (out/f'{mode.upper()}_RESULTS.md').write_text('\n'.join(lines)+'\n')
        dump(out/'status.json',dict(status='complete',mode=mode,utc=now()))
    except BaseException as e:
        dump(out/'status.json',dict(status='failed',mode=mode,error=repr(e),utc=now()));raise

if __name__=='__main__':main()
