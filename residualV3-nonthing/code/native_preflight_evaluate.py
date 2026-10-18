"""One RAW+A..G comparison on one GPU; identical native math and sampler."""
import sys
from pathlib import Path
RUNTIME=Path(__file__).resolve().parent/'runtime_bundle'
sys.path.insert(0,str(RUNTIME))
__package__='repair'
import argparse,fcntl,json,os,signal,socket,subprocess,threading,time
from pathlib import Path
from device import assignment,bind
from runtime.utils import ROOT,HERE,EVAL,environment,resources,members,stop,dump,sha,now,healthy
from runtime.benchmark import parallel_requests
import runtime.benchmark as _benchmark
_original_request = _benchmark.request
def _thinking_request(model, prompt, index, temp, length):
    value = _original_request(model, prompt, index, temp, length)
    value['chat_template_kwargs'] = {'enable_thinking': False}
    return value
_benchmark.request = _thinking_request
from runtime.http import _http_json,_http_text,_metrics_delta
from runtime.counters import snapshot,delta
from runtime.batch_preheat import coverage


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--training",type=Path,required=True)
    ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--concurrency',type=int,choices=[1,40],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--tokens',type=int,default=4096)
    ap.add_argument('--prompts',type=int,required=True);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--rounds',type=int,choices=[1,2],default=1);ap.add_argument('--arms',nargs='+',default=None);ap.add_argument('--temperatures',nargs='+',type=int,choices=[0,1],default=[0,1]);a=ap.parse_args()
    assert a.smoke, "Preflight permits correctness smoke only"
    identity=assignment(a.gpu)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=a.training.resolve()/'native_protocol.json'
    p=json.loads(source.read_text())
    assert a.rounds==1,'User requested one forward round only'
    a.arms=a.arms or p['native_arms']
    if not set(a.arms)<=set(p['native_arms']):raise ValueError('unregistered arm')
    ck=dict(p['native_checkpoints']);completions=p['training_completions']
    assert a.arms==p['native_arms'] and len(a.arms)==2 and a.arms[0]=='RAW', 'Same-card nonthinking RAW and freshly trained no-soft3 required'
    panel=Path(p['panel_paths'][f'final_C{a.concurrency}'])
    prompts=[json.loads(s) for s in panel.read_text().splitlines()][:a.prompts]
    warm=[json.loads(s) for s in Path(p['panel_paths']['warmup']).read_text().splitlines()]
    original_warmup_count=len(warm)
    assert original_warmup_count>0
    warm=(warm*((max(a.concurrency,2)+original_warmup_count-1)//original_warmup_count))[:max(a.concurrency,2)]
    if len(prompts)!=a.prompts or a.tokens>2096:raise ValueError('invalid diagnostic limits')
    sources=dict(p['source_files'])
    if len(set(a.arms))!=len(a.arms) or not a.arms:raise ValueError('unique nonempty arm selection required')
    orders=[a.arms,list(reversed(a.arms))][:a.rounds]
    if a.smoke:orders=[a.arms]
    temperatures=[0,1] if a.smoke else a.temperatures
    if len(set(temperatures))!=len(temperatures):raise ValueError('duplicate temperature')
    protocol=dict(diagnostic_only=a.smoke,smoke=a.smoke,unpaired=False,repair_graph_redundant_fill=True,gpu=a.gpu,device_assignment=identity,concurrency=a.concurrency,max_tokens=a.tokens,max_model_len=4096,
        prompt_count=a.prompts,panel_sha256=sha(panel),prompts=prompts,temperatures=temperatures,orders=orders,training_completion_sha256=completions,
        training_protocol_sha256=sha(source),checkpoints=ck,sources=sources,environment=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_USE_AOT_COMPILE='1',PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES=identity['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID'),
        memory_utilization=.70,kv_cache_memory_bytes=36*(1<<30),acceptance_only=False,speed_evidence=not a.smoke,shared_gpu_allowed=True,inductor_compile_config=dict(combo_kernels=False,benchmark_combo_kernel=False),deterministic_algorithms=True,graphs='FULL_DECODE_ONLY; all request counts 1..C',repeats_are_not_independent_samples=True)
    protocol.update(colocated_training_on_evaluation_gpu=False,training_may_overlap_test=True,all_training_complete_before_test=False, unrelated_long_training_on_other_cards=True,development_panel=True,scope=f'C{a.concurrency}, {a.prompts} prompts, max{a.tokens}; bounded native preflight of saved training snapshot; repeats are not independent samples',target_counter_limit='target eager/fallback zero does not prove prefill/mixed graph; FULL_DECODE_ONLY',tf32_enabled=False)
    protocol.update(comparison='native snapshot preflight only; training ongoing on another GPU; block16=anchor+15 proposals; no speed evidence',architecture_changed=False,training_changed=True,thinking=False,proposal_length=15,block_size=16)
    protocol.update(training_snapshot=p['training_snapshot'],final_weights=False)
    protocol.update(warmup_unique_prompts=original_warmup_count,warmup_requests=len(warm),warmup_repeated_to_reach_concurrency=a.concurrency>original_warmup_count)
    assert a.gpu==p['gpu'] and (a.smoke or a.prompts=={1:200,40:2000}[a.concurrency])
    dump(out/'protocol.json',protocol)
    if not a.smoke:
        gate=out.parent/(out.name+'_smoke')/'summary.json'
        value=json.loads(gate.read_text())
        if value['protocol']['checkpoints']!=ck or value['protocol']['training_protocol_sha256']!=sha(source) or value['protocol']['training_completion_sha256']!=completions:raise RuntimeError('smoke lineage mismatch')
        if value['status']!='complete' or value['protocol']['sources']!=sources or value['protocol']['gpu']!=a.gpu or value['protocol']['concurrency']!=a.concurrency:raise RuntimeError('missing matching native smoke')
    port=19800+a.gpu;windows=[]
    def interrupted(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
      for rnd,arms in enumerate(orders,1):
       for arm in arms:
        if any(sha(f)!=h for f,h in sources.items()):raise RuntimeError('runtime source changed')
        free,pids=resources(a.gpu)
        while free<65536:
            dump(out/'status.json',dict(status='waiting_memory',arm=arm,free_mib=free,host_pids=pids,utc=now(),foreign_processes_untouched=True))
            with (out/'memory_wait.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=now(),arm=arm,free_mib=free,host_pids=pids))+'\n')
            time.sleep(15)
            free,pids=resources(a.gpu)
        memory_utilization=.70
        print(now(),'MODEL',arm,'PHASE','SMOKE' if a.smoke else 'FORMAL',flush=True)
        trial=out/f'round_{rnd}'/arm;trial.mkdir(parents=True)
        for name in ['execution_counters.bin','target_execution_counters.bin']:(trial/name).write_bytes(bytes(32))
        (trial/'graph_captures.jsonl').write_text('')
        env=bind(environment(),identity);env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_USE_AOT_COMPILE='1',PYTHONHASHSEED='0',VLLM_USE_V2_MODEL_RUNNER='1',
            INPUT_ABLATION_CONFIG=json.dumps(p['native_configs'].get(arm,{})),RUNTIME_VARIANT='optimized',HIDDEN_ARM='RAW' if arm=='RAW' else 'EMBEDDING',ORIGINAL_RUNTIME_ARM='RAW' if arm=='RAW' else 'EMBEDDING',ABLATION_LABEL=arm,NATIVE_PROPOSAL_LENGTH='15',
            TOKEN_FEATURE_SOURCE='RAW' if arm=='RAW' else 'EMBEDDING',D1_BASIS_PATH=p['basis_path'],D1_BASIS_SHA256=p['basis_sha256'],HIDDEN_HEAD_CONFIG=ck[arm]['head_config_path'],HIDDEN_HEAD_CONFIG_SHA256=ck[arm]['head_config_sha256'],RUNTIME_REAL_PROFILE='0',
            LOCAL_ARM='raw_graph' if arm=='RAW' else 'dense_combined',LOCAL_TRIAL=str(trial),
            LOCAL_CONTRACT_CHECKS='1' if a.smoke else '0',LOCAL_MAX_CONCURRENCY=str(a.concurrency),NATIVE_FULL_GRAPH='1',
            REPAIR_CHECKPOINT=ck[arm]['path'],STATIC_M_PATH=p['m_ids_path'],STATIC_M_SHA256=p['m_ids_sha256'],STATIC_M_SIZE=str(p['m_size']),
            XDG_CACHE_HOME=str(out/'cache'),TRITON_CACHE_DIR=str(out/'triton_cache'))
        cmd=[str(EVAL/'bin/python'),'-B','-m','repair.native_launch','serve',str(Path('/data/weijiahong/huggingface/models/Qwen3-8B')),
            '--host','127.0.0.1','--port',str(port),'--device-ids','0','--tensor-parallel-size','1','--dtype','bfloat16','--generation-config','vllm','--seed','0',
            '--max-model-len','4096','--max-num-seqs',str(a.concurrency),'--max-num-batched-tokens','8192','--gpu-memory-utilization',str(memory_utilization),
            '--kv-cache-memory-bytes',str(36*(1<<30)),'--no-enable-prefix-caching','--no-async-scheduling','--compilation-config',json.dumps(dict(inductor_compile_config=dict(combo_kernels=False,benchmark_combo_kernel=False),cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=[16*i for i in range(1,a.concurrency+1)])),
            '--speculative-config',json.dumps(dict(method='dflash',model=str((HERE/'body_only_draft').resolve()),num_speculative_tokens=15,enforce_eager=False,draft_sample_method='probabilistic'))]
        dump(trial/'command.json',dict(command=cmd,environment={k:env[k] for k in ('TOKEN_FEATURE_SOURCE','VLLM_USE_AOT_COMPILE','PYTHONHASHSEED','RUNTIME_VARIANT','ORIGINAL_RUNTIME_ARM','HIDDEN_ARM','INPUT_ABLATION_CONFIG','HIDDEN_HEAD_CONFIG','D1_BASIS_PATH','D1_BASIS_SHA256','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','CUBLAS_WORKSPACE_CONFIG','TORCHINDUCTOR_DETERMINISTIC')},initial_free_mib=free,initial_host_pids=pids,actual_memory_utilization=memory_utilization))
        proc=None;done=threading.Event();samples=[];errors=[];watcher=None
        def inspect():
            available,ids=resources(a.gpu)
            foreign=sorted(set(ids)-set(members(proc.pid)))
            samples.append(dict(utc=now(),free_mib=available,foreign_pids=foreign))
            # Foreign processes are recorded; they do not stop owned work.
        def watch():
            while not done.wait(5):
                try:inspect()
                except Exception as e:errors.append(repr(e))
        trial_rows=[]
        try:
          with (trial/'vllm.log').open('x') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            watcher=threading.Thread(target=watch,daemon=True);watcher.start()
            dump(out/'status.json',dict(status='starting',round=rnd,arm=arm,pid=proc.pid,utc=now()))
            for _ in range(600):
                if errors:raise RuntimeError(f'isolation lost at startup: {errors}')
                if proc.poll() is not None:raise RuntimeError(f'server failed: {trial}')
                if healthy(port):break
                time.sleep(1)
            else:raise TimeoutError('health')
            endpoint=f'http://127.0.0.1:{port}';model=_http_json(endpoint+'/v1/models')['data'][0]['id']
            for t in temperatures:
                # Graphs are captured at startup for every size; sweep only if coverage is missing.
                captures=[json.loads(x) for x in (trial/'graph_captures.jsonl').read_text().splitlines()]
                gate=coverage(captures,a.concurrency)
                for size in gate['missing_batch_sizes']:parallel_requests(endpoint,model,warm[:size],t,32,size)
                for _ in range(2):parallel_requests(endpoint,model,warm[:max(a.concurrency,2)],t,32 if a.smoke else 512,a.concurrency)
                gate=coverage([json.loads(x) for x in (trial/'graph_captures.jsonl').read_text().splitlines()],a.concurrency)
                if not gate['passed']:raise RuntimeError('incomplete Graph coverage')
                dump(trial/f'warmup_T{t}.json',gate)
                inspect();before=_http_text(endpoint+'/metrics');b=snapshot(trial/'execution_counters.bin');tb=snapshot(trial/'target_execution_counters.bin')
                dump(out/'status.json',dict(status='measuring',round=rnd,arm=arm,T=t,utc=now()))
                start=time.perf_counter();rows=parallel_requests(endpoint,model,warm[:a.concurrency] if a.smoke else prompts,t,a.tokens,a.concurrency);elapsed=time.perf_counter()-start
                if errors:raise RuntimeError(f'isolation monitor failed: {errors}')
                metrics=_metrics_delta(before,_http_text(endpoint+'/metrics'));counts=delta(b,snapshot(trial/'execution_counters.bin'));target=delta(tb,snapshot(trial/'target_execution_counters.bin'));inspect()
                tokens=sum(r['response']['usage']['completion_tokens'] for r in rows)
                if metrics['num_drafts']<=0 or abs(metrics['acceptance_length']-1-sum(metrics[f'acceptance_at_pos_{j}'] for j in range(15)))>1e-8:raise RuntimeError('acceptance counters invalid')
                for c in (counts,target):
                    if c['captures'] or c['fallbacks'] or c['eager'] or c['replays']<=0:raise RuntimeError(f'Graph gate failed: {c}')
                record=dict(round=rnd,arm=arm,T=t,requests=len(rows),completion_tokens=tokens,wall_seconds=elapsed,tokens_per_second=tokens/elapsed,
                    native_metrics=metrics,execution_counters=counts,target_execution_counters=target,diagnostic_only=a.smoke,smoke=a.smoke)
                (trial/f'T{t}_requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
                trial_rows.append(record);print(now(),rnd,arm,t,'A',metrics['acceptance_length'],'TPS',tokens/elapsed,flush=True)
        finally:
            done.set()
            if watcher:watcher.join(timeout=8)
            if proc:stop(proc)
            dump(trial/'isolation.json',dict(samples=samples,errors=errors))
        isolated=bool(samples) and not errors and not any(x['foreign_pids'] for x in samples)
        if errors:raise RuntimeError('GPU monitor query failed; preserve evidence')
        if a.smoke:
            contracts=[json.loads(x) for x in (trial/'sampling_contract.jsonl').read_text().splitlines()]
            if not contracts or not all(x['graph_eager_equal'] and x['processed_q_equal'] and x['feedback_equal'] for x in contracts) or not any(x['temperature']>0 and x['non_top1']>0 for x in contracts):raise RuntimeError('native independent-reference parity/stochastic witness missing')
        if arm!='RAW':
            contract=json.loads((trial/'model_contract.json').read_text());cfg=p['architecture_by_arm'][arm]
            expected_branches=[n for n,enabled in [('hidden',cfg['use_hidden']),('hard_previous',True),('q',True),('soft_previous',cfg['use_soft'])] if enabled]
            assert contract['input_feature_source']=='EMBEDDING' and contract['head_config_sha256']==ck[arm]['head_config_sha256']
            assert contract['checkpoint_sha256']==ck[arm]['sha256'] and contract['blocks']==cfg['depth']
            assert contract['correction_support']==cfg['q_topk'] and contract['sampling_support']=='full original vocabulary'
            assert contract['branches']==expected_branches and contract['scale_mode']==cfg['scale_mode'] and contract['fixed_scale']==cfg['fixed_scale']
        for w in trial_rows:w['isolation_observed']=isolated;w['speed_evidence']=not a.smoke and isolated;w['performance_contaminated']=not isolated;w['acceptance_only']=False;w['actual_memory_utilization']=memory_utilization
        dump(trial/'windows.json',trial_rows);windows.extend(trial_rows)
        dump(trial/'complete.json',dict(status='complete',artifacts={f.name:sha(f) for f in trial.iterdir() if f.is_file()},utc=now()))
        dump(out/'summary.json',dict(status='running',protocol=protocol,windows=windows))
      assert len(windows)==2*len(a.arms) and {(w['arm'],w['T']) for w in windows}=={(arm,t) for arm in a.arms for t in (0,1)}
      dump(out/'summary.json',dict(status='complete',protocol=protocol,windows=windows))
      dump(out/'status.json',dict(status='complete',utc=now()))
    except BaseException as exc:
      dump(out/'status.json',dict(status='failed',error=repr(exc),utc=now()));raise

if __name__=='__main__':main()
