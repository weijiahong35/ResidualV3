"""Formal E2 independent DFlash versus frozen HEAD2_NOFILL; no cutoff."""
import sys
from pathlib import Path
RUNTIME=Path(__file__).resolve().parent/'native_runtime'
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


def tail_snapshot(endpoint):
    return _http_json(endpoint+'/repair_tail_stats',timeout=45)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--training",type=Path,required=True)
    ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--concurrency',type=int,choices=[1,32],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--tokens',type=int,default=4096)
    ap.add_argument('--prompts',type=int,required=True);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--formal-path-smoke',action='store_true')
    ap.add_argument('--rounds',type=int,choices=[1,2],default=1);ap.add_argument('--arms',nargs='+',default=None);ap.add_argument('--temperatures',nargs='+',type=int,choices=[0,1],default=[0,1]);a=ap.parse_args()
    identity=assignment(a.gpu)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=a.training.resolve()/'native_protocol.json'
    p=json.loads(source.read_text())
    assert a.rounds==1,'User requested one forward round only'
    a.arms=a.arms or p['native_arms']
    if not set(a.arms)<=set(p['native_arms']):raise ValueError('unregistered arm')
    ck=dict(p['native_checkpoints']);completions=p['training_completions']
    assert p['native_arms']==['DFLASH','HEAD2_NOFILL']
    # C1 covers exact short prefixes. C32 rounds ragged totals by at most7
    # rows while retaining every old uniform full-graph size.
    capture_sizes=list(range(1,17)) if a.concurrency==1 else sorted(set(range(1,17))|set(range(24,16*a.concurrency+1,8)))
    assert not a.formal_path_smoke or a.smoke, 'formal-path-smoke is diagnostic only'
    panel=Path(p['panel_paths'][f'final_C{a.concurrency}'])
    prompts=[json.loads(s) for s in panel.read_text().splitlines()][:a.prompts]
    warm=[json.loads(s) for s in Path(p['panel_paths']['warmup']).read_text().splitlines()]
    original_warmup_count=len(warm)
    assert original_warmup_count>0
    warm=(warm*((max(a.concurrency,2)+original_warmup_count-1)//original_warmup_count))[:max(a.concurrency,2)]
    if len(prompts)!=a.prompts or a.tokens>2096:raise ValueError('invalid diagnostic limits')
    sources=dict(p['source_files'])
    phase='smoke' if a.smoke else 'formal'
    kv_gib=p['kv_cache_gib_by_phase'][phase][f'C{a.concurrency}']
    kv_bytes=kv_gib*(1<<30)
    requested_utilization=.92
    if len(set(a.arms))!=len(a.arms) or not a.arms:raise ValueError('unique nonempty arm selection required')
    orders=[a.arms,list(reversed(a.arms))][:a.rounds]
    if a.smoke:orders=[a.arms]
    temperatures=[0,1] if a.smoke else a.temperatures
    if len(set(temperatures))!=len(temperatures):raise ValueError('duplicate temperature')
    protocol=dict(diagnostic_only=a.smoke,smoke=a.smoke,unpaired=False,repair_graph_redundant_fill=False,gpu=a.gpu,device_assignment=identity,concurrency=a.concurrency,max_tokens=a.tokens,max_model_len=4096,
        prompt_count=a.prompts,panel_sha256=sha(panel),prompts=prompts,temperatures=temperatures,orders=orders,training_completion_sha256=completions,
        training_protocol_sha256=sha(source),checkpoints=ck,sources=sources,environment=dict(VLLM_BATCH_INVARIANT='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_USE_AOT_COMPILE='1',PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES=identity['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID'),
        target_batch_invariant=True,memory_utilization=requested_utilization,kv_cache_memory_bytes=kv_bytes,acceptance_only=False,speed_evidence=not a.smoke,shared_gpu_allowed=False,inductor_compile_config=dict(combo_kernels=False,benchmark_combo_kernel=False),deterministic_algorithms=True,graphs='FULL_AND_PIECEWISE: native full uniform target decode, native piecewise ragged target decode; fixed full draft graphs',repeats_are_not_independent_samples=True)
    protocol.update(colocated_training_on_evaluation_gpu=False,training_may_overlap_test=False,all_training_complete_before_test=True, unrelated_long_training_on_other_cards=True,development_panel=False,scope=f"{p['dataset']}, C{a.concurrency}, {a.prompts} unique first-turn questions, max{a.tokens}; one forward round",target_counter_limit='target eager/fallback zero does not prove prefill/mixed graph; FULL_DECODE_ONLY',tf32_enabled=False)
    protocol.update(comparison=p['implementation_comparison'],architecture_changed=True,training_changed=False,thinking=False,proposal_length=15,block_size=16,
        repair_prefix_length_by_arm=p['repair_prefix_length_by_arm'],adaptive_q_gate=False, predictor_sha256=sha(p['predictor_path']), delay_positions=1, repair_depth=2, source_trained_repair_depth=3, two_layer_retrained=False)
    for key,path in p['panel_paths'].items():assert sha(path)==p['panel_sha256'][key],key
    for arm in a.arms:
        assert sha(ck[arm]['path'])==ck[arm]['sha256'] and sha(ck[arm]['head_config_path'])==ck[arm]['head_config_sha256']
    protocol.update(formal_path_smoke=a.formal_path_smoke,recovery_parent=p.get('recovery_parent'))
    protocol.update(short_development_only=p.get('short_development_only',False),skip_empty_fill_by_arm=p['skip_empty_fill_by_arm'],implementation_comparison=p['implementation_comparison'])
    protocol.update(latency_diagnostic=bool(p.get('latency_diagnostic')),
                    speed_evidence=not a.smoke and not p.get('latency_diagnostic'),
                    target_prefix_mode=p.get('target_prefix_mode','ADAPTIVE'),
                    physical_target_compaction=False)
    if p.get('short_development_only'):
        protocol['scope']=f'Short implementation development: C{a.concurrency}, {a.prompts} fixed prompts, max{a.tokens}; not a full formal evaluation'
    protocol.update(warmup_unique_prompts=original_warmup_count,warmup_requests=len(warm),warmup_repeated_to_reach_concurrency=a.concurrency>original_warmup_count)
    assert a.gpu==p['gpu_by_concurrency'][f'C{a.concurrency}'] and (a.smoke or a.prompts==p['prompts_per_concurrency'][f'C{a.concurrency}'])
    protocol.update(dataset=p['dataset'], dataset_provenance=p['dataset_provenance'], dataset_scope=p['dataset_scope'], unique_questions=a.prompts, repeated_questions=False, dspark_alignment=p['dspark_alignment'])
    dump(out/'protocol.json',protocol)
    if not a.smoke:
        gate=out.parent/(out.name+'_smoke')/'summary.json'
        value=json.loads(gate.read_text())
        if value['protocol']['checkpoints']!=ck or value['protocol']['training_protocol_sha256']!=sha(source) or value['protocol']['training_completion_sha256']!=completions:raise RuntimeError('smoke lineage mismatch')
        if value['status']!='complete' or value['protocol']['sources']!=sources or value['protocol']['gpu']!=a.gpu or value['protocol']['concurrency']!=a.concurrency:raise RuntimeError('missing matching native smoke')
    port=p['server_port_base']+a.gpu;windows=[]
    with socket.socket() as check_socket:
        check_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        check_socket.bind(('127.0.0.1',port))
    def interrupted(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
      for rnd,arms in enumerate(orders,1):
       for arm in arms:
        repair_on=p['repair_enabled_by_arm'][arm]
        for filename,digest in p['body_checksums'][arm].items():
            assert sha(Path(p['body_models'][arm])/filename)==digest, (arm,filename)
        if any(sha(f)!=h for f,h in sources.items()):raise RuntimeError('runtime source changed')
        free,pids=resources(a.gpu)
        while pids or free<78000:
            dump(out/'status.json',dict(status='waiting_memory',arm=arm,free_mib=free,host_pids=pids,utc=now(),foreign_processes_untouched=True))
            with (out/'memory_wait.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=now(),arm=arm,free_mib=free,host_pids=pids))+'\n')
            time.sleep(15)
            free,pids=resources(a.gpu)
        memory_utilization=requested_utilization
        print(now(),'MODEL',arm,'PHASE','SMOKE' if a.smoke else ('SHORT' if p.get('short_development_only') else 'FORMAL'),flush=True)
        trial=out/f'round_{rnd}'/arm;trial.mkdir(parents=True)
        for name in ['execution_counters.bin','target_execution_counters.bin']:(trial/name).write_bytes(bytes(32))
        (trial/'graph_captures.jsonl').write_text('')
        env=bind(environment(),identity);env.update(VLLM_BATCH_INVARIANT='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_USE_AOT_COMPILE='1',PYTHONHASHSEED='0',VLLM_USE_V2_MODEL_RUNNER='1',
            TAIL_RAGGED_FULL_GRAPH='1' if p.get('ragged_full_graph') and arm.endswith('_CUT') else '0',
            TAIL_SPARSE_Q45=str(p.get('sparse_q45',1)),
            TAIL_LATENCY_DIAGNOSTIC='1' if p.get('latency_diagnostic') else '0',
            TAIL_HIGH_PRIORITY_MAIN=str(p.get('high_priority_main',0)),
            TAIL_SPARSE_PREP=p.get('sparse_prepare','topk'),
            TAIL_CACHE_POSITIONS=str(p.get('cache_positions',15)),
            TAIL_FUSED_FEATURE_ASSEMBLY=str(p.get('fused_feature_assembly',0)),
            TAIL_FUSED_SCAN=str(p.get('fused_scan',0)),
            TAIL_EARLY_BATCH_SIZE=str(p.get('early_batch_size',9)),
            TAIL_DENSE_GUARANTEED=str(p.get('dense_guaranteed',0)),
            TAIL_FUSED_READOUT=str(p.get('fused_readout',0)),
            TAIL_FUSED_FINAL=str(p.get('fused_final',0)),
            TAIL_REUSE_SAMPLE_MAX=str(p.get('reuse_sample_max',0)),
            TAIL_FUSED_INPUT=str(p.get('fused_input',0)),
            HEAD_SKIP_EMPTY_FILL=str(p['skip_empty_fill_by_arm'][arm]),
            TAIL_SERIAL_EARLY_ENTROPY=str(p.get('serial_early_entropy',0)),
            INPUT_ABLATION_CONFIG=json.dumps(p['native_configs'].get(arm,{})),RUNTIME_VARIANT='optimized',HIDDEN_ARM='EMBEDDING' if repair_on else 'RAW',ORIGINAL_RUNTIME_ARM='EMBEDDING' if repair_on else 'RAW',ABLATION_LABEL=arm,NATIVE_PROPOSAL_LENGTH='15',
            HEAD_STRUCTURE_MODE=p['head_structure_mode_by_arm'][arm],HEAD_LATENCY_OPT_VARIANT=p['optimization_by_arm'][arm],HEAD_SMALL_TILE_MODE=p['head_small_tile_mode'],TAIL_MODE=p['tail_mode_by_arm'][arm],TARGET_PREFIX_MODE=(p.get('target_prefix_mode','ADAPTIVE') if arm.endswith('_CUT') else 'FULL'),TAIL_PREDICTOR=p['predictor_path'],TAIL_AUX_CHECKPOINT=p['auxiliary_checkpoint'],TAIL_AUX_BRANCH=p['auxiliary_branch_by_arm'][arm],TOKEN_FEATURE_SOURCE='EMBEDDING' if repair_on else 'RAW',D1_BASIS_PATH=p['basis_path'],D1_BASIS_SHA256=p['basis_sha256'],HIDDEN_HEAD_CONFIG=ck[arm]['head_config_path'],HIDDEN_HEAD_CONFIG_SHA256=ck[arm]['head_config_sha256'],RUNTIME_REAL_PROFILE='0',
            LOCAL_ARM='dense_combined' if repair_on else 'raw_graph',LOCAL_TRIAL=str(trial),
            LOCAL_CONTRACT_CHECKS='1' if a.smoke and not a.formal_path_smoke else '0',LOCAL_MAX_CONCURRENCY=str(a.concurrency),NATIVE_FULL_GRAPH='1',
            REPAIR_CHECKPOINT=ck[arm]['path'],STATIC_M_PATH=p['m_ids_path'],STATIC_M_SHA256=p['m_ids_sha256'],STATIC_M_SIZE=str(p['m_size']),
            XDG_CACHE_HOME=str(Path(p['shared_eval_cache'])/'xdg'),TRITON_CACHE_DIR=str(Path(p['shared_eval_cache'])/'triton'))
        cmd=[str(EVAL/'bin/python'),'-B','-m','repair.target_prefix_launch','serve',str(Path('/data/weijiahong/huggingface/models/Qwen3-8B')),
            '--host','127.0.0.1','--port',str(port),'--device-ids','0','--tensor-parallel-size','1','--dtype','bfloat16','--generation-config','vllm','--seed','0',
            '--max-model-len','4096','--max-num-seqs',str(a.concurrency),'--max-num-batched-tokens','8192','--gpu-memory-utilization',str(memory_utilization),
            '--kv-cache-memory-bytes',str(kv_bytes),'--no-enable-prefix-caching','--no-async-scheduling','--compilation-config',json.dumps(dict(inductor_compile_config=dict(combo_kernels=False,benchmark_combo_kernel=False),cudagraph_mode='FULL_AND_PIECEWISE',cudagraph_capture_sizes=capture_sizes)),
            '--speculative-config',json.dumps(dict(method='dflash',model=p['body_models'][arm],num_speculative_tokens=15,enforce_eager=False,draft_sample_method='probabilistic'))]
        recorded_env={k:env[k] for k in ('VLLM_BATCH_INVARIANT','HEAD_LATENCY_OPT_VARIANT','HEAD_SMALL_TILE_MODE','HEAD_STRUCTURE_MODE','TAIL_MODE','TAIL_PREDICTOR','TOKEN_FEATURE_SOURCE','VLLM_USE_AOT_COMPILE','PYTHONHASHSEED','RUNTIME_VARIANT','ORIGINAL_RUNTIME_ARM','HIDDEN_ARM','INPUT_ABLATION_CONFIG','HIDDEN_HEAD_CONFIG','D1_BASIS_PATH','D1_BASIS_SHA256','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','CUBLAS_WORKSPACE_CONFIG','TORCHINDUCTOR_DETERMINISTIC')}
        recorded_env['HEAD_SKIP_EMPTY_FILL']=env['HEAD_SKIP_EMPTY_FILL']
        recorded_env.update({k:v for k,v in env.items() if k.startswith('TAIL_') or k.startswith('TARGET_PREFIX_')})
        dump(trial/'command.json',dict(command=cmd,environment=recorded_env,initial_free_mib=free,initial_host_pids=pids,actual_memory_utilization=memory_utilization))
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
            for _ in range(1500):
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
                tail_before=tail_snapshot(endpoint) if repair_on else None
                prefix_before=_http_json(endpoint+'/target_prefix_stats',timeout=45)
                if p.get('latency_diagnostic'):_http_json(endpoint+'/repair_short_latency?reset=true',timeout=45)
                start=time.perf_counter();rows=parallel_requests(endpoint,model,prompts,t,a.tokens,a.concurrency);elapsed=time.perf_counter()-start
                if p.get('latency_diagnostic'):
                    dump(trial/f'T{t}_actual_latency.json',_http_json(endpoint+'/repair_short_latency',timeout=45))
                tail_after=tail_snapshot(endpoint) if repair_on else None
                prefix_after=_http_json(endpoint+'/target_prefix_stats',timeout=45)
                if errors:raise RuntimeError(f'isolation monitor failed: {errors}')
                metrics=_metrics_delta(before,_http_text(endpoint+'/metrics'));counts=delta(b,snapshot(trial/'execution_counters.bin'));target=delta(tb,snapshot(trial/'target_execution_counters.bin'));inspect()
                tokens=sum(r['response']['usage']['completion_tokens'] for r in rows)
                if metrics['num_drafts']<=0 or abs(metrics['acceptance_length']-1-sum(metrics[f'acceptance_at_pos_{j}'] for j in range(15)))>1e-8:raise RuntimeError('acceptance counters invalid')
                for c in (counts,target):
                    if c['captures'] or c['fallbacks'] or c['eager'] or c['replays']<=0:raise RuntimeError(f'Graph gate failed: {c}')
                record=dict(round=rnd,arm=arm,T=t,tail_mode=p['tail_mode_by_arm'][arm],requests=len(rows),completion_tokens=tokens,wall_seconds=elapsed,tokens_per_second=tokens/elapsed,
                    native_metrics=metrics,execution_counters=counts,target_execution_counters=target,diagnostic_only=a.smoke,smoke=a.smoke)
                record['target_prefix_counters']={k:prefix_after[k]-prefix_before[k] for k in prefix_after if isinstance(prefix_after[k],int)}
                record['target_prefix_histogram']=[v-u for u,v in zip(prefix_before['histogram'],prefix_after['histogram'])]
                if record['target_prefix_counters']['decode_eager_calls']:raise RuntimeError('Target verification decode silently fell back to eager')
                if arm.endswith('_CUT'):record['genuine_target_shortening_observed']=any(record['target_prefix_histogram'][1:15])
                record['target_prefix_mode']=p.get('target_prefix_mode','ADAPTIVE') if arm.endswith('_CUT') else 'FULL'
                if record['target_prefix_mode']=='DEVICE':
                    record['genuine_target_shortening_observed']=False
                    record['device_logical_prefix_with_fixed_target_padding']=True
                (trial/f'T{t}_requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
                if arm.endswith('_CUT'):
                    hist=[b-a for a,b in zip(tail_before['counts'],tail_after['counts'])]
                    assert all(v>=0 for v in hist)
                    record['repair_cutoff_counts']={str(j):v for j,v in enumerate(hist)}
                    record['mean_repair_positions']=sum(j*v for j,v in enumerate(hist))/max(sum(hist),1)
                    record['tail_statistics_blocks']=sum(hist)
                else:
                    record['mean_repair_positions']=15 if repair_on else 0
                dump(trial/f'T{t}_window_provisional.json',dict(record,isolation_pending=True))
                trial_rows.append(record);print(now(),rnd,arm,t,'A',metrics['acceptance_length'],'TPS',tokens/elapsed,flush=True)
        finally:
            done.set()
            if watcher:watcher.join(timeout=8)
            if proc:stop(proc)
            dump(trial/'isolation.json',dict(samples=samples,errors=errors))
        isolated=bool(samples) and not errors and not any(x['foreign_pids'] for x in samples)
        if errors:raise RuntimeError('GPU monitor query failed; preserve evidence')
        if a.smoke and not a.formal_path_smoke:
            contracts=[json.loads(x) for x in (trial/'sampling_contract.jsonl').read_text().splitlines()]
            if not contracts or not all(x['graph_eager_equal'] and x['processed_q_equal'] and x['feedback_equal'] for x in contracts) or not any(x['temperature']>0 and x['non_top1']>0 for x in contracts):raise RuntimeError('native independent-reference parity/stochastic witness missing')
            if repair_on and p['optimization_by_arm'][arm] != 'baseline':
                parity=[json.loads(x) for x in (trial/'head_implementation_parity.jsonl').read_text().splitlines()]
                checks=('logits_bitwise_equal','sampled_ids_bitwise_equal','actual_predecessors_bitwise_equal','verifier_q_bitwise_equal')
                if arm.endswith('_CUT'):checks=checks+('raw_suffix_logits_bitwise_equal',)
                assert parity and all(all(x[k] for k in checks) for x in parity), 'optimized real-state parity failed'
                assert {0.,1.} <= {x['temperature'] for x in parity}, 'T0/T1 real parity witnesses missing'
                if p.get('target_prefix_mode')=='DEVICE' and arm.endswith('_CUT'):
                    device_parity=[json.loads(x) for x in (trial/'device_prefix_sampler_parity.jsonl').read_text().splitlines()]
                    assert device_parity and all(x['prefix_sampler_bitwise'] for x in device_parity)
        if repair_on:
            contract=json.loads((trial/'model_contract.json').read_text());cfg=p['architecture_by_arm'][arm]
            expected_branches=[n for n,enabled in [('hidden',cfg['use_hidden']),('hard_previous',True),('q',True),('soft_previous',cfg['use_soft'])] if enabled]
            assert contract['input_feature_source']=='EMBEDDING' and contract['head_config_sha256']==ck[arm]['head_config_sha256']
            assert contract['checkpoint_sha256']==ck[arm]['sha256'] and contract['blocks']==cfg['depth']
            assert contract['correction_support']==cfg['q_topk'] and contract['sampling_support']=='full original vocabulary'
            assert contract['branches']==expected_branches and contract['scale_mode']==cfg['scale_mode'] and contract['fixed_scale']==cfg['fixed_scale']
            assert contract['tail_mode']==p['tail_mode_by_arm'][arm] and contract['tail_predictor_sha256']==p['predictor_sha256']
            assert contract['auxiliary_checkpoint_sha256']==p['auxiliary_checkpoint_sha256']
            assert contract['auxiliary_branch']==p['auxiliary_branch_by_arm'][arm]
        else:
            contract=json.loads((trial/'model_contract.json').read_text())
            assert contract['blocks']==0 and contract['correction_support']==0
        # Update inherited descriptive labels without changing native computation.
        contract.update(graph_scope='target decode Graph; draft query + raw LM + optional repair + actual 15-proposal stock sampling Graph; context KV preparation remains eager', training_lineage=p['long_training'],
            evaluation_arm=arm,
            repair_initialization='first two repair layers of E2 three-layer checkpoint; third layer omitted for inference, no two-layer retraining' if repair_on else 'not applicable',
            body_weight_lineage=p['body_models'][arm],
            body_weight_checksums=p['body_checksums'][arm])
        dump(trial/'model_contract.json',contract)
        dump(trial/'loaded_body_lineage.json',dict(body_model=p['body_models'][arm],checksums=p['body_checksums'][arm],repair_enabled=repair_on,utc=now()))
        for w in trial_rows:w['isolation_observed']=isolated;w['speed_evidence']=not a.smoke and isolated and not p.get('latency_diagnostic');w['latency_diagnostic']=bool(p.get('latency_diagnostic'));w['performance_contaminated']=not isolated;w['acceptance_only']=False;w['actual_memory_utilization']=memory_utilization
        dump(trial/'windows.json',trial_rows);windows.extend(trial_rows)
        dump(trial/'complete.json',dict(status='complete',artifacts={f.name:sha(f) for f in trial.iterdir() if f.is_file()},utc=now()))
        dump(out/'summary.json',dict(status='running',protocol=protocol,windows=windows))
      assert len(windows)==2*len(a.arms) and {(w['arm'],w['T']) for w in windows}=={(arm,t) for arm in a.arms for t in (0,1)}
      dump(out/'summary.json',dict(status='complete',protocol=protocol,windows=windows))
      dump(out/'status.json',dict(status='complete',utc=now()))
    except BaseException as exc:
      dump(out/'status.json',dict(status='failed',error=repr(exc),utc=now()));raise

if __name__=='__main__':main()
