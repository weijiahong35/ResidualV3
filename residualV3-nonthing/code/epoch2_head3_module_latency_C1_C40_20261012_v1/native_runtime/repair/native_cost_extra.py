"""Additional real-service spans and same-worker actual-state head replay.

All measurements are diagnostic. No timing result from this module is TPS evidence.
"""
import json,os,statistics,time
from pathlib import Path
import torch


def captured_span(model,name,n,fn):
    if not torch.cuda.is_current_stream_capturing():return fn()
    a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True)
    a.record();out=fn();b.record()
    if not hasattr(model,'_short_latency_spans'):model._short_latency_spans={}
    model._short_latency_spans.setdefault(n,{})[name]=(a,b)
    return out


def install_extra(timer):
    from runtime import native
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    prior=DFlashSpeculator.propose
    def propose(self,input_batch,*args,**kwargs):
        n=input_batch.num_reqs
        lengths=input_batch.seq_lens_cpu_upper_bound[:n]
        meta=dict(n=n,input_tokens=input_batch.num_tokens,context_min=int(lengths.min()),context_max=int(lengths.max()))
        native.MODEL._profile_current_propose_meta=meta
        return timer('draft_propose_total',lambda:prior(self,input_batch,*args,**kwargs),meta)
    DFlashSpeculator.propose=propose
    original=native.ExecutionDraft.precompute_and_store_context_kv
    def context(self,*args,**kwargs):
        return timer('draft_context_kv',lambda:original(self,*args,**kwargs),{})
    native.ExecutionDraft.precompute_and_store_context_kv=context


def actual_head_probe(n):
    from runtime import native
    from .native_launch import run_eager
    model=native.MODEL
    assert n in model.integrated_inputs
    saved=model._cost_real_state
    assert saved['n']==n
    inp=saved['inputs'];fp64=saved['fp64']
    q=torch.empty(saved['q_shape'],device='cuda',dtype=torch.float32)
    torch.cuda.synchronize()
    expected_ids=saved['ids'];expected_q=saved['q']
    idx=inp[4].long();step=inp[7].long()
    spans={k:dict(v) for k,v in getattr(model,'_short_latency_spans',{}).items()}
    with torch.inference_mode():
        def fn():return run_eager(model.head,model.lm_head.weight,model.token_basis,inp,q,arm=model.arm,capture=False,fp64=fp64)
        out=fn()
        checks=dict(tokens=torch.equal(out[0],expected_ids),verifier_q=torch.equal(q[idx,step],expected_q))
        assert all(checks.values()),checks
        for _ in range(12):fn()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):result=fn()
        for _ in range(40):g.replay()
        times=[]
        for _ in range(7):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(25):g.replay()
            b.record();b.synchronize();times.append(a.elapsed_time(b)/25)
        checks.update(graph_tokens=torch.equal(result[0],expected_ids),graph_q=torch.equal(q[idx,step],expected_q))
        assert all(checks.values()),checks
        # Detail events apply only to this same-worker replay of the saved state.
        # The real request workload above is not run a second time.
        previous_detail=os.environ.get('HEAD_PROFILE_DETAIL','0')
        module_times={};detail_times=[]
        try:
            os.environ['HEAD_PROFILE_DETAIL']='1'
            detail_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(detail_graph):detail_result=fn()
            detail_spans={name:pair for name,pair in model._short_latency_spans[n].items() if name.startswith('head.')}
            assert detail_spans,'No head module event spans captured'
            for _ in range(40):detail_graph.replay()
            for _ in range(7):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                for _ in range(25):detail_graph.replay()
                end.record();end.synchronize();detail_times.append(start.elapsed_time(end)/25)
                values={}
                for name,(a,b) in detail_spans.items():
                    category=name.split('#')[0]
                    values[category]=values.get(category,0.)+a.elapsed_time(b)
                for name,value in values.items():module_times.setdefault(name,[]).append(value)
            checks.update(detail_graph_tokens=torch.equal(detail_result[0],expected_ids),detail_graph_q=torch.equal(q[idx,step],expected_q))
            assert all(checks.values()),checks
        finally:
            os.environ['HEAD_PROFILE_DETAIL']=previous_detail
    model._short_latency_spans=spans
    return dict(scope='Head+sampling and module spans, same worker cloned real service state with original strides; first full-size block only; repeated one real state; detail events perturb the replay, not another service workload',C=n,fp64_gumbel=fp64,median_ms=statistics.median(times),repeats_ms=times,checks=checks,
        module_replay_ms={name:dict(mean_ms=statistics.mean(values),median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),observations=len(values),repeats_ms=values) for name,values in module_times.items()},
        detail_graph_total_median_ms=statistics.median(detail_times),detail_graph_total_repeats_ms=detail_times,
        detail_event_overhead_observed_ms=statistics.median(detail_times)-statistics.median(times),
        module_spans_nested_do_not_sum=True,real_workload_repeated_for_modules=False,
        snapshot_sequence=saved['sequence'],actual_state_metadata=saved['actual_state_metadata'],input_shapes=[list(v.shape) for v in inp],input_strides=[list(v.stride()) for v in inp],head_skip_empty_fill=os.environ.get('HEAD_SKIP_EMPTY_FILL','0'),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),fill_flag_restored=torch.utils.deterministic.fill_uninitialized_memory)


def save_real_state(model,n,sequence):
    if n!=int(os.environ['LOCAL_MAX_CONCURRENCY']) or getattr(model,'_cost_real_state',None) is not None:return
    metadata=dict(getattr(model,'_profile_current_propose_meta',{}))
    if metadata.get('n',n)!=n:return
    inp,q,result,fp64=model.integrated_inputs[n]
    copies=[]
    for x in inp:
        y=torch.empty_strided(x.shape,x.stride(),dtype=x.dtype,device=x.device)
        y.copy_(x);copies.append(y)
    model._cost_real_state=dict(n=n,inputs=tuple(copies),ids=result[0].clone(),q=q[inp[4].long(),inp[7].long()].clone(),q_shape=list(q.shape),fp64=fp64,sequence=sequence,actual_state_metadata=metadata)
