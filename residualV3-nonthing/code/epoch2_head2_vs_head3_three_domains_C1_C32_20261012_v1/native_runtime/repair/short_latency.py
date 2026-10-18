"""Opt-in real-service CUDA/CPU spans, excluded from throughput evidence."""
import json,os,time
from collections import defaultdict
import torch

ENABLED=False
ROWS=[]
PENDING=[]
LAST_DRAFT=None
ORIGIN=None


def _collect_draft():
    global LAST_DRAFT
    if LAST_DRAFT is None:return
    n,spans,ready=LAST_DRAFT
    if not ready.query():return
    row=dict(kind='captured_draft_components',n=n)
    for name,(start,end) in spans.items():
        assert end.query()
        row[name+'_ms']=start.elapsed_time(end)
    ROWS.append(row);LAST_DRAFT=None


def _timed(kind,fn,metadata):
    if not ENABLED:return fn()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();clock=time.perf_counter()
    result=fn()
    finish=time.perf_counter();cpu=(finish-clock)*1000;end.record()
    PENDING.append((dict(kind=kind,cpu_ms=cpu,cpu_begin_seconds=clock,cpu_end_seconds=finish,**metadata),start,end))
    return result


def snapshot(reset=False):
    global ENABLED,LAST_DRAFT,ORIGIN
    torch.cuda.synchronize()
    _collect_draft()
    for row,start,end in PENDING:
        ROWS.append(dict(row,gpu_stream_ms=start.elapsed_time(end),
                         gpu_begin_ms=ORIGIN.elapsed_time(start),gpu_end_ms=ORIGIN.elapsed_time(end)))
    PENDING.clear()
    from runtime.native import MODEL
    by_kind=defaultdict(list)
    for row in ROWS:by_kind[row['kind']].append(row)
    summary={}
    for kind,rows in by_kind.items():
        numeric={k for row in rows for k,v in row.items() if k.endswith('_ms') and isinstance(v,(float,int))}
        summary[kind]=dict(count=len(rows),**{k:sum(row.get(k,0.) for row in rows)/len(rows) for k in numeric})
    result=dict(diagnostic_only=True,not_throughput_evidence=True,summary=summary,rows=list(ROWS),
                fp64_gumbel=getattr(MODEL,'_short_latency_fp64',None))
    if reset:
        ROWS.clear();LAST_DRAFT=None;ENABLED=True
        ORIGIN=torch.cuda.Event(enable_timing=True);ORIGIN.record();ORIGIN.synchronize()
    else:ENABLED=False
    return result


def install():
    if os.environ.get('TAIL_LATENCY_DIAGNOSTIC')!='1':return
    from runtime import native,native_graph
    from . import native_launch
    original_forward=native.ExecutionDraft.forward
    original_execute=native.ExecutionDraft.execute
    original_head=native.run_eager
    def captured(self,name,n,fn):
        if not torch.cuda.is_current_stream_capturing():return fn()
        start=torch.cuda.Event(enable_timing=True,external=True);end=torch.cuda.Event(enable_timing=True,external=True)
        start.record();result=fn();end.record()
        if not hasattr(self,'_short_latency_spans'):self._short_latency_spans={}
        self._short_latency_spans.setdefault(n,{})[name]=(start,end)
        return result
    def forward(self,input_ids,*args,**kwargs):
        n=input_ids.numel()//16
        return captured(self,'body',n,lambda:original_forward(self,input_ids,*args,**kwargs))
    def execute(self,hidden,*args,**kwargs):
        self._short_latency_fp64=bool(kwargs.get('fp64',args[-1] if args else False))
        return captured(self,'projection_and_head',hidden.shape[0]//15,lambda:original_execute(self,hidden,*args,**kwargs))
    def head(*args,**kwargs):
        model=native.MODEL;n=args[3][0].shape[0]
        return captured(model,'repair_and_sampling' if args[0] is not None else 'raw_sampling',n,lambda:original_head(*args,**kwargs))
    native.ExecutionDraft.forward=forward;native.ExecutionDraft.execute=execute
    native.run_eager=head;native_launch.run_eager=head;native_graph.run_eager=head
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager,ModelCudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
    original_full=CudaGraphManager.run_fullgraph
    def full(self,desc,*args,**kwargs):
        global LAST_DRAFT
        is_draft=isinstance(self,DFlashCudaGraphManager)
        if ENABLED and is_draft:_collect_draft()
        result=_timed('draft_full_graph' if is_draft else 'target_full_graph',
              lambda:original_full(self,desc,*args,**kwargs),dict(n=desc.num_reqs,tokens=desc.num_tokens))
        if ENABLED and is_draft:
            model=native.MODEL;n=desc.num_reqs
            ready=torch.cuda.Event();ready.record()
            LAST_DRAFT=(n,model._short_latency_spans[n],ready)
        return result
    CudaGraphManager.run_fullgraph=full
    original_pw=ModelCudaGraphManager.run_pw_graph
    def piecewise(self,*args,**kwargs):
        return _timed('target_piecewise',lambda:original_pw(self,*args,**kwargs),{})
    ModelCudaGraphManager.run_pw_graph=piecewise
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    for name in ('execute_model','sample_tokens','sample','prepare_inputs','prepare_attn',
                 'update_requests','add_requests','postprocess_sampled','take_draft_token_ids'):
        original=getattr(GPUModelRunner,name)
        def wrapper(self,*args,_original=original,_name=name,**kwargs):
            return _timed('runner_'+_name,lambda:_original(self,*args,**kwargs),{})
        setattr(GPUModelRunner,name,wrapper)
    # This diagnostic is opt-in and never used as throughput evidence. Distinguish
    # the asynchronous proposal copy from its host-side wait and list handling.
    from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
    for name in ('set_draft_tokens','get_draft_tokens'):
        original=getattr(DraftTokensHandler,name)
        def transfer(self,*args,_original=original,_name=name,**kwargs):
            return _timed('draft_transfer_'+_name,lambda:_original(self,*args,**kwargs),{})
        setattr(DraftTokensHandler,name,transfer)
    from vllm.v1.worker.gpu_worker import Worker
    def collect(self,reset=False):return snapshot(reset)
    Worker.repair_short_latency=collect
    from fastapi import APIRouter,Request
    from vllm.entrypoints.openai import api_server
    previous_build=api_server.build_app;router=APIRouter()
    @router.get('/repair_short_latency')
    async def stats(request:Request,reset:bool=False):
        values=await request.app.state.engine_client.collective_rpc(method='repair_short_latency',kwargs={'reset':reset},timeout=30.)
        assert len(values)==1
        return values[0]
    def build(*args,**kwargs):
        app=previous_build(*args,**kwargs);app.include_router(router);return app
    api_server.build_app=build
