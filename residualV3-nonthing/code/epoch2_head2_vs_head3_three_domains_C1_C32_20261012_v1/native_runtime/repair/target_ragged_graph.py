"""Experimental full graphs for <=16-token speculative verification prefixes.

FA2 has a special packed-GQA path for max_query_len==1. Extra captures always
have max_query_len>=2; dispatch is restricted to speculative decode with a
maximum query length in[2,16]. Real-state hidden/aux parity is mandatory in
the diagnostic service gate. Installed vLLM files are never changed.
"""
import os,json,sys
from pathlib import Path
import torch


def install():
    if os.environ.get('TAIL_RAGGED_FULL_GRAPH','0')!='1':return
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager,BatchExecutionDescriptor
    from . import target_prefix_bridge as bridge
    previous_init=ModelCudaGraphManager._init_candidates
    def init(self):
        previous_init(self)
        self._repair_ragged_full=[]
        spec=self.vllm_config.speculative_config
        if not spec or spec.method!='dflash' or self.decode_query_len!=16:return
        # Request padding is an independent cost from token padding. The
        # previous token-only table could pad a one-request prefix to five
        # requests, and almost every C40 batch to forty requests. Use a small
        # two-dimensional table and preserve the original uniform graphs.
        req_bins=sorted(set(range(1,min(10,self.max_num_reqs)+1)) |
                        {min(self.max_num_reqs,r) for r in (12,16,20,24,28,32,36,40)})
        sizes=sorted(self.compilation_config.cudagraph_capture_sizes or [])
        for reqs in req_bins:
            for query in range(11,17):
                # Real-state PW reference must also have a startup-captured
                # token bucket. Arbitrary new token counts otherwise trigger
                # a lazy PW capture on the default stream in diagnostics.
                size=next(s for s in sizes if s>=reqs*query)
                desc=BatchExecutionDescriptor(CUDAGraphMode.FULL,size,reqs,None,0)
                if desc not in self._capture_descs.setdefault(CUDAGraphMode.FULL,[]):
                    self._capture_descs[CUDAGraphMode.FULL].append(desc)
                if desc not in self._repair_ragged_full:self._repair_ragged_full.append(desc)
        self._repair_ragged_full.sort(key=lambda d:(d.num_reqs,d.num_tokens))
        self._capture_descs[CUDAGraphMode.FULL].sort(key=lambda d:d.num_tokens,reverse=True)
    ModelCudaGraphManager._init_candidates=init
    previous_dispatch=ModelCudaGraphManager.dispatch
    def dispatch(self,num_reqs,num_tokens,uniform_token_count,num_active_loras):
        ordinary=previous_dispatch(self,num_reqs,num_tokens,uniform_token_count,num_active_loras)
        if not self._graphs_captured or not bridge.IN_DECODE or num_active_loras or not 2<=bridge.DECODE_MAX_QUERY<=16:
            return ordinary
        if ordinary.cg_mode==CUDAGraphMode.FULL:return ordinary
        compatible=[d for d in self._repair_ragged_full if d.num_tokens>=num_tokens and d.num_reqs>=num_reqs]
        if compatible:
            return min(compatible,key=lambda d:(d.num_tokens-num_tokens+16*(d.num_reqs-num_reqs),d.num_reqs,d.num_tokens))
        return ordinary
    ModelCudaGraphManager.dispatch=dispatch
    previous_run=ModelCudaGraphManager.run_fullgraph
    def run(self,desc,*args,**kwargs):
        result=previous_run(self,desc,*args,**kwargs)
        if not bridge.IN_DECODE or desc.uniform_token_count is not None or os.environ.get('LOCAL_CONTRACT_CHECKS')!='1':return result
        # The runner has already filled this full graph's static input/attention
        # buffers. Compare the existing PW path using those EXACT buffers,
        # including KV writes at the same positions; inspect only valid rows.
        frame=sys._getframe(1)
        assert frame.f_code.co_name=='execute_model'
        values=frame.f_locals;runner=values['self'];batch=values['input_batch'];valid=batch.num_tokens
        actual=[x[:valid].clone() for x in [result[0],*result[1]]] if isinstance(result,tuple) else [result[:valid].clone()]
        from vllm.forward_context import set_forward_context,BatchDescriptor
        with set_forward_context(values['attn_metadata'],self.vllm_config,num_tokens=desc.num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                num_tokens_across_dp=values['num_tokens_across_dp'],
                batch_descriptor=BatchDescriptor(num_tokens=desc.num_tokens,has_lora=False,num_active_loras=0),
                slot_mapping=values['slot_mappings_by_layer'],is_padding=batch.is_padding):
            reference=self.run_pw_graph(runner.model,values['model_inputs'])
        expected=[reference[0][:valid],*[x[:valid] for x in reference[1]]] if isinstance(reference,tuple) else [reference[:valid]]
        checks=[torch.equal(a,b) for a,b in zip(actual,expected)]
        errors=[float((a.float()-b.float()).abs().max()) for a,b in zip(actual,expected)]
        row=dict(actual_rows=valid,graph_rows=desc.num_tokens,padded_requests=desc.num_reqs,
                 max_query_len=bridge.DECODE_MAX_QUERY,hidden_and_aux_bitwise=checks,max_abs_errors=errors)
        with (Path(os.environ['LOCAL_TRIAL'])/'target_ragged_graph_parity.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        assert len(checks)==len(expected) and all(checks),row
        return result
    ModelCudaGraphManager.run_fullgraph=run
