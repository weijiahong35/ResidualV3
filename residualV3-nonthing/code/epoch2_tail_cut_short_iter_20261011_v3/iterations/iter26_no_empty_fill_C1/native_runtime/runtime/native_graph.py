"""Host-side counts of actual vLLM target/draft Graph captures and replays."""
import json
import os
from pathlib import Path
import torch
from .execution import run_eager


def check_replay(model,n):
    inp,q,result,fp64=model.integrated_inputs[n]
    ids,scores,previous=result
    original_ids=ids.clone()
    reference_q=torch.empty_like(q)
    reference=run_eager(model.head,model.lm_head.weight,model.token_basis,inp,reference_q,
        arm=model.arm,fp64=fp64,capture=True)
    for actual,expected in zip(result,reference):
        if not torch.equal(actual,expected):raise RuntimeError('integrated Graph/eager parity failed')
    raw,hidden,anchor,positions,idx,temp,seeds,step=inp
    mask=torch.ones_like(idx,dtype=torch.bool)
    if model.head is not None and os.environ.get('TAIL_MODE','FULL15')!='FULL15':
        length=inp[0].shape[1]
        cuts=model.head._tail_last_stats[-1]
        mask=(torch.arange(length,device=idx.device)[None,:]<cuts[:,None]).reshape(-1)
    if not torch.equal(q[idx[mask].long(),step[mask].long()],reference_q[idx[mask].long(),step[mask].long()]):
        raise RuntimeError('integrated Graph processed q parity failed')
    if not torch.equal(previous[:,0],anchor) or not torch.equal(previous[:,1:],original_ids[:,:-1]):
        raise RuntimeError('integrated Graph actual lag-1 failed')
    active_t=temp[idx.long()]
    row=dict(batch_size=n,graph_eager_equal=True,processed_q_equal=True,feedback_equal=True,
        temperature=float(active_t.max()),non_top1=int((original_ids!=scores.argmax(-1)).sum()))
    with (Path(os.environ['LOCAL_TRIAL'])/'sampling_contract.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')


def install_graph_counters():
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
    old_capture=CudaGraphManager.capture
    old_replay=CudaGraphManager.run_fullgraph
    def counter(manager):
        from .native import MODEL
        if MODEL is None:raise RuntimeError('Graph capture before repair model initialization')
        return MODEL.execution_counts if isinstance(manager,DFlashCudaGraphManager) else MODEL.target_execution_counts
    def capture(self,*args,**kwargs):
        before=len(self.graphs)
        result=old_capture(self,*args,**kwargs)
        counts=counter(self)
        for _ in range(len(self.graphs)-before):counts.inc('captures')
        if isinstance(self,DFlashCudaGraphManager) and not self.graphs:
            raise RuntimeError('DFlash Graph unsupported; refusing silent eager testing')
        return result
    def replay(self,desc,*args,**kwargs):
        adaptive_tail = isinstance(self,DFlashCudaGraphManager) and os.environ.get('TAIL_MODE','FULL15') not in ('FULL15','BODY_ONLY')
        if adaptive_tail:
            from .native import MODEL
            from head_latency.tail_graph_state import bind_tail_state
            n=desc.num_reqs or desc.num_tokens//(1+int(os.environ['NATIVE_PROPOSAL_LENGTH']))
            exported=bind_tail_state(MODEL.head,n)
        result=old_replay(self,desc,*args,**kwargs)
        counter(self).inc('replays')
        if isinstance(self,DFlashCudaGraphManager) and os.environ['LOCAL_CONTRACT_CHECKS']=='1':
            from .native import MODEL
            tail_witness=tuple(v.clone() for v in exported) if adaptive_tail else None
            check_replay(MODEL,desc.num_reqs or desc.num_tokens//(1+int(os.environ['NATIVE_PROPOSAL_LENGTH'])))
            if adaptive_tail:
                eager_tail=MODEL.head._tail_last_stats
                for actual,expected in zip(tail_witness,eager_tail):
                    assert torch.equal(actual,expected), 'exported graph tail state differs from eager'
            n=desc.num_reqs or desc.num_tokens//(1+int(os.environ['NATIVE_PROPOSAL_LENGTH']))
            inp,q,state,fp64=MODEL.integrated_inputs[n]
            MODEL.diagnostic_latest=(tuple(x.detach().clone() for x in inp),q,tuple(x.detach().clone() if x is not None else None for x in state),fp64)
        if adaptive_tail:
            # Smoke eager references allocate another state. Restore this graph's
            # own output views before transferring prefix lengths to the scheduler.
            bind_tail_state(MODEL.head,n)
        return result
    CudaGraphManager.capture=capture
    CudaGraphManager.run_fullgraph=replay
