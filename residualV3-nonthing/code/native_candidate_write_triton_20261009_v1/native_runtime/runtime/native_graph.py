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
    if not torch.equal(q[idx.long(),step.long()],reference_q[idx.long(),step.long()]):
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
        result=old_replay(self,desc,*args,**kwargs)
        counter(self).inc('replays')
        if isinstance(self,DFlashCudaGraphManager) and os.environ['LOCAL_CONTRACT_CHECKS']=='1':
            from .native import MODEL
            check_replay(MODEL,desc.num_reqs or desc.num_tokens//(1+int(os.environ['NATIVE_PROPOSAL_LENGTH'])))
        return result
    CudaGraphManager.capture=capture
    CudaGraphManager.run_fullgraph=replay
