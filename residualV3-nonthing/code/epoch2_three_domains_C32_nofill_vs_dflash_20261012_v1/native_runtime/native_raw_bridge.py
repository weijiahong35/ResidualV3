"""Evaluation-only stock full-vocab baseline; no changes to frozen training code."""
import json
import os
from pathlib import Path
import torch


def install():
    from runtime import execution
    original_run = execution.run_eager
    execution.ARMS = (*execution.ARMS, 'raw_graph')
    execution.GRAPH_ARMS = (*execution.GRAPH_ARMS, 'raw_graph')

    def run(head,weight,basis,inputs,q,*,arm,fp64=False,capture=False):
        if arm != 'raw_graph':
            return original_run(head,weight,basis,inputs,q,arm=arm,fp64=fp64,capture=capture)
        # Exactly stock DraftModelSpeculator.sample_draft's one batched Gumbel call.
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
        raw,hidden,anchor,positions,idx,temp,seeds,step=inputs
        n,length,vocab=raw.shape
        ids=gumbel_sample(raw.reshape(n*length,vocab),idx,temp,seeds,positions+1,
            apply_temperature=True,output_processed_logits=q,
            output_processed_logits_col=step,use_fp64=fp64).reshape(n,length)
        previous=torch.cat((anchor[:,None],ids[:,:-1]),dim=1) if capture else None
        return ids,raw if capture else None,previous
    execution.run_eager=run
    from runtime import native
    original_execute=native.ExecutionDraft.execute
    def execute(self,hidden,positions,idx,temp,seeds,step,q,fp64):
        if self.arm != 'raw_graph':
            return original_execute(self,hidden,positions,idx,temp,seeds,step,q,fp64)
        if self.head is not None:raise RuntimeError('raw baseline unexpectedly has repair head')
        with torch.autocast('cuda',enabled=False):
            # Retain the original model logits processor, not repair's projection path.
            raw=self.compute_logits(hidden)
            length=int(os.environ['NATIVE_PROPOSAL_LENGTH']);block_size=length+1
            n=raw.shape[0]//length
            assert 1<=n<=int(os.environ['LOCAL_MAX_CONCURRENCY'])
            anchor=self.query[:n*block_size].reshape(n,block_size)[:,0].long()
            inp=(raw.reshape(n,length,-1),hidden.reshape(n,length,-1),anchor,positions,idx,temp,seeds,step)
            capture=os.environ['LOCAL_CONTRACT_CHECKS']=='1'
            result=run(None,self.lm_head.weight,None,inp,q,arm=self.arm,capture=capture,fp64=fp64)
            if torch.cuda.is_current_stream_capturing():
                self.integrated_inputs[n]=(inp,q,result,fp64)
                with (Path(os.environ['LOCAL_TRIAL'])/'graph_captures.jsonl').open('a') as f:
                    f.write(json.dumps(dict(batch_size=n,q_shape=list(q.shape),q_data_ptr=int(q.data_ptr())))+'\n')
            else:self.execution_counts.inc('eager')
        return result,anchor
    native.ExecutionDraft.execute=execute
    old_after=native.ExecutionDraft.after_share
    def after(self,target):
        old_after(self,target)
        path=Path(os.environ['LOCAL_TRIAL'])/'model_contract.json'
        value=json.loads(path.read_text());value['repair_enabled']=self.head is not None
        if self.arm=='raw_graph':
            value.update(baseline='original full-vocabulary DFlash; no repair; stock batched Gumbel sampler',
                         fixed_M_runtime_projection='none',graph_requested=True)
        path.write_text(json.dumps(value,indent=2)+'\n')
    native.ExecutionDraft.after_share=after
    native.install()

if __name__ in ('__main__','__mp_main__'):install()
if __name__=='__main__':
    from vllm.entrypoints.cli.main import main
    main()
