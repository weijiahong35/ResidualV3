"""Process-local native integration. Original stock sampler, explicit repair Graph."""
import hashlib,json,os
from pathlib import Path
import torch
from safetensors.torch import load_file
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from .precision import build_token_basis,POLICY
from .static_vocab import load_ids,make_static_basis,sha as static_sha
from .execution import run_eager,RepairGraph,GRAPH_ARMS,ARMS
HERE=Path(__file__).resolve().parent
MODEL=None
from .observe import Observer,scope,marker
from .counters import ExecutionCounters

class ExecutionDraft(DFlashQwen3ForCausalLM):
    def load_weights(self, weights):
        raise RuntimeError('Install repair.native_launch before model construction')
    def after_share(self,target):
        global MODEL
        MODEL=self;self.observer=Observer()
        self.raw_events=[torch.cuda.Event(enable_timing=True) for _ in range(2)] if os.environ.get("DIAG_KIND")=="profile" else None
        target=target.get_language_model() if hasattr(target,'get_language_model') else target
        assert self.lm_head is target.lm_head and self.draft_id_to_target_id is None
        self.token_basis=build_token_basis(self.lm_head.weight,self.repair_basis) if self.head is not None else None
        m_path=Path(os.environ['STATIC_M_PATH'])
        ids=load_ids(m_path,os.environ['STATIC_M_SHA256'],self.lm_head.weight.shape[0],
                     int(os.environ['STATIC_M_SIZE'])).to(self.lm_head.weight.device)
        self.register_buffer('static_ids',ids,persistent=False)
        membership=torch.zeros(self.lm_head.weight.shape[0],device=ids.device,dtype=torch.bool)
        membership[ids]=True
        self.register_buffer('static_membership_mask',membership,persistent=False)
        if self.arm=='static_dense':
            self.register_buffer('static_basis',make_static_basis(self.token_basis,ids),persistent=False)
            self.head.static_ids=self.static_ids
            self.head.static_basis=self.static_basis
            # Full C is used once to construct B_M; runtime retains only B_M.
            self.token_basis=None
        self.execution_counts=ExecutionCounters(Path(os.environ['LOCAL_TRIAL'])/'execution_counters.bin')
        self.target_execution_counts=ExecutionCounters(Path(os.environ['LOCAL_TRIAL'])/'target_execution_counters.bin') if os.environ.get('NATIVE_FULL_GRAPH')=='1' else None
        self.integrated_inputs={}
        record=dict(arm=self.arm,precision_policy=POLICY,graph_requested=self.arm in GRAPH_ARMS and os.environ.get('DIAG_KIND')!='joint_collect',
            graph_scope=('target decode Graph; draft query + raw LM + repair + seven actual samples in one Graph; context KV preparation remains eager' if os.environ.get('NATIVE_FULL_GRAPH')=='1' else 'repair only'),
            first_previous='verified anchor LM-head',prefix='actual proposal LM-head rows, anchor excluded',
            topk_tie_policy='preserve original Top16 features; no TopK repair candidate search',
            graph_batch_policy=('one native draft query/repair Graph per configured batch size; count actual captures/replays' if os.environ.get('NATIVE_FULL_GRAPH')=='1' else 'separate repair Graph per observed batch size'),
            fallback_policy='fail on incompatible shape/q storage; no silent eager fallback',
            fixed_M_size=int(ids.numel()),fixed_M_sha256=static_sha(m_path),
            fixed_M_runtime_projection='only B_M rows' if self.arm=='static_dense' else 'full token basis')
        (Path(os.environ['LOCAL_TRIAL'])/'model_contract.json').write_text(json.dumps(record,indent=2))
    def forward(self,input_ids,positions,inputs_embeds=None):
        self.query=input_ids
        if os.environ.get("DIAG_KIND"):self.observer.begin()
        with scope("drafter_body"):return super().forward(input_ids,positions,inputs_embeds)
    def execute(self,hidden,positions,idx,temp,seeds,step,q,fp64):
        with torch.autocast('cuda',enabled=False):
            if self.raw_events:self.raw_events[0].record()
            with scope('raw_lm_projection'):raw=torch.mm(hidden,self.lm_head.weight.T)
            if self.raw_events:self.raw_events[1].record()
            n=raw.shape[0]//7
            limit=int(os.environ.get('LOCAL_MAX_CONCURRENCY','20'))
            assert 1<=n<=limit,f'batch size {n} outside 1..{limit}'
            anchor=self.query[:n*8].reshape(n,8)[:,0].long()
            inp=(raw.reshape(n,7,-1),hidden.reshape(n,7,-1),anchor,positions,idx,temp,seeds,step)
            capture=os.environ['LOCAL_CONTRACT_CHECKS']=='1'
            if os.environ.get('NATIVE_FULL_GRAPH')=='1':
                # Sampling is inside vLLM's outer DFlash graph. Never nest RepairGraph.
                result=run_eager(self.head,self.lm_head.weight,self.token_basis,inp,q,arm=self.arm,capture=capture,fp64=fp64)
                if torch.cuda.is_current_stream_capturing():
                    self.integrated_inputs[n]=(inp,q,result,fp64)
                    event=dict(batch_size=n,q_shape=list(q.shape),q_data_ptr=int(q.data_ptr()))
                    with (Path(os.environ['LOCAL_TRIAL'])/'graph_captures.jsonl').open('a') as stream:
                        stream.write(json.dumps(event)+'\n')
                else:
                    self.execution_counts.inc('eager')
            elif self.arm in GRAPH_ARMS and os.environ.get('DIAG_KIND') != 'joint_collect':
                key=(n,tuple(q.shape),q.data_ptr())
                graph=self.repair_graphs.get(key)
                if graph is None:
                    if len(self.repair_graphs)>=limit:raise RuntimeError('more repair Graph signatures than permitted batch sizes')
                    graph=RepairGraph(self.head,self.lm_head.weight,self.token_basis,arm=self.arm,capture=capture,fp64=fp64)
                    self.repair_graphs[key]=graph
                self.repair_graph=graph
                before=graph.captures
                result=graph(inp,q)
                if graph.captures!=before:
                    self.execution_counts.inc('captures')
                    event=dict(batch_size=n,q_shape=list(q.shape),q_data_ptr=int(q.data_ptr()),
                               graph_count=len(self.repair_graphs))
                    with (Path(os.environ['LOCAL_TRIAL'])/'graph_captures.jsonl').open('a') as stream:
                        stream.write(json.dumps(event)+'\n')
                self.execution_counts.inc('replays')
                if capture:
                    comparison_q=torch.empty_like(q)
                    reference=run_eager(self.head,self.lm_head.weight,self.token_basis,inp,comparison_q,arm=self.arm,capture=True,fp64=fp64)
                    assert all(torch.equal(x,y) for x,y in zip(result,reference)),'Graph != same-variant eager'
                    active_idx=idx.long();active_step=step.long()
                    assert torch.equal(q[active_idx,active_step],comparison_q[active_idx,active_step]),'Graph active q != eager'
            else:
                self.execution_counts.inc('eager')
                result=run_eager(self.head,self.lm_head.weight,self.token_basis,inp,q,arm=self.arm,capture=capture,fp64=fp64)
        self.copy_events=self.repair_graph.replay_events if self.repair_graph else None
        if os.environ.get("DIAG_KIND")=="profile":
            self.last_proposal_ids=result[0]
            self.observer.after(self)
        return result,anchor

def install():
    from vllm.model_executor.models import ModelRegistry
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    original=dflash.load_dflash_model
    def load(target,config):
        model=original(target,config);model.after_share(target);return model
    dflash.load_dflash_model=load;ModelRegistry.register_model('DFlashDraftModel',ExecutionDraft)
    def sample(self,hidden_states,positions,idx_mapping,temperature,seeds,draft_step,draft_logits):
        assert draft_logits is not None
        (ids,scores,previous),anchor=self.model.execute(hidden_states,positions,idx_mapping,temperature,seeds,draft_step,draft_logits,self.use_fp64_gumbel)
        if os.environ['LOCAL_CONTRACT_CHECKS']=='1' and os.environ.get('NATIVE_FULL_GRAPH')!='1':
            flat=scores.reshape(-1,scores.shape[-1]);q=torch.empty_like(draft_logits)
            expected=gumbel_sample(flat,idx_mapping,temperature,seeds,positions+1,apply_temperature=True,
                output_processed_logits=q,output_processed_logits_col=draft_step,use_fp64=self.use_fp64_gumbel)
            assert torch.equal(ids.reshape(-1),expected)
            active=idx_mapping.unique().long();assert torch.equal(q[active],draft_logits[active])
            assert torch.equal(previous[:,0],anchor) and torch.equal(previous[:,1:],ids[:,:-1])
            row=dict(step_vs_batch_equal=True,processed_q_equal=True,feedback_equal=True,
                graph_eager_equal=self.model.arm in GRAPH_ARMS,batch_size=ids.shape[0],
                non_top1=int((ids!=scores.argmax(-1)).sum()),
                temperature=float(temperature[idx_mapping[0]].item()),bf16_scores=scores.dtype==torch.bfloat16)
            with (Path(os.environ['LOCAL_TRIAL'])/'sampling_contract.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return ids.reshape(-1)
    DraftModelSpeculator.sample_draft=sample
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    old_execute=GPUModelRunner.execute_model
    def execute(self,*args,**kwargs):
        with scope('target_execute_including_prepare'):return old_execute(self,*args,**kwargs)
    GPUModelRunner.execute_model=execute
    old_sample=GPUModelRunner.sample
    def verify(self,*args,**kwargs):
        with scope('target_sample_verify'):return old_sample(self,*args,**kwargs)
    GPUModelRunner.sample=verify
    if os.environ.get('NATIVE_FULL_GRAPH')=='1':
        from .native_graph import install_graph_counters
        install_graph_counters()
