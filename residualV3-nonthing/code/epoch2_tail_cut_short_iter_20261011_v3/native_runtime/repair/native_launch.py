import os,json
from pathlib import Path
import torch
from .model import DualFeedbackRepair,rollout

def run_eager(head,weight,basis,inputs,q,*,arm,fp64=False,capture=False):
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    raw,hidden,anchor,positions,idx,temp,seeds,step=inputs;n,length,v=raw.shape
    if arm=='raw_graph':
        ids=gumbel_sample(raw.reshape(-1,v),idx,temp,seeds,positions+1,apply_temperature=True,output_processed_logits=q,output_processed_logits_col=step,use_fp64=fp64).reshape(n,length)
        return ids,raw if capture else None,torch.cat([anchor[:,None],ids[:,:-1]],1) if capture else None
    assert arm=='dense_combined'
    mappings=idx.reshape(n,length).T.contiguous();pos=(positions.reshape(n,length).T+1).contiguous();cols=step.reshape(n,length).T.contiguous()
    from head_latency.structure_fusion import sample as fused_sample,FUSED_MODES
    sampler=fused_sample if getattr(head,'_fusion_mode','baseline') in FUSED_MODES else gumbel_sample
    def choose(z,j,active=None,entropy_max=None):
        if entropy_max is not None:
            from head_latency.tail_sampler_max import sample as sampled_max
            return sampled_max(z,mappings[j],temp,seeds,pos[j],raw_maximum=entropy_max,active=active,override=head._tail_sampling_override,
                apply_temperature=True,output_processed_logits=q,output_processed_logits_col=cols[j],use_fp64=fp64)
        if active is not None:
            from head_latency.tail_fast import sample as gated_sample
            return gated_sample(z,mappings[j],temp,seeds,pos[j],active=active,override=head._tail_sampling_override,apply_temperature=True,output_processed_logits=q,output_processed_logits_col=cols[j],use_fp64=fp64)
        return sampler(z,mappings[j],temp,seeds,pos[j],apply_temperature=True,output_processed_logits=q,output_processed_logits_col=cols[j],use_fp64=fp64)
    mode=os.environ.get('TAIL_MODE','FULL15')
    if mode!='FULL15':
        from head_latency.tail_pipeline import rollout as tail_rollout
        head._tail_temperatures=temp;head._tail_mapping=mappings[0]
        result=tail_rollout(head,raw,hidden,weight,basis,anchor,choose,capture=capture,mode=mode)
    else:
        result=rollout(head,raw,hidden,weight,basis,anchor,choose,capture=capture)
    return result.tokens,result.logits,result.predecessors

def install():
    global rollout
    from head_latency.optimized import install as install_head_optimization
    implementation = install_head_optimization()
    if implementation is not None:
        rollout = implementation
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from runtime import native_graph
    original_check=native_graph.check_replay
    def independent_check(model,n):
        original_check(model,n)
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
        inputs,processed,result,fp64=model.integrated_inputs[n]
        ids,scores,previous=result
        raw,hidden,anchor,positions,idx,temp,seeds,step=inputs
        reference_q=torch.empty_like(processed)
        expected=gumbel_sample(scores.reshape(-1,scores.shape[-1]),idx,temp,seeds,positions+1,
            apply_temperature=True,output_processed_logits=reference_q,
            output_processed_logits_col=step,use_fp64=fp64)
        kept=torch.ones_like(idx,dtype=torch.bool)
        if model.head is not None and os.environ.get('TAIL_MODE','FULL15')!='FULL15':
            length=raw.shape[1];cuts=model.head._tail_last_stats[-1]
            kept=(torch.arange(length,device=idx.device)[None,:]<cuts[:,None]).reshape(-1)
        assert torch.equal(ids.reshape(-1)[kept],expected[kept]),'stock sampler differs on a submitted prefix'
        rows=idx[kept].long();cols=step[kept].long()
        assert torch.equal(processed[rows,cols],reference_q[rows,cols])
        temperature=temp[rows]
        divisor=torch.where(temperature>0,temperature,torch.ones_like(temperature))
        expected_logits=scores.reshape(-1,scores.shape[-1])[kept].float()/divisor[:,None]
        actual_logits=processed[rows,cols].float()
        t01=bool(((temperature==0)|(temperature==1)).all())
        if t01:
            assert torch.equal(actual_logits,expected_logits),'T0/T1 verifier sees incorrect proposal scores'
        else:
            # vLLM startup dummy requests use T=.9. Triton reciprocal division
            # can differ from torch.div by an FP32 ulp. Stock-sampler parity
            # above remains EXACT for all temperatures.
            torch.testing.assert_close(actual_logits,expected_logits,rtol=4*torch.finfo(torch.float32).eps,atol=1e-6)
        processed_reference_max_abs=float((actual_logits-expected_logits).abs().max())
        if model.head is not None and not getattr(model,'four_input_checked',False):
            from .validate import input_effects
            with torch.no_grad():
                evidence=input_effects(model.head,raw[:1,1],hidden[:1,1],model.head.feature_weight[previous[:1,1]],model.token_basis,scores[:1,0],model.head.q_feature_ids(raw[:1,0]))
            (Path(os.environ['LOCAL_TRIAL'])/'real_input_effects.json').write_text(json.dumps(evidence,indent=2)+'\n')
            model.four_input_checked=True
        with (Path(os.environ['LOCAL_TRIAL'])/'processed_proposal_contract.jsonl').open('a') as f:
            f.write(json.dumps(dict(batch_sampler_equal=True,processed_logits_are_corrected_scores=True,
                                   proposal_support='full vocabulary',temperature=float(temperature.max()),
                                   t01_reference_exact=t01,torch_reference_max_abs=processed_reference_max_abs))+'\n')
        from head_latency.tail_reference import verify_real_block
        verify_real_block(model,n)
    # Load the stock full-vocabulary RAW/native Graph bridge directly. The old
    # optimized patch assumes historical input dimensions; use the shared new
    # rollout for both native and training instead.
    import runpy
    torch.use_deterministic_algorithms(True)
    root=Path(__file__).resolve().parents[1]
    runpy.run_path(str(root/'native_raw_bridge.py'),run_name='__mp_main__')
    from runtime import native
    from .tail_stats import install as install_tail_stats
    install_tail_stats()
    from runtime.utils import sha
    config=Path(os.environ['HIDDEN_HEAD_CONFIG']);assert sha(config)==os.environ['HIDDEN_HEAD_CONFIG_SHA256']
    cfg=json.loads(config.read_text())
    # Replace only this process's head loading; frozen train-only basis, fresh repair checkpoint.
    from safetensors.torch import load_file
    def load_weights(self,weights):
        result=super(native.ExecutionDraft,self).load_weights(weights)
        self.head=None;self.arm=os.environ['LOCAL_ARM']
        self.repair_graph=None;self.repair_graphs={};self.online_weight_version=0
        if self.arm not in ('raw','raw_graph'):
            self.head=DualFeedbackRepair(**cfg).to(device=self.lm_head.weight.device,dtype=torch.bfloat16).eval()
            state=load_file(os.environ['REPAIR_CHECKPOINT'])
            self.head.load_state_dict({(k[5:] if k.startswith('head.') else k):v for k,v in state.items()},strict=True)
            self.head.requires_grad_(False)
        if self.head is not None:
            basis_path=Path(os.environ['D1_BASIS_PATH'])
            assert sha(basis_path)==os.environ['D1_BASIS_SHA256']
            self.register_buffer('repair_basis',load_file(str(basis_path))['repair_basis'].to(device=self.lm_head.weight.device,dtype=torch.bfloat16))
        return result
    native.ExecutionDraft.load_weights=load_weights
    native.run_eager=run_eager;native_graph.run_eager=run_eager;native_graph.check_replay=independent_check
    prior=native.ExecutionDraft.after_share
    def after(self,target):
        prior(self,target)
        signature=None
        if self.head is not None:
            from runtime.precision import build_token_basis
            from .run import feature_signature
            base=target.get_language_model() if hasattr(target,'get_language_model') else target
            source=os.environ['TOKEN_FEATURE_SOURCE']
            assert source in ('LMHEAD','EMBEDDING')
            weight=base.lm_head.weight if source=='LMHEAD' else base.model.embed_tokens.weight
            weight=weight[:cfg['vocab_size']].detach()
            basis=self.token_basis if source=='LMHEAD' else build_token_basis(weight,self.repair_basis)
            self.head.bind_features(weight,basis)
            from head_latency.trained_tail import prepare as prepare_tail
            predictor=json.loads(Path(os.environ['TAIL_PREDICTOR']).read_text())
            branch=os.environ.get('TAIL_AUX_BRANCH','Q45_MLP')
            prepare_tail(self.head,predictor['policies'].get(branch,predictor['calibration_policy']))
            signature=feature_signature(weight,basis)
        if torch.cuda.device_count()!=1 or torch.cuda.current_device()!=0:
            raise RuntimeError('one visible CUDA device required')
        actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
        if actual!=os.environ['REPAIR_GPU_UUID']:raise RuntimeError('GPU UUID mismatch')
        path=Path(os.environ['LOCAL_TRIAL'])/'model_contract.json';data=json.loads(path.read_text())
        data.update(architecture='embedding_ablation_v1' if self.head is not None else 'RAW',repair_initialization='same random initialization followed by paired joint short training',
            input_feature_source=os.environ['TOKEN_FEATURE_SOURCE'],input_feature_signature=signature,
            structure_mode=getattr(self.head,'_fusion_mode','baseline') if self.head is not None else 'none',checkpoint_sha256=sha(os.environ['REPAIR_CHECKPOINT']),head_config_sha256=sha(config),
            q_conditioning='original q -> frozen EMBEDDING basis -> distribution projection; optional corrected soft lag1 T=1',blocks=cfg['depth'],branches=list(self.head.branch_names) if self.head is not None else [],scale_mode=cfg['scale_mode'],fixed_scale=cfg['fixed_scale'],
            prefix='actual hard lag1; optional repaired soft lag1; no older prefix aggregate',dose='0.8*sigmoid(Linear256,1(latent))',soft_feature_temperature=1,position='backbone only; no repair position',
            topk_tie_policy='torch.topk on raw logits selects shared q input/correction IDs; full vocabulary sampling',
            correction_support=cfg['q_topk'],sampling_support='full original vocabulary',basis='fixed initial pretrained mismatch basis rank512',basis_sha256=os.environ['D1_BASIS_SHA256'],fixed_M_runtime_projection='unused compatibility metadata',
            sequential_work='hard lag1, optional soft lag1, fusion, configured SwiGLU depth, direction/dose, Top256 correction and full stock sampler',
            native_parity='same mathematical recurrence, eager versus CUDA Graph; stock sampler q/IDs/feedback checked',
            parameter_count=sum(v.numel() for v in self.head.parameters()) if self.head is not None else 0)
        if self.head is None:
            data.update(q_conditioning='none; original RAW DFlash',blocks=0,correction_support=0,basis='unused',repair_initialization='not applicable',dose='none',soft_feature_temperature=None,prefix='RAW parallel proposals')
        data.update(tail_mode=os.environ.get('TAIL_MODE','FULL15'), tail_predictor_sha256=sha(os.environ['TAIL_PREDICTOR']),tail_delay_positions=1,tail_mma_skip='device branch per original16-row tile; launch slots and small norm operations remain')
        data.update(auxiliary_checkpoint_sha256=sha(os.environ['TAIL_AUX_CHECKPOINT']),auxiliary_branch=os.environ.get('TAIL_AUX_BRANCH','Q45_MLP'),
                    confidence_input='repaired unit-temperature q45 plus optional repair output_norm representation',auxiliary_training_changed=False)
        path.write_text(json.dumps(data,indent=2)+'\n')
    native.ExecutionDraft.after_share=after
if __name__ in ('__main__','__mp_main__'):install()
if __name__=='__main__':
    from vllm.entrypoints.cli.main import main
    main()
