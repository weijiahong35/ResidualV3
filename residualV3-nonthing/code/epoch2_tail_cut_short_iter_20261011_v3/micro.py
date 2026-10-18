"""Frozen trained head: exact recurrence, one-step delay, real skipped MMA and Graph timing."""
import argparse,json,os,sys
from pathlib import Path
C=Path(__file__).resolve().parent;sys.path.insert(0,str(C/'native_runtime'))
# Match native_evaluate's fixed-order kernels even when launched directly.
# init_batch_invariance is otherwise a silent no-op without this flag.
os.environ.setdefault('VLLM_BATCH_INVARIANT','1')
assert os.environ['VLLM_BATCH_INVARIANT']=='1'
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from runtime.utils import dump,sha,now
from runtime.precision import build_token_basis
from repair.model import DualFeedbackRepair
from head_latency.optimized import prepare_variant
from head_latency.structure_fusion import rollout as full,sample
from head_latency.trained_tail import prepare
import head_latency.trained_tail as trained_tail
from head_latency.tail_pipeline import rollout
from head_latency.tail_reference import reference
from head_latency.small_tile import set_mode
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);a=ap.parse_args();root=a.root
    p=json.loads((root/'preregistration.json').read_text());predictor=json.loads((root/'calibration/initial_policy.json').read_text())
    os.environ['TAIL_AUX_CHECKPOINT']=p['auxiliary_checkpoint'];os.environ['TAIL_AUX_BRANCH']='Q45_MLP'
    for f,digest in p['source_files'].items():assert sha(f)==digest,f
    actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-');assert actual==p['gpu_uuid']
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance
    init_batch_invariance();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    checks=[];rows=[]
    with torch.inference_mode():
      ck=p['checkpoint'];cfg=json.loads(Path(ck['head_config_path']).read_text())
      head=DualFeedbackRepair(**cfg).to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
      assert sha(ck['path'])==ck['sha256'];head.load_state_dict({k.removeprefix('head.'):v for k,v in load_file(ck['path']).items()})
      target=Path(p['target_model']);index=json.loads((target/'model.safetensors.index.json').read_text())['weight_map'];values=[]
      for key in ('model.embed_tokens.weight','lm_head.weight'):
        with safe_open(target/index[key],framework='pt',device='cpu') as f:values.append(f.get_tensor(key).to('cuda',dtype=torch.bfloat16))
      embedding,weight=values;basis=load_file(p['basis_path'])['repair_basis'].to('cuda',dtype=torch.bfloat16)
      head.bind_features(embedding,build_token_basis(embedding,basis));outbasis=build_token_basis(weight,basis)
      os.environ['LOCAL_MAX_CONCURRENCY']='0';os.environ['HEAD_SMALL_TILE_MODE']='all16_64';os.environ['HEAD_STRUCTURE_MODE']='lean_combined'
      prepare_variant(head,'small_tile')
      from head_latency.predecessor import prepare_lookup
      prepare_lookup(head,exact=True,batch_sizes=(18,))
      prepare(head,predictor['calibration_policy'])
      # An independently loaded original readout evaluates the same repaired
      # logits and latents. Check features/scores before relying on stop masks.
      import importlib.util
      old_path=Path(p['parent_experiment']).parents[1]/'code/epoch2_trained_aux_tail_cut_20261010_v1/native_runtime/head_latency/trained_tail.py'
      spec=importlib.util.spec_from_file_location('head_latency.original_trained_tail',old_path)
      old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
      fast=trained_tail.predict;readout_checks=[]
      def checked(head,z,latent,active=None,**kw):
        result=fast(head,z,latent,active,**kw)
        if not torch.cuda.is_current_stream_capturing():
          gold=old.predict(head,z,latent);mask=active.bool() if active is not None else torch.ones(z.shape[0],device=z.device,dtype=torch.bool)
          assert torch.equal(result[0][mask],gold[0][mask]) and torch.equal(result[1][mask],gold[1][mask]),'frozen readout or q45 changed on a kept row'
          readout_checks.append(dict(n=z.shape[0],kept_rows=int(mask.sum()),scores_exact=True,maxq_exact=True))
        return result
      trained_tail.predict=checked
      for n in (1,7,18,40):
       for t in (0,1):
        torch.manual_seed(42+n+t)
        raw=(torch.randn(n,15,head.vocab_size,device='cuda')*3).to(torch.bfloat16)
        hidden=torch.randn(n,15,head.hidden_size,device='cuda',dtype=torch.bfloat16)
        anchor=torch.randint(head.vocab_size,(n,),device='cuda');temp=torch.full((n,),float(t),device='cuda')
        seeds=torch.arange(n,device='cuda',dtype=torch.int64)+42;maps=torch.arange(n,device='cuda',dtype=torch.int32)
        pos=torch.arange(15,device='cuda',dtype=torch.int64)[None,:].expand(n,-1).contiguous()+123
        q=torch.empty((n,15,head.vocab_size),device='cuda',dtype=torch.float32);refq=torch.empty_like(q)
        head._tail_temperatures=temp;head._tail_mapping=maps
        def choose(z,j,active=None):
          args=dict(apply_temperature=True,output_processed_logits=q,output_processed_logits_col=torch.full((n,),j,device='cuda',dtype=torch.int32),use_fp64=False)
          if active is not None:
            from head_latency.tail_fast import sample as gated_sample
            return gated_sample(z,maps,temp,seeds,pos[:,j].contiguous(),active=active,override=head._tail_sampling_override,**args)
          return sample(z,maps,temp,seeds,pos[:,j].contiguous(),**args)
        def gold(z,j):return gumbel_sample(z,maps,temp,seeds,pos[:,j].contiguous(),apply_temperature=True,output_processed_logits=refq,output_processed_logits_col=torch.full((n,),j,device='cuda',dtype=torch.int32),use_fp64=False)
        for cutoff in (15,12,10,0,-1):
         head._tail_thresholds.fill_(.9 if cutoff==-1 else 0.)
         os.environ['TAIL_FORCE_CUTOFF']=str(max(cutoff,0));os.environ['TAIL_INSTRUMENT']='1'
         seq=rollout(head,raw,hidden,weight,outbasis,anchor,choose,mode='SEQUENTIAL')
         seqaux=tuple(v.clone() for v in head._tail_last_stats);seq_q=q.clone()
         pipe=rollout(head,raw,hidden,weight,outbasis,anchor,choose,mode='PIPELINE');torch.cuda.synchronize()
         pipeaux=tuple(v.clone() for v in head._tail_last_stats)
         assert torch.equal(seq.tokens,pipe.tokens) and torch.equal(seq.logits,pipe.logits) and torch.equal(seq_q,q),'stream outputs differ'
         assert all(torch.equal(x,y) for x,y in zip(seqaux,pipeaux)),'stream decisions differ'
         cuts=pipeaux[-1]
         set_mode(head,'baseline');head._fusion_mode='baseline'
         expected=reference(head,raw,hidden,weight,outbasis,anchor,gold,cuts)
         set_mode(head,'all16_64');head._fusion_mode='lean_combined'
         kept=torch.arange(15,device='cuda')[None,:]<cuts[:,None]
         valid=dict(logits=torch.equal(pipe.logits,expected.logits),tokens=torch.equal(pipe.tokens[kept],expected.tokens[kept]),previous=torch.equal(pipe.predecessors[kept],expected.predecessors[kept]),actual_verifier_q=torch.equal(q[kept],refq[kept]),streams_exact=True)
         if not all(valid.values()):
            mismatch=pipe.logits!=expected.logits
            print('PARITY_DETAIL',dict(n=n,T=t,cutoff=cutoff,cuts=cuts.tolist(),different_logits=int(mismatch.sum()),logit_max_abs=float((pipe.logits.float()-expected.logits.float()).abs().max()),positions=mismatch.any(-1).nonzero().tolist(),q_max_abs=float((q[kept]-refq[kept]).abs().max())),flush=True)
         assert all(valid.values()),(n,t,cutoff,valid)
         mask=torch.arange(15,device='cuda')[None,:]>=cuts[:,None]
         assert torch.equal(pipe.logits[mask],raw[mask]),'raw suffix modified'
         maxq=(pipe.logits.float().amax(-1)-pipe.logits.float().logsumexp(-1)).exp()
         alive=pipeaux[0][:,:15].bool()
         torch.testing.assert_close(pipeaux[2][alive],maxq[alive],rtol=2e-5,atol=2e-6)
         if cutoff>0:assert bool((cuts==cutoff).all()),('one-position cutoff error',cutoff,cuts)
         if cutoff==0:assert bool((cuts==15).all()),'disabled learned gate must preserve full15'
         if cutoff==-1:
          from repair.trained_capture import cutoff as cpu_cutoff
          expected_cuts=[cpu_cutoff(v,.9) for v in pipeaux[-2][:,:,t].cpu().tolist()]
          assert cuts.tolist()==expected_cuts,'GPU prefix decision differs from independent CPU causal policy'
         check=dict(n=n,T=t,forced_cutoff=cutoff,cuts=cuts.tolist(),math_tile_counts=pipeaux[3].tolist(),checks=valid)
         checks.append(check);print(now(),'CHECK',check,flush=True)
         if n in (1,40):
          os.environ['TAIL_INSTRUMENT']='0'
          for mode in ('FULL15','SEQUENTIAL','PIPELINE'):
           def fn():
            return full(head,raw,hidden,weight,outbasis,anchor,choose,capture=False) if mode=='FULL15' else rollout(head,raw,hidden,weight,outbasis,anchor,choose,mode=mode,capture=False)
           for _ in range(3):fn()
           torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
           with torch.cuda.graph(g):value=fn()
           for _ in range(5):g.replay()
           if mode!='FULL15':
            from head_latency.tail_graph_state import bind_tail_state
            exported=bind_tail_state(head,n)
            torch.cuda.synchronize()
            assert all(torch.equal(exported[i],pipeaux[i]) for i in (0,1,2,4,5)), 'captured auxiliary export differs from eager'
           start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
           for _ in range(30):g.replay()
           end.record();end.synchronize();ms=start.elapsed_time(end)/30
           row=dict(n=n,T=t,forced_cutoff=cutoff,mode=mode,head_and_actual_sampling_ms=ms)
           rows.append(row);print(now(),'TIMING',row,flush=True)
           del g,value
         del seq,pipe,expected,seqaux,pipeaux,seq_q
        del raw,hidden,q,refq;torch.cuda.empty_cache()
    # Forced10 has fewer expensive arithmetic tiles than forced15 at every n/T.
    for n in (1,7,18,40):
      for t in (0,1):
        a=next(v for v in checks if v['n']==n and v['T']==t and v['forced_cutoff']==15)
        b=next(v for v in checks if v['n']==n and v['T']==t and v['forced_cutoff']==10)
        assert all(x>y for x,y in zip(a['math_tile_counts'],b['math_tile_counts']))
    dump(root/'MICRO_GATE.json',dict(status='verified',checks=checks,timings=rows,readout_checks=readout_checks,gpu_uuid=actual,
         graph_auxiliary_export_checked=True,
         limitations=['Synthetic state timings are not native service TPS.','Launch slots and full preparation remain. C40 savings depend on fully inactive16-row tiles.'],utc=now()))
    print(now(),'MICRO_GATE VERIFIED',flush=True)

if __name__=='__main__':main()
