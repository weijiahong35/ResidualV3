"""Isolated controls for the frozen tail pipeline; no live runtime edits."""
import os,sys,json,inspect,hashlib,fcntl,subprocess
from pathlib import Path
P=Path('/data/weijiahong/speculators-main-lora')
C=Path(os.environ.get('HEAD_PROBE_CODE',str(P/'residualV3-nonthing/code/epoch2_tail_cut_short_iter_20261011_v3')))
R=P/'residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3'
OUT=R/'probes'/os.environ.get('HEAD_PROBE_TAG','full_head_02')
os.environ['VLLM_BATCH_INVARIANT']='1'
sys.path.insert(0,str(C/'native_runtime'))
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from runtime.precision import build_token_basis
from repair.model import DualFeedbackRepair
from head_latency.optimized import prepare_variant
from head_latency.structure_fusion import rollout as full,sample
from head_latency.trained_tail import prepare
import head_latency.trained_tail as trained
import head_latency.tail_pipeline as pipeline
import head_latency.tail_pipeline_v2 as old_pipeline


def reference_call(fn):
    values={k:os.environ.get(k,'0') for k in ('TAIL_FUSED_READOUT','TAIL_FUSED_FINAL','TAIL_REUSE_SAMPLE_MAX','TAIL_FUSED_INPUT')}
    os.environ.update({k:'0' for k in values})
    try:return fn()
    finally:os.environ.update(values)


def early_reference(fn):
    value=os.environ.get('TAIL_EARLY_BATCH_SIZE','9')
    os.environ['TAIL_EARLY_BATCH_SIZE']='9'
    try:return fn()
    finally:os.environ['TAIL_EARLY_BATCH_SIZE']=value


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'PROBE_CONFIG.json').write_text(json.dumps({k:v for k,v in os.environ.items() if k.startswith('TAIL_') or k.startswith('HEAD_PROBE_')},indent=2)+'\n')
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    index=next(line.split(',')[0].strip() for line in inventory.splitlines() if gpu in line)
    lease=(R.parent/f'.native_gpu_{index}.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert gpu not in processes,'probe requires an empty GPU'
    assert os.environ['CUDA_VISIBLE_DEVICES']==gpu
    p=json.loads((R/'preregistration.json').read_text())
    for f,h in p['source_files'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    branch=os.environ.get('HEAD_PROBE_BRANCH','Q45_MLP')
    os.environ.update(TAIL_AUX_CHECKPOINT=p['auxiliary_checkpoint'],TAIL_AUX_BRANCH=branch,
        LOCAL_MAX_CONCURRENCY='0',HEAD_SMALL_TILE_MODE='all16_64',HEAD_STRUCTURE_MODE='lean_combined',TAIL_INSTRUMENT='0')
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance
    init_batch_invariance();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    rows=[];checks=[]
    with torch.inference_mode():
      ck=p['checkpoint'];cfg=json.loads(Path(ck['head_config_path']).read_text())
      head=DualFeedbackRepair(**cfg).to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
      head.load_state_dict({k.removeprefix('head.'):v for k,v in load_file(ck['path']).items()})
      target=Path(p['target_model']);index=json.loads((target/'model.safetensors.index.json').read_text())['weight_map'];values=[]
      for key in ('model.embed_tokens.weight','lm_head.weight'):
        with safe_open(target/index[key],framework='pt',device='cpu') as f:values.append(f.get_tensor(key).to('cuda',dtype=torch.bfloat16))
      embedding,weight=values;basis=load_file(p['basis_path'])['repair_basis'].to('cuda',dtype=torch.bfloat16)
      head.bind_features(embedding,build_token_basis(embedding,basis));outbasis=build_token_basis(weight,basis)
      prepare_variant(head,'small_tile')
      prepare(head,json.loads(Path(p['predictor_path']).read_text())['calibration_policy'])
      original_signal=trained.signal
      for n in tuple(map(int,os.environ.get('HEAD_PROBE_BATCHES','1,40').split(','))):
       for t in (0,1):
        torch.manual_seed(42+n+t)
        raw=(torch.randn(n,15,head.vocab_size,device='cuda')*3).to(torch.bfloat16)
        hidden=torch.randn(n,15,head.hidden_size,device='cuda',dtype=torch.bfloat16)
        anchor=torch.randint(head.vocab_size,(n,),device='cuda');temp=torch.full((n,),float(t),device='cuda')
        seeds=torch.arange(n,device='cuda',dtype=torch.int64)+42;maps=torch.arange(n,device='cuda',dtype=torch.int32)
        pos=torch.arange(15,device='cuda',dtype=torch.int64)[None,:].expand(n,-1).contiguous()+123
        q=torch.empty((n,15,head.vocab_size),device='cuda',dtype=torch.float32)
        head._tail_temperatures=temp;head._tail_mapping=maps
        columns=[torch.full((n,),j,device='cuda',dtype=torch.int32) for j in range(15)]
        def choose(z,j,active=None,entropy_max=None):
            args=dict(apply_temperature=True,output_processed_logits=q,output_processed_logits_col=columns[j],use_fp64=False)
            if entropy_max is not None:
                from head_latency.tail_sampler_max import sample as sampled_max
                return sampled_max(z,maps,temp,seeds,pos[:,j].contiguous(),raw_maximum=entropy_max,active=active,
                    override=head._tail_sampling_override,**args)
            if active is not None:
                from head_latency.tail_fast import sample as gated
                return gated(z,maps,temp,seeds,pos[:,j].contiguous(),active=active,override=head._tail_sampling_override,**args)
            return sample(z,maps,temp,seeds,pos[:,j].contiguous(),**args)
        for cutoff in (0,10,-1):
         head._tail_thresholds.fill_(.9 if cutoff==-1 else 0.)
         os.environ['TAIL_FORCE_CUTOFF']=str(max(cutoff,0))
         gold=reference_call(lambda:old_pipeline.rollout(head,raw,hidden,weight,outbasis,anchor,choose,mode='PIPELINE'))
         goldq=q.clone();goldstate=tuple(x.clone() for x in head._tail_last_stats)
         mask=torch.arange(15,device='cuda')[None,:]<goldstate[-1][:,None]
         for name,fn in [('SPARSE_V3',pipeline.rollout)]:
          result=fn(head,raw,hidden,weight,outbasis,anchor,choose,mode='PIPELINE')
          state=tuple(x.clone() for x in head._tail_last_stats)
          equal=dict(logits=torch.equal(gold.logits,result.logits),tokens=torch.equal(gold.tokens[mask],result.tokens[mask]),
                     predecessors=torch.equal(gold.predecessors[mask],result.predecessors[mask]),q=torch.equal(goldq[mask],q[mask]),
                     cutoffs=torch.equal(goldstate[-1],state[-1]),used_scores=torch.equal(goldstate[-2][:,:13],state[-2][:,:13]),
                     maximum_q=torch.equal(goldstate[2][:,:13],state[2][:,:13]),
                     survival=torch.equal(goldstate[1][:,:14],state[1][:,:14]))
          print('CHECK',n,t,cutoff,name,equal,flush=True)
          if not all(equal.values()):
           from head_latency.q45_features import features as stock_features
           from head_latency.tail_sparse_q45 import features as sparse_features
           from head_latency.trained_tail import readout
           length=9;cache,ci=head._tail_sparse_cache
           qref=torch.stack([stock_features(result.logits[:,j].contiguous())[0] for j in range(length)],0)
           qb=sparse_features(result.logits.transpose(0,1)[:length].contiguous().reshape(length*n,-1),torch.ones(length*n,device='cuda',dtype=torch.int32),cache[:length].reshape(length*n,256),ci[:length].reshape(length*n,-1)).reshape(length,n,45)
           prob_ref=torch.stack([readout(head,qref[j],hidden[:,j,:256]).detach() for j in range(length)],0)
           prob_batch=readout(head,qref.reshape(length*n,45),hidden[:,:length,:256].transpose(0,1).reshape(length*n,256)).reshape(length,n,-1)
           debug=dict(case=dict(n=n,T=t,cutoff=cutoff),checks=equal,q_features_equal=torch.equal(qref,qb),q_features_max_error=float((qref-qb).abs().max()),readout_batch_equal=torch.equal(prob_ref,prob_batch),readout_max_error=float((prob_ref-prob_batch).abs().max()),score_max_error=float((goldstate[-2][:,:13]-state[-2][:,:13]).abs().max()))
           (OUT/'FAILED.json').write_text(json.dumps(debug,indent=2)+'\n');print('DEBUG',debug,flush=True)
           torch.save(dict(qref=qref.cpu(),qbatch=qb.cpu(),prob_ref=prob_ref.cpu(),prob_batch=prob_batch.cpu()),OUT/'DEBUG_small_tensors.pt')
           raise AssertionError(debug)
          checks.append(dict(n=n,T=t,cutoff=cutoff,mode=name,checks=equal))
         variants=[('FULL15',lambda:full(head,raw,hidden,weight,outbasis,anchor,choose,capture=False)),
                   ('PIPELINE_V2',lambda:reference_call(lambda:old_pipeline.rollout(head,raw,hidden,weight,outbasis,anchor,choose,capture=False,mode='PIPELINE'))),
                   ('SPARSE_V3',lambda:pipeline.rollout(head,raw,hidden,weight,outbasis,anchor,choose,capture=False,mode='PIPELINE'))]
         if os.environ.get('HEAD_PROBE_FINAL_COMPARE')=='1':
            def final_reference(fn):
                value=os.environ.get('TAIL_FUSED_FINAL','0');os.environ['TAIL_FUSED_FINAL']='0'
                try:return fn()
                finally:os.environ['TAIL_FUSED_FINAL']=value
            variants.insert(1,('FULL15_OLD_FINAL',lambda:final_reference(lambda:full(head,raw,hidden,weight,outbasis,anchor,choose,capture=False))))
            variants.insert(-1,('SPARSE_OLD_FINAL',lambda:final_reference(lambda:pipeline.rollout(head,raw,hidden,weight,outbasis,anchor,choose,capture=False,mode='PIPELINE'))))
         if os.environ.get('HEAD_PROBE_SAMPLE_MAX_COMPARE')=='1':
            def sampler_reference(fn):
                value=os.environ.get('TAIL_REUSE_SAMPLE_MAX','0');os.environ['TAIL_REUSE_SAMPLE_MAX']='0'
                try:return fn()
                finally:os.environ['TAIL_REUSE_SAMPLE_MAX']=value
            variants.insert(-1,('SPARSE_OLD_SAMPLE_MAX',lambda:sampler_reference(lambda:pipeline.rollout(head,raw,hidden,weight,outbasis,anchor,choose,capture=False,mode='PIPELINE'))))
         if os.environ.get('HEAD_PROBE_INPUT_COMPARE')=='1':
            def input_reference(fn):
                value=os.environ.get('TAIL_FUSED_INPUT','0');os.environ['TAIL_FUSED_INPUT']='0'
                try:return fn()
                finally:os.environ['TAIL_FUSED_INPUT']=value
            variants.insert(1,('FULL15_OLD_INPUT',lambda:input_reference(lambda:full(head,raw,hidden,weight,outbasis,anchor,choose,capture=False))))
            variants.insert(-1,('SPARSE_OLD_INPUT',lambda:input_reference(lambda:pipeline.rollout(head,raw,hidden,weight,outbasis,anchor,choose,capture=False,mode='PIPELINE'))))
         if os.environ.get('HEAD_PROBE_EARLY_COMPARE')=='1':
            variants.insert(2,('SPARSE_EARLY9',lambda:early_reference(lambda:pipeline.rollout(head,raw,hidden,weight,outbasis,anchor,choose,capture=False,mode='PIPELINE'))))
         for name,fn in variants:
          for _ in range(12):fn()
          torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
          with torch.cuda.graph(graph):value=fn()
          for _ in range(40):graph.replay()
          repeats=[]
          for _ in range(3):
           start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
           for _ in range(25):graph.replay()
           end.record();end.synchronize();repeats.append(start.elapsed_time(end)/25)
          ms=sorted(repeats)[1]
          row=dict(n=n,T=t,cutoff=cutoff,mode=name,ms=ms,repeats_ms=repeats);rows.append(row);print('TIMING',row,flush=True)
          if name=='SPARSE_V3':
           from head_latency.tail_graph_state import bind_tail_state
           state=bind_tail_state(head,n);assert torch.equal(state[-1],goldstate[-1]),'graph cutoff differs'
           if n==40 and t==0 and cutoff==10 and os.environ.get('HEAD_PROBE_PROFILE')=='1':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
             for _ in range(5):graph.replay()
             torch.cuda.synchronize()
            prof.export_chrome_trace(str(OUT/'cuda_trace.json'))
            (OUT/'KERNELS.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
          del graph,value
         trained.signal=original_signal
         del gold,result,goldq,goldstate,state
        del raw,hidden,q;torch.cuda.empty_cache()
    (OUT/'SUMMARY.json').write_text(json.dumps(dict(status='verified',branch=branch,checks=checks,timings=rows,
        scope='synthetic frozen head, sparse feature parity versus untouched v2; no target/service TPS'),indent=2)+'\n')
    print('FULL_HEAD_VERIFIED',flush=True)

if __name__=='__main__':main()
