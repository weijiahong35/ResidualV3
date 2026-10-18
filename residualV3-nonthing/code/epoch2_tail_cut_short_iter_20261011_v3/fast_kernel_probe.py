"""Kept-row feature and native sampling equivalence before integration."""
import argparse,json,sys,os
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'native_runtime'))
import torch
from head_latency.q45_features import features as old_features
from head_latency.tail_fast import features,sample
from head_latency.structure_fusion import sample as old_sample
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

def timing(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):result=fn()
    for _ in range(3):g.replay()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(30):g.replay()
    end.record();end.synchronize()
    value=start.elapsed_time(end)/30
    del result,g
    return value

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    checks=[];times=[]
    torch.set_num_threads(4)
    with torch.inference_mode():
      for n in (1,7,40):
       for scale in (0.,3.,30.):
        torch.manual_seed(42+n)
        z=(torch.randn(n,151936,device='cuda')*scale).to(torch.bfloat16)
        storage=torch.zeros((n,17),device='cuda',dtype=torch.int32);storage[:,0]=1
        if n>1:storage[::2,0]=0
        active=storage[:,0];mask=active.bool()
        old,_=old_features(z);new=features(z,active)
        torch.cuda.synchronize()
        exact=torch.equal(old[mask],new[mask])
        error=float((old[mask]-new[mask]).abs().max()) if bool(mask.any()) else 0.
        assert exact,('features',n,scale,error)
        checks.append(dict(kind='features',n=n,scale=scale,active_exact=exact,max_error=error))
        print('CHECK',checks[-1],flush=True)
        if scale!=3.:continue
        for live in ('all','mixed','none'):
          storage[:,0]=1 if live=='all' else 0
          if live=='mixed':storage[::2,0]=1
          act=storage[:,0];m=act.bool()
          times.append(dict(n=n,live=live,old_feature_ms=timing(lambda:old_features(z)),new_feature_ms=timing(lambda:features(z,act))))
          print('TIMING',times[-1],flush=True)
          for t in (0.,1.,.9):
           for fp64 in (False,True):
            maps=torch.arange(n,device='cuda',dtype=torch.int32).flip(0)
            temp=torch.full((n,),t,device='cuda');seed=torch.arange(n,device='cuda',dtype=torch.int64)+42
            pos=torch.arange(n,device='cuda',dtype=torch.int64)+123;col=torch.arange(n,device='cuda',dtype=torch.int32)%3
            q=torch.full((n,3,z.shape[1]),-12345.,device='cuda',dtype=torch.float32);ref=torch.empty_like(q)
            args=dict(apply_temperature=True,output_processed_logits_col=col,use_fp64=fp64)
            gold=gumbel_sample(z,maps,temp,seed,pos,output_processed_logits=ref,**args)
            result=sample(z,maps,temp,seed,pos,active=act,output_processed_logits=q,**args)
            assert torch.equal(result[m],gold[m]),('sampling',n,t,fp64,live)
            assert torch.equal(q[maps[m].long(),col[m].long()],ref[maps[m].long(),col[m].long()])
            assert bool((result[~m]==0).all())
            assert bool((q[maps[~m].long(),col[~m].long()]==-12345.).all())
            checks.append(dict(kind='sample',n=n,live=live,T=t,fp64=fp64,kept_ids_exact=True,kept_q_exact=True,inactive_q_untouched=True))
            override=torch.ones((),device='cuda',dtype=torch.int32)
            warm=sample(z,maps,temp,seed,pos,active=act,override=override,output_processed_logits=q,**args)
            assert torch.equal(warm,gold) and torch.equal(q[maps.long(),col.long()],ref[maps.long(),col.long()]),'internal warmup must fill full native q'
            checks.append(dict(kind='internal_warmup',n=n,live=live,T=t,fp64=fp64,all_ids_exact=True,all_q_exact=True))
            # vLLM captures once, then changes this flag between internal
            # warmup and real requests. The choice must remain a GPU input.
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured=sample(z,maps,temp,seed,pos,active=act,override=override,output_processed_logits=q,**args)
            for enabled in (0,1,0):
                override.fill_(enabled);q.fill_(-12345.);graph.replay()
                keep=torch.ones_like(m) if enabled else m
                assert torch.equal(captured[keep],gold[keep]),'captured warmup flag changed kept IDs'
                assert torch.equal(q[maps[keep].long(),col[keep].long()],ref[maps[keep].long(),col[keep].long()]),'captured warmup flag changed native q'
                assert bool((q[maps[~keep].long(),col[~keep].long()]==-12345.).all()),'captured real request wrote inactive q'
            checks.append(dict(kind='warmup_graph_transition',n=n,live=live,T=t,fp64=fp64,transitions=[0,1,0],ids_and_q_exact=True))
            del graph,captured
          args=dict(apply_temperature=True,use_fp64=False)
          times[-1].update(old_sample_ms=timing(lambda:old_sample(z,maps,temp,seed,pos,**args)),new_sample_ms=timing(lambda:sample(z,maps,temp,seed,pos,active=act,**args)))
    a.root.mkdir(parents=True,exist_ok=True)
    (a.root/'FAST_KERNEL_GATE.json').write_text(json.dumps(dict(status='verified',checks=checks,timings=times),indent=2)+'\n')
    print('FAST_KERNEL_GATE VERIFIED',flush=True)

if __name__=='__main__':main()
