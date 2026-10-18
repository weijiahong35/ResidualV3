"""Small deterministic GPU gate and feature-only timing; no training changes."""
import os,sys,json,fcntl,time,hashlib,subprocess
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'native_runtime'))
os.environ['VLLM_BATCH_INVARIANT']='1'
import torch
from head_latency.q45_features import features as stock
from head_latency.tail_sparse_q45 import prepare,features
from head_latency.tail_q45_hist import features as histogram


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
    mapping=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    index=next(x.split(',')[0].strip() for x in mapping.splitlines() if os.environ['CUDA_VISIBLE_DEVICES'] in x)
    lease=(root.parents[2]/f'.native_gpu_{index}.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] not in inventory,'requires an empty GPU'
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance
    init_batch_invariance();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    rows=[];timings=[]
    def timed(fn,repeats=25):
        for _ in range(3):fn()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):result=fn()
        for _ in range(3):g.replay()
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        for _ in range(repeats):g.replay()
        b.record();b.synchronize();value=a.elapsed_time(b)/repeats
        del g,result
        return value
    with torch.inference_mode():
      for n in (1,40):
       for scale in (0.,3.,15.):
        torch.manual_seed(42+n)
        raw=(torch.randn(n,15,151936,device='cuda')*scale).to(torch.bfloat16)
        ids=raw.topk(256,-1).indices
        cache,ci=prepare(raw,ids)
        masked=raw.clone().scatter(-1,ids,float('-inf'))
        expected=masked.topk(256,-1).values.float().transpose(0,1).contiguous()
        assert torch.equal(cache,expected),'unmodified Top256 cache differs'
        del masked,expected
        for j in (0,7,14):
            z=raw[:,j].clone().contiguous()
            chosen=ids[:,j]
            updates=(z.gather(-1,chosen).float()+torch.randn_like(chosen,dtype=torch.float32)*2).to(torch.bfloat16)
            z.scatter_(-1,chosen,updates)
            for kind in ('all','mixed','none'):
                active=torch.ones(n,device='cuda',dtype=torch.int32)
                if kind=='mixed':active[::2]=0
                if kind=='none':active.zero_()
                old,_=stock(z);new=features(z,active,cache[j],ci[j]);mask=active.bool()
                eq=torch.equal(old[mask],new[mask]);err=(old[mask]-new[mask]).abs().max().item() if mask.any() else 0.
                row=dict(n=n,scale=scale,position=j,active=kind,bitwise_equal=eq,max_error=err)
                rows.append(row);print('CHECK',row,flush=True)
                assert eq,row
            if scale==3. and j==7:
                active.fill_(1)
                for name,fn in [('stock',lambda:stock(z)),('histogram',lambda:histogram(z,active)),('sparse',lambda:features(z,active,cache[j],ci[j]))]:
                    ms=timed(fn);row=dict(n=n,component=name,ms=ms);timings.append(row);print('TIMING',row,flush=True)
        if scale==3.:
            ms=timed(lambda:prepare(raw,ids));row=dict(n=n,component='sparse_prepare_full15',ms=ms);timings.append(row);print('TIMING',row,flush=True)
        del raw,ids,cache,ci,z;torch.cuda.empty_cache()
    value=dict(status='verified',checks=rows,timings=timings,scope='synthetic repaired sparse support, feature-only GPU timing')
    (root/'SUMMARY.json').write_text(json.dumps(value,indent=2)+'\n')
    print('SPARSE_FEATURE_VERIFIED',flush=True)


if __name__=='__main__':main()
