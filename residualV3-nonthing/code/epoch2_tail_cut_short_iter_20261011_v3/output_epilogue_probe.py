"""Exact output epilogue checks and graph microtiming on an empty GPU."""
import os,sys,json,fcntl,subprocess
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'native_runtime'))


def main():
    import torch
    from head_latency.output_epilogue import finish
    from runtime.components import normalized_rows
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    index=next(s.split(',')[0].strip() for s in inventory.splitlines() if gpu in s)
    repo=Path('/data/weijiahong/speculators-main-lora')
    lease=(repo/'residualV3-nonthing/results'/f'.native_gpu_{index}.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert gpu not in ps,'requires an empty GPU'
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    def stock(x):
        return normalized_rows(x[:,:512],eps=1e-6),(.8*torch.sigmoid(x[:,512].float())).to(x.dtype)
    checks=[];timings=[]
    with torch.inference_mode():
      witness=repo/'residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_19_epilogue_diagnosis/EPILOGUE_DIAGNOSIS_tensors.pt'
      if witness.exists():
        x=torch.load(witness,map_location='cpu',weights_only=True)['output'].to('cuda')
        a,b=stock(x);c,d=finish(x,512,1e-6,.8)
        assert torch.equal(a,c) and torch.equal(b,d),'previous real-data BF16 norm boundary differs'
        checks.append(dict(case='previous_N40_BF16_boundary_failure',direction_equal=True,dose_equal=True))
      for n in (1,2,3,7,18,40):
       for seed in (42,57,81):
        torch.manual_seed(n+seed)
        for scale in (0.,1e-9,1e-5,.01,1.,10.,1000.):
         x=(torch.randn(n,513,device='cuda')*scale).to(torch.bfloat16)
         a,b=stock(x);c,d=finish(x,512,1e-6,.8)
         row=dict(n=n,seed=seed,scale=scale,direction_equal=torch.equal(a,c),dose_equal=torch.equal(b,d),
                  direction_max_error=float((a-c).abs().max()),dose_max_error=float((b-d).abs().max()))
         checks.append(row)
         if not row['direction_equal'] or not row['dose_equal']:
            (out/'FAILED.json').write_text(json.dumps(row,indent=2)+'\n')
            torch.save(dict(input=x.cpu(),reference_direction=a.cpu(),actual_direction=c.cpu(),reference_dose=b.cpu(),actual_dose=d.cpu()),out/'FAILED_tensors.pt')
            raise AssertionError(row)
       if n in (1,40):
        for name,fn in (('stock',lambda:stock(x)),('fused',lambda:finish(x,512,1e-6,.8))):
         for _ in range(5):fn()
         graph=torch.cuda.CUDAGraph()
         with torch.cuda.graph(graph):y=fn()
         for _ in range(10):graph.replay()
         values=[]
         for _ in range(3):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(200):graph.replay()
            end.record();end.synchronize();values.append(start.elapsed_time(end)/200)
         timings.append(dict(n=n,mode=name,ms=sorted(values)[1],repeats_ms=values))
         del graph,y
    result=dict(status='verified',checks=checks,timings=timings,scope='direction/dose epilogue only; not whole-head or TPS')
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print('EPILOGUE_VERIFIED',len(checks),timings,flush=True)


if __name__=='__main__':main()
