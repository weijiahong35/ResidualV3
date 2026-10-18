"""Exact q45 readout parity and CUDA-graph timing, on an empty leased GPU."""
import os, sys, json, fcntl, subprocess, hashlib
from pathlib import Path
from types import SimpleNamespace
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'native_runtime'))


def main():
    import torch
    from torch import nn
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance
    from head_latency.tail_readout_fused import readout
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    index=next(s.split(',')[0].strip() for s in lines.splitlines() if gpu in s)
    repo=Path('/data/weijiahong/speculators-main-lora')
    lease=(repo/'residualV3-nonthing/results'/f'.native_gpu_{index}.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert gpu not in ps,'requires an empty GPU'
    init_batch_invariance();torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    prereg=json.loads((repo/'residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/preregistration.json').read_text())
    ck=Path(prereg['auxiliary_checkpoint'])
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==prereg['auxiliary_checkpoint_sha256']
    state=torch.load(ck,map_location='cpu',weights_only=True)
    net=nn.Sequential(nn.Linear(45,32),nn.SiLU(),nn.Linear(32,2))
    net.load_state_dict({k.removeprefix('Q45_MLP.net.'):v for k,v in state['model'].items() if k.startswith('Q45_MLP.net.')})
    head=SimpleNamespace(_tail_branch='Q45_MLP',_tail_readouts={'Q45_MLP':net.to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)},
          _tail_mean=state['mean'].to('cuda'),_tail_scale=state['scale'].to('cuda'))
    def stock(x):
        normalized=(x-head._tail_mean[:45])/head._tail_scale[:45]
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):y=net(normalized)
        return y.float().sigmoid()
    checks=[];timings=[]
    with torch.inference_mode():
      for n in (1,2,3,7,18,40,9,18,27,63,162,360):
        for seed in (42,57,81):
          torch.manual_seed(seed+n)
          for scale in (.01,1.,10.):
            x=torch.randn((n,45),device='cuda')*scale*head._tail_scale[:45]+head._tail_mean[:45]
            expected=stock(x);actual=readout(head,x)
            equal=torch.equal(expected,actual)
            row=dict(n=n,seed=seed,scale=scale,bitwise_equal=equal,max_error=float((expected-actual).abs().max()))
            checks.append(row)
            if not equal:
                (out/'FAILED.json').write_text(json.dumps(row,indent=2)+'\n')
                torch.save(dict(x=x.cpu(),expected=expected.cpu(),actual=actual.cpu()),out/'FAILED_tensors.pt')
                raise AssertionError(row)
        if n in (1,40,9,360):
          for name,fn in (('stock',lambda:stock(x)),('fused',lambda:readout(head,x))):
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
    result=dict(status='verified',checks=checks,timings=timings,scope='q45-only readout, exact outputs; no target or end-to-end timing')
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print('READOUT_VERIFIED',len(checks),timings,flush=True)


if __name__=='__main__':main()
