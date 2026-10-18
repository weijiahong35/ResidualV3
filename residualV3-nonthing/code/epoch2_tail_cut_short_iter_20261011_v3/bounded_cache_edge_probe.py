"""Exact bounded-cache fast path and fallback coverage before service adoption."""
import os,sys,json,fcntl,subprocess
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'native_runtime'))
import torch
from head_latency.tail_bounded_cache import prepare

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    index=next(x.split(',')[0].strip() for x in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines() if gpu in x)
    lease=(out.parents[2]/f'.native_gpu_{index}.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert gpu not in apps,'requires empty GPU'
    torch.set_num_threads(4);rows=[]
    with torch.inference_mode():
        for n in (1,40):
            torch.manual_seed(123)
            random=(torch.randn(n,1,151936,device='cuda')*3).to(torch.bfloat16)
            cases={'random':random,'negative':-random.abs()-1,'all_ties':torch.zeros_like(random)}
            separated=torch.full_like(random,-100);separated[:,:,:256]=100
            cases['fallback_gap']=separated
            boundary=torch.full_like(random,-1);boundary[:,:,:256]=1
            cases['fallback_sign_boundary']=boundary
            sparse=separated.clone();sparse[:,:,256:400]=99
            cases['fallback_insufficient_nearby']=sparse
            for name,z in cases.items():
                ids=z.topk(256,-1).indices;actual,_=prepare(z,ids)
                expected=z.clone().scatter_(-1,ids,float('-inf')).topk(256,-1).values.float().transpose(0,1).contiguous()
                eq=torch.equal(actual,expected)
                row=dict(n=n,case=name,bitwise_equal=eq,max_error=float((actual-expected).abs().max()))
                rows.append(row);print(row,flush=True);assert eq,row
    (out/'SUMMARY.json').write_text(json.dumps(dict(status='verified',checks=rows,explicit_fallback_cases=True),indent=2)+'\n')
if __name__=='__main__':main()
