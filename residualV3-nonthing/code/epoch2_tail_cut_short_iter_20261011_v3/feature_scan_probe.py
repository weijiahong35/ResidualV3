"""Check the fused scan against the installed deterministic ATen reduction tree."""
import os,sys,json,subprocess,fcntl,time
from pathlib import Path
import torch

CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'native_runtime'))
from head_latency.tail_sparse_q45 import _scan_assemble

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    index=next(s.split(',')[0].strip() for s in inventory.splitlines() if gpu in s)
    lease=(out.parents[2]/f'.native_gpu_{index}.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert gpu not in processes,'requires empty GPU'
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    checks=[]
    with torch.inference_mode():
      for group in range(2,41):
        for repeat in (1,9):
          n=group*repeat;torch.manual_seed(42+n)
          probs=torch.rand((n,256),device='cuda')
          probs[::3,::7]=0.;probs[::5,::11]=1e-30
          logs=probs.clamp_min(1e-38).log();stats=torch.rand((2,n),device='cuda')
          cum=torch.cat([probs[j:j+group].cumsum(-1) for j in range(0,n,group)],0)
          four=torch.stack((probs[:,0],probs[:,1],probs[:,0]-probs[:,1],stats[1]),-1)
          c=torch.stack([cum[:,i] for i in (0,1,3,7,15,31,63,127,255)],-1)
          expected=torch.cat((four,c,logs[:,:32].clamp_min(-69.07755278982137)),-1)
          result=torch.empty_like(expected);logx=max(4,min(9,(9+8-(group-1).bit_length())//2))
          _scan_assemble[(n,)](probs,logs,stats,result,n,logx,2*(1<<logx),num_warps=4)
          equal=torch.equal(result,expected)
          row=dict(group=group,rows=n,bitwise_equal=equal,max_error=float((expected-result).abs().max()))
          checks.append(row);print('CHECK',row,flush=True)
          if not equal:
            (out/'FAILED.json').write_text(json.dumps(row,indent=2)+'\n')
            raise AssertionError(row)
    (out/'SUMMARY.json').write_text(json.dumps(dict(status='verified',checks=checks),indent=2)+'\n')
    print('SCAN_ASSEMBLY_VERIFIED',flush=True)

if __name__=='__main__':main()
