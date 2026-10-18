"""Installed sampler versus device-cut padded-layout sampler on identical RNG."""
import os,sys,json,subprocess,fcntl
from pathlib import Path
C=Path(__file__).resolve().parent;sys.path.insert(0,str(C/'native_runtime'))
import torch
from repair.device_prefix_sampler_utils import rejection_sample as limited
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample as stock

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);gpu=os.environ['CUDA_VISIBLE_DEVICES']
    index=next(x.split(',')[0].strip() for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines() if gpu in x)
    lease=(out.parents[2]/f'.native_gpu_{index}.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert gpu not in apps
    torch.set_num_threads(4);rows=[]
    with torch.inference_mode():
      for n in (1,40):
       for temperature in (0.,1.):
        for kind in ('identical','different','mixed_prefill'):
            v=2048;torch.manual_seed(4242)
            idx=torch.randperm(n,device='cuda',dtype=torch.int32)
            q=torch.randn(n,15,v,device='cuda');p=torch.randn(n*16,v,device='cuda')
            sampled=torch.randint(v,(n*16,),device='cuda',dtype=torch.int64)
            if kind=='identical':
                p.reshape(n,16,v)[:,:15]=q[idx.long()]
                sampled.reshape(n,16)[:,1:]=q[idx.long()].argmax(-1)
            cuts=torch.randint(10,16,(n,),device='cuda',dtype=torch.int32)
            if kind=='mixed_prefill':cuts[idx[::3].long()]=0
            lengths=cuts[idx.long()].cpu().tolist();sel=[];ends=[0]
            for i,length in enumerate(lengths):sel.extend(range(i*16,i*16+length+1));ends.append(ends[-1]+length+1)
            selection=torch.tensor(sel,device='cuda',dtype=torch.int64);cu=torch.arange(n+1,device='cuda',dtype=torch.int32)*16
            pos=(torch.arange(16,device='cuda')[None,:]+1000+torch.arange(n,device='cuda')[:,None]*200).reshape(-1)
            maps=idx[:,None].expand(n,16).contiguous().reshape(-1);local=torch.arange(16,device='cuda',dtype=torch.int32).repeat(n)
            temps=torch.full((n,),temperature,device='cuda');seeds=torch.arange(n,device='cuda',dtype=torch.int64)+123
            actual=limited(p,q,sampled,cu,pos,idx,maps,local,temps,seeds,15,prefix_cuts=cuts)
            expected=stock(p[selection],q,sampled[selection],torch.tensor(ends,device='cuda',dtype=torch.int32),pos[selection],idx,maps[selection],local[selection],temps,seeds,15)
            mask=torch.arange(16,device='cuda')[None,:]<actual[1][:,None]
            eq=torch.equal(actual[1],expected[1]) and torch.equal(actual[0][mask],expected[0][mask])
            row=dict(n=n,T=temperature,case=kind,bitwise_equal=eq,counts=actual[1].tolist());rows.append(row);print(row,flush=True);assert eq,row
    (out/'SUMMARY.json').write_text(json.dumps(dict(status='verified',checks=rows,scope='exact sampler/RNG parity including prefix bonus and mixed prefill'),indent=2)+'\n')
if __name__=='__main__':main()
