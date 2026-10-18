"""GPU prefix boundary control, without synchronizing proposal lengths on CPU.

Target still executes its fixed padded graph. This is explicitly a control for
host synchronization, NOT physical target-token compaction. The valid-prefix
sampler retains the installed probability-ratio/RNG/bonus logic.
"""
import os,json
from pathlib import Path
import torch

def cuts(model):
    if not hasattr(model,'_device_prefix_cuts'):
        model._device_prefix_cuts=torch.full((int(os.environ['LOCAL_MAX_CONCURRENCY']),),15,
                                           device='cuda',dtype=torch.int32)
    return model._device_prefix_cuts

def install():
    if os.environ.get('TARGET_PREFIX_MODE')!='DEVICE':return
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as module
    original=module.rejection_sample
    from .device_prefix_sampler_utils import rejection_sample as limited
    def sample(target_logits,draft_logits,draft_sampled,cu,pos,idx,expanded_idx,local_pos,temp,seeds,steps,
               synthetic=None,*,use_fp64=False,use_block_verification=False):
        from runtime.native import MODEL
        prefix=cuts(MODEL)
        result=limited(target_logits,draft_logits,draft_sampled,cu,pos,idx,expanded_idx,local_pos,temp,seeds,steps,
                       synthetic,use_fp64=use_fp64,use_block_verification=use_block_verification,prefix_cuts=prefix)
        if os.environ.get('LOCAL_CONTRACT_CHECKS')=='1':
            # Actual target logits and RNG state, independently compacted to
            # the submitted logical prefix. Diagnostic synchronization only.
            boundaries=cu.tolist();limits=prefix[idx.long()].tolist();selected=[];ends=[0]
            for begin,end,limit in zip(boundaries,boundaries[1:],limits):
                length=min(end-begin,int(limit)+1)
                selected.extend(range(begin,begin+length));ends.append(ends[-1]+length)
            selection=torch.tensor(selected,device=cu.device,dtype=torch.int64)
            reference=original(target_logits[selection],draft_logits,draft_sampled[selection],
                torch.tensor(ends,device=cu.device,dtype=cu.dtype),pos[selection],idx,
                expanded_idx[selection],local_pos[selection],temp,seeds,steps,synthetic,
                use_fp64=use_fp64,use_block_verification=use_block_verification)
            assert torch.equal(result[1],reference[1]),'device prefix sampled count differs'
            mask=torch.arange(steps+1,device=cu.device)[None,:]<result[1][:,None]
            assert torch.equal(result[0][mask],reference[0][mask]),'device prefix sampled tokens differ'
            row=dict(prefix_sampler_bitwise=True,requests=len(limits),cutoffs=limits,
                     actual_submitted=[b-a-1 for a,b in zip(ends,ends[1:])],
                     target_fixed_padded_graph=True)
            with (Path(os.environ['LOCAL_TRIAL'])/'device_prefix_sampler_parity.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
        return result
    module.rejection_sample=sample
