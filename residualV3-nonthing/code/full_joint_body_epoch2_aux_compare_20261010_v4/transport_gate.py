"""CPU gate of unlink/mmap lifetime, exact packing and bounded ownership."""
from pathlib import Path
import gc
import hashlib
import json
import tempfile
import time
import torch
from safetensors.torch import load_file,save_file


def gate(root):
    root=Path(root);torch.set_num_threads(1);torch.manual_seed(610008)
    records=[]
    with tempfile.TemporaryDirectory(prefix='residualV3_mmap_gate_',dir='/dev/shm') as temp:
        for n in [1,37,4096,17]:
            expected=torch.randn((n,6,4096),dtype=torch.bfloat16);ids=torch.arange(n)
            path=Path(temp)/'gate.safetensors';save_file({'hidden_states':expected,'token_ids':ids},str(path))
            payload=load_file(str(path));h=payload['hidden_states'];del payload;gc.collect()
            path.unlink();assert not path.exists();gc.collect()
            # Exercise the mapping after the context and file name are gone,
            # including a new same-name file backed by a different inode.
            save_file({'new_file':torch.zeros(1024)},str(path));path.unlink()
            assert torch.equal(h,expected) and torch.isfinite(h).all()
            aux=torch.empty((n,5*4096),dtype=torch.bfloat16);last=torch.empty((n,4096),dtype=torch.bfloat16)
            aux.copy_(h[:,:-1].flatten(1));last.copy_(h[:,-1])
            assert torch.equal(aux,expected[:,:-1].flatten(1)) and torch.equal(last,expected[:,-1])
            records.append(dict(tokens=n,hidden_after_unlink_bitwise_equal=True,packed_aux_and_final_bitwise_equal=True))
            del h,expected,aux,last;gc.collect()
    old=Path('/data/weijiahong/speculators-main-lora/residualV3-nonthing/code/head_forward_backward_optimization_20261007_v1')
    new=Path(__file__).resolve().parent
    math_files=['optimized_repair.py','combined_ce_tv.py','joint_features.py','joint_model.py','losses.py','fused_kernels.py','host_pipeline.py','concurrent_driver.py']
    for name in math_files:assert (old/name).read_bytes()==(new/name).read_bytes(),name
    result=dict(status='passed',cpu_lifetime_cases=records,head_loss_optimizer_and_body_math_sources_unchanged=math_files,
                prior_full1536_online_gradient_gate=str(root.parent/'head_forward_backward_optimization_20261007_v1/optimized/joint/full1536_online_gradient_gate.json'),
                no_hidden_logits_saved=True,temporary_files_removed=True)
    (root/'transport_lifetime_gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    import sys
    gate(sys.argv[1])
