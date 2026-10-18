"""Keep one CUDA context and its genuine training allocation across DP1/DP2."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback

CODE=Path(__file__).resolve().parent
RUNTIME=CODE.parent/'runtime_bundle'
sys.path[:0]=[str(CODE),str(RUNTIME),str(RUNTIME/'src'),str(RUNTIME/'hs_connectors/src')]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    from runtime.utils import dump,now,sha
    import torch
    protocol=json.loads((a.root/'protocol.json').read_text())
    original=sha(Path(protocol['source_checkpoint']))
    try:
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(68*2**30/torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        workspace=torch.empty(67*2**30,device='cuda',dtype=torch.uint8);del workspace
        os.environ['STUDY_WORKSPACE_READY']='1'
        dump(a.root/'native_preheat.json',dict(budget_gib=68,preheat_gib=67,reserved_bytes=torch.cuda.memory_reserved(),utc=now()))
        print(now(),'NATIVE_WORKSPACE_READY',torch.cuda.memory_reserved(),flush=True)
        from benchmark import run
        for dp in [1,2]:
            service=a.root/f'service_dp{dp}'
            endpoint=json.loads((service/'endpoint.json').read_text());assert endpoint['dp_size']==dp
            dump(a.root/'status.json',dict(status='running',teacher_dp_size=dp,worker_pid=os.getpid(),utc=now()))
            os.environ['DO_FULL_HEAD_GATE']='1' if dp==1 else '0'
            run(a.root/f'dp{dp}',Path(protocol['source_run']),service,allocator='native')
            gc.collect();torch.cuda.synchronize()
            dump(a.root/f'dp{dp}_memory_after.json',dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),utc=now()))
        assert sha(Path(protocol['source_checkpoint']))==original,'Source checkpoint changed'
        dump(a.root/'worker_complete.json',dict(status='complete',source_checkpoint_sha256=original,utc=now()))
    except BaseException as exc:
        dump(a.root/'worker_failure.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),utc=now()))
        raise

if __name__=='__main__':main()
