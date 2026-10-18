"""Pair fixed arena and native allocation under fresh per-request KV salts."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import time

CODE=Path(__file__).resolve().parent
RUNTIME=CODE.parent/'runtime_bundle'
sys.path[:0]=[str(CODE),str(RUNTIME),str(RUNTIME/'src'),str(RUNTIME/'hs_connectors/src')]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True); ap.add_argument('--service',type=Path,required=True)
    ap.add_argument('--allocator',choices=['fixed','native'],required=True); ap.add_argument('--gate',type=Path,required=True)
    a=ap.parse_args(); a.output.mkdir(exist_ok=False)
    from runtime.utils import dump,now,sha
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    library=None
    try:
        gate=json.loads(a.gate.read_text())
        assert gate['status']=='passed' and gate['full1536_memory_passed']
        dump(a.output/'gate_reused.json',dict(source=str(a.gate),sha256=sha(a.gate),unchanged_base_math=True, same_allocator=False,
             fresh_cache_salt_changes_KV_reuse_only=True, experimental_parallel_heads_separately_checked=True,scope=gate['scope'],utc=now()))
        if a.allocator=='fixed':
            import arena
            allocator,library=arena.install(68)
            dump(a.output/'small_gate.json',arena.small_gate(library))
            print(now(),'CONSTANT_WORKSPACE',arena.stats(library),flush=True)
        if a.allocator=='native':
            torch.cuda.set_device(0)
            torch.cuda.set_per_process_memory_fraction(68*2**30/torch.cuda.get_device_properties(0).total_memory)
            workspace=torch.empty(67*2**30,device='cuda',dtype=torch.uint8)
            del workspace
            dump(a.output/'native_preheat.json',dict(budget_gib=68,preheat_gib=67,
                 reserved_bytes=torch.cuda.memory_reserved(),expandable_segments=True,utc=now()))
            print(now(),'NATIVE_WORKSPACE_READY',torch.cuda.memory_reserved(),flush=True)
        deadline=time.monotonic()+900
        while not(a.service/'endpoint.json').exists():
            assert time.monotonic()<deadline,'Teacher startup timeout'
            if (a.service/'status.json').exists():
                assert json.loads((a.service/'status.json').read_text())['status']!='failed'
            time.sleep(2)
        from benchmark import run
        run(a.output/'profiles',a.source,a.service,library,a.allocator)
        dump(a.output/'worker_complete.json',dict(status='complete',pid=os.getpid(),utc=now()))
    except BaseException as exc:
        dump(a.output/'worker_failure.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),utc=now()))
        raise


if __name__=='__main__':main()

