import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

CODE=Path(__file__).resolve().parent
RUNTIME=CODE.parent/'runtime_bundle'
sys.path[:0]=[str(CODE),str(RUNTIME),str(RUNTIME/'src'),str(RUNTIME/'hs_connectors/src')]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True); ap.add_argument('--service',type=Path,required=True)
    a=ap.parse_args()
    from runtime.utils import dump,now
    try:
        # Acquire the learner workspace while the DP teacher is still warming up.
        # A persistent global allocator also covers the autograd worker threads.
        import torch
        torch.backends.cuda.matmul.allow_tf32=False
        import arena
        allocator,library=arena.install(68)
        gate=arena.small_gate(library)
        dump(a.output/'arena_small_gate.json',dict(**gate,pid=os.getpid(),utc=now()))
        print(now(),'FIXED_WORKSPACE_ACQUIRED',arena.stats(library),flush=True)
        deadline=time.monotonic()+900
        while not(a.service/'endpoint.json').exists():
            assert time.monotonic()<deadline,'DP teacher startup timeout'
            if (a.service/'status.json').exists():
                status=json.loads((a.service/'status.json').read_text())
                assert status['status']!='failed',status
            time.sleep(2)
        # The validated routine performs the full1536 memory/gradient checks.
        import validate_arena
        validate_arena.install_arena=lambda gib:(allocator,library)
        # install is imported inside main, so redirect its module symbol too.
        arena.install=lambda gib:(allocator,library)
        import sys
        sys.argv=[str(CODE/'validate_arena.py'),'--run',str(a.source),'--service',str(a.service),'--output',str(a.output/'full_gate')]
        validate_arena.main()
        import gc
        gc.collect()
        print(now(),"POST_GATE_GC",arena.stats(library),flush=True)
        from benchmark import run
        run(a.output/'arena_benchmark',a.source,a.service,library,'fixed_arena68')
        dump(a.output/'worker_complete.json',dict(status='complete',pid=os.getpid(),utc=now()))
    except BaseException as exc:
        dump(a.output/'worker_failure.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),utc=now()))
        raise


if __name__=='__main__':main()
