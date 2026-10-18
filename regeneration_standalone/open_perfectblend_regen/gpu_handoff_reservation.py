"""Hold a small CUDA allocation on an already-owned GPU during server replacement."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from regenlib.checkpoint import durable_json as write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--status-file', type=Path, required=True)
    parser.add_argument('--lifetime-seconds', type=int, default=1800)
    args = parser.parse_args()
    # Use the same PyTorch/CU13 runtime and UUID pinning as the serving process.
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu_uuid
    import torch
    assert torch.cuda.device_count() == 1

    def interrupted(sig, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    allocation = torch.empty(16 * 2**20, device='cuda:0', dtype=torch.uint8)
    torch.cuda.synchronize()
    try:
        write_json(args.status_file, {'state': 'holding', 'pid': os.getpid(),
            'gpu_uuid': args.gpu_uuid, 'allocation_mib': 16,
            'expires_unix': time.time() + args.lifetime_seconds})
        deadline = time.monotonic() + args.lifetime_seconds
        while time.monotonic() < deadline:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        del allocation
        torch.cuda.empty_cache()
        write_json(args.status_file, {'state': 'released', 'pid': os.getpid(), 'gpu_uuid': args.gpu_uuid})


if __name__ == '__main__':
    main()
