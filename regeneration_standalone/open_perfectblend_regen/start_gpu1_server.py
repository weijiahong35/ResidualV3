"""Start one assigned idle GPU server using existing CU13 libraries."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from dispatch import gpu_inventory
from profiles import PROFILES, add_profile_argument, reservation_matches
from speculation_policy import selected_tokens, with_tokens
spec = importlib.util.spec_from_file_location('backend', ROOT / 'run.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--port', type=int)
    add_profile_argument(parser)
    parser.add_argument('--reservation-pid', type=int)
    options = parser.parse_args()
    profile = PROFILES[options.profile]
    gpu = gpu_inventory(str(options.gpu))[0]
    uuid = gpu['uuid']
    directory = profile.worker_dir(options.gpu)
    directory.mkdir(exist_ok=True)
    previous = directory / 'server.json'
    if previous.exists():
        pid = json.loads(previous.read_text())['pid']
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            stat = Path(f'/proc/{pid}/stat')
            if stat.exists() and stat.read_text().split(') ', 1)[1][0] != 'Z':
                raise RuntimeError(f'Previous server PID {pid} is still running')
    reserved = (reservation_matches(options.reservation_pid, uuid) and
                gpu['compute_pids'] == [options.reservation_pid] and gpu['free_mib'] >= .97 * gpu['total_mib'])
    if not (gpu['idle'] or reserved) or not gpu['compatible']:
        raise RuntimeError(f'GPU {options.gpu} is not idle/compatible: {gpu}')
    sys.argv = ['run.py', '--dataset', 'open-perfectblend', '--dataset-files',
                str(HERE / 'pool_500k_seed42/prompts/train/shard-000.jsonl'),
                '--gpu', uuid, '--thinking', profile.thinking, '--draft-model', str(profile.draft_model),
                '--concurrency', '128', '--max-model-len', str(profile.max_model_len),
                '--max-tokens', str(profile.max_model_len),
                '--port', str(options.port or 18080 + options.gpu), '--gpu-memory-utilization', '0.93']
    args = backend.parse_args()
    plan = backend.make_plan(args)
    command = plan['server_argv'] + ['--max-num-batched-tokens', '8192']
    native_tokens = plan['speculative_config']['num_speculative_tokens']
    tokens = selected_tokens(profile, options.gpu, native_tokens)
    command, actual_speculation = with_tokens(command, tokens)
    import socket
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(('127.0.0.1', args.port))
    with (directory / 'vllm.log').open('ab') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=backend.environment(args),
                                stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    backend.write_json(previous, {'pid': proc.pid, 'gpu_index': options.gpu, 'gpu_uuid': args.gpu,
        'port': args.port, 'command': command, 'environment': plan['server_environment'],
        'target_model': str(args.model), 'draft_model': str(args.draft_model),
        'profile': profile.name, 'max_model_len': profile.max_model_len,
        'num_speculative_tokens': tokens, 'actual_speculative_config': actual_speculation,
        'state': 'starting', 'note': 'Client concurrency starts at 40; server capacity allows tuning without reload.'})
    print(json.dumps({'server_pid': proc.pid, 'log': str(directory / 'vllm.log')}, indent=2))

if __name__ == '__main__':
    main()
