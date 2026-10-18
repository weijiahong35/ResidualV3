#!/usr/bin/env python3
"""Regenerate fixed pool shards with one explicitly assigned GPU/server."""
import argparse
from contextlib import ExitStack
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
# Load by path: this file and the shared launcher are both named run.py.
spec = importlib.util.spec_from_file_location('standalone_backend', ROOT / 'run.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
from pool import DEFAULT_POOL, sha256_file, write_json
from regenlib.checkpoint import repair_output_tail, load_skipped
from profiles import PROFILES, DEFAULT_PROFILE
from pool_manifest import load_pool_manifest


def shard_numbers(expression, available):
    if expression == 'all':
        return sorted(available)
    selected = set()
    for part in expression.split(','):
        if '-' in part:
            start, end = map(int, part.split('-', 1))
            if end < start:
                raise ValueError('Shard range must be ascending')
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected or selected - set(available):
        raise ValueError(f'Unknown shard numbers: {sorted(selected - set(available))}')
    return sorted(selected)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool-dir', type=Path, default=DEFAULT_POOL)
    parser.add_argument('--output-dir', type=Path, help='Separate generation root for a different model/sampling/length recipe; prompts remain shared')
    parser.add_argument('--split', choices=['train', 'validation'], default='train')
    parser.add_argument('--shards', default='all', help='all, 0, 0-9, or 0-9,20-29 (zero based)')
    parser.add_argument('--max-samples', type=int, help='Total target conversations across selected shards; existing results count toward this cap')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--dry-run', action='store_true', help='Default; does not start any GPU process')
    parser.add_argument('--gpu', help='Exactly one explicitly assigned GPU index/UUID')
    parser.add_argument('--thinking', choices=['on', 'off'], default='off')
    parser.add_argument('--env-dir', type=Path, default=Path(os.getenv('VLLM_ENV_DIR', backend.DEFAULT_ENV)))
    parser.add_argument('--model', type=Path, default=backend.MODELS / 'Qwen3-8B')
    parser.add_argument('--draft-model', type=Path)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--max-model-len', type=int)
    parser.add_argument('--concurrency', type=int, default=40)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.93)
    parser.add_argument('--port', type=int, default=18081)
    parser.add_argument('--ready-timeout', type=int, default=900)
    parser.add_argument('--sampling-params', default='{}')
    args = parser.parse_args()
    profile = PROFILES['thinking_ctx8192' if args.thinking == 'on' else DEFAULT_PROFILE]
    args.draft_model = args.draft_model or profile.draft_model
    args.max_model_len = args.max_model_len or profile.max_model_len
    args.max_tokens = args.max_tokens or args.max_model_len
    if args.run and args.gpu is None:
        parser.error('--run requires explicit --gpu; existing experiment GPUs are never selected automatically')
    if args.max_samples is not None and args.max_samples < 1:
        parser.error('max-samples must be positive')
    args.pool_dir = args.pool_dir.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    return args


def make_plan(args):
    manifest_path = args.pool_dir / 'manifest.json'
    manifest, contract_manifest_sha, active_manifest_sha = load_pool_manifest(
        args.pool_dir, prefer_base=args.thinking == 'on')
    candidates = {shard['shard']: shard for shard in manifest['shards'] if shard['split'] == args.split}
    chosen = shard_numbers(args.shards, candidates)
    available = sum(candidates[number]['rows'] for number in chosen)
    if args.max_samples is not None and args.max_samples > available:
        raise ValueError(f'max-samples exceeds selected pool size: {available}')
    remaining = args.max_samples or available
    jobs = []
    thinking = 'thinking' if args.thinking == 'on' else 'nonthinking'
    generation_root = args.output_dir or args.pool_dir / f'regenerated_ctx{args.max_model_len}'
    # Reuse the existing target sampling and stock DFlash server configuration.
    common = ['run.py', '--dataset', 'open-perfectblend', '--dataset-files',
              str(args.pool_dir / candidates[chosen[0]]['path']), '--thinking', args.thinking,
              '--dataset-revision', manifest['parameters']['revision']]
    for name in ['env_dir', 'model', 'draft_model', 'max_tokens', 'max_model_len',
                 'concurrency', 'gpu_memory_utilization', 'port', 'ready_timeout', 'sampling_params']:
        common += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    if args.gpu:
        common += ['--gpu', args.gpu]
    previous = sys.argv
    try:
        sys.argv = common
        runtime = backend.parse_args()
    finally:
        sys.argv = previous
    base = backend.make_plan(runtime)
    generation_params = dict(base['sampling_params'])
    generation_params.setdefault('seed', manifest['parameters']['seed'])
    for number in chosen:
        if remaining <= 0:
            break
        shard = candidates[number]
        source = args.pool_dir / shard['path']
        if sha256_file(source) != shard['sha256']:
            raise ValueError(f'Prepared prompt shard changed: {source}')
        target = min(remaining, shard['rows'])
        output = generation_root / thinking / args.split / f'shard-{number:03d}.jsonl'
        command = [str(runtime.env_dir / 'bin/python'), str(ROOT / 'regenerate.py'),
                   '--dataset', 'open-perfectblend', '--dataset-files', str(source),
                   '--dataset-revision', manifest['parameters']['revision'],
                   '--model', str(runtime.model), '--split', 'train',
                   '--endpoint', f'http://127.0.0.1:{runtime.port}/v1/chat/completions',
                   '--concurrency', str(runtime.concurrency), '--max-tokens', str(runtime.max_tokens),
                   '--max-model-len', str(runtime.max_model_len),
                   '--sampling-params', json.dumps(generation_params), '--outfile', str(output), '--resume',
                   '--minimum-valid-tokens', '8', '--skip-context-overflow']
        jobs.append({'shard': number, 'split': args.split, 'source': str(source),
                     'source_sha256': shard['sha256'], 'output': str(output),
                     'target_conversations': target, 'full_shard_conversations': shard['rows'],
                     'client_argv': command})
        remaining -= target
    model_hashes = {str(path): sha256_file(path) for model in [runtime.model, runtime.draft_model]
                    for path in sorted(model.iterdir()) if path.suffix in {'.json', '.jinja'}}
    weight_files = {str(path): {'resolved_path': str(path.resolve()),
                               'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                    for model in [runtime.model, runtime.draft_model]
                    for path in sorted(model.iterdir()) if path.suffix in {'.safetensors', '.bin'}}
    contract = {'pool_manifest_sha256': contract_manifest_sha,
                'target_model': str(runtime.model), 'draft_model': str(runtime.draft_model),
                'model_config_hashes': model_hashes, 'model_weight_files': weight_files,
                'speculative_config': base['speculative_config'],
                'sampling_params': generation_params, 'max_tokens': runtime.max_tokens,
                'max_model_len': runtime.max_model_len,
                'minimum_valid_tokens': 8, 'skip_context_overflow': True,
                'checkpoint_sha256': sha256_file(ROOT / 'regenlib/checkpoint.py'),
                'client_sha256': sha256_file(ROOT / 'regenerate.py'),
                'registry_sha256': sha256_file(ROOT / 'regenlib/configs.py')}
    plan = {'mode': 'run' if args.run else 'dry_run_no_gpu_process', 'pool_dir': str(args.pool_dir),
            'active_pool_manifest_sha256': active_manifest_sha,
            'pool_extension': manifest.get('extension'),
            'generation_root': str(generation_root),
            'split': args.split, 'thinking': thinking, 'target_conversations': sum(job['target_conversations'] for job in jobs),
            'server_argv': base['server_argv'], 'server_environment': base['server_environment'],
            'contract': contract, 'jobs': jobs}
    return runtime, plan


def completed_count(job):
    """Verify resumed output belongs to this exact shard before counting it."""
    path = Path(job['output'])
    if not path.exists():
        return 0
    with Path(job['source']).open() as handle:
        expected = {json.loads(line)['id'] for line in handle if line.strip()}
    seen = set()
    with path.open('rb') as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.endswith(b'\n'):
                raise ValueError(f'Incomplete JSONL tail: {path}:{lineno}')
            row = json.loads(line)
            primary_id = row.get('primary_id')
            if primary_id not in expected:
                raise ValueError(f'Output conversation does not belong to shard {job["shard"]}: {primary_id}')
            if not row.get('input_ids') or len(row['input_ids']) != len(row.get('loss_mask', [])):
                raise ValueError(f'Invalid generated sample: {path}:{lineno}')
            if row.get('metadata', {}).get('conversation_complete', True):
                seen.add(primary_id)
    return len(seen)


def skipped_count(job):
    with Path(job['source']).open() as handle:
        expected = {json.loads(line)['id'] for line in handle if line.strip()}
    skipped = load_skipped(job['output'])
    if skipped - expected:
        raise ValueError('Terminal skip does not belong to this shard')
    return len(skipped)


def execute(args, runtime, plan):
    logdir = args.pool_dir / 'logs' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'_{os.getpid()}')
    logdir.mkdir(parents=True, exist_ok=False)
    manifest_path = logdir / 'run_manifest.json'
    state = {**plan, 'state': 'starting', 'log_dir': str(logdir)}
    write_json(manifest_path, state)
    server = client = None
    with ExitStack() as stack:
        for job in plan['jobs']:
            path = Path(job['output'])
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.with_suffix('.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract_path = Path(plan['generation_root']) / plan['thinking'] / 'generation_contract.json'
        with contract_path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if contract_path.exists():
                if json.loads(contract_path.read_text()) != plan['contract']:
                    raise ValueError('Existing generation settings differ; use a separate pool/output directory to preserve provenance')
            else:
                if any(Path(job['output']).exists() for job in plan['jobs']):
                    raise ValueError('Existing output has no generation contract; inspect it before resuming')
                write_json(contract_path, plan['contract'])
        pending = []
        for job in plan['jobs']:
            repair_output_tail(job['output'])
            job['existing'] = completed_count(job)
            job['skipped'] = skipped_count(job)
            if job['existing'] + job['skipped'] < job['target_conversations']:
                pending.append(job)
        if not pending:
            state.update(state='complete', note='Selected conversation targets already exist; no server started')
            write_json(manifest_path, state)
            print(state['note'])
            return
        # No CUDA probe: only the explicit device is passed to the child process.
        import socket
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', runtime.port))
        check = "import sys,importlib.metadata as m;sys.path.insert(0," + repr(str(ROOT / '.client_deps')) + ");import datasets,aiohttp,transformers;assert m.version('vllm')=='0.26.0';assert '+cu130' in m.version('torch')"
        subprocess.run([str(runtime.env_dir / 'bin/python'), '-c', check],
                       env=backend.environment(runtime, client=True), check=True)
        try:
            with (logdir / 'vllm.log').open('wb') as server_log:
                server = subprocess.Popen(plan['server_argv'], env=backend.environment(runtime), cwd=ROOT,
                                          stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(state='starting_server', server_pid=server.pid)
                write_json(manifest_path, state)
                backend.wait_ready(server, runtime)
                for job in pending:
                    state.update(state='generating', active_shard=job['shard'])
                    write_json(manifest_path, state)
                    summary = logdir / f'{job["split"]}-{job["shard"]:03d}-summary.json'
                    command = job['client_argv'] + ['--limit', str(job['target_conversations'] - job['existing'] - job['skipped']),
                                                  '--run-manifest', str(manifest_path), '--summary-file', str(summary)]
                    status_path = Path(job['output']).with_suffix('.status.json')
                    status = {'state': 'generating', 'completed_conversations': job['existing'],
                              'expected_conversations': job['full_shard_conversations'], 'run_manifest': str(manifest_path)}
                    write_json(status_path, status)
                    print(f'Generating {job["split"]}/{job["shard"]:03d}: {job["existing"]}/{job["target_conversations"]}; logs={logdir}', flush=True)
                    with (logdir / f'{job["split"]}-{job["shard"]:03d}-client.log').open('wb') as client_log:
                        client = subprocess.Popen(command, env=backend.environment(runtime, client=True), cwd=ROOT,
                                                  stdout=client_log, stderr=subprocess.STDOUT, start_new_session=True)
                        while client.poll() is None:
                            if server.poll() is not None:
                                raise RuntimeError('vLLM exited during generation')
                            time.sleep(1)
                    completed = completed_count(job)
                    skipped = skipped_count(job)
                    status['completed_conversations'] = completed
                    status['skipped_conversations'] = skipped
                    status['state'] = 'complete' if completed + skipped >= job['full_shard_conversations'] else 'partial'
                    status['output_bytes'] = Path(job['output']).stat().st_size
                    if summary.is_file():
                        status['latest_summary'] = json.loads(summary.read_text())
                    write_json(status_path, status)
                    if client.returncode or completed + skipped < job['target_conversations']:
                        raise RuntimeError(f'Shard {job["shard"]}: {completed}/{job["target_conversations"]}; inspect client log/errors and rerun the same command')
                state['state'] = 'complete'
        except BaseException as exc:
            state.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
            raise
        finally:
            backend.stop_group(client)
            backend.stop_group(server)
            write_json(manifest_path, state)
    data_paths = {'train': [], 'validation': []}
    for split in data_paths:
        directory = Path(plan['generation_root']) / plan['thinking'] / split
        data_paths[split] = [str(path) for path in sorted(directory.glob('shard-*.jsonl')) if '.' not in path.stem]
    write_json(Path(plan['generation_root']) / plan['thinking'] / 'training_data_paths.json', data_paths)
    if args.max_samples is not None and args.max_samples <= 10000:
        total = sum(completed_count(job) for job in plan['jobs'])
        output_bytes = sum(Path(job['output']).stat().st_size for job in plan['jobs'])
        source_manifest = json.loads((args.pool_dir / 'manifest.json').read_text())
        report = {'measured_conversations': total, 'measured_output_bytes': output_bytes,
                  'estimated_full_train_output_bytes': round(output_bytes / max(total, 1) * source_manifest['parameters']['train_size']),
                  'note': 'Extrapolated from this mixed pilot; actual output lengths can vary by source and number of turns'}
        stats = [json.loads((logdir / f'{job["split"]}-{job["shard"]:03d}-summary.json').read_text())
                 for job in pending]
        seconds = sum(item['elapsed_seconds'] for item in stats)
        tokens = sum(item['completion_tokens'] for item in stats)
        ok = sum(item['ok'] for item in stats)
        report.update(newly_generated_conversations=ok, generation_elapsed_seconds=seconds,
                      completion_tokens=tokens, completion_tokens_per_second=tokens / max(seconds, 1e-9),
                      estimated_full_train_generation_hours=seconds / max(ok, 1) * source_manifest['parameters']['train_size'] / 3600,
                      truncated_conversations=sum(item['truncated'] for item in stats))
        write_json(logdir / 'pilot_report.json', report)
        print(f'Pilot cost report: {logdir / "pilot_report.json"}', flush=True)


def main():
    args = parse_args()
    runtime, plan = make_plan(args)
    if not args.run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    execute(args, runtime, plan)


if __name__ == '__main__':
    main()
