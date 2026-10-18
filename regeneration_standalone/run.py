#!/usr/bin/env python3
"""Standalone stock DFlash regeneration. No GPU work without --run AND --gpu."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from regenlib.configs import DATASET_CONFIGS

HERE = Path(__file__).resolve().parent
DEFAULT_ENV = '/data/weijiahong/miniconda3/envs/llm_dflash_eval_v026'
MODELS = Path('/data/weijiahong/huggingface/models')
DATASETS = {
    'openmathinstruct2': 'openmath_qwen3_8b_onpolicy',
    'opencodeinstruct': 'opencode_qwen3_8b_onpolicy',
    'gsm8k': 'gsm8k_qwen3_8b',
    'humaneval': 'humaneval_qwen3_8b',
    'mt_bench': 'mt_bench_qwen3_8b',
    'open-perfectblend': 'open_perfectblend_qwen3_8b',
}
BENCHMARK_DATASETS = {'gsm8k', 'humaneval', 'mt_bench'}
BENCHMARK_SIZES = {
    'gsm8k': {'train': 7473, 'test': 1319},
    'humaneval': {'test': 164},
    'mt_bench': {'train': 80},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Actually start generation; requires --gpu')
    mode.add_argument('--dry-run', action='store_true', help='Print commands only (default)')
    p.add_argument('--gpu', help='Explicit physical GPU index or UUID; no automatic selection')
    p.add_argument('--env-dir', type=Path, default=Path(os.getenv('VLLM_ENV_DIR', DEFAULT_ENV)))
    p.add_argument('--model', type=Path, default=MODELS / 'Qwen3-8B')
    p.add_argument('--draft-model', type=Path, default=MODELS / 'Qwen3-8B-speculator.dflash')
    p.add_argument('--split', help='Raw dataset split; GSM8K defaults to train; use --split test for evaluation')
    p.add_argument('--thinking', choices=['on', 'off'], help='Explicit Qwen thinking switch; new benchmark entries default to on')
    p.add_argument('--max-samples', type=int, help='Target total unique conversations, including existing data; benchmarks default to full split, other datasets to 10000')
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--concurrency', type=int, help='Default: OpenMath 50, other datasets 40')
    p.add_argument('--gpu-memory-utilization', type=float, default=0.93)
    p.add_argument('--port', type=int, default=18081)
    p.add_argument('--sampling-params', default='{}', help='Target sampling overrides; otherwise checkpoint defaults')
    p.add_argument('--dataset-path', type=Path, help='Optional local raw HF dataset; never a regenerated answer file')
    p.add_argument('--dataset-files', type=Path, nargs='+', help='Fixed prompt JSONL files prepared by a pool builder')
    p.add_argument('--dataset-revision', help='HF revision; benchmark entries otherwise reuse download_manifest.json')
    p.add_argument('--ready-timeout', type=int, default=900)
    args = p.parse_args()
    default_split = DATASET_CONFIGS[args.dataset].split
    args.split = args.split or default_split
    if args.dataset in BENCHMARK_DATASETS:
        if args.split not in BENCHMARK_SIZES[args.dataset]:
            p.error(f'Unsupported split for {args.dataset}: {args.split}')
        args.thinking = args.thinking or 'on'
    if args.dataset == 'open-perfectblend':
        args.thinking = args.thinking or 'on'
        if not args.dataset_files:
            p.error('Open-PerfectBlend requires prepared --dataset-files; use open_perfectblend_regen/prepare.sh first')
    if args.dataset_path and args.dataset_files:
        p.error('dataset-path and dataset-files are mutually exclusive')
    if args.max_samples is None:
        args.max_samples = BENCHMARK_SIZES.get(args.dataset, {}).get(args.split, 10000)
    if args.dataset in BENCHMARK_DATASETS and args.max_samples > BENCHMARK_SIZES[args.dataset][args.split]:
        p.error('max-samples exceeds the available conversations in this benchmark split')
    if args.concurrency is None:
        args.concurrency = 50 if args.dataset == 'openmathinstruct2' else 40
    if min(args.max_samples, args.max_tokens, args.concurrency, args.ready_timeout) < 1:
        p.error('sample/token/concurrency/timeout limits must be positive')
    if not 0 < args.max_tokens <= args.max_model_len:
        p.error('Require 0 < max-tokens <= max-model-len')
    if not 0 < args.gpu_memory_utilization < 1 or not 1 <= args.port <= 65535:
        p.error('Invalid memory fraction or port')
    if args.gpu is not None and not re.fullmatch(r'(\d+|GPU-[a-zA-Z0-9-]+|MIG-[a-zA-Z0-9/-]+)', args.gpu):
        p.error('--gpu must identify exactly one GPU')
    if args.run and args.gpu is None:
        p.error('--run requires explicit --gpu; GPU availability is never inferred')
    try:
        args.sampling_params = json.loads(args.sampling_params)
    except ValueError as exc:
        p.error(str(exc))
    if not isinstance(args.sampling_params, dict):
        p.error('sampling-params must be an object')
    if args.thinking is not None:
        thinking = args.thinking == 'on'
        kwargs = args.sampling_params.get('chat_template_kwargs', {})
        if not isinstance(kwargs, dict):
            p.error('chat_template_kwargs must be an object')
        if 'enable_thinking' in kwargs and kwargs['enable_thinking'] != thinking:
            p.error('enable_thinking conflicts with --thinking')
        if args.sampling_params.get('reasoning_effort') == 'none' and thinking:
            p.error('reasoning_effort=none conflicts with thinking=on')
        if args.dataset in BENCHMARK_DATASETS or args.dataset == 'open-perfectblend':
            defaults = {'temperature': 0.6 if thinking else 0.7,
                        'top_p': 0.95 if thinking else 0.8, 'top_k': 20, 'min_p': 0.0}
            args.sampling_params = {**defaults, **args.sampling_params}
        args.sampling_params['chat_template_kwargs'] = {**kwargs, 'enable_thinking': thinking}
    reserved = {'max_tokens', 'max_completion_tokens', 'n', 'stream', 'return_token_ids',
                'model', 'messages', 'truncate_prompt_tokens'} & args.sampling_params.keys()
    if reserved:
        p.error(f'Reserved sampling keys: {sorted(reserved)}')
    for name in ['env_dir', 'model', 'draft_model', 'dataset_path']:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.dataset_files:
        args.dataset_files = [value.expanduser().resolve() for value in args.dataset_files]
    return args


def environment(args, *, client=False):
    env = os.environ.copy()
    # Do not inherit experimental patch hooks or a foreign CUDA library path.
    for name in list(env):
        if name.startswith(('VLLM_', 'RESIDUAL', 'DFLASH_')) or name in {
            'PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_LIBRARY_PATH',
            'USE_TORCH', 'USE_TF', 'USE_FLAX', 'CUDA_VISIBLE_DEVICES',
        }:
            env.pop(name, None)
    env.update({
        'PATH': str(args.env_dir / 'bin') + os.pathsep + env.get('PATH', ''),
        'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1',
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'CUDA_VISIBLE_DEVICES': '' if client else (args.gpu or '<explicit GPU required>'),
        'LD_LIBRARY_PATH': f'{args.env_dir}/cuda-compat:{args.env_dir}/lib',
        'VLLM_ENABLE_CUDA_COMPATIBILITY': '1',
        'VLLM_CUDA_COMPATIBILITY_PATH': str(args.env_dir / 'cuda-compat'),
    })
    if client:
        env.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0')
    return env


def read_config(model):
    return json.loads((model / 'config.json').read_text())


def make_plan(args):
    draft = read_config(args.draft_model)
    block_size = draft.get('block_size')
    if (draft.get('speculators_model_type') != 'dflash' or
            type(block_size) is not int or block_size < 2 or
            draft.get('draft_vocab_size') not in (32000, 151936)):
        raise ValueError('Require a supported pretrained Qwen3-8B DFlash checkpoint')
    spec = {'method': 'dflash', 'model': str(args.draft_model),
            'num_speculative_tokens': block_size - 1, 'draft_sample_method': 'greedy'}
    dataset_dir = HERE / f'{args.dataset}_regen'
    preset = 'gsm8k_test' if args.dataset == 'gsm8k' and args.split == 'test' else args.dataset
    config = DATASET_CONFIGS[preset]
    revision = args.dataset_revision
    download_manifest = HERE / 'download_manifest.json'
    if not revision and not args.dataset_path and args.dataset in BENCHMARK_DATASETS and download_manifest.is_file():
        for entry in json.loads(download_manifest.read_text()).get('datasets', []):
            if entry['hf_path'] == config.hf_path and entry['split'] == args.split:
                revision = entry['revision']
                break
    stem = DATASETS[args.dataset]
    if args.dataset in BENCHMARK_DATASETS:
        stem += f'_{args.split}_{"thinking" if args.thinking == "on" else "nonthinking"}_onpolicy'
    else:
        if args.split != config.split:
            if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.split):
                raise ValueError('Split must be a plain name when used in output filenames')
            stem += f'_{args.split}'
        if args.thinking is not None:
            stem += f'_{"thinking" if args.thinking == "on" else "nonthinking"}'
    output = dataset_dir / 'regenerated' / f'{stem}.jsonl'
    python = str(args.env_dir / 'bin/python')
    server = [python, '-m', 'vllm.entrypoints.cli.main', 'serve', str(args.model),
              '--host', '127.0.0.1', '--port', str(args.port),
              '--tensor-parallel-size', '1', '--data-parallel-size', '1',
              '--dtype', 'bfloat16', '--max-model-len', str(args.max_model_len),
              '--max-num-seqs', str(args.concurrency),
              '--gpu-memory-utilization', str(args.gpu_memory_utilization),
              '--speculative-config', json.dumps(spec), '--disable-uvicorn-access-log']
    client = [python, str(HERE / 'regenerate.py'), '--dataset', preset, '--split', args.split,
              '--model', str(args.model), '--endpoint', f'http://127.0.0.1:{args.port}/v1/chat/completions',
              '--concurrency', str(args.concurrency), '--max-tokens', str(args.max_tokens),
              '--max-model-len', str(args.max_model_len), '--sampling-params', json.dumps(args.sampling_params), '--outfile', str(output), '--resume']
    if args.dataset_path:
        client += ['--dataset-path', str(args.dataset_path)]
    if args.dataset_files:
        client += ['--dataset-files', *map(str, args.dataset_files)]
    if revision:
        client += ['--dataset-revision', revision]
    return {'dataset_dir': str(dataset_dir), 'output': str(output),
            'dataset': args.dataset, 'dataset_preset': preset, 'dataset_split': args.split,
            'dataset_hf_path': config.hf_path, 'dataset_subset': config.subset,
            'dataset_revision': revision,
            'thinking': args.thinking or 'checkpoint/template default',
            'dataset_role': 'benchmark_train_split' if args.dataset == 'gsm8k' and args.split == 'train' else ('evaluation' if args.dataset in BENCHMARK_DATASETS else 'training'),
            'server_argv': server, 'client_argv': client, 'speculative_config': spec,
            'max_samples': args.max_samples, 'max_tokens': args.max_tokens,
            'max_model_len': args.max_model_len, 'sampling_params': args.sampling_params,
            'target_generation_config': 'auto (checkpoint defaults unless explicitly overridden)',
            'target_vocabulary': 'full original target vocabulary',
            'draft_vocabulary': f'pretrained {draft["draft_vocab_size"]} rows; original checkpoint mapping',
            'custom_repair': False, 'training': False,
            'server_environment': {k: environment(args)[k] for k in [
                'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'LD_LIBRARY_PATH',
                'VLLM_ENABLE_CUDA_COMPATIBILITY', 'VLLM_CUDA_COMPATIBILITY_PATH']}}


def count_existing(path):
    """Fail on corrupt/incomplete JSONL rather than appending to a broken tail."""
    seen = set()
    if not path.exists():
        return 0
    with path.open('rb') as f:
        for lineno, line in enumerate(f, 1):
            if not line.endswith(b'\n'):
                raise ValueError(f'Incomplete JSONL tail: {path}:{lineno}; preserve and inspect it before resuming')
            row = json.loads(line)
            key = row.get('primary_id') or row.get('id')
            if not key or not row.get('input_ids') or len(row['input_ids']) != len(row.get('loss_mask', [])):
                raise ValueError(f'Invalid boundary sample at {path}:{lineno}')
            seen.add(str(key))
    return len(seen)


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def stop_group(proc):
    if proc is None:
        return
    # Also clean surviving workers when the process-group leader already exited.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def wait_ready(proc, args):
    deadline = time.monotonic() + args.ready_timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError('vLLM exited before readiness; inspect this run\'s vllm.log')
        try:
            with opener.open(f'http://127.0.0.1:{args.port}/health', timeout=2) as response:
                if response.status == 200 and proc.poll() is None:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError('vLLM readiness timed out')


def run(args, plan):
    python = args.env_dir / 'bin/python'
    for p in [python, args.env_dir / 'cuda-compat/libcuda.so.1', args.env_dir / 'lib/libcudart.so.13']:
        if not p.exists():
            raise FileNotFoundError(p)
    # Metadata and CPU-only imports; no import vllm/torch or CUDA probe.
    check = "import sys,importlib.metadata as m;sys.path.insert(0," + repr(str(HERE / '.client_deps')) + ");import datasets,aiohttp,transformers;assert m.version('vllm')=='0.26.0';assert '+cu130' in m.version('torch');print('CPU dependency check passed')"
    subprocess.run([str(python), '-c', check], env=environment(args, client=True), check=True, cwd=HERE)
    dataset_dir, output = Path(plan['dataset_dir']), Path(plan['output'])
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Record the continuation provenance without splitting the data file.
        hashes = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [HERE / 'run.py', HERE / 'regenerate.py', *sorted((HERE / 'regenlib').glob('*.py'))]}
        model_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for model in [args.model, args.draft_model]
                        for p in model.iterdir() if p.suffix in {'.json', '.jinja'}}
        contract = {k: plan[k] for k in ['speculative_config', 'max_tokens', 'max_model_len', 'sampling_params']}
        contract.update(target_model=str(args.model), dataset_path=str(args.dataset_path),
                        dataset=plan['dataset_hf_path'], subset=plan['dataset_subset'],
                        dataset_revision=plan['dataset_revision'],
                        split=args.split, thinking=plan['thinking'],
                        source_hashes=hashes, model_config_hashes=model_hashes)
        existing = count_existing(output)
        if existing >= args.max_samples:
            print(f'Already have {existing} conversations; no server started')
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        # Never attach the client to another service accidentally.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port))
        run_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'_{os.getpid()}'
        logdir = dataset_dir / 'logs' / run_id
        logdir.mkdir(parents=True, exist_ok=False)
        manifest_path = logdir / 'run_manifest.json'
        manifest = dict(plan, contract=contract, started_at=run_id, state='starting', existing=existing)
        client_cmd = plan['client_argv'] + ['--limit', str(args.max_samples-existing),
                                         '--run-manifest', str(manifest_path),
                                         '--summary-file', str(logdir / 'client_summary.json')]
        manifest['client_argv'] = client_cmd
        write_json(manifest_path, manifest)
        server = client = None
        try:
            with (logdir / 'vllm.log').open('wb') as sf, (logdir / 'client.log').open('wb') as cf:
                server = subprocess.Popen(plan['server_argv'], env=environment(args), cwd=HERE,
                                          stdout=sf, stderr=subprocess.STDOUT, start_new_session=True)
                manifest.update(server_pid=server.pid, state='starting_server')
                write_json(manifest_path, manifest)
                wait_ready(server, args)
                manifest['state'] = 'generating'
                write_json(manifest_path, manifest)
                print(f'Run logs: {logdir}', flush=True)
                client = subprocess.Popen(client_cmd, env=environment(args, client=True), cwd=HERE,
                                          stdout=cf, stderr=subprocess.STDOUT, start_new_session=True)
                # Catch server failure even if an outstanding HTTP request hangs.
                while client.poll() is None:
                    if server.poll() is not None:
                        raise RuntimeError('vLLM exited during generation')
                    time.sleep(1)
                if client.returncode:
                    raise RuntimeError(f'Client exited {client.returncode}; inspect client.log')
                final_count = count_existing(output)
                if final_count < args.max_samples:
                    raise RuntimeError(f'Only {final_count}/{args.max_samples} conversations; input exhausted or rows failed')
                manifest.update(state='complete', final_count=final_count)
        except BaseException as exc:
            manifest.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
            raise
        finally:
            stop_group(client)
            stop_group(server)
            write_json(manifest_path, manifest)


def main():
    args = parse_args()
    plan = make_plan(args)
    if not args.run:
        print(json.dumps(dict(plan, mode='dry_run_no_gpu_process'), indent=2, ensure_ascii=False))
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(args, plan)


if __name__ == '__main__':
    main()
