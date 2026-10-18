"""Run one assigned GPU and claim unfinished shards from the shared pool."""
import argparse
from contextlib import ExitStack
import datetime as dt
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('pool_launcher', HERE / 'run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
backend = launcher.backend
from pool import write_json
from regenlib.checkpoint import repair_output_tail
from dispatch import gpu_inventory, owned_process, export_training_paths, pool_progress
from profiles import PROFILES, DEFAULT_PROFILE, add_profile_argument, reservation_matches
from concurrency_policy import select_concurrency, tuning_candidates, record_server_failure
from pool_manifest import load_pool_manifest
from worker_recovery import client_complete, should_release_server, reserve_startup
from speculation_policy import execution_record

DIRECTORY = HERE / 'gpu1_pilot_20261004'
UUID = 'GPU-250cd769-9b6a-e76d-92cb-681686776938'
PORT = 18081
GPU_CONFIGS = {1: 'GPU-250cd769-9b6a-e76d-92cb-681686776938',
               3: 'GPU-e2c0a9aa-5376-9fd2-4aff-5db3ae4260cd'}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def metrics():
    with OPENER.open(f'http://127.0.0.1:{PORT}/metrics', timeout=5) as response:
        body = response.read().decode()
    values = {}
    for line in body.splitlines():
        if not line.startswith('vllm:'):
            continue
        name = re.split('[{ ]', line, 1)[0]
        try:
            values[name] = values.get(name, 0) + float(line.rsplit(' ', 1)[1])
        except ValueError:
            pass
    return values


def gpu():
    result = subprocess.check_output(['nvidia-smi', '-i', UUID,
        '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
    utilization, used, total = map(float, result.strip().split(','))
    return {'utilization_percent': utilization, 'memory_used_mib': used, 'memory_total_mib': total}


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except (ProcessLookupError, FileNotFoundError):
        return False


def occupied_gpu(uuid):
    record = gpu_inventory(uuid)[0]
    return record['compute_pids'], record['free_mib']


def plans(gpu_index=1, shard_selection=None, include_validation=None, gpu_uuid=None, port=None,
          profile_name=DEFAULT_PROFILE):
    shard_selection = shard_selection or 'all'
    include_validation = False if include_validation is None else include_validation
    gpu_uuid = gpu_uuid or GPU_CONFIGS.get(gpu_index) or gpu_inventory(str(gpu_index))[0]['uuid']
    profile = PROFILES[profile_name]
    result = []
    for split in ['train', *(['validation'] if include_validation else [])]:
        previous = sys.argv
        try:
            sys.argv = ['run.py', '--run', '--gpu', gpu_uuid, '--port', str(port or 18080 + gpu_index),
                        '--concurrency', '128', '--split', split,
                        '--thinking', profile.thinking,
                        '--max-model-len', str(profile.max_model_len), '--max-tokens', str(profile.max_model_len),
                        '--draft-model', str(profile.draft_model), '--output-dir', str(profile.generation_dir),
                        '--shards', shard_selection if split == 'train' else 'all']
            args = launcher.parse_args()
            runtime, plan = launcher.make_plan(args)
        finally:
            sys.argv = previous
        result.append((runtime, plan))
    return result


def available_port(gpu_index):
    for candidate in range(18080 + gpu_index, 20080 + gpu_index, 100):
        try:
            with socket.socket() as check:
                check.bind(('127.0.0.1', candidate))
        except OSError:
            continue
        return candidate
    raise RuntimeError('No available local server port for assigned GPU')


class AttachedClient:
    """Monitor an already running detached client without cancelling its requests."""
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        if not process_alive(self.pid):
            self.returncode = 0
        return self.returncode

    def wait(self, timeout=None):
        deadline = time.monotonic() + timeout if timeout is not None else float('inf')
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired('attached regeneration client', timeout)
            time.sleep(.2)
        return self.returncode


def initial_job_counts(job, previous_state, adopting):
    # A live client may be midway through an append. Its supervisor handoff
    # must neither repair/truncate that file nor scan a temporarily partial row.
    # Reuse the original client baseline; live progress supplies subsequent commits.
    if adopting:
        return previous_state['existing_conversations'], previous_state.get('existing_skipped_conversations', 0)
    repair_output_tail(Path(job['output']))
    return launcher.completed_count(job), launcher.skipped_count(job)


def main():
    global DIRECTORY, UUID, PORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retune', action='store_true', help='Repeat the 40-first live tuning schedule')
    parser.add_argument('--gpu', type=int, required=True, help='GPU explicitly assigned by scheduler or user')
    parser.add_argument('--gpu-uuid', help='Pinned GPU UUID assigned by the scheduler')
    parser.add_argument('--port', type=int, help='Optional explicit server port; otherwise reuse or find a free port')
    parser.add_argument('--shards', help='Default: all unfinished training shards; evaluation is online')
    add_profile_argument(parser)
    parser.add_argument('--reservation-pid', type=int, help='Verified same-GPU reservation for rolling handoff')
    options = parser.parse_args()
    profile = PROFILES[options.profile]
    DIRECTORY = profile.worker_dir(options.gpu)
    UUID = options.gpu_uuid or gpu_inventory(str(options.gpu))[0]['uuid']
    DIRECTORY.mkdir(exist_ok=True)
    previous_server = json.loads((DIRECTORY / 'server.json').read_text()) if (DIRECTORY / 'server.json').exists() else {}
    PORT = options.port or (previous_server['port'] if previous_server.get('gpu_uuid') == UUID and
        owned_process(previous_server.get('pid'), ['vllm.entrypoints.cli.main']) else available_port(options.gpu))
    if options.gpu != 1 and not (DIRECTORY / 'concurrency_results.json').exists() and not options.retune:
        source_tuning = profile.worker_dir(1) / 'concurrency_results.json'
        if source_tuning.exists() and json.loads(source_tuning.read_text()).get('selected_concurrency') is not None:
            reused = json.loads(source_tuning.read_text())
            reused['reused_from'] = str(source_tuning)
            reused['note'] = 'Same A100 80GB and models; start from the measured 40 cap. Use --retune to repeat.'
            write_json(DIRECTORY / 'concurrency_results.json', reused)
    lock = (DIRECTORY / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = DIRECTORY / 'status.json'
    previous_state = json.loads(state_path.read_text()) if state_path.exists() else {}
    control = DIRECTORY / 'concurrency.json'
    tuning_path = DIRECTORY / 'concurrency_results.json'
    tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() and not options.retune else {}
    candidates = tuning_candidates(tuning)
    results = tuning.get('results', [])
    chosen = tuning.get('selected_concurrency', 40)
    write_json(control, {'concurrency': chosen})
    state = {'state': 'starting', 'supervisor_pid': os.getpid(), 'gpu_index': options.gpu,
             'gpu_uuid': UUID, 'port': PORT, 'concurrency': chosen, 'thinking': profile.thinking == 'on',
             'profile': profile.name, 'draft_model': str(profile.draft_model),
             'target_conversations': load_pool_manifest(HERE / 'pool_500k_seed42', prefer_base=profile.thinking == 'on')[0]['parameters']['train_size'],
             'train_shards': options.shards or 'shared queue: all unfinished shards',
             'owns_validation': False, 'generation_splits': ['train'], 'evaluation_mode': 'online',
             'scheduling': 'shared shard queue with nonblocking exclusive locks',
             'checkpoint_policy': 'fsync each generated turn; atomic conversation commits; resume IDs',
             'max_model_len': profile.max_model_len,
             'completion_budget': f'{profile.max_model_len} - rendered prompt tokens',
             'minimum_valid_tokens': 8,
             'cuda_policy': 'existing CU13 compatibility libraries; no driver installation'}
    client = None
    server_pid = None
    server_info = None
    server_ready = False
    release_requested = False
    all_processed = False
    startup_guard = None
    deferred = {}
    client_retry_policy = {"attempts_before_deferral": 3, "retry_delay_seconds": 5, "defer_seconds": 300}
    def save(**values):
        state.update(values, updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(state_path, state)
    def interrupted(sig, frame):
        nonlocal release_requested
        release_requested = True
        raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    # Adopt a verified client during scheduler handoff or supervisor recovery.
    # Its checkpoint writer and current GPU requests continue uninterrupted.
    orphan_pid = previous_state.get('client_pid')
    if orphan_pid and process_alive(orphan_pid):
        if not owned_process(orphan_pid, [str(ROOT / 'regenerate.py'), previous_state.get('output', '\0'),
                                         f'127.0.0.1:{PORT}/v1/chat/completions']):
            raise RuntimeError('Recorded client PID no longer matches; refusing adoption')
        client = AttachedClient(orphan_pid)
        state['adopted_client_pid'] = orphan_pid
    save()
    try:
        if not owned_process(previous_server.get('pid'), ['vllm.entrypoints.cli.main']):
            if not reservation_matches(options.reservation_pid, UUID):
                while True:
                    device = gpu_inventory(UUID)[0]
                    if device['idle'] and device['compatible']:
                        break
                    save(state='waiting_gpu', blocking_compute_pids=device['compute_pids'],
                         free_memory_mib=device['free_mib'])
                    time.sleep(2)
                startup_guard = reserve_startup(UUID, DIRECTORY, backend, device, save)
                options.reservation_pid = startup_guard.pid
        all_plans = plans(options.gpu, options.shards, gpu_uuid=UUID, port=PORT, profile_name=profile.name)
        if previous_state.get('output'):
            # Resume the interrupted shard before scanning already finished
            # shards, including when a serving change replaced its client.
            all_plans.sort(key=lambda entry: not any(job['output'] == previous_state['output'] for job in entry[1]['jobs']))
            for _, plan in all_plans:
                def priority(job):
                    progress_file = Path(job['output']).with_suffix('.status.json')
                    progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
                    return (job['output'] != previous_state['output'], progress.get('state') == 'complete', job['shard'])
                plan['jobs'].sort(key=priority)
        runtime = all_plans[0][0]
        server_path = DIRECTORY / 'server.json'
        if not server_path.exists() or not process_alive(json.loads(server_path.read_text())['pid']):
            while True:
                device = gpu_inventory(UUID)[0]
                occupied, free_mib = device['compute_pids'], device['free_mib']
                reserved = (reservation_matches(options.reservation_pid, UUID) and
                            occupied == [options.reservation_pid] and free_mib >= .97 * device['total_mib'])
                if (device['idle'] or reserved) and device['compatible']:
                    break
                save(state='waiting_gpu', blocking_compute_pids=occupied, free_memory_mib=free_mib,
                     note='Waiting for the assigned GPU to become free; other users processes are preserved')
                time.sleep(20)
            start_command = [sys.executable, str(HERE / 'start_gpu1_server.py'), '--gpu', str(options.gpu),
                             '--port', str(PORT), '--profile', profile.name]
            if options.reservation_pid:
                start_command += ['--reservation-pid', str(options.reservation_pid)]
            subprocess.run(start_command, check=True)
        server_info = json.loads(server_path.read_text())
        candidate_pid = server_info['pid']
        if (server_info.get('gpu_uuid') != UUID or server_info.get('port') != PORT or
            not owned_process(candidate_pid, ['vllm.entrypoints.cli.main', str(runtime.model),
                                              str(runtime.draft_model), f'--max-model-len\x00{profile.max_model_len}'])):
            raise RuntimeError('Recorded PID is not the owned Qwen server')
        server_pid = candidate_pid
        save(state='waiting_server', server_pid=server_pid)
        deadline = time.monotonic() + runtime.ready_timeout
        while True:
            if not process_alive(server_pid):
                raise RuntimeError('Owned vLLM server exited; inspect vllm.log')
            try:
                metrics()
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError('Server readiness timed out')
                time.sleep(2)
        server_info['state'] = 'ready'
        server_ready = True
        write_json(server_path, server_info)
        execution = execution_record(server_info, all_plans[0][1]['contract'],
                                     profile.generation_dir / profile.mode / 'speculation_policy.json')
        if execution:
            save(num_speculative_tokens=execution['actual_speculative_config']['num_speculative_tokens'],
                 serving_execution=execution)
        if startup_guard is not None:
            backend.stop_group(startup_guard)
            startup_guard = None
            save(startup_reservation_pid=None)
        while True:
            for runtime, plan in all_plans:
                root = Path(plan['generation_root']) / plan['thinking']
                root.mkdir(parents=True, exist_ok=True)
                contract_path = root / 'generation_contract.json'
                with contract_path.with_suffix('.lock').open('a') as contract_lock:
                    fcntl.flock(contract_lock, fcntl.LOCK_EX)
                    if contract_path.exists() and json.loads(contract_path.read_text()) != plan['contract']:
                        raise RuntimeError('Generation contract changed; do not mix output recipes')
                    if not contract_path.exists():
                        write_json(contract_path, plan['contract'])
                for job in plan['jobs']:
                    if deferred.get(job['output'], 0) > time.monotonic():
                        continue
                    path = Path(job['output'])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.with_suffix('.lock').open('a') as shard_lock:
                        try:
                            fcntl.flock(shard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            if isinstance(client, AttachedClient) and job['output'] == previous_state.get('output'):
                                raise RuntimeError('Adopted client shard is owned by another worker; refusing duplicate writes')
                            continue
                        for attempt in range(client_retry_policy['attempts_before_deferral']):
                            adopting = isinstance(client, AttachedClient) and job['output'] == previous_state.get('output')
                            existing, existing_skipped = initial_job_counts(job, previous_state, adopting)
                            if not adopting and existing + existing_skipped >= job['target_conversations']:
                                write_json(path.with_suffix('.status.json'), {'state': 'complete',
                                    'completed_conversations': existing, 'skipped_conversations': existing_skipped,
                                    'expected_conversations': job['full_shard_conversations']})
                                break
                            if shutil.disk_usage(path.parent).free < 8 * 2**30:
                                raise OSError('Less than 8 GiB free; free space and resume the same command')
                            summary = DIRECTORY / f'{job["split"]}-{job["shard"]:03d}-summary.json'
                            manifest_path = DIRECTORY / f'{job["split"]}-{job["shard"]:03d}-run.json'
                            if not adopting:
                                previous_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                                segments = previous_manifest.get('serving_execution_segments', [])
                                offset = path.stat().st_size if path.exists() else 0
                                old_server = previous_manifest.get('server', {})
                                if not segments and offset and old_server.get('command'):
                                    old_cmd = old_server['command']
                                    segments.append({'output_byte_start': 0, 'output_byte_end': offset,
                                        'server_pid': old_server['pid'],
                                        'actual_speculative_config': json.loads(old_cmd[old_cmd.index('--speculative-config') + 1])})
                                if segments:
                                    segments[-1]['output_byte_end'] = offset
                                if execution:
                                    segments.append({'output_byte_start': offset, 'server_pid': server_info['pid'],
                                        'actual_speculative_config': execution['actual_speculative_config'],
                                        'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
                                write_json(manifest_path, {**plan, 'job': job, 'server': server_info,
                                    'supervisor_pid': os.getpid(), 'live_concurrency_control': str(control),
                                    'serving_execution': execution,
                                    'serving_execution_segments': segments,
                                    'durable_checkpoint': True})
                            command = job['client_argv'] + ['--limit', str(job['target_conversations'] - existing - existing_skipped),
                                '--concurrency-control', str(control), '--summary-file', str(summary),
                                '--run-manifest', str(manifest_path)]
                            status_path = path.with_suffix('.status.json')
                            write_json(status_path, {'state': 'generating', 'completed_conversations': existing,
                                'skipped_conversations': existing_skipped,
                                'expected_conversations': job['full_shard_conversations'], 'run_manifest': str(manifest_path)})
                            if adopting:
                                existing = previous_state['existing_conversations']
                                existing_skipped = previous_state.get('existing_skipped_conversations', 0)
                            else:
                                Path(str(path) + '.progress.json').unlink(missing_ok=True)
                                with (DIRECTORY / f'{job["split"]}-{job["shard"]:03d}-client.log').open('ab') as log:
                                    client = subprocess.Popen(command, env=backend.environment(runtime, client=True),
                                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                            save(state='tuning' if candidates else 'generating',
                                 active_split=job['split'], active_shard=job['shard'], client_pid=client.pid,
                                 output=str(path), existing_conversations=existing,
                                 existing_skipped_conversations=existing_skipped)
                            step = 0
                            step_started = time.monotonic()
                            start = None
                            observations = []
                            preemptions_before = metrics().get('vllm:num_preemptions_total', 0)
                            if step < len(candidates):
                                chosen = candidates[step]
                                write_json(control, {'concurrency': chosen})
                            while client.poll() is None:
                                if step >= len(candidates):
                                    chosen = int(json.loads(control.read_text())['concurrency'])
                                if not process_alive(server_pid):
                                    raise RuntimeError('Owned server exited during generation')
                                if shutil.disk_usage(path.parent).free < 8 * 2**30:
                                    raise OSError('Disk budget reached; checkpoints saved; free space and resume')
                                current = metrics()
                                now = time.monotonic()
                                observation = {'unix_time': time.time(), 'concurrency': chosen, **gpu(),
                                    'generation_tokens': current.get('vllm:generation_tokens_total', 0),
                                    'running_requests': current.get('vllm:num_requests_running', 0),
                                    'preemptions': current.get('vllm:num_preemptions_total', 0),
                                    'draft_tokens': current.get('vllm:spec_decode_num_draft_tokens_total', 0),
                                    'accepted_tokens': current.get('vllm:spec_decode_num_accepted_tokens_total', 0)}
                                with (DIRECTORY / 'telemetry.jsonl').open('a') as telemetry:
                                    telemetry.write(json.dumps(observation) + '\n')
                                progress_path = Path(str(path) + '.progress.json')
                                progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                                completed = existing + progress.get('ok', 0)
                                skipped = existing_skipped + progress.get('skipped_context', 0)
                                save(concurrency=chosen, completed_conversations_in_shard=completed,
                                     skipped_conversations_in_shard=skipped,
                                     client_progress=progress, gpu=observation, tuning_results=results,
                                     disk_free_gib=round(shutil.disk_usage(path.parent).free / 2**30, 2))
                                write_json(status_path, {'state': 'generating', 'completed_conversations': completed,
                                    'skipped_conversations': skipped,
                                    'expected_conversations': job['full_shard_conversations'],
                                    'run_manifest': str(manifest_path), 'live_status': str(state_path)})
                                if step < len(candidates):
                                    if now - step_started >= 25 and start is None:
                                        start = (now, current)
                                    if start:
                                        observations.append(observation)
                                    if start and now - start[0] >= 65:
                                        delta = current.get('vllm:generation_tokens_total', 0) - start[1].get('vllm:generation_tokens_total', 0)
                                        tps = delta / (now - start[0])
                                        preemptions = current.get('vllm:num_preemptions_total', 0) - preemptions_before
                                        result = {'concurrency': chosen, 'measured_seconds': now - start[0],
                                            'generation_tokens_per_second': tps,
                                            'mean_gpu_utilization_percent': sum(o['utilization_percent'] for o in observations) / len(observations),
                                            'preemptions': preemptions,
                                            'score_tokens_per_second': tps if preemptions <= 2 else 0,
                                            'note': '25s settling + 65s measurement on the same shuffled shard; completed outputs are retained.'}
                                        results.append(result)
                                        tuning.update(results=results)
                                        write_json(tuning_path, tuning)
                                        print(json.dumps(result), flush=True)
                                        step += 1
                                        if step < len(candidates):
                                            chosen = candidates[step]
                                        else:
                                            stable = [r for r in results if r['score_tokens_per_second'] > 0]
                                            if not stable:
                                                raise RuntimeError('Every concurrency preempted heavily; inspect telemetry before continuing')
                                            chosen = select_concurrency(results)
                                            tuning.update(selected_concurrency=chosen,
                                                selection='smallest stable concurrency within 3% of best measured token throughput')
                                            write_json(tuning_path, tuning)
                                            save(state='generating', selected_concurrency=chosen)
                                        write_json(control, {'concurrency': chosen})
                                        step_started = now
                                        start = None
                                        observations = []
                                        preemptions_before = current.get('vllm:num_preemptions_total', 0)
                                time.sleep(5)
                            completed = launcher.completed_count(job)
                            skipped = launcher.skipped_count(job)
                            write_json(status_path, {'state': 'complete' if completed + skipped >= job['target_conversations'] else 'partial',
                                'skipped_conversations': skipped,
                                'completed_conversations': completed, 'expected_conversations': job['full_shard_conversations'],
                                'run_manifest': str(manifest_path), 'latest_summary': json.loads(summary.read_text()) if summary.exists() else {}})
                            if not client_complete(completed, skipped, job['target_conversations']):
                                if not process_alive(server_pid):
                                    raise RuntimeError('Owned server exited before the client could resume')
                                retry = {'attempt': attempt + 1, 'client_returncode': client.returncode,
                                         'completed_conversations': completed, 'skipped_conversations': skipped,
                                         'remaining_conversations': job['target_conversations'] - completed - skipped,
                                         'server_pid': server_pid, 'server_retained': True,
                                         'updated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
                                with (DIRECTORY / 'client_recovery.jsonl').open('a') as recovery_log:
                                    recovery_log.write(json.dumps({'shard': job['shard'], **retry}) + '\n')
                                    recovery_log.flush()
                                    os.fsync(recovery_log.fileno())
                                client = None
                                save(state='retrying_client', client_pid=None, client_retry=retry,
                                     client_retry_policy=client_retry_policy)
                                if attempt + 1 >= client_retry_policy['attempts_before_deferral']:
                                    deferred[job['output']] = time.monotonic() + client_retry_policy['defer_seconds']
                                    write_json(status_path, {'state': 'partial', 'completed_conversations': completed,
                                        'skipped_conversations': skipped, 'expected_conversations': job['full_shard_conversations'],
                                        'run_manifest': str(manifest_path), 'retry_pending': retry})
                                    save(state='client_retry_deferred', deferred_shard=job['shard'])
                                    break
                                time.sleep(client_retry_policy['retry_delay_seconds'])
                                continue
                            export_training_paths(root)
                            candidates = tuning_candidates(tuning)
                            deferred.pop(job['output'], None)
                            client = None
                            break
            progress = pool_progress(generation_dir=profile.generation_dir, thinking=profile.mode)
            if progress['all_processed']:
                all_processed = True
                break
            save(state='waiting_for_shard', client_pid=None,
                 remaining_conversations=progress['remaining_conversations'],
                 deferred_shards=len(deferred), note='Keep the GPU service resident while waiting for unfinished shards')
            time.sleep(5)
        save(state='complete')
    except BaseException as error:
        server_connection_failed = server_ready and isinstance(error, (urllib.error.URLError, ConnectionError, TimeoutError))
        if not isinstance(error, KeyboardInterrupt) and server_pid and (not process_alive(server_pid) or server_connection_failed):
            tuning.update(results=results)
            tuning = record_server_failure(tuning, chosen, error)
            write_json(tuning_path, tuning)
            write_json(control, {'concurrency': tuning['selected_concurrency']})
        save(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        raise
    finally:
        backend.stop_group(client)
        if startup_guard is not None:
            backend.stop_group(startup_guard)
        release_server = server_pid and should_release_server(
            release_requested, all_processed, process_alive(server_pid))
        if server_pid and not release_server:
            save(server_retained=True,
                 note='Healthy model service retained after a supervisor failure; scheduler will reattach')
        if release_server:
            # Clean owned workers even when the validated API leader has died.
            try:
                os.killpg(server_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            else:
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    try:
                        os.killpg(server_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(.5)
                try:
                    os.killpg(server_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        lock.close()


if __name__ == '__main__':
    main()
