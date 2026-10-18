"""Publish a verified append-only pool and replace only its CPU supervisors.

The API servers and detached checkpoint-writing clients keep running throughout.
Every signal is limited to a recorded process with matching UID, command and
process start time. Freezing all supervisors protects shard ownership during
the handoff; no CUDA reservation or model reload is needed.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from extend_pool import publish
from pool import DEFAULT_POOL, write_json
from pool_manifest import load_pool_manifest
from profiles import PROFILES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read_json(path):
    return json.loads(path.read_text())


def identity(pid, required):
    directory = Path(f'/proc/{pid}')
    try:
        fields = (directory / 'stat').read_text().split(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        command = (directory / 'cmdline').read_bytes().decode().split('\0')
        # During process exit /proc/stat can still report a live state even
        # though cmdline has already been cleared. Such a process cannot be
        # signalled as an owned worker and should be treated as finished.
        if not any(command):
            return None
        if directory.stat().st_uid != os.getuid() or any(token not in command for token in required):
            raise RuntimeError(f'Process {pid} does not match the authorized regeneration task')
        return {'pid': pid, 'start_ticks': fields[19], 'required': required, 'command': command}
    except FileNotFoundError:
        return None


def matching(record):
    current = identity(record['pid'], record['required'])
    if current is not None and current['start_ticks'] != record['start_ticks']:
        raise RuntimeError(f'PID {record["pid"]} has been reused')
    return current is not None


def send(record, sig):
    if matching(record):
        os.kill(record['pid'], sig)


def wait_for(predicate, timeout, description):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.5)
    raise TimeoutError(description)


def stopped(record):
    return matching(record) and Path(f'/proc/{record["pid"]}/stat').read_text().split(') ', 1)[1][0] in 'Tt'


def start(command, logfile):
    with logfile.open('ab') as handle:
        return subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)


def activate(stage, pool_dir, force_reload=False, audit_directory=None, night_max_workers=None):
    profile = PROFILES['nonthinking_ctx4096']
    if pool_dir.resolve() != DEFAULT_POOL.resolve():
        raise ValueError('Live activation is limited to the active pool')
    audit_dir = audit_directory or stage.parent
    audit_dir.mkdir(parents=True, exist_ok=True)
    report_path = audit_dir / 'activation_status.json'
    report = {'state': 'checking', 'controller_pid': os.getpid(), 'events': [], 'workers': {}}

    def save(state, **values):
        report.update(state=state, updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), **values)
        write_json(report_path, report)
        print(json.dumps({'state': state, **values}), flush=True)

    scheduler_dir = profile.scheduler_dir
    scheduler_state = read_json(scheduler_dir / 'status.json')
    previous_day_limit = scheduler_state['capacity_schedule']['day_limit']
    previous_night_limit = scheduler_state['capacity_schedule']['night_limit']
    expected_scope = ['--gpus', 'all', '--max-workers', str(previous_day_limit),
                      '--night-max-workers', str(previous_night_limit),
                      '--profile', profile.name]
    old_scheduler = identity(scheduler_state['scheduler_pid'],
        [str(HERE / 'scheduler.py'), *expected_scope])
    if old_scheduler is None or scheduler_state['profile'] != profile.name:
        raise RuntimeError('The active nonthinking scheduler is not running')
    validation = read_json(stage.parent / 'plan_validation.json')
    for mode in ['nonthinking', 'thinking']:
        contract = pool_dir / f'regenerated_ctx{4096 if mode == "nonthinking" else 8192}' / mode / 'generation_contract.json'
        if hashlib.sha256(contract.read_bytes()).hexdigest() != validation[mode]['contract_file_sha256']:
            raise ValueError(f'{mode} generation contract changed since dry-run validation')
    extended, _, _ = load_pool_manifest(stage)
    target = extended['parameters']['train_size']
    if validation['nonthinking']['train_conversations'] != target:
        raise ValueError('The verified plan and prepared manifest have different targets')
    if not validation['nonthinking']['existing_generation_contract_identical']:
        raise ValueError('The new plan would change the existing generation recipe')

    paused, new_children, new_scheduler = [], [], None
    requested_night_limit = previous_night_limit if night_max_workers is None else night_max_workers
    scheduler_command = [sys.executable, str(HERE / 'scheduler.py'), '--gpus', 'all',
        '--max-workers', str(previous_day_limit), '--night-max-workers', str(requested_night_limit),
        '--profile', profile.name]
    report['day_max_workers'] = previous_day_limit
    report['night_max_workers'] = requested_night_limit
    save('pausing_cpu_supervisors', old_scheduler=old_scheduler, target_conversations=target)
    try:
        send(old_scheduler, signal.SIGSTOP)
        paused.append(old_scheduler)
        wait_for(lambda: stopped(old_scheduler), 10, 'Scheduler did not pause')
        scheduler_state = read_json(scheduler_dir / 'status.json')
        workers = {}
        for key in sorted(scheduler_state['workers'], key=int):
            index = int(key)
            directory = profile.worker_dir(index)
            state = read_json(directory / 'status.json')
            old = identity(state['supervisor_pid'], [str(HERE / 'pilot_gpu1.py'),
                '--gpu', str(index), '--gpu-uuid', state['gpu_uuid'], '--profile', profile.name])
            if old is None or state.get('state') != 'generating':
                raise RuntimeError(f'Worker {index} is not stable for live adoption')
            send(old, signal.SIGSTOP)
            paused.append(old)
            wait_for(lambda: stopped(old), 10, f'Worker {index} did not pause')
            # Read the last status after the supervisor is definitely stopped.
            state = read_json(directory / 'status.json')
            server = identity(state['server_pid'], ['vllm.entrypoints.cli.main'])
            client = identity(state['client_pid'], [str(ROOT / 'regenerate.py'), state['output'],
                f'http://127.0.0.1:{state["port"]}/v1/chat/completions'])
            if server is None or client is None:
                raise RuntimeError(f'Worker {index} server/client exited before adoption')
            workers[index] = {'old_supervisor': old, 'server': server, 'client': client, 'before': state}
        report['workers'] = {str(k): v for k, v in workers.items()}
        write_json(audit_dir / 'before_activation.json', scheduler_state)
        save('publishing_verified_pool')
        publish(pool_dir, stage)
        for index, record in workers.items():
            state = record['before']
            if not force_reload and state.get('target_conversations') == target:
                # A resumed activation may find an already upgraded worker.
                # Keep it frozen with its shard lock instead of replacing it.
                record['new_supervisor'] = record['old_supervisor']
                record['after_adoption'] = state
                save('worker_already_expanded', gpu_index=index,
                     preserved_supervisor_pid=state['supervisor_pid'])
                continue
            # SIGTERM would execute the old supervisor's cleanup and stop both
            # its client and API server. Kill only the paused CPU supervisor.
            send(record['old_supervisor'], signal.SIGKILL)
            wait_for(lambda: not matching(record['old_supervisor']), 10,
                     f'Old supervisor {index} did not release its lock')
            command = [sys.executable, str(HERE / 'pilot_gpu1.py'), '--gpu', str(index),
                       '--gpu-uuid', state['gpu_uuid'], '--profile', profile.name]
            directory = profile.worker_dir(index)
            child = start(command, directory / 'supervisor.log')
            new_children.append(child)
            write_json(directory / 'supervisor_pid.json', {'pid': child.pid, 'gpu_index': index,
                'command': command, 'handoff': str(report_path)})

            def adopted():
                if child.poll() is not None:
                    raise RuntimeError(f'New worker {index} exited; inspect supervisor.log')
                current = read_json(directory / 'status.json')
                if current.get('supervisor_pid') != child.pid or current.get('state') != 'generating':
                    return False
                if current.get('target_conversations') != target or current.get('server_pid') != state['server_pid']:
                    raise RuntimeError('Worker did not reuse its original server and expanded pool')
                if matching(record['client']) and current.get('client_pid') != state['client_pid']:
                    raise RuntimeError('Live checkpoint client was not adopted')
                if not matching(record['server']):
                    raise RuntimeError('Original API server stopped during the handoff')
                record['after_adoption'] = current
                return True

            wait_for(adopted, 180, f'New worker {index} did not adopt the existing processes')
            # /proc/cmdline can briefly be empty immediately after exec. The
            # ready worker status ensures Python has reached the adoption loop.
            new_record = identity(child.pid, [str(HERE / 'pilot_gpu1.py'), '--gpu', str(index),
                                             '--profile', profile.name])
            if new_record is None:
                raise RuntimeError(f'Adopted worker {index} exited before it could be paused')
            record['new_supervisor'] = new_record
            send(new_record, signal.SIGSTOP)
            paused.append(new_record)
            wait_for(lambda: stopped(new_record), 10, f'New worker {index} did not pause')
            save('worker_adopted', gpu_index=index, new_supervisor_pid=child.pid,
                 preserved_server_pid=state['server_pid'], preserved_client_pid=state['client_pid'])

        send(old_scheduler, signal.SIGKILL)
        wait_for(lambda: not matching(old_scheduler), 10, 'Old scheduler did not release its lock')
        for record in paused:
            send(record, signal.SIGCONT)
        new_scheduler = start(scheduler_command, scheduler_dir / 'scheduler.log')
        write_json(scheduler_dir / 'scheduler_pid.json', {'pid': new_scheduler.pid,
            'command': scheduler_command, 'handoff': str(report_path)})

        def scheduler_adopted():
            if new_scheduler.poll() is not None:
                raise RuntimeError('New scheduler exited; inspect scheduler.log')
            state = read_json(scheduler_dir / 'status.json')
            if state.get('scheduler_pid') != new_scheduler.pid or state.get('state') not in (
                    'running', 'waiting_for_second_gpu'):
                return False
            if state['target_conversations'] != target or state['progress']['total_shards'] != validation['nonthinking']['train_shards']:
                raise RuntimeError('Scheduler did not adopt the expanded pool')
            for index, record in workers.items():
                worker = state['workers'].get(str(index), {})
                if worker.get('supervisor_pid') != record['new_supervisor']['pid']:
                    return False
                if worker.get('server_pid') != record['server']['pid']:
                    raise RuntimeError('Scheduler worker uses a different API server')
            report['after_scheduler'] = state
            return True

        wait_for(scheduler_adopted, 60, 'Scheduler did not adopt the live supervisors')
        for mode in ['nonthinking', 'thinking']:
            contract = pool_dir / f'regenerated_ctx{4096 if mode == "nonthinking" else 8192}' / mode / 'generation_contract.json'
            assert hashlib.sha256(contract.read_bytes()).hexdigest() == validation[mode]['contract_file_sha256']
        save('active', scheduler_pid=new_scheduler.pid,
             train_shards=validation['nonthinking']['train_shards'],
             original_api_servers_preserved=True, generation_contracts_unchanged=True)
    except BaseException as error:
        save('activation_failed', error=repr(error))
        raise
    finally:
        # Unfreeze only identities from this task. On a failed handoff the
        # existing scheduler or its replacement can resume the saved work.
        for record in paused:
            try:
                send(record, signal.SIGCONT)
            except RuntimeError as error:
                print(f'Skipping changed process identity during recovery: {error}', flush=True)
        if not matching(old_scheduler) and new_scheduler is None:
            recovery = start(scheduler_command, scheduler_dir / 'scheduler.log')
            write_json(scheduler_dir / 'scheduler_pid.json', {'pid': recovery.pid,
                'command': scheduler_command, 'recovery': str(report_path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool-dir', type=Path, default=DEFAULT_POOL)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--reload-workers', action='store_true', help='Reload CPU supervisors while retaining their live API servers/clients')
    parser.add_argument('--audit-dir', type=Path, help='Separate audit directory for a CPU-only supervisor update')
    parser.add_argument('--night-max-workers', type=int, choices=[1, 2, 3, 4, 5, 6],
                        help='Change the night limit during CPU scheduler handoff; default: preserve the active limit')
    args = parser.parse_args()
    activate(args.stage.resolve(), args.pool_dir.resolve(), args.reload_workers,
             args.audit_dir.resolve() if args.audit_dir else None, args.night_max_workers)


if __name__ == '__main__':
    main()
