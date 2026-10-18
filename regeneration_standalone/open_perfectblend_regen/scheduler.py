#!/usr/bin/env python3
"""Use authorized idle GPUs for a shared, resumable Open-PerfectBlend queue."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dispatch import DEFAULT_POOL, gpu_inventory, alive, owned_process, pool_progress, verify_outputs, export_training_paths
from pool import write_json
from profiles import PROFILES, add_profile_argument
from worker_recovery import refill_poll_delay
BEIJING = ZoneInfo('Asia/Shanghai')
NIGHT_END_HOUR = 8
NIGHT_WINDOW = f'00:00-{NIGHT_END_HOUR:02d}:00'
REDUCTION_LEAD_SECONDS = 60
DAY_MIN_WORKERS = 2


def capacity_at(now=None, day_limit=3, night_limit=6):
    """Daily Beijing window; reserve one minute for releasing excess GPU services."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        raise ValueError('Capacity policy requires a timezone-aware time')
    local = now.astimezone(BEIJING)
    day_start = local.replace(hour=NIGHT_END_HOUR, minute=0, second=0, microsecond=0)
    drain_start = day_start - dt.timedelta(seconds=REDUCTION_LEAD_SECONDS)
    night = local.hour < NIGHT_END_HOUR
    draining = night and local >= drain_start and night_limit > day_limit
    permitted = night_limit if night else day_limit
    effective = day_limit if draining else permitted
    if night:
        next_change = drain_start if local < drain_start and night_limit > day_limit else day_start
        next_limit = day_limit
    else:
        next_change = (local + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        next_limit = night_limit
    return {'timezone': 'Asia/Shanghai', 'local_time': local.isoformat(),
            'night_window': NIGHT_WINDOW, 'day_limit': day_limit, 'night_limit': night_limit,
            'permitted_limit': permitted, 'effective_limit': effective, 'draining_before_daytime': draining,
            'reduction_lead_seconds': REDUCTION_LEAD_SECONDS,
            'next_change_at_local': next_change.isoformat(), 'next_limit': next_limit,
            'seconds_until_next_change': max(0.0, (next_change - local).total_seconds())}


def poll_delay(poll_seconds, schedule):
    return min(poll_seconds, max(.1, schedule['seconds_until_next_change']))


def allocation_at(schedule, devices, worker_indices):
    """Use the full daily limit; an idle GPU reserve is no longer required."""
    workers = set(worker_indices)
    idle = sorted(device['index'] for device in devices
                  if device['idle'] and device['compatible'] and device['index'] not in workers)
    daytime = not (dt.datetime.fromisoformat(schedule['local_time']).hour < NIGHT_END_HOUR
                   and not schedule['draining_before_daytime'])
    limit = schedule['effective_limit']
    minimum = min(DAY_MIN_WORKERS, limit)
    return {'phase': 'daytime' if daytime else 'night',
            'minimum_retained_workers': minimum, 'target_workers': limit,
            'idle_gpu_indices': idle, 'idle_gpus': len(idle),
            'reserved_idle_gpus_above_minimum': 0,
            'release_worker_when_all_gpus_busy': False}


def gpu_choices(value):
    if value == 'all':
        return value
    try:
        values = sorted({int(part) for part in value.split(',')})
    except ValueError as error:
        raise argparse.ArgumentTypeError('Use all or comma-separated GPU indices') from error
    if not values or min(values) < 0:
        raise argparse.ArgumentTypeError('GPU indices must be nonnegative')
    return ','.join(map(str, values))


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', required=True, type=gpu_choices,
                        help='Authorized scope: all or comma-separated indices')
    parser.add_argument('--poll-seconds', type=float, default=20)
    parser.add_argument('--max-workers', type=int, choices=[2, 3, 4], default=3,
                        help=f'Daytime maximum outside Beijing {NIGHT_WINDOW}; retain 2, default: 3')
    parser.add_argument('--night-max-workers', type=int, choices=[1, 2, 3, 4, 5, 6], default=6,
                        help=f'Beijing {NIGHT_WINDOW} GPU limit; default: 6')
    add_profile_argument(parser)
    options = parser.parse_args()
    if options.poll_seconds < 1 or options.max_workers is not None and options.max_workers < 1:
        parser.error('poll-seconds and max-workers must be positive')
    profile = PROFILES[options.profile]
    directory = profile.scheduler_dir
    directory.mkdir(exist_ok=True)
    lock = (directory / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous_state = read_json(directory / 'status.json')
    root = profile.output_root
    root.mkdir(parents=True, exist_ok=True)
    state = {'state': 'starting', 'scheduler_pid': os.getpid(), 'gpu_scope': options.gpus,
             'max_workers_limit': capacity_at(day_limit=options.max_workers, night_limit=options.night_max_workers)['effective_limit'],
             'generation_splits': ['train'], 'evaluation_mode': 'online',
             'target_conversations': pool_progress(generation_dir=profile.generation_dir, thinking=profile.mode)['splits']['train']['expected_conversations'],
             'scheduling': 'idle GPU discovery + shared unfinished shard queue',
             'profile': profile.name, 'thinking': profile.thinking == 'on',
             'draft_model': str(profile.draft_model),
             'concurrency_per_gpu': '40-first adaptive tuning', 'max_model_len': profile.max_model_len,
             'completion_budget': f'{profile.max_model_len} - rendered prompt tokens',
             'completion_file': str(root / 'completion.json'),
             'gpu_inventory': previous_state.get('gpu_inventory', []),
             'workers': previous_state.get('workers', {})}
    children = {}
    cooldowns = {}
    stopping = set()
    completion = False
    release_requested = False

    def save(**values):
        schedule = capacity_at(day_limit=options.max_workers, night_limit=options.night_max_workers)
        state.update(values, updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        allocation = allocation_at(schedule, state.get('gpu_inventory', []),
                                   map(int, state.get('workers', {})))
        state.update(max_workers_limit=allocation['target_workers'], capacity_schedule=schedule,
                     allocation=allocation)
        write_json(directory / 'status.json', state)

    def interrupt(sig, frame):
        nonlocal release_requested
        release_requested = True
        raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    save()
    write_json(directory / 'gpu_policy.json', {'timezone': 'Asia/Shanghai',
        'daily_windows': [{'start': '00:00', 'end': f'{NIGHT_END_HOUR:02d}:00', 'max_workers': options.night_max_workers},
                          {'start': f'{NIGHT_END_HOUR:02d}:00', 'end': '24:00', 'max_workers': options.max_workers}],
        'daytime_policy': {'minimum_retained_workers': DAY_MIN_WORKERS,
                           'maximum_workers': options.max_workers,
                           'idle_gpus_required_to_add_third': 1,
                           'reserved_idle_gpus_above_minimum': 0,
                           'release_extra_worker_when_no_idle_gpu': False},
        'only_idle_gpus': True, 'reduction_lead_seconds': REDUCTION_LEAD_SECONDS})
    write_json(root / 'regeneration_scope.json', {'generation_splits': ['train'],
        'evaluation_mode': 'online', 'validation_regeneration': False,
        'target_conversations': state['target_conversations'],
        'target_shards': pool_progress(generation_dir=profile.generation_dir, thinking=profile.mode)['total_shards']})
    try:
        while True:
            progress = pool_progress(generation_dir=profile.generation_dir, thinking=profile.mode)
            write_json(root / 'progress.json', progress)
            if progress['all_processed']:
                save(state='verifying', progress=progress)
                try:
                    verified = verify_outputs(generation_dir=profile.generation_dir, thinking=profile.mode,
                                              max_model_len=profile.max_model_len)
                except RuntimeError:
                    time.sleep(2)
                    continue
                if not verified['all_processed']:
                    raise RuntimeError('Counters say complete but output ID coverage is incomplete')
                export_training_paths(root)
                write_json(root / 'completion.json', verified)
                save(state=verified['state'], verification=verified, progress=progress)
                completion = True
                break
            (root / 'completion.json').unlink(missing_ok=True)
            try:
                devices = gpu_inventory(options.gpus)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                # A slow query, including one paused during a CPU handoff,
                # must not terminate independent checkpoint-writing workers.
                # Do not allocate from stale idle-card observations.
                save(state='gpu_inventory_retry', progress=progress,
                     gpu_inventory_error=repr(error), workers_retained=True)
                time.sleep(2)
                continue
            workers = {}
            reusable_servers = {}
            for device in devices:
                index = device['index']
                worker_dir = profile.worker_dir(index)
                server = read_json(worker_dir / 'server.json')
                reusable_servers[index] = server.get('gpu_uuid') == device['uuid'] and owned_process(server.get('pid'),
                    ['vllm.entrypoints.cli.main', str(HERE.parent.parent.parent / 'huggingface/models/Qwen3-8B'),
                     str(profile.draft_model.resolve()), f'--max-model-len\x00{profile.max_model_len}'])
                worker_state = read_json(worker_dir / 'status.json')
                pid = worker_state.get('supervisor_pid')
                if owned_process(pid, [str(HERE / 'pilot_gpu1.py')]):
                    if worker_state.get('profile') != profile.name:
                        raise RuntimeError('Worker belongs to a different generation profile')
                    if worker_state.get('scheduling') != 'shared shard queue with nonblocking exclusive locks':
                        raise RuntimeError('Legacy fixed-shard supervisor still active; perform the verified handoff first')
                    workers[index] = worker_state
                previous = children.get(index)
                if previous is not None and previous.poll() is not None:
                    if previous.returncode:
                        cooldowns[index] = time.monotonic() + 60
                    del children[index]
            pending_shards = sum(not shard['complete'] for shard in progress['shards'])
            schedule = capacity_at(day_limit=options.max_workers, night_limit=options.night_max_workers)
            current_limit = allocation_at(schedule, devices, workers)['target_workers']
            if len(workers) > current_limit:
                keep = sorted(workers, key=lambda index: (
                    workers[index].get('state') != 'generating',
                    workers[index].get('active_split') != 'train',
                    workers[index].get('active_shard', 999), index))[:current_limit]
                for index, worker in workers.items():
                    pid = worker['supervisor_pid']
                    if index not in keep and pid not in stopping and owned_process(pid, [str(HERE / 'pilot_gpu1.py')]):
                        os.kill(pid, signal.SIGTERM)
                        stopping.add(pid)
                        print(json.dumps({'event': 'worker_release_requested', 'gpu': index,
                                          'pid': pid, 'target_workers': current_limit,
                                          'retained_gpu_indices': keep}), flush=True)
                save(state='reducing_workers', workers={str(index): worker for index, worker in workers.items()},
                     retained_gpu_indices=keep, progress=progress, gpu_inventory=devices)
                time.sleep(2)
                continue
            max_workers = min(current_limit, pending_shards)
            disk_free = shutil.disk_usage(root).free / 2**30
            if disk_free >= 8:
                for device in sorted(devices, key=lambda device: (not reusable_servers[device['index']], device['index'])):
                    live_schedule = capacity_at(day_limit=options.max_workers, night_limit=options.night_max_workers)
                    live_limit = allocation_at(live_schedule, devices, workers)['target_workers']
                    if len(workers) >= live_limit:
                        break
                    index = device['index']
                    if index in workers or index in children or len(workers) >= max_workers:
                        continue
                    if cooldowns.get(index, 0) > time.monotonic() and not reusable_servers[index]:
                        continue
                    worker_dir = profile.worker_dir(index)
                    owned_server = reusable_servers[index]
                    if not device['compatible'] or not (device['idle'] or owned_server):
                        continue
                    if len(workers) >= DAY_MIN_WORKERS and allocation_at(live_schedule, devices, workers)['phase'] == 'daytime':
                        # Recheck the assigned card immediately before adding another worker.
                        devices = gpu_inventory(options.gpus)
                        current_device = next((entry for entry in devices if entry['index'] == index), None)
                        live_schedule = capacity_at(day_limit=options.max_workers, night_limit=options.night_max_workers)
                        if (allocation_at(live_schedule, devices, workers)['target_workers'] <= len(workers)
                                or current_device is None
                                or not current_device['compatible']
                                or not (current_device['idle'] or owned_server)):
                            continue
                    worker_dir.mkdir(exist_ok=True)
                    command = [sys.executable, str(HERE / 'pilot_gpu1.py'), '--gpu', str(index),
                               '--gpu-uuid', device['uuid'], '--profile', profile.name]
                    with (worker_dir / 'supervisor.log').open('ab') as log:
                        child = subprocess.Popen(command, cwd=HERE.parent, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    children[index] = child
                    workers[index] = {'supervisor_pid': child.pid, 'gpu_index': index, 'state': 'starting'}
                    write_json(worker_dir / 'supervisor_pid.json', {'pid': child.pid, 'gpu_index': index,
                                                                 'command': command, 'scheduler_pid': os.getpid()})
                    print(json.dumps({'event': 'worker_started', 'gpu': index, 'pid': child.pid}), flush=True)
            required = min(DAY_MIN_WORKERS, current_limit, pending_shards)
            shortfall = max(0, required - len(workers))
            status = ('blocked_disk' if disk_free < 8 else
                      'waiting_for_second_gpu' if shortfall else
                      'running' if workers else 'waiting_for_idle_gpu')
            save(state=status,
                 workers={str(index): worker for index, worker in workers.items()},
                 gpu_inventory=devices, disk_free_gib=round(disk_free, 2), progress=progress,
                 minimum_required_workers=required, minimum_worker_shortfall=shortfall,
                 minimum_worker_status='waiting for an authorized idle GPU; occupied cards cannot be reclaimed'
                     if shortfall else 'minimum met',
                 refill_poll_seconds=refill_poll_delay(options.poll_seconds, len(workers), required))
            time.sleep(poll_delay(refill_poll_delay(options.poll_seconds, len(workers), required),
                capacity_at(day_limit=options.max_workers, night_limit=options.night_max_workers)))
    except BaseException as error:
        save(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        raise
    finally:
        if not completion and release_requested:
            for worker in state.get('workers', {}).values():
                pid = worker.get('supervisor_pid')
                if owned_process(pid, [str(HERE / 'pilot_gpu1.py')]):
                    os.kill(pid, signal.SIGTERM)
        lock.close()


if __name__ == '__main__':
    main()
