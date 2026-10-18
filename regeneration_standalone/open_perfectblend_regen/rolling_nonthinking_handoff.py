"""Replace owned thinking workers one GPU at a time, then start the new scheduler."""
import argparse
import datetime as dt
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import subprocess
import sys
import time

from dispatch import alive, owned_process, gpu_inventory, pool_progress, export_training_paths
from pool import write_json
from profiles import PROFILES, reservation_matches

HERE = Path(__file__).resolve().parent
OLD = PROFILES['thinking_ctx8192']
NEW = PROFILES['nonthinking_ctx4096']
DIRECTORY = HERE / 'nonthinking_ctx4096_handoff_20261005'
spec = importlib.util.spec_from_file_location('handoff_launcher', HERE / 'run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def until(predicate, seconds, description):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(1)
    raise TimeoutError(description)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', required=True, help='Existing owned thinking worker GPUs to replace')
    parser.add_argument('--wait-for-download', type=int, default=0, help='Seconds to wait for the pinned model download')
    args = parser.parse_args()
    indices = [int(part) for part in args.gpus.split(',')]
    state = read(DIRECTORY / 'status.json')
    lock = (DIRECTORY / 'handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.wait_for_download:
        state.update(state='waiting_for_model', handoff_pid=os.getpid())
        write_json(DIRECTORY / 'status.json', state)
        until(lambda: (HERE / 'nonthinking_model_download.json').is_file(),
              args.wait_for_download, 'Pinned model download did not finish')
    old_scheduler = state['old_scheduler_pid']
    assert owned_process(old_scheduler, [str(HERE / 'scheduler.py')]), 'Old scheduler identity changed'
    assert Path(f'/proc/{old_scheduler}/stat').read_text().split(') ', 1)[1][0] == 'T', 'Old scheduler must be paused'
    model = read(HERE / 'nonthinking_model_download.json')
    assert model['revision'] == '1a11b170eb65c8a62c80ecd01dfe22a5907298e6'
    assert (NEW.draft_model / 'model.safetensors').is_file(), 'New weight download incomplete'
    config = read(NEW.draft_model / 'config.json')
    assert config['block_size'] == 16 and config['draft_vocab_size'] == 151936
    state.update(state='rolling_handoff', target_profile=NEW.name, gpu_indices=indices,
                 replaced_gpu_indices=state.get('replaced_gpu_indices', []), model=model)

    def save(**values):
        state.update(values, updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(DIRECTORY / 'status.json', state)
        print(json.dumps({k: state.get(k) for k in ['state', 'stage', 'active_gpu',
            'replaced_gpu_indices', 'updated_at_utc']}, ensure_ascii=False), flush=True)

    try:
        for index in indices:
            if index in state['replaced_gpu_indices']:
                worker = read(NEW.worker_dir(index) / 'status.json')
                assert owned_process(worker.get('supervisor_pid'), [str(HERE / 'pilot_gpu1.py'), NEW.name])
                continue
            old_dir = OLD.worker_dir(index)
            worker = read(old_dir / 'status.json')
            old_pid, uuid = worker['supervisor_pid'], worker['gpu_uuid']
            server_pid = worker['server_pid']
            assert owned_process(old_pid, [str(HERE / 'pilot_gpu1.py'), uuid]), 'Thinking worker identity changed'
            assert owned_process(server_pid, ['vllm.entrypoints.cli.main', str(OLD.draft_model)])
            current = gpu_inventory(str(index))[0]
            assert current['uuid'] == uuid and all(p['ownership'] == 'self' for p in current['compute_processes'])
            save(active_gpu=index, stage='reserve_owned_gpu')
            guard_status = DIRECTORY / f'gpu{index}-reservation.json'
            guard_command = [sys.executable, str(HERE / 'gpu_handoff_reservation.py'),
                '--gpu-uuid', uuid, '--status-file', str(guard_status)]
            guard_env = launcher.backend.environment(SimpleNamespace(env_dir=Path(sys.executable).parent.parent, gpu=uuid))
            with (DIRECTORY / f'gpu{index}-reservation.log').open('ab') as log:
                guard = subprocess.Popen(guard_command, cwd=HERE, env=guard_env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

            def reserved():
                if guard.poll() is not None:
                    raise RuntimeError(f'Reservation exited; inspect gpu{index}-reservation.log')
                value = read(guard_status)
                return value.get('state') == 'holding' and value.get('pid') == guard.pid

            until(reserved, 90, 'GPU reservation did not initialize; thinking worker was not stopped')
            assert reservation_matches(guard.pid, uuid)
            save(stage='stop_one_thinking_worker', reservation_pid=guard.pid, old_supervisor_pid=old_pid)
            os.kill(old_pid, signal.SIGTERM)
            until(lambda: not alive(old_pid), 90, 'Thinking worker shutdown timed out')
            until(lambda: gpu_inventory(str(index))[0]['compute_pids'] == [guard.pid],
                  45, 'GPU still has unexpected compute processes; refusing replacement')
            if worker.get('output'):
                job = {'output': worker['output'], 'source': str(HERE / 'pool_500k_seed42/prompts/train' /
                    f'shard-{worker["active_shard"]:03d}.jsonl'), 'shard': worker['active_shard']}
                launcher.repair_output_tail(Path(job['output']))
                previous = read(Path(job['output']).with_suffix('.status.json'))
                count, skipped = launcher.completed_count(job), launcher.skipped_count(job)
                write_json(Path(job['output']).with_suffix('.status.json'), {**previous,
                    'state': 'complete' if count + skipped == previous['expected_conversations'] else 'partial',
                    'completed_conversations': count, 'skipped_conversations': skipped,
                    'paused_for_profile': NEW.name})
            write_json(OLD.output_root / 'progress.json', pool_progress(
                generation_dir=OLD.generation_dir, thinking=OLD.mode))
            save(stage='start_same_gpu_nonthinking')
            new_dir = NEW.worker_dir(index)
            new_dir.mkdir(exist_ok=True)
            command = [sys.executable, str(HERE / 'pilot_gpu1.py'), '--gpu', str(index),
                '--gpu-uuid', uuid, '--profile', NEW.name, '--reservation-pid', str(guard.pid)]
            with (new_dir / 'supervisor.log').open('ab') as log:
                replacement = subprocess.Popen(command, cwd=HERE.parent,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(new_dir / 'supervisor_pid.json', {'pid': replacement.pid,
                'gpu_index': index, 'command': command, 'handoff_pid': os.getpid()})
            save(stage='wait_nonthinking_generation', new_supervisor_pid=replacement.pid)

            def generating():
                value = read(new_dir / 'status.json')
                if replacement.poll() is not None or value.get('state') == 'failed':
                    raise RuntimeError(f'Non-thinking worker failed on GPU {index}: {value.get("error")}')
                return (value.get('supervisor_pid') == replacement.pid and
                        value.get('thinking') is False and value.get('max_model_len') == 4096 and
                        value.get('state') in ('tuning', 'generating') and
                        value.get('client_progress', {}).get('ok', 0) >= 1)

            until(generating, 1500, 'New service did not save a non-thinking conversation')
            export_training_paths(NEW.output_root)
            os.kill(guard.pid, signal.SIGTERM)
            guard.wait(timeout=15)
            state['replaced_gpu_indices'].append(index)
            save(stage='gpu_replaced_and_generating', reservation_pid=None)
        save(stage='start_nonthinking_scheduler')
        NEW.scheduler_dir.mkdir(exist_ok=True)
        previous_scheduler = read(NEW.scheduler_dir / 'status.json').get('scheduler_pid')
        if owned_process(previous_scheduler, [str(HERE / 'scheduler.py'), NEW.name]):
            new_scheduler_pid = previous_scheduler
        else:
            command = [sys.executable, str(HERE / 'scheduler.py'), '--gpus', 'all',
                '--max-workers', '3', '--night-max-workers', '4', '--profile', NEW.name]
            with (NEW.scheduler_dir / 'scheduler.log').open('ab') as log:
                new_scheduler = subprocess.Popen(command, cwd=HERE.parent, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            new_scheduler_pid = new_scheduler.pid
            write_json(NEW.scheduler_dir / 'scheduler_pid.json', {'pid': new_scheduler_pid, 'command': command})
        until(lambda: read(NEW.scheduler_dir / 'status.json').get('state') == 'running' and
              read(NEW.scheduler_dir / 'status.json').get('scheduler_pid') == new_scheduler_pid,
              90, 'New scheduler did not adopt replacement workers')
        assert owned_process(old_scheduler, [str(HERE / 'scheduler.py')])
        os.kill(old_scheduler, signal.SIGTERM)
        os.kill(old_scheduler, signal.SIGCONT)
        until(lambda: not alive(old_scheduler), 30, 'Old paused scheduler did not exit')
        old_progress = pool_progress(generation_dir=OLD.generation_dir, thinking=OLD.mode)
        write_json(OLD.output_root / 'paused_generation.json', {
            'state': 'paused', 'reason': 'User requested non-thinking generation first',
            'data_retained': True, 'progress': old_progress,
            'resume_profile': OLD.name, 'updated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
        save(state='complete', stage='nonthinking_scheduler_running', new_scheduler_pid=new_scheduler_pid,
             old_scheduler_paused=False, thinking_progress=old_progress,
             nonthinking_progress=pool_progress(generation_dir=NEW.generation_dir, thinking=NEW.mode))
    except BaseException as error:
        save(state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
