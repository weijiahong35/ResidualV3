"""Resume owned crashed workers with reservations and a persisted safe batch size."""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from dispatch import alive, gpu_inventory, owned_process
from pool import write_json
from profiles import PROFILES, reservation_matches

HERE = Path(__file__).resolve().parent
DIRECTORY = HERE / 'cuda_crash_recovery_20261005'
PROFILE = PROFILES['nonthinking_ctx4096']


def read(path):
    return json.loads(path.read_text())


def until(predicate, seconds, message):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(1)
    raise TimeoutError(message)


def main():
    scheduler = read(PROFILE.scheduler_dir / 'status.json')['scheduler_pid']
    assert owned_process(scheduler, [str(HERE / 'scheduler.py'), PROFILE.name])
    assert Path(f'/proc/{scheduler}/stat').read_text().split(') ', 1)[1][0] == 'T'
    guards = read(DIRECTORY / 'guards.json')
    state = {'state': 'recovering', 'pid': os.getpid(), 'scheduler_pid': scheduler, 'restored': []}

    def save(**values):
        state.update(values, updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(DIRECTORY / 'status.json', state)
        print(json.dumps(state), flush=True)

    try:
        for index in range(4):
            guard = guards[str(index)]
            assert reservation_matches(guard['pid'], guard['uuid'])
            directory = PROFILE.worker_dir(index)
            worker = read(directory / 'status.json')
            old = worker['supervisor_pid']
            # A worker that drained successfully after the live cap was lowered
            # keeps its requests, server, and shard writer uninterrupted.
            if (owned_process(old, [str(HERE / 'pilot_gpu1.py'), PROFILE.name]) and
                    worker.get('state') == 'generating' and worker.get('concurrency') == 40 and
                    owned_process(worker.get('server_pid'), ['vllm.entrypoints.cli.main']) and
                    worker.get('client_progress', {}).get('concurrency_cap') == 40):
                replacement_pid = old
            else:
                if owned_process(old, [str(HERE / 'pilot_gpu1.py'), PROFILE.name]):
                    save(stage='stop_one_owned_worker', gpu=index)
                    os.kill(old, signal.SIGTERM)
                    until(lambda: not alive(old), 90, 'Owned supervisor did not stop')
                until(lambda: gpu_inventory(guard['uuid'])[0]['compute_pids'] == [guard['pid']],
                      45, 'Unexpected compute process on reserved GPU')
                command = [sys.executable, str(HERE / 'pilot_gpu1.py'), '--gpu', str(index),
                           '--gpu-uuid', guard['uuid'], '--profile', PROFILE.name,
                           '--reservation-pid', str(guard['pid'])]
                with (directory / 'supervisor.log').open('ab') as log:
                    proc = subprocess.Popen(command, cwd=HERE.parent, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                replacement_pid = proc.pid
                write_json(directory / 'supervisor_pid.json', {'pid': replacement_pid, 'command': command})
                save(stage='wait_for_saved_output', gpu=index, worker_pid=replacement_pid)

                def generating():
                    current = read(directory / 'status.json')
                    if proc.poll() is not None:
                        raise RuntimeError(f'Worker exited on GPU {index}: {current.get("error")}')
                    return (current.get('supervisor_pid') == replacement_pid and
                            current.get('state') == 'generating' and current.get('concurrency') == 40 and
                            current.get('client_progress', {}).get('ok', 0) >= 1)

                until(generating, 900, 'Worker did not save a resumed conversation')
            assert reservation_matches(guard['pid'], guard['uuid'])
            os.kill(guard['pid'], signal.SIGTERM)
            until(lambda: not alive(guard['pid']), 20, 'Reservation did not release')
            state['restored'].append(index)
            save(stage='worker_generating', gpu=index, worker_pid=replacement_pid)
        assert owned_process(scheduler, [str(HERE / 'scheduler.py'), PROFILE.name])
        os.kill(scheduler, signal.SIGCONT)
        save(state='complete', stage='scheduler_resumed')
    except BaseException as error:
        save(state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
