"""Reuse completed training and begin native evaluation on an empty whole GPU."""
import argparse
import fcntl
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path


def choose_empty(candidates):
    devices = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    tasks = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    occupied = {}
    for line in tasks.splitlines():
        uuid, pid = [part.strip() for part in line.split(',')]
        occupied.setdefault(uuid, []).append(int(pid))
    observed = {}
    for line in devices.splitlines():
        index, uuid, free = [part.strip() for part in line.split(',')]
        observed[int(index)] = dict(uuid=uuid, free_mib=int(free), host_pids=occupied.get(uuid, []))
    for index in candidates:
        item = observed[index]
        if not item['host_pids'] and item['free_mib'] >= 65536:
            return index, observed
    return None, observed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output.resolve()
    p = json.loads((out/'native_protocol.json').read_text())
    sys.path.insert(0, p['runtime'])
    from runtime.utils import dump, sha, now, stop, resources
    from device import assignment
    lock = (out/'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out/'evaluation_launch.json').exists(), 'Do not relaunch completed evidence'
    for field in ['source_files', 'asset_sha256']:
        for name, expected in p[field].items():
            assert sha(name) == expected, name
    assert json.loads(Path(p['training_summary']).read_text())['status'] == 'complete'
    dump(out/'evaluation_launch.json', dict(controller_pid=os.getpid(), utc=now(), training_restarted=False))
    active = None
    stage = 'native_evaluation'
    finished = False

    def state(status, **extras):
        dump(out/'evaluation_status.json', dict(status=status, stage=stage, controller_pid=os.getpid(),
             gpu=p.get('gpu'), uuid=p.get('uuid'), utc=now(), training_restarted=False, **extras))

    def interrupt(sig, frame):
        raise KeyboardInterrupt(str(sig))

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    try:
        gpu=p['gpu']
        identity=assignment(gpu)
        assert identity['uuid']==p['uuid']
        print(now(), 'FIXED_EVALUATION_GPU', gpu, identity['uuid'], flush=True)
        command_index = 0
        resource_retries = 0
        while command_index < len(p['evaluation_commands']):
            item = p['evaluation_commands'][command_index]
            stage = item['stage']
            log_path = out/(item['log_name']+'.log')
            with log_path.open('x') as log:
                active = subprocess.Popen(item['command'], cwd=Path(p['runtime']), env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with (out/'process_history.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(pid=active.pid, command=item['command'], utc=now()))+'\n')
            while active.poll() is None:
                free, ids = resources(gpu)
                with (out/'gpu_monitor.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(utc=now(), free_mib=free,
                        observed_host_pids=ids, foreign_processes_untouched=True))+'\n')
                state('running', child_pid=active.pid, log=str(out/(item['log_name']+'.log')))
                time.sleep(5)
            if active.returncode != 0:
                concurrency = 'C40' if stage.startswith('C40') else 'C1'
                evidence = log_path.read_text()
                for path in (out/concurrency).rglob('vllm.log'):
                    evidence += path.read_text(errors='replace')
                resource_failure = any(marker in evidence for marker in ['CUDA out of memory', 'Error in memory profiling.', 'is less than desired GPU memory utilization'])
                if not resource_failure:
                    raise AssertionError(f'{stage} failed; preserve logs')
                resource_retries += 1
                archive = out/'failed_resource_attempts'/f'{concurrency}_{resource_retries:03d}'
                archive.mkdir(parents=True, exist_ok=False)
                if (out/concurrency).exists():
                    shutil.move(str(out/concurrency), str(archive/concurrency))
                for sibling in p['evaluation_commands']:
                    if sibling['stage'].startswith(concurrency):
                        old_log = out/(sibling['log_name']+'.log')
                        if old_log.exists():
                            shutil.move(str(old_log), str(archive/old_log.name))
                dump(archive/'failure.json', dict(stage=stage, exit_code=active.returncode, utc=now(), training_restarted=False, foreign_processes_untouched=True))
                command_index = next(i for i, candidate in enumerate(p['evaluation_commands']) if candidate['stage'] == concurrency+'_smoke')
                print(now(), 'RESOURCE_RETRY', resource_retries, 'PRESERVED', archive, flush=True)
                while True:
                    free, ids = resources(gpu)
                    if free >= 65536 and not ids:
                        break
                    state('waiting_gpu_after_resource_failure', free_mib=free, host_pids=ids, resource_retries=resource_retries)
                    time.sleep(15)
                continue
            command_index += 1
        finished = True
        stage = 'finished'
        state('complete', results=str(out/'RESULTS_CN.txt'))
    except BaseException as exc:
        state('failed', error=repr(exc))
        raise
    finally:
        if active and active.poll() is None:
            stop(active)
        dump(out/'evaluation_cleanup.json', dict(finished=finished, training_unchanged=True,
             foreign_processes_untouched=True, utc=now()))


if __name__ == '__main__':
    main()
