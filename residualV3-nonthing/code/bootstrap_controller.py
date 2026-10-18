"""Own a real online teacher, then frozen bounded training and native tests.

The teacher uses the unchanged completed Parallel_TF runtime. The separately
prepared learner is frozen and checked before it executes. Only owned process
groups are stopped; other GPU PIDs are recorded, never a stop condition.
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output.resolve()
    p = json.loads((out / 'bootstrap.json').read_text())
    base = Path(p['runtime'])
    sys.path.insert(0, str(base))
    from runtime.utils import dump, sha, now, resources, environment, healthy, stop, EVAL
    from device import assignment, bind
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'launch.json').exists(), 'Preserve earlier launch evidence.'
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected, path
    identity = assignment(p['gpu'])
    assert identity['uuid'] == p['uuid']
    children = []
    server = None
    stage = 'online_teacher_startup'
    complete = False
    transfer = Path(p['transfer_root'])

    def status(state, **extra):
        dump(out / 'status.json', dict(status=state, stage=stage, gpu=p['gpu'], uuid=p['uuid'], controller_pid=os.getpid(), utc=now(), **extra))

    def interrupt(sig, frame):
        raise KeyboardInterrupt(str(sig))

    def spawn(command, name, env):
        with (out / (name + '.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=base, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        with (out / 'process_history.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(name=name, pid=process.pid, command=command, utc=now())) + '\n')
        return process

    def observe():
        free, ids = resources(p['gpu'])
        with (out / 'gpu_monitor.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(utc=now(), free_mib=free, observed_host_pids=ids, foreign_processes_untouched=True)) + '\n')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        free, ids = resources(p['gpu'])
        assert not ids and free > 75000, 'Initially requires an empty whole card.'
        dump(out / 'launch.json', dict(pid=os.getpid(), gpu_identity=identity, bootstrap_sha256=sha(out / 'bootstrap.json'), initial_free_mib=free, initial_host_pids=ids, utc=now()))
        transfer.mkdir(exist_ok=False)
        serving = bind(environment(), identity)
        serving.update(PYTHONUNBUFFERED='1', VLLM_USE_V2_MODEL_RUNNER='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8', TRITON_CACHE_DIR=str(out / 'teacher_triton'))
        command = [str(EVAL / 'bin/python'), '-B', str(base / 'scripts/launch_vllm.py'), p['target_model'], '--target-layer-ids', '2', '10', '18', '26', '34', '--hidden-states-path', str(transfer), '--', '--host', '127.0.0.1', '--port', str(p['port']), '--tensor-parallel-size', '1', '--data-parallel-size', '1', '--dtype', 'bfloat16', '--max-num-seqs', '16', '--max-model-len', '4097', '--max-num-batched-tokens', '32768', '--gpu-memory-utilization', '.43', '--kv-cache-memory-bytes', str(12*(1<<30)), '--enable-prefix-caching', '--enforce-eager', '--disable-uvicorn-access-log']
        dump(out / 'teacher_command.json', dict(command=command, gpu_identity=identity, utc=now()))
        server = spawn(command, 'teacher_vllm', serving)
        deadline = time.monotonic() + 900
        while not healthy(p['port']):
            assert server.poll() is None, 'Online teacher exited; see teacher_vllm.log.'
            assert time.monotonic() < deadline, 'Online teacher startup timeout.'
            observe()
            status('starting', server_pid=server.pid)
            time.sleep(3)
        stage = 'await_frozen_learner'
        deadline = time.monotonic() + 3600
        while not (out / 'ready.json').exists():
            assert server.poll() is None
            assert time.monotonic() < deadline, 'Learner preparation timeout.'
            observe()
            status('teacher_ready', server_pid=server.pid)
            time.sleep(5)
        ready = json.loads((out / 'ready.json').read_text())
        assert ready['cpu_validation_passed']
        for path, expected in ready['source_sha256'].items():
            assert sha(path) == expected, path
        for path, expected in ready['asset_sha256'].items():
            assert sha(path) == expected, path
        training = bind(os.environ.copy(), identity)
        training.pop('LD_PRELOAD', None)
        training.update(PYTHONPATH=f'{base}:{base}/src:{base}/hs_connectors/src', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_CACHE_DIR=str(out / 'training_inductor'), TRITON_CACHE_DIR=str(out / 'training_triton'))
        stage = 'nonthinking_basis_and_nosoft3_training'
        learner = spawn(ready['training_command'], 'training', training)
        while learner.poll() is None:
            assert server.poll() is None, 'Online teacher died during learning.'
            observe()
            status('running', server_pid=server.pid, learner_pid=learner.pid)
            time.sleep(5)
        assert learner.returncode == 0, 'Training failed; see training.log.'
        stop(server)
        server = None
        assert not list(transfer.iterdir()), 'Online hidden transport not consumed.'
        for item in ready['after_training']:
            stage = item['stage']
            process = spawn(item['command'], item['log_name'], training)
            while process.poll() is None:
                observe()
                status('running', child_pid=process.pid)
                time.sleep(5)
            assert process.returncode == 0, f'{stage} failed; see {item["log_name"]}.log.'
        for path, expected in {**p['source_sha256'], **ready['source_sha256']}.items():
            assert sha(path) == expected, path
        complete = True
        stage = 'finished'
        status('complete', results=str(out / 'RESULTS_CN.txt'))
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise
    finally:
        for process in reversed(children):
            if process.poll() is None:
                stop(process)
        remaining = []
        if transfer.exists():
            for path in transfer.iterdir():
                remaining.append(path.name)
                assert path.is_file()
                path.unlink()
            transfer.rmdir()
        dump(out / 'cleanup.json', dict(finished=complete, owned_groups_stopped=True, foreign_processes_untouched=True, temporary_files_removed=remaining, utc=now()))


if __name__ == '__main__':
    main()
