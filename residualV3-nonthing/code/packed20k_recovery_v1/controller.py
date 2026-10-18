"""Hold the old automatic tests, train a packed 20k arm, then test both."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

RECOVERY_CODE = Path(__file__).resolve().parent
CODE = RECOVERY_CODE.parent / 'packed20k_split_v3'
sys.path.insert(0, str(CODE / 'runtime_bundle'))
from runtime.utils import dump, sha, now, environment, healthy, stop, members, EVAL
from device import assignment, bind

PYTHON = '/data/weijiahong/miniconda3/envs/dflash/bin/python'


def process_state(pid):
    try:
        data = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return data[0], int(data[19])
    except FileNotFoundError:
        return None, None


def inventory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    tasks = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    occupied = {}
    for row in tasks.splitlines():
        uuid, pid = [value.strip() for value in row.split(',')]
        occupied.setdefault(uuid, []).append(int(pid))
    result = {}
    for row in rows.splitlines():
        index, uuid, free, utilization = [value.strip() for value in row.split(',')]
        result[int(index)] = dict(uuid=uuid, free_mib=int(free), utilization=int(utilization), host_pids=occupied.get(uuid, []))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    p = json.loads((out / 'plan.json').read_text())
    baseline = Path(p['baseline_run'])
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = out / 'recovery_20261007_v1'
    recovery = json.loads((attempt / 'recovery_protocol.json').read_text())
    assert not (attempt / 'launch.json').exists(), 'Do not duplicate recovery'
    for pid in recovery['previous_owned_pids']:
        assert process_state(pid)[0] in (None, 'Z'), f'Previous owned process still alive: {pid}'
    for path, expected in recovery['source_sha256'].items():
        assert sha(path) == expected, path
    assert sha(out / 'training_protocol.json') == recovery['training_protocol_sha256']
    assert sha(out / 'training/recovery_checkpoints/resume_latest.pt') == recovery['checkpoint_sha256']
    assert not (out / 'training/summary.json').exists()
    ready = json.loads((out / 'ready.json').read_text())
    assert sha(out / 'plan.json') == ready['plan_sha256']
    for field in ['source_files', 'asset_sha256']:
        for path, expected in ready[field].items():
            assert sha(path) == expected, path
    assert p['pack_tokens'] == 65536 and p['max_anchors'] == 1536
    assert p['epochs'] == 3 and p['gradient_accumulation_packs'] == 1
    assert json.loads((baseline / 'evaluation_hold_20261006.json').read_text())['evaluation_allowed'] is False
    dump(attempt / 'launch.json', dict(controller_pid=os.getpid(), plan_sha256=sha(out / 'plan.json'), recovery_protocol_sha256=sha(attempt / 'recovery_protocol.json'), utc=now()))
    active = server = None
    owned = []
    transfer = Path(p['transfer_root'])
    baseline_released = False
    stage = 'waiting_low_load_gpu_pair'
    completed = False

    def state(status, **extra):
        dump(out / 'status.json', dict(status=status, stage=stage, controller_pid=os.getpid(),
            baseline_evaluation_held=True, full_data_training_deferred=True, recovery_attempt=str(attempt), utc=now(), **extra))

    def release_baseline_if_done():
        nonlocal baseline_released
        if baseline_released:
            return True
        summary = baseline / 'training/summary.json'
        if not summary.exists():
            learner_state, start = process_state(p['baseline_learner_pid'])
            assert start == p['baseline_learner_start_ticks'] and learner_state not in (None, 'Z'), 'Baseline stopped before completion; no evaluation allowed'
            return False
        report = json.loads(summary.read_text())
        assert report['status'] == 'complete' and report['steps'] == 60000
        learner_state, start = process_state(p['baseline_learner_pid'])
        if start == p['baseline_learner_start_ticks'] and learner_state not in (None, 'Z'):
            return False  # Let native_protocol export and hidden cleanup finish.
        assert (baseline / 'native_protocol.json').exists()
        pid = p['baseline_controller_pid']
        controller_state, start = process_state(pid)
        if start == p['baseline_controller_start_ticks'] and controller_state not in (None, 'Z'):
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            assert str(baseline) in command and 'bootstrap_controller.py' in command
            assert controller_state == 'T', 'Old auto-evaluation controller must stay held'
            # A pending TERM is delivered on CONT before the old controller can
            # advance. Its existing finally block cleans up only its own server.
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
            deadline = time.monotonic() + 60
            while process_state(pid)[0] not in (None, 'Z'):
                assert time.monotonic() < deadline, 'Old controller cleanup timeout'
                time.sleep(.5)
        for candidate in baseline.glob('single_card_C*_formal.log'):
            raise RuntimeError(f'Unexpected early formal evaluation: {candidate}')
        dump(out / 'baseline_training_complete_evaluation_held.json', dict(training_summary_sha256=sha(summary),
            auto_controller_intentionally_closed_after_training=True, no_training_interrupted=True,
            old_weights_and_optimizer_retained=True, native_evaluation_not_started=True, utc=now()))
        baseline_released = True
        print(now(), 'BASELINE_COMPLETE', 'old automatic tests held; target server released', flush=True)
        return True

    def wait_pair():
        last = None
        while True:
            release_baseline_if_done()
            devices = inventory()
            candidates = []
            for learner_gpu in p['gpu_preference']:
                learner = devices[learner_gpu]
                if learner['free_mib'] < p['learner_min_free_mib'] or learner['utilization'] > p['maximum_start_utilization']:
                    continue
                for teacher_gpu in p['gpu_preference']:
                    teacher = devices[teacher_gpu]
                    if teacher_gpu == learner_gpu or teacher['free_mib'] < p['teacher_min_free_mib'] or teacher['utilization'] > p['maximum_start_utilization']:
                        continue
                    score = (learner['utilization'] + teacher['utilization'], -learner['free_mib'], -teacher['free_mib'])
                    candidates.append((score, learner_gpu, teacher_gpu))
            if candidates:
                _, learner_gpu, teacher_gpu = min(candidates)
                return learner_gpu, teacher_gpu, devices
            state('waiting', gpu_inventory=devices, shared_gpu_allowed=True)
            key = tuple((gpu, tuple(row['host_pids'])) for gpu, row in devices.items())
            if key != last:
                print(now(), stage, 'waiting sufficient free memory and low utilization', flush=True)
                last = key
            time.sleep(5)

    def wait_evaluation_card():
        while True:
            devices = inventory()
            eligible = [gpu for gpu in p['gpu_preference'] if devices[gpu]['free_mib'] >= 67584]
            if eligible:
                # Prefer a clean card for speed, otherwise record contamination.
                gpu = min(eligible, key=lambda index: (bool(devices[index]['host_pids']), devices[index]['utilization'], -devices[index]['free_mib']))
                return gpu, devices
            state('waiting', gpu_inventory=devices, reason='actual native evaluation memory requirement')
            time.sleep(5)

    def spawn(command, name, env):
        with (attempt / (name + '.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owned.append(process)
        with (out / 'process_history.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(name=name, pid=process.pid, command=command, utc=now())) + '\n')
        return process

    def interrupt(sig, frame):
        raise KeyboardInterrupt(str(sig))

    def monitor(gpu, processes):
        row = inventory()[gpu]
        own_pids = set()
        for process in processes:
            own_pids.update(members(process.pid))
        row['foreign_pids'] = sorted(set(row['host_pids']) - own_pids)
        with (out / 'gpu_monitor.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(utc=now(), gpu=gpu, observed=row, foreign_processes_untouched=True)) + '\n')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        learner_gpu, teacher_gpu, devices = wait_pair()
        identity = assignment(learner_gpu)
        teacher_identity = assignment(teacher_gpu)
        env = bind(environment(), teacher_identity)
        env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8', TRITON_CACHE_DIR=str(out / 'teacher_triton'), VLLM_USE_V2_MODEL_RUNNER='0')
        transfer.mkdir(exist_ok=False)
        cmd = [str(EVAL / 'bin/python'), '-B', str(CODE / 'runtime_bundle/scripts/launch_vllm.py'), p['target_model'],
            '--target-layer-ids', '2', '10', '18', '26', '34', '--hidden-states-path', str(transfer), '--',
            '--host', '127.0.0.1', '--port', str(p['port']), '--tensor-parallel-size', '1', '--dtype', 'bfloat16',
            '--max-num-seqs', '16', '--max-model-len', '4097', '--max-num-batched-tokens', '32768',
            '--gpu-memory-utilization', '.43', '--kv-cache-memory-bytes', str(12 * (1 << 30)),
            '--enable-prefix-caching', '--enforce-eager', '--disable-uvicorn-access-log']
        dump(attempt / 'hardware.json', dict(learner=identity, teacher=teacher_identity, initial_inventory=devices, teacher_and_draft_colocated=False, shared_gpu_allowed=True, utc=now()))
        print(now(), 'GPU_PAIR', 'learner', learner_gpu, identity['uuid'], 'teacher', teacher_gpu, teacher_identity['uuid'], flush=True)
        server = spawn(cmd, 'teacher_vllm', env)
        stage = 'teacher_startup'
        deadline = time.monotonic() + 900
        while not healthy(p['port']):
            assert server.poll() is None, 'Target startup failed; see teacher_vllm.log'
            assert time.monotonic() < deadline, 'Target startup timeout'
            release_baseline_if_done()
            state('starting', learner_gpu=learner_gpu, teacher_gpu=teacher_gpu, learner_uuid=identity['uuid'], teacher_uuid=teacher_identity['uuid'], teacher_pid=server.pid)
            time.sleep(3)
        # Keep the exact original protocol hash required by the full checkpoint.
        protocol = json.loads((out / 'training_protocol.json').read_text())
        assert sha(out / 'training_protocol.json') == recovery['training_protocol_sha256']
        train_env = bind(os.environ.copy(), identity)
        train_env.pop('LD_PRELOAD', None)
        train_env.update(PYTHONPATH=f"{CODE}/runtime_bundle:{CODE}/runtime_bundle/src:{CODE}/runtime_bundle/hs_connectors/src",
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
            TORCHINDUCTOR_CACHE_DIR=str(out / 'training_inductor'), TRITON_CACHE_DIR=str(out / 'training_triton'))
        train_env['PACKED_RECOVERY_ATTEMPT'] = str(attempt)
        active = spawn([PYTHON, '-B', str(RECOVERY_CODE / 'train_resume.py'), '--protocol', str(out / 'training_protocol.json')], 'training', train_env)
        stage = 'packed_20k_training'
        while active.poll() is None:
            assert server.poll() is None, 'Target died during packed training'
            release_baseline_if_done()
            monitor(teacher_gpu, [server])
            monitor(learner_gpu, [active])
            state('running', learner_gpu=learner_gpu, teacher_gpu=teacher_gpu, learner_uuid=identity['uuid'], teacher_uuid=teacher_identity['uuid'], teacher_pid=server.pid, learner_pid=active.pid)
            time.sleep(5)
        assert active.returncode == 0, 'Packed training failed; weights and logs retained'
        stop(server)
        server = None
        assert not list(transfer.iterdir()), 'Hidden transfer remains'
        transfer.rmdir()
        stage = 'waiting_both_training_complete'
        while not release_baseline_if_done():
            state('waiting', packed_training_complete=True, baseline_training_complete=False)
            time.sleep(5)
        stage = 'waiting_gpu_capacity_for_paired_tests'
        gpu, devices = wait_evaluation_card()
        identity = assignment(gpu)
        old = json.loads((baseline / 'native_protocol.json').read_text())
        original, packed = 'N_per_sequence_20k', 'N_packed65536_a1536_20k'
        checkpoints = {}
        for arm, root in [(original, baseline), (packed, out)]:
            weight = root / 'training/epoch_03.safetensors'
            summary = root / 'training/summary.json'
            assert json.loads(summary.read_text())['status'] == 'complete'
            checkpoints[arm] = dict(path=str(weight), sha256=sha(weight), head_config_path=p['head_config_path'], head_config_sha256=sha(p['head_config_path']))
        checkpoints['RAW'] = dict(checkpoints[original])
        native = dict(old, gpu=gpu, uuid=identity['uuid'], native_arms=['RAW', original, packed],
            native_checkpoints=checkpoints, native_configs={arm: {} for arm in checkpoints},
            architecture_by_arm={arm: p['architecture'] for arm in checkpoints},
            source_files={**old['source_files'], **ready['source_files'], **recovery['source_sha256']},
            training_completions={arm: sha(root / 'training/summary.json') for arm, root in [(original, baseline), (packed, out)]},
            comparison='same 20k x3, frozen target/draft, identical basis and initial head; per-sequence versus whole-pack quota/update cadence',
            paired_training_roots=[str(baseline), str(out)])
        native['asset_sha256'] = {**old['asset_sha256'], **ready['asset_sha256'], **{record['path']: record['sha256'] for record in checkpoints.values()}}
        dump(out / 'native_protocol.json', native)
        dump(out / 'evaluation_gate.json', dict(both_training_complete=True, gpu_assignment=identity,
            training_summary_sha256=native['training_completions'], initial_inventory=devices, utc=now()))
        evaluation_env = os.environ.copy()
        evaluation_env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        for concurrency, prompts in [(1, 200), (40, 2000)]:
            for smoke in [True, False]:
                stage = f'C{concurrency}_' + ('smoke' if smoke else 'formal')
                destination = out / f'C{concurrency}' / ('formal_smoke' if smoke else 'formal')
                cmd = [PYTHON, '-B', str(CODE / 'evaluate.py'), '--training', str(out), '--gpu', str(gpu),
                    '--concurrency', str(concurrency), '--output', str(destination), '--tokens', str(32 if smoke else 2096),
                    '--prompts', str(concurrency if smoke else prompts), '--rounds', '1']
                if smoke:
                    cmd.append('--smoke')
                active = spawn(cmd, f'single_card_{stage}', evaluation_env)
                while active.poll() is None:
                    monitor(gpu, [active])
                    state('evaluating', gpu=gpu, child_pid=active.pid, log=str(attempt / f'single_card_{stage}.log'))
                    time.sleep(5)
                assert active.returncode == 0, f'{stage} failed; preserve formal and smoke logs'
        stage = 'report'
        active = spawn([PYTHON, '-B', str(CODE / 'report.py'), '--output', str(out)], 'report', evaluation_env)
        assert active.wait() == 0
        completed = True
        stage = 'complete'
        state('complete', results=str(out / 'RESULTS_CN.txt'))
    except BaseException as exc:
        state('failed', error=repr(exc))
        raise
    finally:
        for process in reversed(owned):
            if process.poll() is None:
                stop(process)
        if transfer.exists():
            for path in transfer.iterdir():
                assert path.is_file()
                path.unlink()
            transfer.rmdir()
        dump(out / 'cleanup.json', dict(complete=completed, only_owned_processes_stopped=True,
            original_training_not_interrupted=True, old_and_new_weights_retained=True, utc=now()))


if __name__ == '__main__':
    main()
