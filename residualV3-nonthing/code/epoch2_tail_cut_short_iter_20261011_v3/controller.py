"""Run C1 and C40 on separate empty GPUs, preserving per-phase live logs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'native_runtime'))
from runtime.utils import dump, sha, now, resources, stop, EVAL, environment
from device import assignment,bind
PYTHON = '/data/weijiahong/miniconda3/envs/dflash/bin/python'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    root = ap.parse_args().root.resolve()
    lock = (root / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = json.loads((root / 'native_protocol.json').read_text())
    for path, digest in p['source_files'].items():
        assert sha(path) == digest, path
    dump(root / 'controller_pid.json', dict(pid=os.getpid(), utc=now()))
    active, states, failures = {}, {}, {}
    mutex = threading.Lock()
    halt = threading.Event()

    def interrupted(sig, frame):
        halt.set()
        raise KeyboardInterrupt(str(sig))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def update(label, state):
        with mutex:
            states[label] = state
            dump(root / 'status.json', dict(status='evaluating', lanes=dict(states), utc=now()))

    def lane(c):
        label = f'C{c}'
        gpu = p['gpu_by_concurrency'][label]
        lease = (root.parent / f'.native_gpu_{gpu}.lock').open('a')
        try:
            fcntl.flock(lease, fcntl.LOCK_EX)
            while True:
                free, pids = resources(gpu)
                if not pids and free >= 78000:
                    break
                update(label, dict(status='waiting_empty_gpu', gpu=gpu, free_mib=free, pids=pids))
                if halt.wait(5):
                    return
            if c==1 and not (root/'MICRO_GATE.json').exists():
                command=[str(EVAL/'bin/python'),'-B',str(CODE/'micro.py'),'--root',str(root)]
                env=bind(environment(),assignment(gpu))
                env.update(PYTHONUNBUFFERED='1',VLLM_BATCH_INVARIANT='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_DETERMINISTIC='1',HEAD_SMALL_TILE_MODE='all16_64',HEAD_STRUCTURE_MODE='lean_combined',OMP_NUM_THREADS='4')
                with (root/'micro.log').open('a') as log:
                    proc=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with mutex:active[label]=proc
                update(label,dict(status='GPU_micro_parity_and_timing',gpu=gpu,pid=proc.pid))
                while proc.poll() is None and not halt.wait(3):pass
                assert proc.returncode==0,'CUDA micro gate failed'
                with mutex:active.pop(label,None)
            if c==1 and not (root/'calibration/COMPLETE.json').exists():
                command=[PYTHON,'-B',str(CODE/'calibrate.py'),'--root',str(root/'calibration')]
                env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                with (root/'calibration.log').open('a') as log:
                    proc=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with mutex:active[label]=proc
                update(label,dict(status='native_threshold_calibration',gpu=gpu,pid=proc.pid))
                while proc.poll() is None and not halt.wait(3):
                    detail=root/'calibration/status.json'
                    update(label,dict(status='native_threshold_calibration',gpu=gpu,detail=json.loads(detail.read_text()) if detail.exists() else {}))
                assert proc.returncode==0,'native calibration failed'
                with mutex:active.pop(label,None)
            while not (root/'calibration/COMPLETE.json').exists():
                if failures:raise RuntimeError(failures)
                update(label,dict(status='waiting_calibration',gpu=gpu))
                if halt.wait(3):return
            while not (root/'MICRO_GATE.json').exists():
                if failures:raise RuntimeError(failures)
                update(label,dict(status='waiting_GPU_micro_gate',gpu=gpu))
                if halt.wait(3):return
            with mutex:
                updated=json.loads((root/'native_protocol.json').read_text())
                updated['predictor_sha256']=sha(updated['predictor_path'])
                dump(root/'native_protocol.json',updated)
            for smoke in (True, False):
                phase = 'smoke' if smoke else 'formal'
                out = root / label / ('formal_smoke' if smoke else 'formal')
                command = [PYTHON, '-B', str(CODE / 'native_evaluate.py'), '--training', str(root / label),
                           '--gpu', str(gpu), '--concurrency', str(c), '--output', str(out),
                           '--tokens', str(32 if smoke else 2096),
                           '--prompts', str(c if smoke else p['prompts_per_concurrency'][label]), '--rounds', '1']
                if smoke:
                    command.append('--smoke')
                    # Exercise changed integration first; formal order stays
                    # registered and unchanged for the matched comparisons.
                    command.extend(['--arms','Q45_CUT','Q45_LATENT_CUT','DFLASH','FULL15'])
                update(label, dict(status='starting', phase=phase, gpu=gpu))
                env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
                with (root / f'single_card_{label}_{phase}.log').open('a') as log:
                    proc = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with mutex:
                    active[label] = proc
                dump(root / label / f'launch_{phase}.json', dict(pid=proc.pid, command=command, utc=now()))
                while proc.poll() is None and not halt.wait(3):
                    f = out / 'status.json'
                    update(label, dict(status='running', phase=phase, gpu=gpu,
                                       detail=json.loads(f.read_text()) if f.exists() else {}))
                assert not halt.is_set()
                assert proc.returncode == 0, (label, phase, proc.returncode)
                with mutex:
                    active.pop(label, None)
            update(label, dict(status='complete', gpu=gpu))
        except BaseException as exc:
            with mutex:
                failures[label] = repr(exc)
            update(label, dict(status='failed', gpu=gpu, error=repr(exc)))
        finally:
            lease.close()

    def serial_lanes():
        lane(1)
        if not failures:lane(40)
    same_gpu=p['gpu_by_concurrency']['C1']==p['gpu_by_concurrency']['C40']
    threads = ([threading.Thread(target=serial_lanes)] if same_gpu else
               [threading.Thread(target=lane, args=(c,)) for c in (1, 40)])
    try:
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=.5)
        assert not failures, failures
        windows = []
        for c in (1, 40):
            result = json.loads((root / f'C{c}/formal/summary.json').read_text())
            assert result['status'] == 'complete' and len(result['windows']) == 8
            windows.extend(dict(concurrency=c, **w) for w in result['windows'])
        dump(root / 'summary.json', dict(status='complete', windows=windows, utc=now()))
        lines = ['Epoch2已训练预测头尾部截断：GSM8K nonthinking；block16；C1=200/C40=1319；T0/T1；max2096。',
                 '设置 | 方法 | 接收长度 | 吞吐token/s | 隔离核验']
        for w in windows:
            lines.append(f"C{w['concurrency']} T{w['T']} | {w['arm']} | {w['native_metrics']['acceptance_length']:.6f} | {w['tokens_per_second']:.3f} | {'通过' if w['speed_evidence'] else '污染，TPS不作结论'}")
        (root / 'RESULTS_CN.txt').write_text('\n'.join(lines) + '\n')
        dump(root / 'status.json', dict(status='complete', utc=now()))
    except BaseException as exc:
        dump(root / 'status.json', dict(status='failed', error=repr(exc), failures=failures, utc=now()))
        raise
    finally:
        halt.set()
        with mutex:
            owned = list(active.values())
        for proc in owned:
            if proc.poll() is None:
                stop(proc)
        for t in threads:
            if t.is_alive():
                t.join(timeout=30)


if __name__ == '__main__':
    main()
