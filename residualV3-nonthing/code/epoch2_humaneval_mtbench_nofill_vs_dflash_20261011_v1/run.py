"""New authorized E2 cross-domain queue. Host GPU view; never resume old queues."""
from pathlib import Path
import fcntl, json, os, signal, subprocess, sys, time

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'native_runtime'))
from runtime.utils import dump, sha, now, resources, stop
from device import assignment

PYTHON = '/data/weijiahong/miniconda3/envs/dflash/bin/python'


def main():
    root = Path(sys.argv[1]).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    assert plan['eligible_gpus'] == [0, 3, 4, 5, 6, 7]
    assert plan['arms'] == ['DFLASH', 'FULL15_NOFILL']
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((root / 'status.json').read_text())['status'] == 'prepared', 'No automatic restart'
    assert all(sha(f) == h for f, h in plan['source_files'].items()), 'Frozen code changed'
    dump(root / 'controller_identity.json', dict(pid=os.getpid(), uid=os.getuid(),
         argv=sys.argv, code=str(CODE), utc=now(), host_view_required=True))
    def interrupted(sig, frame): raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')

    def report():
        subprocess.run([PYTHON, '-B', str(CODE / 'update_report.py'), str(root)],
                       check=True, cwd=CODE, env=env, stdout=subprocess.DEVNULL)

    held = None
    try:
        for dataset in plan['datasets']:
            for c in plan['concurrency']:
                candidate = None
                while candidate is None:
                    checks = []
                    for gpu in plan['eligible_gpus']:
                        free, pids = resources(gpu)
                        checks.append(dict(gpu=gpu, free_mib=free, host_pids=pids))
                        if pids or free < plan['empty_gpu_free_mib']: continue
                        lease = (root.parent / f'.native_gpu_{gpu}.lock').open('a')
                        try: fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            lease.close(); continue
                        free, pids = resources(gpu)
                        if pids or free < plan['empty_gpu_free_mib']:
                            lease.close(); continue
                        held = lease; candidate = gpu; break
                    dump(root / 'HOST_GPU_GATE.json', dict(utc=now(), host_checks=checks,
                         selected_gpu=candidate, excluded_gpus=[1, 2], foreign_processes_untouched=True))
                    if candidate is None:
                        state = dict(status='waiting_empty_gpu', dataset=dataset, concurrency=c,
                                     controller_pid=os.getpid(), utc=now(), host_checks=checks)
                        dump(root / 'status.json', state)
                        with (root / 'resource_wait.jsonl').open('a') as f: f.write(json.dumps(state) + '\n')
                        print(now(), 'WAIT', dataset, f'C{c}', [(x['gpu'], x['host_pids']) for x in checks], flush=True)
                        report(); time.sleep(30)
                disk_bytes = int(subprocess.check_output(['du', '-x', '-B1', '-s', '/data/weijiahong'], text=True).split()[0])
                assert disk_bytes + 6 * (1 << 30) <= plan['user_disk_limit_bytes'], 'Disk reserve exceeded'
                dump(root / 'DISK_GATE.json', dict(utc=now(), allocated_bytes=disk_bytes,
                     allocated_GiB=disk_bytes / (1 << 30), limit_GiB=600, reserve_GiB=6))
                training = root / dataset / f'C{c}'
                p = json.loads((root / dataset / 'native_protocol.template.json').read_text())
                assert sha(root / dataset / 'native_protocol.template.json') == plan['datasets'][dataset]['template_sha256']
                p.update(gpu_by_concurrency={f'C{c}': candidate}, gpu_uuid=assignment(candidate)['uuid'])
                dump(training / 'native_protocol.json', p)
                for phase in ('smoke', 'formal'):
                    count = min(plan['datasets'][dataset]['count'], max(c, 2)) if phase == 'smoke' else plan['datasets'][dataset]['count']
                    tokens = 64 if phase == 'smoke' else 2096
                    out = training / ('formal_smoke' if phase == 'smoke' else 'formal')
                    cmd = [PYTHON, '-B', str(CODE / 'native_evaluate.py'), '--training', str(training),
                           '--gpu', str(candidate), '--concurrency', str(c), '--output', str(out),
                           '--prompts', str(count), '--tokens', str(tokens), '--rounds', '1',
                           '--arms', *plan['arms']]
                    if phase == 'smoke': cmd.append('--smoke')
                    proc = None
                    try:
                        with (root / f'{dataset}_C{c}_{phase}.log').open('x') as log:
                            proc = subprocess.Popen(cmd, cwd=CODE, env=env, stdout=log,
                                                    stderr=subprocess.STDOUT, start_new_session=True)
                            dump(root / 'status.json', dict(status='running', dataset=dataset,
                                 concurrency=c, gpu=candidate, phase=phase, evaluator_pid=proc.pid,
                                 controller_pid=os.getpid(), command=cmd, utc=now()))
                            print(now(), 'RUN', dataset, f'C{c}', phase, 'GPU', candidate, 'PID', proc.pid, flush=True)
                            while proc.poll() is None:
                                report(); time.sleep(30)
                            if proc.returncode: raise RuntimeError(f'{dataset} C{c} {phase} exited {proc.returncode}; see {log.name}')
                    finally:
                        if proc is not None and proc.poll() is None: stop(proc)
                    report()
                held.close(); held = None
        dump(root / 'status.json', dict(status='verifying', formal_windows_completed=16,
             formal_windows_expected=16, controller_pid=os.getpid(), utc=now()))
        report()
        subprocess.run([PYTHON, '-B', str(CODE / 'verify.py'), str(root)], check=True, cwd=CODE, env=env)
        dump(root / 'status.json', dict(status='complete', formal_windows_completed=16,
             formal_windows_expected=16, controller_pid=os.getpid(), utc=now()))
        report()
        print(now(), 'COMPLETE AND AUDITED', flush=True)
    except BaseException as exc:
        dump(root / 'status.json', dict(status='failed', error=repr(exc), utc=now(), controller_pid=os.getpid()))
        report(); raise
    finally:
        if held is not None: held.close()


if __name__ == '__main__': main()
