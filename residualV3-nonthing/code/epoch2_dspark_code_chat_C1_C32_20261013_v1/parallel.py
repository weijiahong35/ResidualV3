"""E2 DSpark code/chat panels; at most two exclusive empty GPU workers."""
from pathlib import Path
import fcntl,json,os,signal,subprocess,sys,threading,time
ORIGINAL=Path(__file__).resolve().parent
sys.path.insert(0,str(ORIGINAL/'native_runtime'))
from runtime.utils import dump,sha,now,resources
from device import assignment
PYTHON='/data/weijiahong/miniconda3/envs/dflash/bin/python'
ROOT=Path(sys.argv[1]).resolve()
CONTROL=ROOT/'parallel'
def read(p): return json.loads(p.read_text())

def process(pid):
    path = Path('/proc') / str(pid)
    if not path.exists(): return None
    try:
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z': return None
        return dict(uid=path.stat().st_uid, command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except FileNotFoundError: return None

def environment():
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    return env

class RetryOccupation(Exception):
    pass

def terminate_owned_evaluator(proc):
    """Let the frozen evaluator clean up its separately grouped server."""
    identity = process(proc.pid)
    if proc.poll() is not None: return
    assert identity and identity['uid'] == os.getuid()
    assert str(ORIGINAL / 'native_evaluate.py') in identity['command']
    assert str(ROOT) in identity['command']
    os.kill(proc.pid, signal.SIGTERM)
    proc.wait(timeout=60)

def archive_attempt(dataset, concurrency, reason, extra=None):
    folder = ROOT / dataset / f'C{concurrency}'
    stamp = now().replace(':', '').replace('+', '_')
    prior = folder / 'prior_attempts' / ('occupation_' + stamp)
    prior.mkdir(parents=True)
    paths = [folder / 'native_protocol.json', folder / 'formal_smoke', folder / 'formal']
    paths += [ROOT / f'{dataset}_C{concurrency}_{phase}.log' for phase in ('smoke', 'formal')]
    checksums = {}
    for path in paths:
        if not path.exists(): continue
        files = [path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
        for file in files:
            key = str(file.relative_to(path)) if path.is_dir() else ''
            checksums[str(Path(path.name) / key)] = sha(file)
        path.rename(prior / path.name)
    assert all(sha(prior / f) == h for f, h in checksums.items())
    dump(prior / 'RECOVERY.json', dict(utc=now(), reason=reason, details=extra,
         archived_file_sha256=checksums, no_valid_completed_formal_repeated=True,
         performance_evidence_discarded=True, native_evaluator_unchanged=True))
    print(now(), 'ARCHIVE', dataset, f'C{concurrency}', reason, str(prior), flush=True)
    return str(prior)

def phase_evidence(output, phase):
    summary = read(output / 'summary.json')
    assert summary['status'] == 'complete' and len(summary['windows']) == 4
    if not all(w['isolation_observed'] and not w['performance_contaminated']
               for w in summary['windows']):
        raise RetryOccupation('foreign occupation recorded in completed phase')
    if phase == 'formal': assert all(w['speed_evidence'] for w in summary['windows'])
    return summary

def main():
    lock=(ROOT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest_path=CONTROL/'DISPATCH_MANIFEST.json'
    manifest=read(manifest_path)
    plan=read(ROOT/'plan.json')
    assert all(sha(f)==h for f,h in plan['source_files'].items())
    assert all(sha(f)==h for f,h in manifest['new_dispatch_sources'].items())
    dump(CONTROL/'controller_identity.json',dict(pid=os.getpid(),uid=os.getuid(),command=sys.argv,utc=now(),new_dspark_code_chat_queue=True))
    worker_slots = threading.Semaphore(plan['maximum_simultaneous_owned_gpus'])
    cancel=threading.Event()
    def interrupted(sig,frame):
        cancel.set()
        raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def worker(dataset):
        worker_slots.acquire()
        state_file = CONTROL / dataset / 'status.json'
        lease = None
        try:
            for c in plan['concurrency']:
                training = ROOT / dataset / f'C{c}'
                if (training / 'formal/summary.json').exists():
                    evidence = phase_evidence(training / 'formal', 'formal')
                    assert len(evidence['windows']) == 4
                    print(now(), 'KEEP COMPLETE', dataset, f'C{c}', flush=True)
                    continue
                assert not (training / 'formal_smoke').exists()
                assert not (training / 'formal').exists()
                attempt = 0
                while True:
                    gpu = None
                    while gpu is None:
                        if cancel.is_set(): raise KeyboardInterrupt('cancelled')
                        policy = read(CONTROL / 'allocation_policy.json')
                        assert policy['eligible_gpus'] == list(range(8))
                        assert policy['all_empty_GPUs_user_authorized'] is True
                        checks = []
                        choices = [7, 0, 1, 2, 3, 4, 5, 6]
                        for g in choices:
                            free, ids = resources(g)
                            checks.append(dict(gpu=g, free_mib=free, host_pids=ids))
                            if ids or free < plan['empty_gpu_free_mib']: continue
                            trial_lease = (ROOT.parent / f'.native_gpu_{g}.lock').open('a')
                            try: fcntl.flock(trial_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                trial_lease.close(); continue
                            free, ids = resources(g)
                            if ids or free < plan['empty_gpu_free_mib']:
                                trial_lease.close(); continue
                            lease = trial_lease; gpu = g; break
                        dump(state_file, dict(status='waiting_empty_gpu' if gpu is None else 'assigned',
                             dataset=dataset, concurrency=c, gpu=gpu, host_checks=checks,
                             attempt=attempt + 1, utc=now(), foreign_processes_untouched=True))
                        if gpu is None: cancel.wait(30)
                    attempt += 1
                    disk = int(subprocess.check_output(['du', '-x', '-B1', '-s', '/data/weijiahong'], text=True).split()[0])
                    assert disk + 6 * (1 << 30) <= plan['user_disk_limit_bytes']
                    dump(CONTROL / dataset / f'C{c}_DISK_GATE.json', dict(utc=now(), allocated_bytes=disk,
                         limit_GiB=600, reserve_GiB=6))
                    p = read(ROOT / dataset / 'native_protocol.template.json')
                    assert sha(ROOT / dataset / 'native_protocol.template.json') == plan['datasets'][dataset]['template_sha256']
                    p.update(gpu_by_concurrency={f'C{c}': gpu}, gpu_uuid=assignment(gpu)['uuid'],
                             parallel_dispatch_manifest_sha256=sha(manifest_path),
                             gpu_allocation_policy=policy)
                    dump(training / 'native_protocol.json', p)
                    try:
                        for phase in ('smoke', 'formal'):
                            n = min(plan['datasets'][dataset]['count'], max(c, 2)) if phase == 'smoke' else plan['datasets'][dataset]['count']
                            output = training / ('formal_smoke' if phase == 'smoke' else 'formal')
                            command = [PYTHON, '-B', str(ORIGINAL / 'native_evaluate.py'), '--training', str(training),
                                       '--gpu', str(gpu), '--concurrency', str(c), '--output', str(output),
                                       '--prompts', str(n), '--tokens', '64' if phase == 'smoke' else str(plan['max_new_tokens']),
                                       '--rounds', '1', '--arms', *plan['arms']]
                            if phase == 'smoke': command.append('--smoke')
                            proc = None
                            try:
                                log_path = ROOT / f'{dataset}_C{c}_{phase}.log'
                                with log_path.open('x') as log:
                                    proc = subprocess.Popen(command, cwd=ORIGINAL, env=environment(), stdout=log,
                                                            stderr=subprocess.STDOUT, start_new_session=True)
                                    dump(state_file, dict(status='running', dataset=dataset, concurrency=c,
                                         gpu=gpu, phase=phase, evaluator_pid=proc.pid, log=str(log_path),
                                         command=command, attempt=attempt, utc=now()))
                                    print(now(), 'RUN', dataset, f'C{c}', phase, 'GPU', gpu, 'PID', proc.pid, flush=True)
                                    while proc.poll() is None:
                                        if cancel.wait(2): raise KeyboardInterrupt('cancelled')
                                        state_path = output / 'status.json'
                                        if state_path.exists():
                                            active = read(state_path)
                                            if active['status'] == 'waiting_memory' and active.get('host_pids'):
                                                raise RetryOccupation('foreign occupant between arms; return pair to any-empty-GPU queue')
                                    if proc.returncode:
                                        foreign = set()
                                        for file in output.glob('round_*/**/isolation.json'):
                                            for sample in read(file)['samples']: foreign.update(sample['foreign_pids'])
                                        error = read(output / 'status.json').get('error', '')
                                        if foreign and 'server failed:' in error:
                                            raise RetryOccupation('server startup failed with recorded foreign GPU processes: ' + str(sorted(foreign)))
                                        raise RuntimeError(f'{dataset} C{c} {phase} exited {proc.returncode}; not a recoverable occupation startup failure')
                            finally:
                                if proc and proc.poll() is None: terminate_owned_evaluator(proc)
                            phase_evidence(output, phase)
                        break
                    except RetryOccupation as exc:
                        archive_attempt(dataset, c, str(exc), dict(attempt=attempt, gpu=gpu))
                        dump(state_file, dict(status='requeued_after_foreign_occupation', dataset=dataset,
                             concurrency=c, attempt=attempt, error=str(exc), utc=now(),
                             completed_formal_preserved=True))
                    finally:
                        if lease is not None: lease.close(); lease = None
            dump(state_file, dict(status='complete', dataset=dataset, utc=now()))
        except BaseException as exc:
            dump(state_file, dict(status='failed', dataset=dataset, error=repr(exc), utc=now()))
            print(now(), 'FAILED', dataset, repr(exc), flush=True)
        finally:
            if lease is not None: lease.close()
            worker_slots.release()
    datasets=list(plan['datasets'])
    for d in datasets:dump(CONTROL/d/'status.json',dict(status='queued',dataset=d,utc=now()))
    threads=[threading.Thread(target=worker,args=(d,),daemon=False) for d in datasets]
    for t in threads:t.start()
    while any(t.is_alive() for t in threads):
        states={d:read(CONTROL/d/'status.json') for d in datasets}
        active = any(s['status'] in ('running','assigned') for s in states.values())
        dump(ROOT/'status.json',dict(status='running' if active else 'waiting_empty_gpu',controller_pid=os.getpid(),dataset_states=states,utc=now()))
        subprocess.run([PYTHON,'-B',str(ORIGINAL/'update_report.py'),str(ROOT)],check=True,env=environment(),stdout=subprocess.DEVNULL)
        cancel.wait(15)
    states={d:read(CONTROL/d/'status.json') for d in datasets}
    status='complete' if all(s['status']=='complete' for s in states.values()) else 'failed'
    if status=='complete':subprocess.run([PYTHON,'-B',str(ORIGINAL/'verify.py'),str(ROOT)],check=True,env=environment())
    dump(ROOT/'status.json',dict(status=status,controller_pid=os.getpid(),dataset_states=states,utc=now()))
    subprocess.run([PYTHON,'-B',str(ORIGINAL/'update_report.py'),str(ROOT)],check=True,env=environment())
    print(now(),status.upper(),flush=True)

if __name__=='__main__':main()
