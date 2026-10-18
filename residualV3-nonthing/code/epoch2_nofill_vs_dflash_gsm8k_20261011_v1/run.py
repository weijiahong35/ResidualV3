"""Run the explicitly requested formal comparison, sequentially on empty GPU7."""
from pathlib import Path
import datetime, fcntl, json, os, subprocess, sys, time

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'native_runtime'))
from runtime.utils import resources, dump, now, stop

PYTHON = '/data/weijiahong/miniconda3/envs/dflash/bin/python'
ROOT = Path(sys.argv[1]).resolve()
P = json.loads((ROOT / 'native_protocol.json').read_text())
assert P['user_requested_formal_comparison'] and not P['short_development_only']

def report():
    subprocess.run([sys.executable, '-B', str(ROOT / 'update_report.py')], check=True)

lease = (ROOT.parent / '.native_gpu_7.lock').open('a')
fcntl.flock(lease, fcntl.LOCK_EX)
for c in (1, 40):
    for phase in ('smoke', 'formal'):
        count = max(c, 2) if phase == 'smoke' else P['prompts_per_concurrency'][f'C{c}']
        tokens = 64 if phase == 'smoke' else 2096
        while True:
            free, pids = resources(7)
            if not pids and free >= 78000:
                break
            dump(ROOT / 'status.json', dict(status='waiting_empty_gpu', C=c, phase=phase,
                 free_mib=free, host_pids=pids, foreign_processes_untouched=True, utc=now()))
            time.sleep(15)
        used = int(subprocess.check_output(['du', '-x', '-B1', '-s', '/data/weijiahong'], text=True).split()[0])
        if used + 4 * 1024**3 > P['user_disk_limit_bytes']:
            dump(ROOT / 'status.json', dict(status='disk_limit_blocked', used_bytes=used, utc=now()))
            raise RuntimeError('User disk budget: preserve all existing assets; do not launch')
        out = ROOT / f'C{c}' / ('formal_smoke' if phase == 'smoke' else 'formal')
        cmd = [PYTHON, '-B', str(CODE / 'native_evaluate.py'), '--training', str(ROOT / f'C{c}'),
               '--gpu', '7', '--concurrency', str(c), '--output', str(out), '--prompts', str(count),
               '--tokens', str(tokens), '--rounds', '1', '--arms', 'DFLASH', 'FULL15_NOFILL']
        if phase == 'smoke':
            cmd.append('--smoke')
        with (ROOT / f'single_card_C{c}_{phase}.log').open('x') as log:
            proc = subprocess.Popen(cmd, cwd=CODE,
                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                         OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1'),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(now(), 'START', c, phase, proc.pid, flush=True)
        dump(ROOT / 'status.json', dict(status='running', C=c, phase=phase, gpu=7,
             pid=proc.pid, command=cmd, disk_allocated_bytes=used, utc=now()))
        report()
        try:
            last_report = time.monotonic()
            while proc.poll() is None:
                detail = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {}
                dump(ROOT / 'status.json', dict(status='running', C=c, phase=phase, gpu=7,
                     pid=proc.pid, detail=detail, disk_allocated_bytes=used, utc=now()))
                if time.monotonic() - last_report >= 45:
                    report()
                    last_report = time.monotonic()
                time.sleep(3)
            if proc.returncode:
                dump(ROOT / 'status.json', dict(status='failed', C=c, phase=phase,
                     exit_code=proc.returncode, utc=now()))
                report()
                raise RuntimeError(f'C{c} {phase} failed; original evidence retained')
        finally:
            if proc.poll() is None:
                stop(proc)
        report()
        print(now(), 'FINISH', c, phase, flush=True)
dump(ROOT / 'status.json', dict(status='complete', formal_windows=8, gpu=7, utc=now()))
report()
