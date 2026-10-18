"""Retry incomplete clients while retaining the GPU's healthy model service."""
from pathlib import Path
import json
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def client_complete(completed, skipped, target):
    # The committed source-ID counts, rather than an exit code or a historical
    # error log, establish whether the client needs to resume.
    return completed + skipped >= target


def should_release_server(release_requested, all_processed, server_alive):
    return release_requested or all_processed or not server_alive


def refill_poll_delay(normal, workers, required):
    return min(normal, 2) if workers < required else normal


def reserve_startup(uuid, directory, backend, device, save):
    """Mark a verified idle GPU busy before model imports/loading.

    This allocation prevents a free-GPU scanner from claiming the card during
    startup. It is not OS-enforced exclusion against unrelated processes.
    The caller releases it only after the model service is ready, or on failure.
    """
    if not device['compatible'] or not device['idle'] or device['uuid'] != uuid:
        raise RuntimeError('Startup reservation requires this exact idle compatible GPU')
    status = directory / 'startup_reservation.json'
    status.unlink(missing_ok=True)
    command = [sys.executable, str(HERE / 'gpu_handoff_reservation.py'), '--gpu-uuid', uuid,
               '--status-file', str(status), '--lifetime-seconds', '1800']
    env = backend.environment(SimpleNamespace(env_dir=Path(sys.executable).parent.parent, gpu=uuid))
    with (directory / 'startup_reservation.log').open('ab') as logfile:
        guard = subprocess.Popen(command, cwd=HERE.parent, env=env, stdin=subprocess.DEVNULL,
            stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
    save(state='reserving_gpu', startup_reservation_pid=guard.pid)
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if guard.poll() is not None:
                raise RuntimeError('GPU startup reservation exited; see startup_reservation.log')
            value = json.loads(status.read_text()) if status.exists() else {}
            if value.get('state') == 'holding' and value.get('pid') == guard.pid:
                return guard
            time.sleep(.5)
        raise TimeoutError('GPU startup reservation did not become ready')
    except BaseException:
        backend.stop_group(guard)
        raise
