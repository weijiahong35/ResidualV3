"""Bracket NVML PID reads with ownership snapshots to tolerate owned-process exit."""
from pathlib import Path
from runtime.utils import members,resources

def inspect_owned(gpus,processes):
    roots={p.pid for p in processes}
    before={pid for root in roots for pid in members(root)}
    active={pid for gpu in gpus for pid in resources(gpu)[1]}
    after={pid for root in roots for pid in members(root)}
    # NVML can retain an already-exited process for one query. Only a process
    # explicitly spawned by this controller qualifies; unknown gone PIDs fail.
    vanished_roots={pid for pid in active&roots if not Path(f'/proc/{pid}').exists()}
    foreign=active-before-after-vanished_roots
    if foreign:raise RuntimeError(f'GPU isolation lost; unknown processes: {sorted(foreign)}')
    return dict(active_pids=sorted(active),exited_during_query=sorted((active&before)-after),vanished_owned_roots=sorted(vanished_roots))
