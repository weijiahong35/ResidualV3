"""Observe a real GPU shard completion and validate its subsequent output."""
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from dispatch import owned_process
from pool import write_json
from profiles import PROFILES

HERE = Path(__file__).resolve().parent
PROFILE = PROFILES['nonthinking_ctx4096']
DIRECTORY = HERE / 'shard_transition_watch_20261005'
POOL = HERE / 'pool_500k_seed42'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def validate_completed(shard):
    entry = next(s for s in read(POOL / 'manifest.json')['shards']
                 if s['split'] == 'train' and s['shard'] == shard)
    source = POOL / entry['path']
    if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Source hash differs from manifest')
    expected = {json.loads(line)['id'] for line in source.open()}
    assert len(expected) == entry['rows']
    output = PROFILE.output_root / 'train' / f'shard-{shard:03d}.jsonl'
    validator_path = HERE.parent.parent / 'residualV3_main_v1/runtime/data.py'
    spec = importlib.util.spec_from_file_location('training_validator', validator_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    generated, row_ids, skipped, sources = set(), set(), set(), set()
    pending = None
    maximum = 0
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        for raw in output.open():
            assert raw.endswith('\n'), 'Incomplete row in completed shard'
            row = json.loads(raw)
            primary = row['primary_id']
            assert primary in expected and primary not in generated and row['id'] not in row_ids
            assert pending is None or pending == primary
            assert validator.validate_example(row) and len(row['input_ids']) <= PROFILE.max_model_len
            row_ids.add(row['id'])
            maximum = max(maximum, len(row['input_ids']))
            source_info = row.get('metadata', {}).get('source', {})
            sources.add(source_info.get('original_source', 'unknown') if isinstance(source_info, dict)
                        else str(source_info))
            if row.get('metadata', {}).get('conversation_complete', True):
                generated.add(primary)
                pending = None
            else:
                pending = primary
        assert pending is None
        skip_path = output.with_suffix('.skipped.jsonl')
        if skip_path.exists():
            for raw in skip_path.open():
                row = json.loads(raw)
                key = row['primary_id']
                assert row.get('terminal_skip') is True and row.get('reason')
                assert key in expected and key not in skipped
                skipped.add(key)
        assert not generated & skipped
        assert generated | skipped == expected, 'Source ID coverage is incomplete'
        checkpoint_files = list(Path(str(output) + '.checkpoints').glob('*.json'))
        pending_checkpoints = [p for p in checkpoint_files
                               if read(p)['primary_id'] not in generated | skipped]
        assert not pending_checkpoints, 'Completed shard has pending checkpoints'
    return {'shard': shard, 'output': str(output), 'expected_conversations': len(expected),
            'generated_conversations': len(generated), 'skipped_conversations': len(skipped),
            'training_rows': len(row_ids), 'maximum_tokens': maximum, 'sources': sorted(sources),
            'exact_source_id_coverage': True, 'unique_row_ids': True, 'pending_checkpoints': 0,
            'already_committed_checkpoint_files': len(checkpoint_files)}


def snapshot():
    scheduler = read(PROFILE.scheduler_dir / 'status.json')
    workers = {}
    for i in range(8):
        directory = PROFILE.worker_dir(i)
        value = read(directory / 'status.json')
        if not value:
            continue
        workers[str(i)] = {key: value.get(key) for key in [
            'state', 'supervisor_pid', 'server_pid', 'client_pid', 'active_shard', 'output',
            'concurrency', 'gpu_uuid', 'completed_conversations_in_shard',
            'skipped_conversations_in_shard', 'updated_at_utc', 'error', 'gpu']}
        workers[str(i)]['live_supervisor'] = owned_process(value.get('supervisor_pid'),
            [str(HERE / 'pilot_gpu1.py'), PROFILE.name])
        workers[str(i)]['client_progress'] = value.get('client_progress', {})
    return {'captured_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'scheduler_state': scheduler.get('state'),
            'scheduler_live': owned_process(scheduler.get('scheduler_pid'),
                                           [str(HERE / 'scheduler.py'), PROFILE.name]),
            'progress': scheduler.get('progress', {}).get('splits', {}), 'workers': workers}


def main():
    DIRECTORY.mkdir(exist_ok=True)
    lock = (DIRECTORY / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    first = read(DIRECTORY / 'baseline.json') or snapshot()
    write_json(DIRECTORY / 'baseline.json', first)
    previous = first
    pending_transitions = {}
    last_log = 0
    started = time.time()
    while True:
        current = snapshot()
        for index, worker in current['workers'].items():
            old = previous['workers'].get(index, {})
            old_shard = old.get('active_shard')
            new_shard = worker.get('active_shard')
            if old_shard is not None and new_shard is not None and new_shard != old_shard:
                pending_transitions[index] = {'gpu_index': int(index), 'old_worker': old,
                                              'new_shard': new_shard}
            transition = pending_transitions.get(index)
            if not transition or not worker['live_supervisor'] or worker['state'] != 'generating':
                continue
            if worker['client_progress'].get('ok', 0) < 1 or new_shard != transition['new_shard']:
                continue
            old = transition['old_worker']
            old_status = read(PROFILE.output_root / 'train' / f'shard-{old["active_shard"]:03d}.status.json')
            if old_status.get('state') != 'complete':
                continue
            completed = validate_completed(old['active_shard'])
            output = Path(worker['output'])
            with output.open() as handle:
                sample = json.loads(next(handle))
            assert len(sample['input_ids']) <= PROFILE.max_model_len
            assert len(sample['input_ids']) == len(sample['loss_mask']) and sum(sample['loss_mask']) >= 8
            proof = {'state': 'verified', 'verified_at_utc': current['captured_at_utc'],
                     'monitor_elapsed_seconds': time.time() - started,
                     'gpu_index': int(index), 'gpu_uuid': worker['gpu_uuid'],
                     'previous_shard': completed, 'next_shard': new_shard,
                     'next_output': str(output),
                     'next_saved_conversations': worker['client_progress']['ok'],
                     'same_supervisor': old['supervisor_pid'] == worker['supervisor_pid'],
                     'same_server': old['server_pid'] == worker['server_pid'],
                     'worker': worker}
            write_json(DIRECTORY / 'transition_verified.json', proof)
            write_json(DIRECTORY / 'status.json', proof)
            print(json.dumps(proof), flush=True)
            return
        state = {'state': 'monitoring', 'monitor_pid': os.getpid(),
                 'elapsed_seconds': time.time() - started, **current}
        write_json(DIRECTORY / 'status.json', state)
        if time.time() - last_log >= 60:
            compact = {'utc': current['captured_at_utc'], 'scheduler': current['scheduler_state'],
                       'workers': {i: {k: w.get(k) for k in [
                           'state', 'active_shard', 'concurrency', 'completed_conversations_in_shard',
                           'skipped_conversations_in_shard', 'error']}
                           for i, w in current['workers'].items() if w['live_supervisor']}}
            with (DIRECTORY / 'monitor.jsonl').open('a') as log:
                log.write(json.dumps(compact) + '\n')
            print(json.dumps(compact), flush=True)
            last_log = time.time()
        previous = current
        time.sleep(15)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write_json(DIRECTORY / 'status.json', {'state': 'failed', 'monitor_pid': os.getpid(),
            'error': repr(error), 'captured_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
        raise
