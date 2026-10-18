"""Shared GPU discovery, pool progress and final output verification."""
from collections import Counter
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import subprocess
import xml.etree.ElementTree as ET
from pool_manifest import load_pool_manifest

HERE = Path(__file__).resolve().parent
DEFAULT_POOL = HERE / 'pool_500k_seed42'
ACTIVE_SPLITS = ('train',)


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except (ProcessLookupError, FileNotFoundError):
        return False


def owned_process(pid, required):
    if not alive(pid):
        return False
    proc = Path(f'/proc/{pid}')
    command = (proc / 'cmdline').read_bytes().decode(errors='replace')
    return proc.stat().st_uid == os.getuid() and all(part in command for part in required)


def process_owner(pid):
    """Identify a GPU process by its OS account, including non-regeneration jobs."""
    try:
        uid = Path(f'/proc/{pid}').stat().st_uid
    except OSError:
        return {'pid': pid, 'uid': None, 'user': None, 'ownership': 'unknown'}
    try:
        username = pwd.getpwuid(uid).pw_name
    except KeyError:
        username = str(uid)
    return {'pid': pid, 'uid': uid, 'user': username,
            'ownership': 'self' if uid == os.getuid() else 'other'}


def gpu_inventory(selection='all'):
    command = ['nvidia-smi']
    if selection != 'all':
        command += ['-i', selection]
    command += ['-q', '-x']
    root = ET.fromstring(subprocess.check_output(command, text=True, timeout=15))
    result = []
    for node in root.findall('gpu'):
        def memory(name):
            return float(node.findtext('fb_memory_usage/' + name).split()[0])
        pids = [int(proc.findtext('pid')) for proc in node.findall('processes/process_info')
                if 'C' in proc.findtext('type', '')]
        processes = [process_owner(pid) for pid in pids]
        result.append({'index': int(node.findtext('minor_number')), 'uuid': node.findtext('uuid'),
                       'name': node.findtext('product_name'), 'total_mib': memory('total'),
                       'free_mib': memory('free'), 'compute_pids': pids,
                       'compute_processes': processes,
                       'idle': not pids and memory('free') >= .97 * memory('total'),
                       'compatible': memory('total') >= 79000})
    return result


def output_path(root, shard):
    return root / shard['split'] / f'shard-{shard["shard"]:03d}.jsonl'


def pool_progress(pool_dir=DEFAULT_POOL, generation_dir=None, selected_splits=ACTIVE_SPLITS, thinking='nonthinking'):
    pool_dir = Path(pool_dir).resolve()
    context = 8192 if thinking == 'thinking' else 4096
    root = Path(generation_dir or pool_dir / f'regenerated_ctx{context}') / thinking
    manifest, _, _ = load_pool_manifest(pool_dir, prefer_base=thinking == 'thinking')
    splits = {name: {'expected_conversations': 0, 'generated_conversations': 0,
                     'skipped_conversations': 0, 'remaining_conversations': 0,
                     'complete_shards': 0, 'total_shards': 0} for name in selected_splits}
    shards = []
    for shard in manifest['shards']:
        if shard['split'] not in selected_splits:
            continue
        path = output_path(root, shard)
        status_path = path.with_suffix('.status.json')
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        generated = status.get('completed_conversations', 0)
        skipped = status.get('skipped_conversations', 0)
        remaining = shard['rows'] - generated - skipped
        if min(generated, skipped, remaining) < 0:
            raise ValueError('Invalid shard counters: ' + str(status_path))
        complete = status.get('state') == 'complete' and remaining == 0
        entry = {'split': shard['split'], 'shard': shard['shard'], 'path': str(path),
                 'expected_conversations': shard['rows'], 'generated_conversations': generated,
                 'skipped_conversations': skipped, 'remaining_conversations': remaining,
                 'state': status.get('state', 'pending'), 'complete': complete}
        shards.append(entry)
        counts = splits[shard['split']]
        counts['expected_conversations'] += shard['rows']
        counts['generated_conversations'] += generated
        counts['skipped_conversations'] += skipped
        counts['remaining_conversations'] += remaining
        counts['complete_shards'] += int(complete)
        counts['total_shards'] += 1
    processed = all(counts['remaining_conversations'] == 0 and
                    counts['complete_shards'] == counts['total_shards'] for counts in splits.values())
    skipped_total = sum(counts['skipped_conversations'] for counts in splits.values())
    return {'updated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'pool_dir': str(pool_dir),
            'output_root': str(root), 'active_splits': list(selected_splits), 'evaluation_mode': 'online',
            'all_processed': processed,
            'all_generated_without_skips': processed and skipped_total == 0,
            'state': ('complete_with_skips' if skipped_total else 'complete') if processed else 'in_progress',
            'remaining_conversations': sum(counts['remaining_conversations'] for counts in splits.values()),
            'complete_shards': sum(counts['complete_shards'] for counts in splits.values()),
            'total_shards': len(shards), 'splits': splits, 'shards': shards}


def export_training_paths(root):
    from pool import write_json
    root = Path(root).resolve()
    with (root / 'training_data_paths.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = {split: [str(path) for path in sorted((root / split).glob('shard-*.jsonl'))
                         if '.' not in path.stem and path.stat().st_size]
                 for split in ACTIVE_SPLITS}
        value['validation'] = []
        write_json(root / 'training_data_paths.json', value)


def verify_outputs(pool_dir=DEFAULT_POOL, generation_dir=None, selected_splits=ACTIVE_SPLITS,
                   thinking='nonthinking', max_model_len=None):
    """Verify IDs and training rows from disk; never trust progress counters alone."""
    pool_dir = Path(pool_dir).resolve()
    max_model_len = max_model_len or (8192 if thinking == 'thinking' else 4096)
    root = Path(generation_dir or pool_dir / f'regenerated_ctx{max_model_len}') / thinking
    manifest, _, _ = load_pool_manifest(pool_dir, prefer_base=thinking == 'thinking')
    validator_path = HERE.parent.parent / 'residualV3_main_v1/runtime/data.py'
    spec = importlib.util.spec_from_file_location('training_validator', validator_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    totals = {split: Counter(expected_conversations=0, generated_conversations=0,
                             skipped_conversations=0, training_rows=0, remaining_conversations=0)
              for split in selected_splits}
    incomplete = []
    for shard in manifest['shards']:
        if shard['split'] not in selected_splits:
            continue
        path = output_path(root, shard)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Shard is still running; retry verification later: ' + str(path))
            source = pool_dir / shard['path']
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != shard['sha256']:
                raise ValueError('Prepared prompt shard changed: ' + str(source))
            # JSONL is separated by physical newlines. str.splitlines() also
            # splits valid U+2028/U+2029 characters inside JSON string values.
            with source.open() as source_rows:
                expected = {json.loads(line)['id'] for line in source_rows}
            if len(expected) != shard['rows']:
                raise ValueError('Prepared source ID count differs from manifest: ' + str(source))
            generated, row_ids, pending = set(), set(), None
            row_count = 0
            if path.exists():
                with path.open() as handle:
                    for line in handle:
                        if not line.endswith('\n'):
                            raise ValueError('Incomplete training append: ' + str(path))
                        row = json.loads(line)
                        primary_id = row.get('primary_id')
                        if primary_id not in expected or row['id'] in row_ids or primary_id in generated:
                            raise ValueError('Foreign or duplicate training row: ' + str(path))
                        if pending is not None and pending != primary_id:
                            raise ValueError('Interleaved unfinished conversation: ' + str(path))
                        if not validator.validate_example(row) or len(row['input_ids']) > max_model_len:
                            raise ValueError(f'Invalid {max_model_len}-token training row: ' + str(path))
                        row_ids.add(row['id'])
                        row_count += 1
                        if row.get('metadata', {}).get('conversation_complete', True):
                            generated.add(primary_id)
                            pending = None
                        else:
                            pending = primary_id
            if pending:
                raise ValueError('Unfinished conversation append: ' + str(path))
            skipped = set()
            skip_path = path.with_suffix('.skipped.jsonl')
            if skip_path.exists():
                with skip_path.open() as handle:
                    for line in handle:
                        if not line.endswith('\n'):
                            raise ValueError('Incomplete skip append: ' + str(skip_path))
                        row = json.loads(line)
                        key = row.get('primary_id')
                        if row.get('terminal_skip') is not True or key not in expected or key in skipped or not row.get('reason'):
                            raise ValueError('Invalid terminal skip: ' + str(skip_path))
                        skipped.add(key)
            if generated & skipped:
                raise ValueError('Conversation both generated and skipped: ' + str(path))
            missing = expected - generated - skipped
            checkpoint_count = 0
            for checkpoint in Path(str(path) + '.checkpoints').glob('*.json'):
                key = json.loads(checkpoint.read_text())['primary_id']
                checkpoint_count += int(key in missing)
            totals[shard['split']].update(expected_conversations=len(expected),
                generated_conversations=len(generated), skipped_conversations=len(skipped),
                training_rows=row_count, remaining_conversations=len(missing))
            if missing or checkpoint_count:
                incomplete.append({'split': shard['split'], 'shard': shard['shard'],
                                   'missing_conversations': len(missing), 'pending_checkpoints': checkpoint_count})
    complete = not incomplete
    skipped_total = sum(counts['skipped_conversations'] for counts in totals.values())
    return {'verified_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'state': ('complete_with_skips' if skipped_total else 'complete') if complete else 'incomplete',
            'all_processed': complete, 'all_generated_without_skips': complete and skipped_total == 0,
            'active_splits': list(selected_splits), 'evaluation_mode': 'online',
            'splits': {split: dict(counts) for split, counts in totals.items()},
            'remaining_conversations': sum(counts['remaining_conversations'] for counts in totals.values()),
            'incomplete_shards': incomplete, 'output_root': str(root),
            'validator': str(validator_path), 'method': 'pinned prompt hashes, exact ID coverage, unique commits, training validator'}
