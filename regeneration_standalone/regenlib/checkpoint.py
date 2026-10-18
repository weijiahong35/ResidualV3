"""Durable per-generation progress and recovery of an interrupted JSONL append."""
import hashlib
import json
import os
from pathlib import Path
import time


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def skipped_path(output):
    return Path(output).with_suffix('.skipped.jsonl')


def load_skipped(output):
    path = skipped_path(output)
    if not path.exists():
        return set()
    seen = set()
    with path.open('r+b') as handle:
        while True:
            committed = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                backup = path.with_name(path.name + f'.interrupted-tail-{time.time_ns()}.bin')
                with backup.open('wb') as saved:
                    saved.write(line)
                    saved.flush()
                    os.fsync(saved.fileno())
                handle.truncate(committed)
                handle.flush()
                os.fsync(handle.fileno())
                break
            record = json.loads(line)
            if record.get('terminal_skip') is not True or not record.get('primary_id'):
                raise ValueError('Invalid terminal skip record: ' + str(path))
            seen.add(record['primary_id'])
    return seen


class ConversationCheckpoint:
    def __init__(self, directory, primary_id, recipe):
        key = hashlib.sha256(primary_id.encode()).hexdigest()
        self.path = Path(directory) / (key + '.json')
        self.primary_id = primary_id
        self.recipe = hashlib.sha256(json.dumps(recipe, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def load(self):
        if not self.path.exists():
            return None
        record = json.loads(self.path.read_text())
        if record['primary_id'] != self.primary_id or record['recipe_sha256'] != self.recipe:
            raise ValueError('Checkpoint prompt or generation settings changed: ' + str(self.path))
        return record['progress']

    def save(self, progress):
        durable_json(self.path, {'primary_id': self.primary_id,
                     'recipe_sha256': self.recipe, 'progress': progress})

    def clear(self):
        self.path.unlink(missing_ok=True)


def repair_output_tail(path):
    """Keep committed conversations; quarantine only the final incomplete append.

    Durable writers append all rows of one conversation synchronously and mark
    its last row complete. Interior corruption or interleaved transactions fail.
    Legacy rows without the marker are treated as individually committed.
    """
    path = Path(path)
    if not path.exists():
        return None
    committed = 0
    pending = None
    broken = False
    with path.open('r+b') as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                broken = True
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError('Corrupt committed JSONL: ' + str(path)) from error
            key = row.get('primary_id') or row.get('id')
            if not key or not row.get('input_ids') or len(row['input_ids']) != len(row.get('loss_mask', [])):
                raise ValueError('Invalid generated sample: ' + str(path))
            if pending is not None and pending != key:
                raise ValueError('Interleaved unfinished conversation: ' + str(path))
            if row.get('metadata', {}).get('conversation_complete', True):
                committed = handle.tell()
                pending = None
            else:
                pending = key
        if not broken and pending is None:
            return None
        handle.seek(committed)
        tail = handle.read()
        backup = path.with_name(path.name + f'.interrupted-tail-{time.time_ns()}.bin')
        with backup.open('wb') as saved:
            saved.write(tail)
            saved.flush()
            os.fsync(saved.fileno())
        handle.truncate(committed)
        handle.flush()
        os.fsync(handle.fileno())
    return {'quarantined_tail': str(backup), 'removed_bytes': len(tail)}
