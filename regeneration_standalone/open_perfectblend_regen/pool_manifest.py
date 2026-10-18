"""Validate append-only prompt-pool extensions while keeping recipe provenance."""
import hashlib
import json
from pathlib import Path


def load_pool_manifest(pool_dir, prefer_base=False):
    pool_dir = Path(pool_dir).resolve()
    raw = (pool_dir / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    current_sha = hashlib.sha256(raw).hexdigest()
    extension = manifest.get('extension')
    if not extension:
        return manifest, current_sha, current_sha
    archive = (pool_dir / extension['base_manifest']).resolve()
    if not archive.is_relative_to(pool_dir):
        raise ValueError('Extension base manifest escapes the pool directory')
    base_raw = archive.read_bytes()
    base_sha = hashlib.sha256(base_raw).hexdigest()
    if base_sha != extension['base_manifest_sha256']:
        raise ValueError('Extension base manifest hash changed')
    base = json.loads(base_raw)
    if base.get('extension'):
        raise ValueError('Nested pool extensions are not supported')
    old_params, new_params = dict(base['parameters']), dict(manifest['parameters'])
    old_size, new_size = old_params.pop('train_size'), new_params.pop('train_size')
    if old_params != new_params or new_size <= old_size:
        raise ValueError('Extension changed the source/split/sampling recipe')
    for key in ['dataset', 'source_split', 'source_rows', 'eligible_rows', 'eligible_train_rows',
                'eligible_heldout_rows', 'source_counts_raw', 'skipped', 'benchmark_exclusions',
                'decontamination_scope', 'split_policy', 'assistant_policy']:
        if base.get(key) != manifest.get(key):
            raise ValueError('Extension changed base provenance: ' + key)
    original = {(s['split'], s['shard']): s for s in base['shards']}
    extended = {(s['split'], s['shard']): s for s in manifest['shards']}
    if len(extended) != len(manifest['shards']):
        raise ValueError('Duplicate extended shard numbers')
    if any(extended.get(key) != value for key, value in original.items()):
        raise ValueError('Extension changed or removed an original prompt shard')
    added = [s for key, s in extended.items() if key not in original]
    first = max(s['shard'] for s in base['shards'] if s['split'] == 'train') + 1
    if (any(s['split'] != 'train' for s in added) or
            sorted(s['shard'] for s in added) != list(range(first, first + len(added))) or
            sum(s['rows'] for s in added) != new_size - old_size or
            sum(s['rows'] for s in manifest['shards'] if s['split'] == 'train') != new_size):
        raise ValueError('Extension must append contiguous training shards with exact counts')
    if manifest['source_counts_selected']['validation'] != base['source_counts_selected']['validation']:
        raise ValueError('Extension changed heldout prompts')
    return base if prefer_base else manifest, base_sha, current_sha
