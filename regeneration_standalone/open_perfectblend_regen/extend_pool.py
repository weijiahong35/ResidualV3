"""Prepare and verify additional prompts without resampling an existing pool."""
import argparse
from collections import Counter, defaultdict
import copy
import datetime as dt
import json
import os
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

from pool import (DEFAULT_POOL, EVAL_COMMIT, SOURCE_FAMILIES, allocate, eligible_rows,
                  load_exclusions, partition, prompt_only, require_cached_revision,
                  sha256_file, source_families, verify, write_json)
from pool_manifest import load_pool_manifest


def additional_shards(records, selected_indices, existing_counts, target_size, shard_size, seed):
    by_source, unused = defaultdict(list), defaultdict(list)
    for index, source, group in records:
        by_source[source].append(index)
        if index not in selected_indices:
            unused[source].append(index)
    available = {source: len(indices) for source, indices in by_source.items()}
    final_counts = allocate(target_size, available)
    needed = {source: final_counts[source] - existing_counts.get(source, 0) for source in available}
    if any(n < 0 or n > len(unused[source]) for source, n in needed.items()):
        raise ValueError('Cannot preserve existing rows while meeting final source proportions')
    rng = random.Random(seed)
    chosen = {source: rng.sample(unused[source], needed[source]) for source in sorted(available)}
    remaining, chunks = sum(needed.values()), []
    while remaining:
        take = min(shard_size, remaining)
        quotas = allocate(take, {source: len(indices) for source, indices in chosen.items()})
        chunk = []
        for source in sorted(chosen):
            n = quotas[source]
            if n:
                chunk.extend(chosen[source][-n:])
                del chosen[source][-n:]
        rng.shuffle(chunk)
        chunks.append(chunk)
        remaining -= take
    return chunks, needed, final_counts


def prepare(pool_dir, target_size):
    from datasets import load_dataset
    pool_dir = pool_dir.resolve()
    manifest, anchor, active_hash = load_pool_manifest(pool_dir)
    if manifest.get('extension'):
        raise ValueError('Pool has already been extended; inspect the active manifest')
    label = f'{target_size // 1000}k' if target_size % 1000 == 0 else str(target_size)
    directory = pool_dir.parent / f'extension_{label}_seed42_20261005'
    directory.mkdir(exist_ok=True)
    stage = directory / 'prepared_pool'
    if (stage / 'verification.json').exists():
        candidate, _, _ = load_pool_manifest(stage)
        assert candidate['parameters']['train_size'] == target_size
        assert candidate['extension']['base_manifest_sha256'] == anchor
        print('Verified extension staging already exists:', stage, flush=True)
        return stage
    if stage.exists():
        raise ValueError('Incomplete extension staging exists; inspect its status before retrying')

    def save(state, **values):
        write_json(directory / 'status.json', {'state': state, 'pid': os.getpid(), **values,
            'updated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()})

    save('loading_cached_source', target_size=target_size, preserved_size=manifest['parameters']['train_size'])
    exclusions, evidence = load_exclusions(pool_dir.parent / 'eval_sources' / EVAL_COMMIT, offline=True)
    if evidence != manifest['benchmark_exclusions']:
        raise ValueError('Evaluation exclusion evidence differs from the original pool')
    revision = manifest['parameters']['revision']
    dataset = load_dataset(manifest['dataset'], split=manifest['source_split'], revision=revision)
    require_cached_revision(dataset, revision)
    save('scanning_source', source_rows=len(dataset))
    records, groups, raw_counts, skipped, examples = eligible_rows(dataset, exclusions)
    train, heldout = partition(records, groups, manifest['parameters']['seed'],
                               manifest['parameters']['holdout_fraction'])
    if (len(train) != manifest['eligible_train_rows'] or len(heldout) != manifest['eligible_heldout_rows']
            or dict(raw_counts) != manifest['source_counts_raw'] or dict(skipped) != manifest['skipped']):
        raise ValueError('Original eligibility/split counts could not be reproduced')
    existing_indices = set()
    for shard in manifest['shards']:
        path = pool_dir / shard['path']
        if sha256_file(path) != shard['sha256']:
            raise ValueError('Original prompt shard changed: ' + str(path))
        if shard['split'] == 'train':
            for line in path.open():
                existing_indices.add(json.loads(line)['pool']['original_row_index'])
    if len(existing_indices) != manifest['parameters']['train_size']:
        raise ValueError('Original training indices contain duplicates or are missing')
    if not existing_indices <= {r[0] for r in train}:
        raise ValueError('Original training prompts are not in the reproduced training split')
    chunks, added_counts, final_counts = additional_shards(train, existing_indices,
        manifest['source_counts_selected']['train'], target_size,
        manifest['parameters']['shard_size'], manifest['parameters']['seed'])
    stage.mkdir()
    archive_relative = f'manifests/base_{manifest["parameters"]["train_size"]}_{anchor}.json'
    archive = stage / archive_relative
    archive.parent.mkdir(parents=True)
    shutil.copyfile(pool_dir / 'manifest.json', archive)
    for shard in manifest['shards']:
        destination = stage / shard['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(pool_dir / shard['path'], destination)
    extended = copy.deepcopy(manifest)
    extended['parameters']['train_size'] = target_size
    extended['source_counts_selected']['train'] = dict(sorted(final_counts.items()))
    extended['source_family_counts_selected']['train'] = source_families(final_counts)
    extended['extended_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    first = max(s['shard'] for s in manifest['shards'] if s['split'] == 'train') + 1
    extended['extension'] = {'base_manifest': archive_relative, 'base_manifest_sha256': anchor,
        'preserved_train_size': manifest['parameters']['train_size'], 'target_train_size': target_size,
        'added_train_shards': list(range(first, first + len(chunks))),
        'source_counts_added': dict(sorted(added_counts.items())), 'seed': manifest['parameters']['seed'],
        'policy': 'Append unused eligible training IDs; preserve original shards and heldout split; final source quotas by largest remainder'}
    for number, indices in enumerate(chunks, first):
        relative = f'prompts/train/shard-{number:03d}.jsonl'
        path = stage / relative
        source_counts, turns = Counter(), 0
        rows = dict(zip(sorted(indices), dataset.select(sorted(indices))))
        with path.open('w', encoding='utf-8') as handle:
            for index in indices:
                row = rows[index]
                messages, users, fingerprint, group = prompt_only(row)
                source = str(row['source'])
                item = {'id': f'opb:{revision[:12]}:{index:07d}', 'source': source, 'conversations': messages,
                    'pool': {'revision': revision, 'source_split': 'train', 'original_row_index': index,
                             'prompt_sha256': fingerprint.hex(), 'group_sha256': group.hex(),
                             'split': 'train', 'shard': number}}
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
                source_counts[source] += 1
                turns += len(users)
            handle.flush()
            os.fsync(handle.fileno())
        families = source_families(source_counts)
        if set(families) != set(SOURCE_FAMILIES):
            raise ValueError('Additional shard is missing one of the eight sources')
        extended['shards'].append({'split': 'train', 'shard': number, 'path': relative,
            'rows': len(indices), 'user_turns': turns, 'bytes': path.stat().st_size,
            'sha256': sha256_file(path), 'source_counts': dict(sorted(source_counts.items())),
            'source_family_counts': families})
        save('writing_additional_shards', completed_additional_shards=number - first + 1,
             total_additional_shards=len(chunks))
        print(f'Additional shard {number:03d}: {len(indices):,} conversations; all eight sources', flush=True)
    write_json(stage / 'manifest.json', extended)
    load_pool_manifest(stage)
    save('verifying_full_pool', stage=str(stage), train_size=target_size)
    verify(SimpleNamespace(pool_dir=stage))
    save('ready_to_publish', stage=str(stage), train_size=target_size,
         added_conversations=sum(added_counts.values()), additional_shards=len(chunks),
         source_counts_added=added_counts, source_counts_final=final_counts)
    return stage


def publish(pool_dir, stage):
    active, anchor, active_sha = load_pool_manifest(pool_dir)
    extended, base_sha, extended_sha = load_pool_manifest(stage)
    if active_sha == extended_sha:
        print('Extension already active', flush=True)
        return
    if active_sha != base_sha or active.get('extension'):
        raise ValueError('Active pool changed since extension preparation')
    report = json.loads((stage / 'verification.json').read_text())
    if report.get('state') != 'verified' or report['conversations']['train'] != extended['parameters']['train_size']:
        raise ValueError('Extension has not passed full verification')
    original = {(s['split'], s['shard']) for s in active['shards']}
    archive = pool_dir / extended['extension']['base_manifest']
    archive.parent.mkdir(exist_ok=True)
    if archive.exists() and sha256_file(archive) != base_sha:
        raise ValueError('Existing manifest archive differs')
    if not archive.exists():
        shutil.copyfile(stage / extended['extension']['base_manifest'], archive)
    for shard in extended['shards']:
        destination = pool_dir / shard['path']
        if (shard['split'], shard['shard']) in original:
            if sha256_file(destination) != shard['sha256']:
                raise ValueError('Original shard changed during preparation')
            continue
        if sha256_file(stage / shard['path']) != shard['sha256']:
            raise ValueError('Prepared additional shard changed')
        if destination.exists():
            if sha256_file(destination) != shard['sha256']:
                raise ValueError('Additional shard destination already differs')
        else:
            os.link(stage / shard['path'], destination)
    # Readers see the old 50 shards until every appended file is durable and
    # this final manifest is replaced atomically. The recipe contract is untouched.
    with (pool_dir / 'manifest.extending.json').open('wb') as handle:
        handle.write((stage / 'manifest.json').read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pool_dir / 'manifest.extending.json', pool_dir / 'manifest.json')
    report['pool_dir'] = str(pool_dir.resolve())
    write_json(pool_dir / 'verification.json', report)
    target = extended['parameters']['train_size']
    label = f'{target // 1000}k' if target % 1000 == 0 else str(target)
    alias = pool_dir.parent / f'pool_{label}_seed42'
    if alias != pool_dir and not alias.exists():
        alias.symlink_to(pool_dir.name, target_is_directory=True)
    print('Published appended training pool:', extended['parameters']['train_size'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'publish'])
    parser.add_argument('--pool-dir', type=Path, default=DEFAULT_POOL)
    parser.add_argument('--target-size', type=int, default=843028)
    parser.add_argument('--stage', type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    if args.command == 'prepare':
        prepare(args.pool_dir, args.target_size)
    else:
        if not args.stage:
            parser.error('publish requires --stage')
        publish(args.pool_dir.resolve(), args.stage.resolve())


if __name__ == '__main__':
    main()
