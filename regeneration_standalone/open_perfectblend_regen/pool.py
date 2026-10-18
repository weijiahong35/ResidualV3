#!/usr/bin/env python3
"""Build a reproducible, prompt-only Open-PerfectBlend pool without CUDA."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import unicodedata
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_POOL = HERE / 'pool_500k_seed42'
EVAL_COMMIT = '005e03b81cec38b7da6399833d609ee89a2587f2'
EVAL_NAMES = ['gsm8k', 'math500', 'aime25', 'humaneval', 'mbpp',
              'livecodebench', 'mt-bench', 'alpaca', 'arena-hard-v2']
SOURCE_FAMILIES = {'MetaMathQA': 'metamathqa', 'Orca Math': 'orca-math',
                   'Evol-CodeAlpaca': 'evol-codealpaca', 'UltraInteract': 'ultrainteract',
                   'UltraChat': 'ultrachat', 'UltraFeedback': 'ultrafeedback',
                   'AutoIF': 'autoif', 'LMSYS Arena': 'lmsys-arena-human-preference'}
os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', CUDA_VISIBLE_DEVICES='')
sys.path.insert(0, str(ROOT / '.client_deps'))
sys.path.insert(0, str(ROOT))
from regenlib.configs import DATASET_CONFIGS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prompt_key(text):
    """Whitespace-normalized exact matching; no claim of semantic deduplication."""
    text = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()
    return hashlib.sha256(text.encode()).digest()


def source_families(counts):
    result = Counter()
    for source, count in counts.items():
        matches = [name for name, marker in SOURCE_FAMILIES.items() if marker in source.lower()]
        if len(matches) != 1:
            raise ValueError(f'Unrecognized or ambiguous Open-PerfectBlend source: {source}')
        result[matches[0]] += count
    return dict(sorted(result.items()))


def require_cached_revision(dataset, revision):
    # Some datasets versions fall back to the latest cache in offline mode.
    # Refuse that fallback if it picked a different commit from the pinned one.
    files = dataset.cache_files
    if not files or any(revision not in Path(entry['filename']).parts for entry in files):
        raise ValueError(f'Cached Arrow data does not match pinned revision {revision}')


def prompt_only(row):
    messages = DATASET_CONFIGS['open-perfectblend'].normalize_fn(row)['conversations']
    if any(message['role'] == 'tool' for message in messages):
        raise ValueError('Tool-result conversations need a separate regeneration policy')
    messages = [message for message in messages if message['role'] in {'system', 'user'}]
    users = [message['content'] for message in messages if message['role'] == 'user']
    if not users:
        raise ValueError('No user prompt')
    fingerprint = hashlib.sha256(b''.join(
        message['role'].encode() + prompt_key(message['content']) for message in messages
    )).digest()
    return messages, users, fingerprint, prompt_key(users[0])


def allocate(total, counts):
    """Largest remainder allocation preserves actual source proportions."""
    available = sum(counts.values())
    if not 0 <= total <= available:
        raise ValueError(f'Need {total} rows but only {available} available')
    if not available:
        return {}
    result = {source: total * count // available for source, count in counts.items()}
    order = sorted(counts, key=lambda source: (-(total * counts[source] % available), source))
    for source in order[:total - sum(result.values())]:
        result[source] += 1
    return result


def select_shards(records, size, shard_size, seed):
    by_source = defaultdict(list)
    for index, source, group in records:
        by_source[source].append(index)
    quotas = allocate(size, {source: len(indices) for source, indices in by_source.items()})
    rng = random.Random(seed)
    selected = {}
    for source in sorted(by_source):
        selected[source] = rng.sample(by_source[source], quotas[source])
    shards = []
    remaining = size
    while remaining:
        take = min(shard_size, remaining)
        chunk_quota = allocate(take, {source: len(indices) for source, indices in selected.items()})
        chunk = []
        for source in sorted(selected):
            n = chunk_quota[source]
            if n:
                chunk.extend(selected[source][-n:])
                del selected[source][-n:]
        rng.shuffle(chunk)
        shards.append(chunk)
        remaining -= take
    return shards, quotas


def load_exclusions(eval_dir, offline=False):
    """Use full official benchmark prompt pools, not only their selected subsets."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    keys = set()
    evidence = []
    for name in EVAL_NAMES:
        path = eval_dir / f'{name}.jsonl'
        url = f'https://raw.githubusercontent.com/deepseek-ai/DeepSpec/{EVAL_COMMIT}/eval_datasets/{name}.jsonl'
        if not path.exists():
            if offline:
                raise FileNotFoundError(f'Missing eval prompts: {path}; prepare once without --offline')
            print(f'Download eval prompts: {name}', flush=True)
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
            tmp = path.with_suffix('.jsonl.tmp')
            tmp.write_bytes(content)
            tmp.replace(path)
        rows = 0
        # JSON strings may contain Unicode line separators; only physical LF
        # delimits JSONL records, so str.splitlines() would corrupt valid rows.
        for line in path.open(encoding='utf-8'):
            if not line.strip():
                continue
            row = json.loads(line)
            turns = row.get('turns')
            if not isinstance(turns, list) or not turns or any(not isinstance(turn, str) for turn in turns):
                raise ValueError(f'Invalid benchmark prompts: {path}')
            keys.update(prompt_key(turn) for turn in turns)
            rows += 1
        evidence.append({'dataset': name, 'rows': rows, 'url': url,
                         'path': str(path), 'sha256': sha256_file(path)})
    # Also check raw prompts from the small HF benchmarks already downloaded.
    # HumanEval's bare code prompt differs from the official formatted question.
    cache_manifest = ROOT / 'download_manifest.json'
    if cache_manifest.exists():
        from datasets import load_dataset
        for entry in json.loads(cache_manifest.read_text()).get('datasets', []):
            name = entry['dataset']
            if name not in {'gsm8k', 'humaneval', 'mt_bench'} or (name == 'gsm8k' and entry['split'] != 'test'):
                continue
            dataset = load_dataset(entry['hf_path'], name=entry['subset'],
                                   split=entry['split'], revision=entry['revision'])
            require_cached_revision(dataset, entry['revision'])
            for row in dataset:
                turns = [row['question']] if name == 'gsm8k' else ([row['prompt']] if name == 'humaneval' else row['prompt'])
                keys.update(prompt_key(turn) for turn in turns)
            evidence.append({'dataset': name + '_hf_raw', 'rows': len(dataset),
                             'revision': entry['revision'], 'hf_path': entry['hf_path']})
    return keys, evidence


def eligible_rows(dataset, exclusions):
    records, groups, seen = [], set(), set()
    raw_counts, skipped = Counter(), Counter()
    examples = []
    for index, row in enumerate(dataset):
        source = sys.intern(str(row.get('source') or 'unknown'))
        raw_counts[source] += 1
        try:
            messages, users, fingerprint, group = prompt_only(row)
        except (ValueError, TypeError, KeyError) as exc:
            skipped['invalid_conversation'] += 1
            if len(examples) < 10:
                examples.append({'index': index, 'error': str(exc)})
            continue
        if any(prompt_key(user) in exclusions for user in users):
            skipped['benchmark_prompt_match'] += 1
            continue
        if fingerprint in seen:
            skipped['duplicate_conversation_prompt'] += 1
            continue
        seen.add(fingerprint)
        groups.add(group)
        records.append((index, source, group))
        if (index + 1) % 100000 == 0:
            print(f'Scanned {index + 1:,}/{len(dataset):,}; eligible={len(records):,}', flush=True)
    return records, groups, raw_counts, skipped, examples


def partition(records, groups, seed, holdout_fraction):
    group_order = sorted(groups)
    random.Random(seed).shuffle(group_order)
    heldout = set(group_order[:math.ceil(len(group_order) * holdout_fraction)])
    train, validation = [], []
    for record in records:
        (validation if record[2] in heldout else train).append(record)
    return train, validation


def prepare(args):
    from datasets import config as datasets_config, load_dataset
    from huggingface_hub import HfApi, constants
    output = args.pool_dir.resolve()
    revision = args.revision
    if not revision and (ROOT / 'download_manifest.json').exists():
        for entry in json.loads((ROOT / 'download_manifest.json').read_text()).get('datasets', []):
            if entry['hf_path'] == 'mlabonne/open-perfectblend':
                revision = entry['revision']
    if not revision:
        if args.offline:
            raise ValueError('No downloaded Open-PerfectBlend revision; run download.sh first')
        revision = HfApi().dataset_info('mlabonne/open-perfectblend').sha
    if not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Pin the dataset to a full 40-character commit SHA')
    parameters = {'revision': revision, 'train_size': args.train_size,
                  'validation_size': args.validation_size, 'shard_size': args.shard_size,
                  'seed': args.seed, 'holdout_fraction': args.holdout_fraction,
                  'eval_commit': EVAL_COMMIT, 'recipe_version': 1}
    manifest_path = output / 'manifest.json'
    if output.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f'Pool directory exists without a complete manifest: {output}')
        manifest = json.loads(manifest_path.read_text())
        if manifest['parameters'] != parameters:
            raise ValueError('Existing pool recipe differs; use a new --pool-dir')
        for shard in manifest['shards']:
            if sha256_file(output / shard['path']) != shard['sha256']:
                raise ValueError(f'Prepared shard changed: {shard["path"]}')
        print(f'Existing pool verified: {manifest_path}', flush=True)
        return
    exclusions, evidence = load_exclusions(HERE / 'eval_sources' / EVAL_COMMIT, args.offline)
    dataset = load_dataset('mlabonne/open-perfectblend', split='train', revision=revision)
    require_cached_revision(dataset, revision)
    print(f'Source: {len(dataset):,} rows; revision={revision}', flush=True)
    records, groups, raw_counts, skipped, examples = eligible_rows(dataset, exclusions)
    if set(source_families(raw_counts)) != set(SOURCE_FAMILIES):
        raise ValueError('Published Open-PerfectBlend is missing one of the eight source families')
    train, validation = partition(records, groups, args.seed, args.holdout_fraction)
    print(f'Eligible train={len(train):,}, heldout={len(validation):,}', flush=True)
    plans = {}
    selected_counts = {}
    for split, rows, size in [('train', train, args.train_size), ('validation', validation, args.validation_size)]:
        plans[split], selected_counts[split] = select_shards(rows, size, args.shard_size, args.seed + (0 if split == 'train' else 1))
    staging = output.with_name(output.name + f'.building-{os.getpid()}')
    staging.mkdir(parents=True, exist_ok=False)
    shards = []
    for split, chunks in plans.items():
        for number, indices in enumerate(chunks):
            relative = f'prompts/{split}/shard-{number:03d}.jsonl'
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            source_counts, user_turns = Counter(), 0
            with path.open('w', encoding='utf-8') as handle:
                # Read in source order for Arrow locality, then restore sample order.
                rows_by_index = dict(zip(sorted(indices), dataset.select(sorted(indices))))
                for index in indices:
                    row = rows_by_index[index]
                    messages, users, fingerprint, group = prompt_only(row)
                    source = str(row.get('source') or 'unknown')
                    primary_id = f'opb:{revision[:12]}:{index:07d}'
                    item = {'id': primary_id, 'source': source, 'conversations': messages,
                            'pool': {'revision': revision, 'source_split': 'train',
                                     'original_row_index': index, 'prompt_sha256': fingerprint.hex(),
                                     'group_sha256': group.hex(), 'split': split, 'shard': number}}
                    handle.write(json.dumps(item, ensure_ascii=False) + '\n')
                    source_counts[source] += 1
                    user_turns += len(users)
            families = source_families(source_counts)
            if len(indices) >= 1000 and set(families) != set(SOURCE_FAMILIES):
                raise ValueError(f'Shard {split}/{number} is missing a required source family')
            shards.append({'split': split, 'shard': number, 'path': relative,
                           'rows': len(indices), 'user_turns': user_turns,
                           'bytes': path.stat().st_size, 'sha256': sha256_file(path),
                           'source_counts': dict(sorted(source_counts.items())),
                           'source_family_counts': families})
            print(f'Prepared {split}/{number:03d}: {len(indices):,} conversations, {user_turns:,} turns', flush=True)
    manifest = {'state': 'prepared_prompts_only', 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                'dataset': 'mlabonne/open-perfectblend', 'source_split': 'train',
                'parameters': parameters, 'source_rows': len(dataset),
                'eligible_rows': len(records), 'eligible_train_rows': len(train),
                'eligible_heldout_rows': len(validation), 'source_counts_raw': dict(sorted(raw_counts.items())),
                'source_counts_selected': selected_counts, 'skipped': dict(skipped),
                'source_family_counts_selected': {split: source_families(counts) for split, counts in selected_counts.items()},
                'invalid_examples': examples, 'benchmark_exclusions': evidence,
                'decontamination_scope': 'NFKC/whitespace-normalized exact user prompts; no semantic/paraphrase guarantee',
                'split_policy': 'First-user-prompt groups; seeded 95/5 group split; same group never crosses train/validation',
                'sampling_policy': 'Actual eligible source proportions, largest remainder; no inferred domain labels',
                'assistant_policy': 'All original assistant answers removed; regenerate every user turn',
                'huggingface_cache': {'hub': str(constants.HF_HUB_CACHE), 'datasets': str(datasets_config.HF_DATASETS_CACHE)},
                'shards': shards}
    write_json(staging / 'manifest.json', manifest)
    staging.replace(output)
    print(f'Pool ready: {manifest_path}', flush=True)


def status(args):
    from dispatch import pool_progress
    pool_dir = args.pool_dir.resolve()
    manifest = json.loads((pool_dir / 'manifest.json').read_text())
    result = {'state': manifest['state'], 'pool_dir': str(pool_dir),
              'revision': manifest['parameters']['revision'],
              'prompt_conversations': {split: sum(shard['rows'] for shard in manifest['shards'] if shard['split'] == split)
                                       for split in ['train', 'validation']},
              'user_turns': {split: sum(shard['user_turns'] for shard in manifest['shards'] if shard['split'] == split)
                             for split in ['train', 'validation']}, 'generation': {}}
    for thinking in ['thinking', 'nonthinking']:
        states = []
        for shard in manifest['shards']:
            context = 8192 if thinking == 'thinking' else 4096
            path = (args.generation_dir or pool_dir / f'regenerated_ctx{context}') / thinking / shard['split'] / f'shard-{shard["shard"]:03d}.status.json'
            if path.is_file():
                states.append(json.loads(path.read_text()))
        result['generation'][thinking] = {'completed_conversations': sum(state.get('completed_conversations', 0) for state in states),
                                         'skipped_conversations': sum(state.get('skipped_conversations', 0) for state in states),
                                         'complete_shards': sum(state.get('state') == 'complete' for state in states),
                                         'tracked_shards': len(states)}
    overall = pool_progress(pool_dir, args.generation_dir)
    overall.pop('shards', None)
    result['overall_generation'] = overall
    print(json.dumps(result, ensure_ascii=False, indent=2))


def verify(args):
    pool_dir = args.pool_dir.resolve()
    manifest = json.loads((pool_dir / 'manifest.json').read_text())
    ids, groups = set(), {'train': set(), 'validation': set()}
    counts, turns, sources = Counter(), Counter(), {'train': Counter(), 'validation': Counter()}
    for shard in manifest['shards']:
        path = pool_dir / shard['path']
        if sha256_file(path) != shard['sha256']:
            raise ValueError(f'Changed prompt shard: {path}')
        rows, shard_turns, shard_sources = 0, 0, Counter()
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.endswith('\n'):
                    raise ValueError(f'Incomplete prompt JSONL: {path}')
                row = json.loads(line)
                messages, users, fingerprint, group = prompt_only(row)
                if any(message['role'] not in {'system', 'user'} for message in row['conversations']):
                    raise ValueError(f'Original assistant answer retained: {row["id"]}')
                if row['id'] in ids:
                    raise ValueError(f'Duplicate selected source ID: {row["id"]}')
                ids.add(row['id'])
                metadata = row['pool']
                if metadata['split'] != shard['split'] or metadata['shard'] != shard['shard']:
                    raise ValueError(f'Wrong pool split/shard: {row["id"]}')
                if metadata['prompt_sha256'] != fingerprint.hex() or metadata['group_sha256'] != group.hex():
                    raise ValueError(f'Wrong prompt fingerprint: {row["id"]}')
                if metadata['revision'] != manifest['parameters']['revision']:
                    raise ValueError(f'Wrong dataset revision: {row["id"]}')
                if row['id'] != f'opb:{metadata["revision"][:12]}:{metadata["original_row_index"]:07d}':
                    raise ValueError(f'Wrong source ID: {row["id"]}')
                groups[shard['split']].add(group)
                shard_sources[row['source']] += 1
                rows += 1
                shard_turns += len(users)
        if rows != shard['rows'] or shard_turns != shard['user_turns'] or shard_sources != shard['source_counts']:
            raise ValueError(f'Wrong counts in manifest: {path}')
        if rows >= 1000 and set(source_families(shard_sources)) != set(SOURCE_FAMILIES):
            raise ValueError(f'Missing source family: {path}')
        counts[shard['split']] += rows
        turns[shard['split']] += shard_turns
        sources[shard['split']].update(shard_sources)
    if groups['train'] & groups['validation']:
        raise ValueError('First-user-prompt group leaked across train/validation')
    if counts != {'train': manifest['parameters']['train_size'], 'validation': manifest['parameters']['validation_size']}:
        raise ValueError(f'Wrong total conversation counts: {counts}')
    if any(sources[split] != manifest['source_counts_selected'][split] for split in sources):
        raise ValueError('Selected source totals differ from manifest')
    report = {'state': 'verified', 'pool_dir': str(pool_dir), 'conversations': dict(counts),
              'user_turns': dict(turns), 'unique_source_ids': len(ids), 'train_validation_group_overlap': 0,
              'source_family_counts': {split: source_families(counter) for split, counter in sources.items()}}
    write_json(pool_dir / 'verification.json', report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'status'])
    parser.add_argument('--pool-dir', type=Path, default=DEFAULT_POOL)
    parser.add_argument('--generation-dir', type=Path, help='Optional generation root; default: inspect separate thinking 8192 and nonthinking 4096 roots')
    parser.add_argument('--train-size', type=int, help='Reuse the existing manifest size; default for a new pool: 843028')
    parser.add_argument('--validation-size', type=int, default=2000)
    parser.add_argument('--shard-size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--holdout-fraction', type=float, default=0.05)
    parser.add_argument('--revision')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.train_size is None:
        manifest_path = args.pool_dir.expanduser() / 'manifest.json'
        args.train_size = (json.loads(manifest_path.read_text())['parameters']['train_size']
                           if manifest_path.exists() else 843028)
    if min(args.train_size, args.validation_size, args.shard_size) < 1 or not 0 < args.holdout_fraction < 1:
        parser.error('Sizes must be positive and holdout-fraction must be between zero and one')
    if args.offline:
        os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    {'prepare': prepare, 'verify': verify, 'status': status}[args.command](args)


if __name__ == '__main__':
    main()
