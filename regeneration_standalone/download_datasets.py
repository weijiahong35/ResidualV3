#!/usr/bin/env python3
"""Download source datasets into the standard Hugging Face cache (CPU only)."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0')
if (HERE / '.client_deps').is_dir():
    sys.path.insert(0, str(HERE / '.client_deps'))

from regenlib.configs import DATASET_CONFIGS

PRESETS = {'gsm8k': 'gsm8k_test', 'humaneval': 'humaneval', 'mt_bench': 'mt_bench',
           'open-perfectblend': 'open-perfectblend'}
DEFAULT_DATASETS = ['gsm8k', 'humaneval', 'mt_bench']
EXPECTED_ROWS = {'gsm8k': {'train': 7473, 'test': 1319},
                 'humaneval': {'test': 164}, 'mt_bench': {'train': 80}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=PRESETS, default=DEFAULT_DATASETS)
    parser.add_argument('--gsm8k-splits', nargs='+', choices=['train', 'test'], default=['train', 'test'])
    parser.add_argument('--manifest', type=Path, default=HERE / 'download_manifest.json')
    args = parser.parse_args()
    from datasets import config as datasets_config, load_dataset
    from huggingface_hub import HfApi, constants

    print(f'Hub cache: {constants.HF_HUB_CACHE}', flush=True)
    print(f'Datasets cache: {datasets_config.HF_DATASETS_CACHE}', flush=True)
    manifest = {'downloaded_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                'hub_cache': str(constants.HF_HUB_CACHE),
                'datasets_cache': str(datasets_config.HF_DATASETS_CACHE), 'datasets': []}
    if args.manifest.is_file():
        previous = json.loads(args.manifest.read_text())
        if previous.get('hub_cache') == manifest['hub_cache'] and previous.get('datasets_cache') == manifest['datasets_cache']:
            manifest['datasets'] = previous.get('datasets', [])
    api = HfApi()
    for name in dict.fromkeys(args.datasets):
        config = DATASET_CONFIGS[PRESETS[name]]
        revision = api.dataset_info(config.hf_path).sha
        splits = args.gsm8k_splits if name == 'gsm8k' else [config.split]
        for split in dict.fromkeys(splits):
            # No cache_dir/local_dir/save_to_disk: keep the default Hub + Arrow caches.
            dataset = load_dataset(config.hf_path, name=config.subset, split=split, revision=revision)
            expected = EXPECTED_ROWS.get(name, {}).get(split)
            if expected is not None and len(dataset) != expected:
                raise ValueError(f'{name}/{split}: expected {expected} rows, got {len(dataset)}')
            prepared = config.normalize_fn(dataset[0]) if config.normalize_fn else dataset[0]
            if not prepared.get('conversations'):
                raise ValueError(f'{name}: normalization produced no conversation')
            manifest['datasets'] = [entry for entry in manifest['datasets']
                if (entry['hf_path'], entry['subset'], entry['split']) != (config.hf_path, config.subset, split)]
            manifest['datasets'].append({'dataset': name, 'hf_path': config.hf_path,
                'subset': config.subset, 'split': split, 'revision': revision,
                'rows': len(dataset), 'columns': dataset.column_names,
                'cache_files': dataset.cache_files})
            print(f'{name}/{split}: {len(dataset)} rows, revision={revision}', flush=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.manifest.with_suffix(args.manifest.suffix + '.tmp')
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(args.manifest)
    print(f'Manifest: {args.manifest}', flush=True)


if __name__ == '__main__':
    main()
