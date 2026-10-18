#!/usr/bin/env python3
from __future__ import annotations
import json
from Baseline_common import DEFAULT_OUT, dump, sha

def main() -> None:
    folder = DEFAULT_OUT / 'dflash2'
    train = folder / 'training_summary.json'
    test = folder / 'offline_summary.json'
    manifest = DEFAULT_OUT / 'token_manifest.json'
    for path in (train, test):
        if not path.exists() or json.loads(path.read_text())['status'] != 'complete':
            raise RuntimeError(f'incomplete baseline: {path}')
    dump(DEFAULT_OUT / 'completion.json', {
        'status': 'complete', 'family': 'dflash2',
        'scope': 'head_only_full_output_vocabulary_offline_baseline',
        'training_summary_sha256': sha(train),
        'offline_summary_sha256': sha(test),
        'token_manifest_sha256': sha(manifest),
        'native_runtime_results': None})
    print(f'[dflash2] complete: {DEFAULT_OUT / "completion.json"}', flush=True)

if __name__ == '__main__':
    main()
