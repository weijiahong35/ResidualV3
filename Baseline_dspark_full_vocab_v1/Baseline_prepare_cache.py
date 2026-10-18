#!/usr/bin/env python3
"""Recover exact teacher-forced block tokens for the frozen full-vocab cache.

No target server or draft forward pass is needed. The original collector seeded
select_anchors with 31000 + source_index. Every recovered anchor ID and valid
mask is compared against the existing compact cache before it is saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from speculators.models.dflash.utils import select_anchors

SOURCE = ROOT / 'eval_results/residualV3_full_vocab_unified_lag1_v3_seed42'
DEFAULT_OUT = ROOT / 'eval_results/Baseline_dspark_full_vocab_v1_seed42'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def recover(split: str, selected: list[dict], cache_rows: list[dict], data_path: Path,
            device: torch.device) -> list[dict]:
    data = load_from_disk(str(data_path))
    if len(selected) != len(cache_rows):
        raise RuntimeError(f'{split}: selected/cache length mismatch')
    recovered = []
    for i, (row, cache_row) in enumerate(zip(selected, cache_rows)):
        if int(row['group_id']) != int(cache_row['group_id']):
            raise RuntimeError(f'{split} group order changed at {i}')
        source_index = int(row['index'])
        item = data[source_index]
        ids = item['input_ids'].long()
        loss_mask = item['loss_mask'].bool()
        if ids.shape != loss_mask.shape or ids.ndim != 1:
            raise RuntimeError(f'{split} prepared row shape changed at {source_index}')
        # CollateFn pads every single-row batch to 8192 before anchor selection.
        if ids.numel() > 8192:
            ids, loss_mask = ids[:8192], loss_mask[:8192]
        elif ids.numel() < 8192:
            pad = 8192 - ids.numel()
            ids = torch.nn.functional.pad(ids, (0, pad))
            loss_mask = torch.nn.functional.pad(loss_mask, (0, pad))
        torch.manual_seed(31000 + source_index)
        torch.cuda.manual_seed_all(31000 + source_index)
        anchors, anchor_valid = select_anchors(loss_mask[None].to(device), 16, 8)
        anchors = anchors.cpu()
        anchor_valid = anchor_valid.cpu()
        positions = anchors[:, None] + torch.arange(8)[None]
        valid = loss_mask[positions] & anchor_valid[:, None]
        valid[:, 0] = False  # sample_from_anchor=False
        real = valid[:, 1]
        token_blocks = ids[positions][real].contiguous()
        valid = valid[real].contiguous()
        cache_path = Path(cache_row['path'])
        if sha(cache_path) != cache_row['sha256']:
            raise RuntimeError(f'{split} compact shard SHA changed: {cache_path}')
        compact = torch.load(cache_path, map_location='cpu', weights_only=True)
        if (int(compact['group_id']) != int(row['group_id'])
                or int(compact['source_index']) != source_index
                or not torch.equal(token_blocks[:, 0], compact['anchor_token_id'])
                or not torch.equal(valid, compact['valid'])
                or len(token_blocks) != len(compact['draft_hidden'])):
            raise RuntimeError(f'{split} exact anchor reconstruction failed: row {i}, source {source_index}')
        recovered.append({'source_index': source_index, 'group_id': int(row['group_id']),
                          'block_token_ids': token_blocks})
        if (i + 1) % 256 == 0:
            print(f'[prepare] {split} {i+1}/{len(selected)} exact anchor checks', flush=True)
    return recovered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True, help='Physical GPU; a free GPU is sufficient')
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if 'CUDA_VISIBLE_DEVICES' in __import__('os').environ:
        raise RuntimeError('Use physical GPU index; CUDA_VISIBLE_DEVICES must be unset')
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    protocol_path = SOURCE / 'preregistration.json'
    compact_path = SOURCE / 'compact_manifest.json'
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(compact_path.read_text())
    if manifest['preregistration_sha256'] != sha(protocol_path):
        raise RuntimeError('source compact manifest and protocol differ')
    if protocol['source_segments'] != {'target_regenerated_openmath': [0, 2367]}:
        raise RuntimeError('source training set is not regenerated-only')
    train_rows = [r for r in manifest['shards'] if r['split'] == 'train']
    final_rows = [r for r in manifest['shards'] if r['split'] == 'final']
    if len(train_rows) != 2368 or len(final_rows) != 64:
        raise RuntimeError('source cache is incomplete')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, rows, cache, data in (
        ('train', protocol['train'], train_rows, Path(protocol['regenerated_prepared_data_path'])),
        ('final', protocol['final'], final_rows, Path(protocol['final_data_path'])),
    ):
        path = args.output_dir / f'block_tokens_{split}.pt'
        if path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if [r['group_id'] for r in payload] != [r['group_id'] for r in rows]:
                raise RuntimeError(f'existing {split} token sidecar differs from protocol')
        else:
            payload = recover(split, rows, cache, data, device)
            tmp = path.with_suffix('.pt.partial')
            torch.save(payload, tmp)
            tmp.replace(path)
        outputs[split] = {'path': str(path), 'sha256': sha(path), 'groups': len(rows)}
    audit = {'status': 'complete', 'format': 'exact_teacher_forced_block_tokens_v1',
             'source_preregistration_sha256': sha(protocol_path),
             'source_compact_manifest_sha256': sha(compact_path),
             'source_regenerated_file_sha256': protocol['regenerated_file_sha256'],
             'train_final_group_overlap': 0, 'physical_gpu': args.gpu,
             'anchor_seed_rule': '31000 + source_index',
             'anchor_ids_and_valid_masks_checked_for_every_group': True,
             'sidecars': outputs}
    save_json(args.output_dir / 'token_manifest.json', audit)
    print('[prepare] exact token sidecars complete', flush=True)


if __name__ == '__main__':
    main()
