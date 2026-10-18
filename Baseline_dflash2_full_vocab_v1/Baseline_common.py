"""Frozen-cache I/O and exact full-target projection shared by both baselines."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from Baseline_models import DFlash2CandidateSelector

SOURCE = ROOT / 'eval_results/residualV3_full_vocab_unified_lag1_v3_seed42'
DEFAULT_OUT = ROOT / 'eval_results/Baseline_dflash2_full_vocab_bf16_v2_seed42'
TARGET = ROOT.parent / 'huggingface/models/Qwen3-8B'
VOCAB = 151936
HIDDEN = 4096
RANK = 256
POSITIONS = 7


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.partial')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def target_head(device: torch.device) -> tuple[torch.Tensor, Path]:
    index = json.loads((TARGET / 'model.safetensors.index.json').read_text())['weight_map']
    path = TARGET / index['lm_head.weight']
    with safe_open(str(path), framework='pt', device='cpu') as f:
        weight = f.get_tensor('lm_head.weight')
    if weight.shape != (VOCAB, HIDDEN):
        raise RuntimeError('target LM head shape changed')
    weight = weight.to(device=device, dtype=torch.bfloat16).detach()
    weight.requires_grad_(False)
    return weight, path


def make_head(family: str, vocab: int = VOCAB, hidden: int = HIDDEN,
              rank: int = RANK, top_k: int = 16) -> torch.nn.Module:
    if family == 'dflash2':
        return DFlash2CandidateSelector(vocab, hidden, rank, top_k)
    raise ValueError(family)


def load_data(out: Path, split: str) -> tuple[list[dict], list[dict], dict]:
    if split not in {'train', 'final'}:
        raise ValueError(split)
    protocol_path = SOURCE / 'preregistration.json'
    compact_path = SOURCE / 'compact_manifest.json'
    protocol = json.loads(protocol_path.read_text())
    compact = json.loads(compact_path.read_text())
    token_manifest = json.loads((out / 'token_manifest.json').read_text())
    if compact['preregistration_sha256'] != sha(protocol_path):
        raise RuntimeError('compact cache protocol mismatch')
    if token_manifest['source_compact_manifest_sha256'] != sha(compact_path):
        raise RuntimeError('token sidecars refer to another cache')
    if token_manifest['source_preregistration_sha256'] != sha(protocol_path):
        raise RuntimeError('token sidecars refer to another split')
    sidecar = token_manifest['sidecars'][split]
    path = Path(sidecar['path'])
    if sha(path) != sidecar['sha256']:
        raise RuntimeError('token sidecar SHA changed')
    tokens = torch.load(path, map_location='cpu', weights_only=True)
    rows = [r for r in compact['shards'] if r['split'] == split]
    if len(rows) != sidecar['groups'] or len(tokens) != len(rows):
        raise RuntimeError('cache/sidecar group count mismatch')
    if [int(r['group_id']) for r in rows] != [int(r['group_id']) for r in protocol[split]]:
        raise RuntimeError('cache/protocol group order mismatch')
    if [int(r['group_id']) for r in rows] != [int(r['group_id']) for r in tokens]:
        raise RuntimeError('token/cache group order mismatch')
    if {int(r['group_id']) for r in protocol['train']} & {int(r['group_id']) for r in protocol['final']}:
        raise RuntimeError('train/final group overlap')
    return rows, tokens, protocol


def group_tensors(row: dict, token_row: dict, device: torch.device,
                  lm_head: torch.Tensor) -> dict[str, torch.Tensor]:
    path = Path(row['path'])
    if sha(path) != row['sha256']:
        raise RuntimeError(f'compact shard SHA changed: {path}')
    item = torch.load(path, map_location='cpu', weights_only=True)
    ids = token_row['block_token_ids'].to(device)
    hidden = item['draft_hidden'].to(device)
    teacher_hidden = item['teacher_hidden'].to(device)
    valid = item['valid'].to(device).bool()
    anchor = item['anchor_token_id'].to(device)
    if (ids.shape != valid.shape or hidden.shape[:2] != valid.shape
            or hidden.shape[-1] != HIDDEN or teacher_hidden.shape != hidden.shape
            or not torch.equal(ids[:, 0], anchor)
            or not ids.ge(0).all() or not ids.lt(VOCAB).all()):
        raise RuntimeError(f'compact/token shape mismatch: {path}')
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                         enabled=device.type == 'cuda'):
        raw = (hidden.reshape(-1, HIDDEN).to(lm_head.dtype) @ lm_head.T)
        teacher = (teacher_hidden.reshape(-1, HIDDEN).to(lm_head.dtype) @ lm_head.T)
    return {'raw': raw.reshape(*valid.shape, VOCAB),
            'teacher': teacher.reshape(*valid.shape, VOCAB),
            'hidden': hidden, 'valid': valid, 'token_ids': ids,
            'group_id': torch.tensor(int(row['group_id']))}
