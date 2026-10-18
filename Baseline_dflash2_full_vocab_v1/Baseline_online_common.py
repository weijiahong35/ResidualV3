#!/usr/bin/env python3
"""Online target hidden extraction with guaranteed delete; live full-vocab body."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import sys
from Baseline_common import SOURCE, ROOT, VOCAB, HIDDEN, sha
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'hs_connectors/src'))

from datasets import load_from_disk
from hs_connectors import FileTransfer
import openai
from safetensors import safe_open
from safetensors.torch import load_file
import torch
from torch import nn
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import generate_hidden_states
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel


BODY = Path(__file__).resolve().parent / 'body_only_draft'
TARGET = ROOT.parent / 'huggingface/models/Qwen3-8B'
LAYER_IDS = (2, 10, 18, 26, 34)
BLOCK = 8


def token_hash(ids: torch.Tensor) -> int:
    ids = torch.as_tensor(ids, dtype=torch.long)[:8192].contiguous()
    v = int.from_bytes(hashlib.sha256(ids.numpy().tobytes()).digest()[:8], 'little')
    return v if v < 2**63 else v - 2**64


def split_rows(split: str) -> tuple[list[dict], object, dict]:
    protocol_path = SOURCE / 'preregistration.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol['source_segments'] != {'target_regenerated_openmath': [0, 2367]}:
        raise RuntimeError('online training source must be regenerated-only')
    if {r['group_id'] for r in protocol['train']} & {r['group_id'] for r in protocol['final']}:
        raise RuntimeError('train/final overlap')
    rows = protocol[split]
    path = Path(protocol['regenerated_prepared_data_path'] if split == 'train' else protocol['final_data_path'])
    data = load_from_disk(str(path))
    expected = protocol['regenerated_prepared_arrow_sha256'] if split == 'train' else protocol['final_data_arrow_sha256']
    for path, digest in expected.items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f'online source Arrow changed: {path}')
    return rows, data, protocol


def get_row(row: dict, data: object, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    item = data[int(row['index'])]
    ids = torch.as_tensor(item['input_ids'], dtype=torch.long)
    loss = torch.as_tensor(item['loss_mask'], dtype=torch.bool)
    if ids.shape != loss.shape or token_hash(ids) != int(row['group_id']):
        raise RuntimeError(f'online trajectory group mismatch: {row["index"]}')
    if ids.numel() < 16 or max_seq_len < 16:
        raise RuntimeError('online sequence too short')
    return ids[:max_seq_len].contiguous(), loss[:max_seq_len].contiguous()


class LiveHiddenFetcher:
    """Never reads cached hidden files; deletes each generated handle in finally."""
    def __init__(self, endpoint: str, hidden_states_path: Path, model_name: str) -> None:
        self.client = openai.OpenAI(base_url=endpoint, api_key='EMPTY', max_retries=0)
        self.model_name = model_name
        self.transfer = FileTransfer(hidden_states_path)
        self.hidden_states_path = hidden_states_path.resolve()
        self.hidden_states_path.mkdir(parents=True, exist_ok=True)
        active = [p for p in self.hidden_states_path.iterdir() if p.is_file()]
        if active:
            raise RuntimeError(f'ephemeral hidden directory must be empty: {active[:3]}')
        listed = self.client.models.list().data
        if len(listed) != 1 or listed[0].id != model_name:
            raise RuntimeError(f'target server model mismatch: {[m.id for m in listed]}')
        self.generated = 0
        self.deleted = 0

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        handle: str | None = None
        try:
            handle = generate_hidden_states(
                self.client, self.model_name, {'input_ids': ids.tolist()},
                timeout=600, max_retries=0)
            path = Path(handle).resolve()
            if not path.is_relative_to(self.hidden_states_path):
                raise RuntimeError(f'hidden handle escaped ephemeral directory: {path}')
            self.generated += 1
            payload = self.transfer.get_generated(handle)
            if payload is None:
                raise RuntimeError(f'generated hidden file missing: {handle}')
            check_hidden_states(payload, ids.tolist())
            hidden = payload['hidden_states']
            if hidden.shape != (ids.numel(), len(LAYER_IDS)+1, HIDDEN):
                raise RuntimeError(f'target hidden shape mismatch: {tuple(hidden.shape)}')
            return hidden
        finally:
            if handle is not None:
                # FileTransfer.delete is the same connector operation used by
                # --on-generate delete. The finally block also handles failures.
                path = Path(handle).resolve()
                if path.is_relative_to(self.hidden_states_path):
                    self.transfer.delete(handle)
                    self.deleted += 1


class MaskedTargetEmbedding(nn.Module):
    def __init__(self, target_embed: nn.Embedding, mask_id: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.target_embed = target_embed
        self.mask_id = mask_id
        self.register_buffer('separate_mask', mask.detach().clone())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ordinary = self.target_embed(ids)
        return torch.where(ids.eq(self.mask_id).unsqueeze(-1),
                           self.separate_mask.to(dtype=ordinary.dtype), ordinary)


def _target_weight(key: str) -> torch.Tensor:
    index = json.loads((TARGET / 'model.safetensors.index.json').read_text())['weight_map']
    path = TARGET / index[key]
    with safe_open(str(path), framework='pt', device='cpu') as f:
        return f.get_tensor(key)


def build_live_body(device: torch.device, *, train_body: bool) -> DFlashDraftModel:
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise RuntimeError('Use physical --gpu; CUDA_VISIBLE_DEVICES must be unset')
    config = DFlashSpeculatorConfig.from_pretrained(str(BODY), local_files_only=True)
    if config.draft_vocab_size != VOCAB or config.block_size != BLOCK or config.sample_from_anchor:
        raise RuntimeError('body-only full-vocab DFlash config changed')
    body = DFlashDraftModel(config)
    state = load_file(str(BODY / 'model.safetensors'), device='cpu')
    missing, unexpected = body.load_state_dict(state, strict=False)
    allowed = {'embed_tokens.weight', 'lm_head.weight', 'verifier_lm_head.weight', 'verifier_norm.weight'}
    if set(missing) - allowed or unexpected:
        raise RuntimeError(f'body transplant mismatch missing={missing} unexpected={unexpected}')
    del state
    embed = _target_weight('model.embed_tokens.weight')
    head = _target_weight('lm_head.weight')
    norm = _target_weight('model.norm.weight')
    if embed.shape != head.shape or embed.shape != (VOCAB,HIDDEN) or norm.shape != (HIDDEN,):
        raise RuntimeError('full target parameter shapes changed')
    target_embed = nn.Embedding(VOCAB,HIDDEN)
    target_embed.weight = nn.Parameter(embed,requires_grad=False)
    target_head = nn.Linear(HIDDEN,VOCAB,bias=False)
    target_head.weight = nn.Parameter(head,requires_grad=False)
    mask_payload = torch.load(BODY/'mask_embedding.pt',map_location='cpu',weights_only=True)
    if int(mask_payload['mask_token_id']) != int(config.mask_token_id):
        raise RuntimeError('separate mask ID mismatch')
    body.embed_tokens = MaskedTargetEmbedding(target_embed,int(config.mask_token_id),mask_payload['embedding'])
    body.lm_head = target_head
    body.verifier_lm_head = target_head
    body.verifier_norm.weight = nn.Parameter(norm,requires_grad=False)
    body.to(device=device,dtype=torch.bfloat16)
    for parameter in body.parameters():
        parameter.requires_grad_(False)
    if train_body:
        for name,parameter in body.named_parameters():
            if name.startswith(('layers.','fc.','hidden_norm.','norm.')):
                parameter.requires_grad_(True)
    if body.lm_head is not body.verifier_lm_head or body.lm_head.weight is body.embed_tokens.target_embed.weight:
        raise RuntimeError('target parameter sharing/distinction contract failed')
    if body.t2d is not None or body.d2t is not None:
        raise RuntimeError('full-vocab online body created a token mapping')
    return body


def live_group(row: dict, data: object, body: DFlashDraftModel,
               fetcher: LiveHiddenFetcher, device: torch.device,
               max_seq_len: int, *, max_anchors: int, train_body: bool) -> dict[str, torch.Tensor]:
    ids_cpu, loss_cpu = get_row(row, data, max_seq_len)
    hidden_cpu = fetcher(ids_cpu)
    ids = ids_cpu.unsqueeze(0).to(device)
    loss = loss_cpu.unsqueeze(0).to(device)
    aux = hidden_cpu[:,:-1].reshape(1,ids.shape[1],len(LAYER_IDS)*HIDDEN).to(device)
    last = hidden_cpu[:,-1].unsqueeze(0).to(device)
    n = ids.shape[1]
    document_ids = torch.zeros_like(ids)
    position_ids = torch.arange(n, device=device).unsqueeze(0)
    with torch.set_grad_enabled(train_body), torch.autocast('cuda', dtype=torch.bfloat16):
        hidden, raw, teacher, valid, anchored = body._backbone_forward(
            aux, ids, loss, last, document_ids, position_ids, max_anchors=max_anchors)
    token_ids = ids[0, anchored].view(max_anchors, BLOCK)
    return {'raw': raw.reshape(max_anchors, BLOCK, VOCAB),
            'teacher': teacher.reshape(max_anchors, BLOCK, VOCAB),
            'hidden': hidden.reshape(max_anchors, BLOCK, HIDDEN),
            'valid': valid.reshape(max_anchors, BLOCK).bool(),
            'token_ids': token_ids, 'group_id': torch.tensor(int(row['group_id']))}
