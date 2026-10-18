#!/usr/bin/env python3
"""Frozen-backbone, full-vocabulary head training shared by separate CLIs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from Baseline_common import (DEFAULT_OUT, SOURCE, VOCAB, dump, group_tensors,
                             load_data, make_head, sha, target_head)


def parser(family: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f'Full-vocab {family} head-only training')
    ap.add_argument('--gpu', type=int, required=True, help='Physical GPU index')
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--groups-per-step', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=.01)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--top-k', type=int, default=16)
    ap.add_argument('--max-groups', type=int, default=0,
                    help='Smoke test only; stored in a separate output directory')
    return ap


def loss_for_group(family: str, model: torch.nn.Module,
                   group: dict[str, torch.Tensor], *, detach_base: bool = True) -> tuple[torch.Tensor, dict]:
    valid = group['valid'][:, 1:]
    if not valid.any():
        raise RuntimeError('cache group has no valid proposal slots')
    unary = group['raw'][:, 1:][valid]
    target = group['teacher'][:, 1:][valid].detach()
    hidden = group['hidden'][:, 1:][valid]
    if detach_base:
        unary, hidden = unary.detach(), hidden.detach()
    prev = group['token_ids'][:, :-1][valid]
    positions = torch.arange(7, device=valid.device)[None].expand_as(valid)[valid]
    weights = torch.exp(-positions.float() / 4.0)
    target_top1 = target.argmax(-1)
    if family == 'dspark':
        scores, confidence = model(unary, hidden, prev)
        p = F.softmax(target.float(), dim=-1)
        q = F.softmax(scores.float(), dim=-1)
        overlap = torch.minimum(p, q).sum(-1)
        tv = 1.0 - overlap
        ce = F.cross_entropy(scores.float(), target_top1, reduction='none')
        bce = F.binary_cross_entropy_with_logits(
            confidence.float(), overlap.detach(), reduction='none')
        loss = ((.1 * ce + .9 * tv + bce) * weights).sum() / weights.sum()
        metrics = {'ce': float(ce.detach().mean()), 'tv': float(tv.detach().mean()),
                   'confidence_bce': float(bce.detach().mean()),
                   'target_top1_in_unary_top16': float(unary.topk(16, -1).indices.eq(target_top1[:, None]).any(-1).float().mean())}
    else:
        candidates, label = model.training_candidates(unary, target_top1)
        scores = model.score_candidates(unary, hidden, prev, candidates)
        ce = F.cross_entropy(scores.float(), label, reduction='none')
        loss = (ce * weights).sum() / weights.sum()
        natural = model.candidates(unary).eq(target_top1[:, None]).any(-1)
        metrics = {'candidate_ce': float(ce.detach().mean()),
                   'target_top1_in_unary_top16': float(natural.float().mean())}
    if not torch.isfinite(loss):
        raise RuntimeError('non-finite head loss')
    return loss, metrics


def train(family: str) -> None:
    args = parser(family).parse_args()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise RuntimeError('Use the physical --gpu index without CUDA_VISIBLE_DEVICES')
    if args.seed != 42 or args.epochs <= 0 or args.groups_per_step <= 0:
        raise ValueError('protocol requires seed 42 and positive training budget')
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{args.gpu}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    family_dir = args.output_dir / family
    if args.max_groups:
        family_dir = family_dir / f'smoke_{args.max_groups}_groups'
    family_dir.mkdir(parents=True, exist_ok=True)
    rows, tokens, _ = load_data(args.output_dir, 'train')
    if args.max_groups:
        rows, tokens = rows[:args.max_groups], tokens[:args.max_groups]
    weight, weight_path = target_head(device)
    model = make_head(family, top_k=args.top_k).to(device=device, dtype=torch.bfloat16).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    protocol = {'family': family, 'precision_policy': 'bf16-head-v2-fp32-prob-loss', 'seed': args.seed, 'epochs': args.epochs,
                'groups_per_step': args.groups_per_step, 'lr': args.lr,
                'weight_decay': args.weight_decay, 'top_k': args.top_k,
                'max_groups': args.max_groups, 'train_groups': len(rows),
                'source_preregistration_sha256': sha(SOURCE / 'preregistration.json'),
                'source_compact_manifest_sha256': sha(SOURCE / 'compact_manifest.json'),
                'token_manifest_sha256': sha(args.output_dir / 'token_manifest.json'),
                'target_lm_head_file': str(weight_path),
                'target_lm_head_file_sha256': sha(weight_path),
                'target_lm_head_shape': list(weight.shape),
                'frozen_target_lm_head': True,
                'frozen_dflash_backbone': True,
                'checkpoint_policy': 'fixed final epoch; no validation selection',
                'source_data': 'target-regenerated OpenMath only',
                'teacher_forced_predecessor_train': True,
                'self_conditioned_predecessor_test': True,
                'dspark_loss': '0.1 CE(target Top1)+0.9 TV(full target distribution)+1.0 confidence BCE(overlap), exp(-proposal_index/4)',
                'dflash2_loss': 'experimental Top16 candidate CE; target injected in last slot for training only, exp(-proposal_index/4)',
                'head_only': True}
    protocol_path = family_dir / 'training_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError('attempted resume with changed training protocol')
    else:
        dump(protocol_path, protocol)
    resume = family_dir / 'resume.pt'
    start_epoch = 0
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['protocol_sha256'] != sha(protocol_path):
            raise RuntimeError('resume/protocol SHA mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], args.gpu)
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        start_epoch = int(state['epoch'])
    for epoch in range(start_epoch, args.epochs):
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        metric_totals: dict[str, float] = {}
        start = time.monotonic()
        for step, i in enumerate(order):
            group = group_tensors(rows[i], tokens[i], device, weight)
            loss, metrics = loss_for_group(family, model, group)
            (loss / args.groups_per_step).backward()
            if (step + 1) % args.groups_per_step == 0 or step + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_total += float(loss.detach())
            for key, value in metrics.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + value
            if (step + 1) % 64 == 0:
                dump(family_dir / 'status.json', {'stage': 'training', 'family': family,
                     'epoch': epoch + 1, 'epochs': args.epochs, 'groups_done': step + 1,
                     'groups_total': len(order), 'mean_loss': loss_total / (step + 1)})
                print(f'[{family}] epoch {epoch+1}/{args.epochs} groups {step+1}/{len(order)} loss={loss_total/(step+1):.5f}', flush=True)
        ckpt = family_dir / f'epoch_{epoch+1:02d}.safetensors'
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(ckpt))
        payload = {'epoch': epoch + 1, 'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state(args.gpu),
                   'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                   'protocol_sha256': sha(protocol_path)}
        tmp = Path(str(resume) + '.partial')
        torch.save(payload, tmp)
        tmp.replace(resume)
        epoch_result = {'epoch': epoch + 1, 'mean_loss': loss_total / len(order),
                        'mean_terms': {k: v / len(order) for k, v in metric_totals.items()},
                        'seconds': time.monotonic() - start,
                        'checkpoint': str(ckpt), 'checkpoint_sha256': sha(ckpt)}
        dump(family_dir / f'epoch_{epoch+1:02d}.json', epoch_result)
        dump(family_dir / 'status.json', {'stage': 'epoch_complete', 'family': family,
             **epoch_result})
    final = family_dir / f'epoch_{args.epochs:02d}.safetensors'
    dump(family_dir / 'training_summary.json',
         {'status': 'complete', 'family': family, 'fixed_checkpoint': str(final),
          'fixed_checkpoint_sha256': sha(final), 'training_protocol_sha256': sha(protocol_path),
          'train_groups': len(rows), 'epochs': args.epochs,
          'final_epoch_metrics': json.loads((family_dir / f'epoch_{args.epochs:02d}.json').read_text())})
    dump(family_dir / 'status.json', {'stage': 'training_complete', 'family': family,
         'fixed_checkpoint': str(final)})
    print(f'[{family}] fixed checkpoint: {final}', flush=True)
