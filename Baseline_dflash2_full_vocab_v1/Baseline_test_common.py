#!/usr/bin/env python3
"""Held-out offline tests; sequential heads self-condition on their greedy Top1 (offline only)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from Baseline_common import (DEFAULT_OUT, SOURCE, VOCAB, dump, group_tensors,
                             load_data, make_head, sha, target_head)


def parser(family: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f'Full-vocab {family} head-only held-out test')
    ap.add_argument('--gpu', type=int, required=True, help='Physical GPU index')
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--top-k', type=int, default=16)
    ap.add_argument('--max-groups', type=int, default=0,
                    help='Smoke test only; reads smoke checkpoint, never writes official result')
    return ap


def sequential(family: str, model: torch.nn.Module, group: dict[str, torch.Tensor]) -> dict:
    raw = group['raw'][:, 1:].to(torch.bfloat16)
    target = group['teacher'][:, 1:].float()
    hidden = group['hidden'][:, 1:]
    valid = group['valid'][:, 1:]
    count = raw.shape[0]
    prev = group['token_ids'][:, 0].long()
    target_top1 = target.argmax(-1)
    raw_top1 = raw.argmax(-1)
    draft_top1 = []
    soft_overlap = []
    raw_overlap = []
    top1_coverage = []
    mass_coverage = []
    confidence_errors = []
    for pos in range(7):
        p = F.softmax(target[:, pos], dim=-1)
        raw_p = F.softmax(raw[:, pos].float(), dim=-1)
        raw_overlap.append(torch.minimum(p, raw_p).sum(-1))
        if family == 'dspark':
            scores, confidence = model(raw[:, pos], hidden[:, pos], prev)
            q = F.softmax(scores.float(), dim=-1)
            prediction = scores.argmax(-1)
            overlap = torch.minimum(p, q).sum(-1)
            top1_coverage.append(torch.ones(count, device=raw.device, dtype=torch.bool))
            mass_coverage.append(torch.ones(count, device=raw.device))
            confidence_errors.append((confidence.float().sigmoid() - overlap).abs())
        else:
            candidates, scores = model.inference(raw[:, pos], hidden[:, pos], prev)
            q = F.softmax(scores.float(), dim=-1)
            prediction = candidates.gather(1, scores.argmax(-1, keepdim=True)).squeeze(-1)
            p_candidates = p.gather(1, candidates)
            overlap = torch.minimum(p_candidates, q).sum(-1)
            top1_coverage.append(candidates.eq(target_top1[:, pos, None]).any(-1))
            mass_coverage.append(p_candidates.sum(-1))
        draft_top1.append(prediction)
        soft_overlap.append(overlap)
        prev = prediction  # inference uses its own Top1, never teacher forcing
    predicted = torch.stack(draft_top1, dim=1)
    soft = torch.stack(soft_overlap, dim=1)
    raw_soft = torch.stack(raw_overlap, dim=1)
    top1_coverage = torch.stack(top1_coverage, dim=1)
    mass_coverage = torch.stack(mass_coverage, dim=1)
    draft_match = predicted.eq(target_top1) & valid
    raw_match = raw_top1.eq(target_top1) & valid
    # Invalid later slots stop a prefix in both arms.
    draft_prefix = draft_match.int().cumprod(1)
    raw_prefix = raw_match.int().cumprod(1)
    draft_soft_prefix = (soft * valid.float()).cumprod(1)
    raw_soft_prefix = (raw_soft * valid.float()).cumprod(1)
    raw_eal = raw_prefix.sum(1).float()
    head_eal = draft_prefix.sum(1).float()
    result = {'group_id': int(group['group_id']), 'blocks': count,
              'valid_proposals': int(valid.sum()),
              'raw_greedy_eal': float(raw_eal.mean()),
              'head_greedy_eal': float(head_eal.mean()),
              'raw_soft_eal': float(raw_soft_prefix.sum(1).mean()),
              'head_soft_eal': float(draft_soft_prefix.sum(1).mean()),
              'raw_top1_accuracy': float(raw_match.float().sum() / valid.sum()),
              'head_top1_accuracy': float(draft_match.float().sum() / valid.sum()),
              'wrong_to_correct': int((~raw_match & draft_match & valid).sum()),
              'correct_to_wrong': int((raw_match & ~draft_match & valid).sum()),
              'harmful_blocks': int(head_eal.lt(raw_eal).sum()),
              'helpful_blocks': int(head_eal.gt(raw_eal).sum()),
              'changed_blocks': int(predicted.ne(raw_top1).logical_and(valid).any(1).sum()),
              'target_top1_candidate_coverage': float((top1_coverage & valid).float().sum() / valid.sum()),
              'target_probability_mass_coverage': float((mass_coverage * valid.float()).sum() / valid.sum()),
              'per_position': []}
    if confidence_errors:
        error = torch.stack(confidence_errors, dim=1)
        result['confidence_abs_error'] = float((error * valid.float()).sum() / valid.sum())
    for pos in range(7):
        v = valid[:, pos]
        if not v.any():
            result['per_position'].append({'position': pos + 1, 'valid': 0})
            continue
        result['per_position'].append({
            'position': pos + 1, 'valid': int(v.sum()),
            'raw_top1_accuracy': float(raw_match[:, pos].float().sum() / v.sum()),
            'head_top1_accuracy': float(draft_match[:, pos].float().sum() / v.sum()),
            'raw_prefix_survival': float(raw_prefix[:, pos].float().mean()),
            'head_prefix_survival': float(draft_prefix[:, pos].float().mean()),
            'raw_soft_prefix': float(raw_soft_prefix[:, pos].mean()),
            'head_soft_prefix': float(draft_soft_prefix[:, pos].mean()),
            'target_top1_candidate_coverage': float((top1_coverage[:, pos] & v).float().sum() / v.sum())})
    return result


def summarize(results: list[dict], family: str) -> dict:
    blocks = np.array([r['blocks'] for r in results], dtype=float)
    raw = np.array([r['raw_greedy_eal'] for r in results])
    head = np.array([r['head_greedy_eal'] for r in results])
    raw_soft = np.array([r['raw_soft_eal'] for r in results])
    head_soft = np.array([r['head_soft_eal'] for r in results])
    weighted = lambda values, ii: float(np.average(values[ii], weights=blocks[ii]))
    all_idx = np.arange(len(results))
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        ii = rng.integers(0, len(results), size=len(results))
        boot.append(weighted(head - raw, ii))
    folds = []
    for repeat in range(5):
        order = np.random.default_rng(42000 + repeat).permutation(len(results))
        for fold, ii in enumerate(np.array_split(order, min(5, len(results)))):
            folds.append({'repeat': repeat, 'fold': fold, 'groups': len(ii),
                          'greedy_eal_delta': weighted(head - raw, ii),
                          'soft_eal_delta': weighted(head_soft - raw_soft, ii)})
    total_valid = sum(r['valid_proposals'] for r in results)
    total_blocks = int(blocks.sum())
    return {'family': family, 'evaluation': 'offline_frozen_hidden_teacher_distribution_T1',
            'native_T0_acceptance': None, 'native_T1_acceptance': None,
            'groups': len(results), 'blocks': total_blocks,
            'raw_greedy_eal': weighted(raw, all_idx),
            'head_greedy_eal': weighted(head, all_idx),
            'greedy_eal_delta': weighted(head - raw, all_idx),
            'raw_soft_eal': weighted(raw_soft, all_idx),
            'head_soft_eal': weighted(head_soft, all_idx),
            'soft_eal_delta': weighted(head_soft - raw_soft, all_idx),
            'grouped_bootstrap_greedy_delta_lcb90': float(np.quantile(boot, .1)),
            'grouped_bootstrap_greedy_delta_ci90': [float(np.quantile(boot, .05)), float(np.quantile(boot, .95))],
            'repeated_grouped_folds_5x5': folds,
            'raw_top1_accuracy': sum(r['raw_top1_accuracy'] * r['valid_proposals'] for r in results) / total_valid,
            'head_top1_accuracy': sum(r['head_top1_accuracy'] * r['valid_proposals'] for r in results) / total_valid,
            'target_top1_candidate_coverage': sum(r['target_top1_candidate_coverage'] * r['valid_proposals'] for r in results) / total_valid,
            'target_probability_mass_coverage': sum(r['target_probability_mass_coverage'] * r['valid_proposals'] for r in results) / total_valid,
            'wrong_to_correct': sum(r['wrong_to_correct'] for r in results),
            'correct_to_wrong': sum(r['correct_to_wrong'] for r in results),
            'harmful_block_rate': sum(r['harmful_blocks'] for r in results) / total_blocks,
            'helpful_block_rate': sum(r['helpful_blocks'] for r in results) / total_blocks,
            'changed_block_rate': sum(r['changed_blocks'] for r in results) / total_blocks,
            'per_position': [{
                'position': pos + 1,
                'valid': sum(r['per_position'][pos]['valid'] for r in results),
                **{key: float(np.average(
                    [r['per_position'][pos].get(key, 0.) for r in results],
                    weights=[r['per_position'][pos]['valid'] for r in results]))
                   for key in ('raw_top1_accuracy', 'head_top1_accuracy',
                               'target_top1_candidate_coverage')},
                **{key: weighted(np.array([r['per_position'][pos].get(key, 0.) for r in results]), all_idx)
                   for key in ('raw_prefix_survival', 'head_prefix_survival',
                               'raw_soft_prefix', 'head_soft_prefix')}
            } for pos in range(7)]}


def test(family: str) -> None:
    args = parser(family).parse_args()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise RuntimeError('Use the physical --gpu index without CUDA_VISIBLE_DEVICES')
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{args.gpu}')
    family_dir = args.output_dir / family
    if args.max_groups:
        family_dir = family_dir / f'smoke_{args.max_groups}_groups'
    training = json.loads((family_dir / 'training_summary.json').read_text())
    ckpt = Path(training['fixed_checkpoint'])
    if ckpt.name != f'epoch_{args.epochs:02d}.safetensors' or sha(ckpt) != training['fixed_checkpoint_sha256']:
        raise RuntimeError('fixed checkpoint or SHA changed')
    protocol_path = family_dir / 'training_protocol.json'
    if sha(protocol_path) != training['training_protocol_sha256']:
        raise RuntimeError('training protocol changed after checkpoint')
    protocol = json.loads(protocol_path.read_text())
    if protocol['family'] != family or protocol['top_k'] != args.top_k or protocol['max_groups'] != args.max_groups:
        raise RuntimeError('evaluation parameters differ from training protocol')
    rows, tokens, _ = load_data(args.output_dir, 'final')
    if args.max_groups:
        rows, tokens = rows[:args.max_groups], tokens[:args.max_groups]
    weight, _ = target_head(device)
    model = make_head(family, top_k=args.top_k).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(load_file(str(ckpt), device='cpu'), strict=True)
    model.eval()
    results = []
    with torch.inference_mode():
        for i, (row, token_row) in enumerate(zip(rows, tokens)):
            group = group_tensors(row, token_row, device, weight)
            results.append(sequential(family, model, group))
            if (i + 1) % 16 == 0:
                print(f'[{family}] final groups {i+1}/{len(rows)}', flush=True)
    summary = summarize(results, family)
    summary.update({'status': 'complete', 'fixed_checkpoint': str(ckpt),
                    'fixed_checkpoint_sha256': sha(ckpt),
                    'source_preregistration_sha256': sha(SOURCE / 'preregistration.json'),
                    'final_group_ids': [r['group_id'] for r in results]})
    dump(family_dir / 'offline_group_results.json', {'groups': results})
    dump(family_dir / 'offline_summary.json', summary)
    print(f'[{family}] offline greedy ΔEAL={summary["greedy_eal_delta"]:.4f}; '
          f'offline soft ΔEAL={summary["soft_eal_delta"]:.4f}', flush=True)

