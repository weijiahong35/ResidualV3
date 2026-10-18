#!/usr/bin/env python3
"""Train a full-vocab baseline from live target and draft forwards, no hidden cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from safetensors.torch import save_file

from Baseline_common import DEFAULT_OUT, SOURCE, dump, make_head, sha
from Baseline_online_common import BODY, TARGET, LiveHiddenFetcher, build_live_body, live_group, split_rows
from Baseline_train_common import loss_for_group

FAMILY = 'dflash2'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True, help='physical training GPU')
    ap.add_argument('--vllm-endpoint', required=True)
    ap.add_argument('--hidden-states-path', type=Path, required=True)
    ap.add_argument('--on-missing', choices=['generate'], default='generate')
    ap.add_argument('--on-generate', choices=['delete'], default='delete')
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--groups-per-step', type=int, default=4)
    ap.add_argument('--body-lr', type=float, default=1e-5)
    ap.add_argument('--head-lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=.01)
    ap.add_argument('--max-anchors', type=int, default=16)
    ap.add_argument('--max-seq-len', type=int, default=8192)
    ap.add_argument('--max-groups', type=int, default=0, help='smoke test; separate output')
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument('--head-only', action='store_true', help='default: freeze body')
    scope.add_argument('--train-body', action='store_true', help='explicit body+head variant')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if 'CUDA_VISIBLE_DEVICES' in os.environ or args.seed != 42:
        raise RuntimeError('Use physical --gpu without CUDA_VISIBLE_DEVICES; seed is 42')
    if args.epochs < 1 or args.max_anchors < 1 or args.groups_per_step < 1:
        raise ValueError('positive epoch, group and anchor budgets required')
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{args.gpu}')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed(42)
    out = args.output_dir / FAMILY / 'online'
    if args.max_groups:
        out = out / f'smoke_{args.max_groups}_groups'
    out.mkdir(parents=True, exist_ok=True)
    rows, data, source_protocol = split_rows('train')
    if args.max_groups:
        rows = rows[:args.max_groups]
    body_trainable = args.train_body
    protocol = {'kind': f'Baseline_{FAMILY}_online_full_vocab_v1', 'family': FAMILY,
                'precision_policy': 'bf16-head-v2-fp32-prob-loss', 'seed': 42, 'training_mode': 'vllm_generate_delete_each_group_and_live_dflash_forward_no_hidden_snapshots',
                'target_model': str(TARGET), 'target_config_sha256': sha(TARGET/'config.json'),
                'vllm_endpoint': args.vllm_endpoint, 'hidden_states_path': str(args.hidden_states_path.resolve()),
                'on_missing': args.on_missing, 'on_generate': args.on_generate,
                'body_seed': str(BODY), 'body_manifest_sha256': sha(BODY/'manifest.json'),
                'source_preregistration_sha256': sha(SOURCE/'preregistration.json'),
                'regenerated_file_sha256': source_protocol['regenerated_file_sha256'],
                'training_rows': len(rows), 'max_groups': args.max_groups, 'epochs': args.epochs,
                'groups_per_step': args.groups_per_step,
                'body_lr': args.body_lr, 'head_lr': args.head_lr,
                'weight_decay': args.weight_decay, 'max_anchors': args.max_anchors,
                'max_seq_len': args.max_seq_len, 'train_body': body_trainable,
                'target_embed_and_head_shared_parameter_objects': True,
                'target_embedding_and_head_distinct': True,
                'target_embed_and_head_frozen': True,
                'separate_mask_embedding': True, 'draft_to_target_mapping': None,
                'checkpoint_policy': 'fixed final epoch before held-out test',
                'train_predecessor': 'teacher forced response token',
                'test_predecessor': 'self-conditioned greedy token'}
    pp = out/'training_protocol.json'
    if pp.exists():
        if json.loads(pp.read_text()) != protocol:
            raise RuntimeError('changed online training protocol on resume')
    else:
        dump(pp, protocol)
    completed = out/'training_summary.json'
    if completed.exists():
        previous = json.loads(completed.read_text())
        fixed = Path(previous['fixed_checkpoint'])
        if (previous['status'] != 'complete' or
            previous['training_protocol_sha256'] != sha(pp) or
            sha(fixed) != previous['fixed_checkpoint_sha256']):
            raise RuntimeError('existing online completion is inconsistent')
        (out/'resume.pt').unlink(missing_ok=True)
        print(f'[{FAMILY} online] fixed checkpoint already complete: {fixed}',flush=True)
        return
    body = build_live_body(device, train_body=body_trainable)
    fetcher = LiveHiddenFetcher(args.vllm_endpoint, args.hidden_states_path, str(TARGET))
    head = make_head(FAMILY).to(device=device, dtype=torch.bfloat16).train()
    body.train(body_trainable)
    groups = [{'params': list(head.parameters()), 'lr': args.head_lr}]
    body_params = [p for p in body.parameters() if p.requires_grad]
    if body_trainable:
        if not body_params:
            raise RuntimeError('online body training was requested but no body parameters are trainable')
        groups.append({'params': body_params, 'lr': args.body_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    resume = out/'resume.pt'
    start_epoch = 0
    if resume.exists():
        payload = torch.load(resume, map_location='cpu', weights_only=False)
        if payload['protocol_sha256'] != sha(pp):
            raise RuntimeError('online resume protocol SHA mismatch')
        head.load_state_dict(payload['head'])
        body.load_state_dict(payload['body'], strict=False)
        optimizer.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['torch_rng'])
        torch.cuda.set_rng_state(payload['cuda_rng'], args.gpu)
        random.setstate(payload['python_rng'])
        np.random.set_state(payload['numpy_rng'])
        start_epoch = int(payload['epoch'])
    for epoch in range(start_epoch, args.epochs):
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(42+epoch)).tolist()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        terms: dict[str,float] = {}
        start = time.monotonic()
        for step, i in enumerate(order):
            row = rows[i]
            torch.manual_seed(31000 + int(row['index']) + 10000*epoch)
            torch.cuda.manual_seed(31000 + int(row['index']) + 10000*epoch)
            group = live_group(row, data, body, fetcher, device, args.max_seq_len,
                               max_anchors=args.max_anchors, train_body=body_trainable)
            if fetcher.generated != fetcher.deleted:
                raise RuntimeError('generated hidden file was not deleted')
            loss, metrics = loss_for_group(FAMILY, head, group, detach_base=not body_trainable)
            (loss / args.groups_per_step).backward()
            if (step+1) % args.groups_per_step == 0 or step+1 == len(order):
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                if body_params:
                    torch.nn.utils.clip_grad_norm_(body_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_total += float(loss.detach())
            for key,value in metrics.items():
                terms[key] = terms.get(key,0.)+value
            if step == 0 or (step+1) % 8 == 0 or step+1 == len(order):
                dump(out/'status.json', {'stage':'online_training','family':FAMILY,
                     'epoch':epoch+1,'epochs':args.epochs,'groups_done':step+1,
                     'groups_total':len(order),'mean_loss':loss_total/(step+1)})
                print(f'[{FAMILY} online] epoch={epoch+1}/{args.epochs} groups={step+1}/{len(order)} loss={loss_total/(step+1):.5f}',flush=True)
        state = {'head': {k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},
                 'body': {k:v.detach().cpu().contiguous() for k,v in body.state_dict().items()
                          if k.startswith(('layers.','fc.','hidden_norm.','norm.'))} if body_trainable else {}}
        tmp = Path(str(resume)+'.partial')
        torch.save({'epoch':epoch+1, **state, 'optimizer':optimizer.state_dict(),
                    'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state(args.gpu),
                    'python_rng':random.getstate(), 'numpy_rng':np.random.get_state(),
                    'protocol_sha256':sha(pp)},tmp)
        tmp.replace(resume)
        record={'epoch':epoch+1,'mean_loss':loss_total/len(rows),
                'mean_terms':{k:v/len(rows) for k,v in terms.items()},
                'seconds':time.monotonic()-start,
                'resume_state':str(resume),'resume_epoch':epoch+1}
        dump(out/f'epoch_{epoch+1:02d}.json',record)
        dump(out/'status.json',{'stage':'online_epoch_complete','family':FAMILY,**record})
    final=out/f'epoch_{args.epochs:02d}.safetensors'
    if not final.exists():
        final_state = {
            'head': {k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},
            'body': {k:v.detach().cpu().contiguous() for k,v in body.state_dict().items()
                     if k.startswith(('layers.','fc.','hidden_norm.','norm.'))} if body_trainable else {}
        }
        final_tmp=Path(str(final)+'.partial')
        save_file({f'{part}.{key}':value for part,part_state in final_state.items()
                   for key,value in part_state.items()},str(final_tmp))
        final_tmp.replace(final)
    dump(out/'training_summary.json',{'status':'complete','family':FAMILY,
         'training_mode':'vllm_generate_delete_each_group_and_live_dflash_forward_no_hidden_snapshots',
         'fixed_checkpoint':str(final),'fixed_checkpoint_sha256':sha(final),
         'training_protocol_sha256':sha(pp),'train_groups':len(rows),
         'epochs':args.epochs,'trained_body':body_trainable,
         'final_epoch_metrics':json.loads((out/f'epoch_{args.epochs:02d}.json').read_text())})
    resume.unlink(missing_ok=True)
    dump(out/'status.json',{'stage':'online_training_complete','fixed_checkpoint':str(final)})
    print(f'[{FAMILY} online] complete {final}',flush=True)

if __name__=='__main__':
    main()
