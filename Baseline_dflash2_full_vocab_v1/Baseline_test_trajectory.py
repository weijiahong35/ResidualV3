#!/usr/bin/env python3
"""Held-out test of a fixed online-trained body plus sequential head."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors.torch import load_file
import torch

from Baseline_common import DEFAULT_OUT, SOURCE, dump, make_head, sha
from Baseline_online_common import TARGET, LiveHiddenFetcher, build_live_body, live_group, split_rows
from Baseline_test_common import sequential, summarize

FAMILY='dflash2'


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--vllm-endpoint',required=True)
    ap.add_argument('--hidden-states-path',type=Path,required=True)
    ap.add_argument('--on-missing',choices=['generate'],default='generate')
    ap.add_argument('--on-generate',choices=['delete'],default='delete')
    ap.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--max-groups',type=int,default=0,help='must match online training smoke count')
    args=ap.parse_args()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise RuntimeError('Use physical --gpu without CUDA_VISIBLE_DEVICES')
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    device=torch.device(f'cuda:{args.gpu}')
    out=args.output_dir/FAMILY/'online'
    if args.max_groups:
        out=out/f'smoke_{args.max_groups}_groups'
    summary=json.loads((out/'training_summary.json').read_text())
    pp=out/'training_protocol.json'
    protocol=json.loads(pp.read_text())
    if summary['status']!='complete' or sha(pp)!=summary['training_protocol_sha256']:
        raise RuntimeError('online training not frozen')
    if protocol['family']!=FAMILY or protocol['on_missing']!='generate' or protocol['on_generate']!='delete':
        raise RuntimeError('online generate/delete protocol mismatch')
    if args.vllm_endpoint!=protocol['vllm_endpoint'] or str(args.hidden_states_path.resolve())!=protocol['hidden_states_path']:
        raise RuntimeError('target server path/endpoint differs from training protocol')
    if (args.max_groups or 0)!=int(protocol.get('max_groups',0)):
        raise RuntimeError('online smoke group count differs from training')
    ckpt=Path(summary['fixed_checkpoint'])
    if sha(ckpt)!=summary['fixed_checkpoint_sha256']:
        raise RuntimeError('online fixed checkpoint changed')
    rows,data,_=split_rows('final')
    if args.max_groups:
        rows=rows[:args.max_groups]
    body=build_live_body(device,train_body=False)
    fetcher=LiveHiddenFetcher(args.vllm_endpoint,args.hidden_states_path,str(TARGET))
    head=make_head(FAMILY).to(device=device, dtype=torch.bfloat16)
    state=load_file(str(ckpt),device='cpu')
    head_state={k[5:]:v for k,v in state.items() if k.startswith('head.')}
    body_state={k[5:]:v for k,v in state.items() if k.startswith('body.')}
    head.load_state_dict(head_state,strict=True)
    if bool(body_state)!=bool(protocol['train_body']):
        raise RuntimeError('checkpoint body state differs from training protocol')
    if body_state:
        missing,unexpected=body.load_state_dict(body_state,strict=False)
        if unexpected or set(body_state)!=set(body.state_dict())-set(missing):
            raise RuntimeError(f'online body checkpoint mismatch: {unexpected}')
    head.eval(); body.eval()
    results=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            torch.manual_seed(51000+int(row['index']))
            torch.cuda.manual_seed(51000+int(row['index']))
            group=live_group(row,data,body,fetcher,device,protocol['max_seq_len'],
                             max_anchors=protocol['max_anchors'],train_body=False)
            if fetcher.generated != fetcher.deleted:
                raise RuntimeError('generated hidden file was not deleted')
            results.append(sequential(FAMILY,head,group))
            if (i+1)%16==0 or i+1==len(rows):
                print(f'[{FAMILY} online test] final groups {i+1}/{len(rows)}',flush=True)
    report=summarize(results,FAMILY)
    report.update({'status':'complete',
                   'evaluation':'offline_live_target_and_dflash_forward_fixed_trajectory_groups',
                   'native_T0_acceptance':None,'native_T1_acceptance':None,
                   'source_preregistration_sha256':sha(SOURCE/'preregistration.json'),
                   'fixed_checkpoint':str(ckpt),'fixed_checkpoint_sha256':sha(ckpt),
                   'target_and_draft_recomputed_live':True,
                   'hidden_snapshot_read':False,
                   'trained_body':protocol['train_body']})
    dump(out/'offline_group_results.json',{'groups':results})
    dump(out/'offline_summary.json',report)
    dump(out/'completion.json',{'status':'complete','family':FAMILY,
         'scope':'online_training_and_live_forward_heldout_offline_test',
         'training_summary_sha256':sha(out/'training_summary.json'),
         'offline_summary_sha256':sha(out/'offline_summary.json'),
         'native_runtime_results':None})
    print(f'[{FAMILY} online test] greedy ΔEAL={report["greedy_eal_delta"]:.4f} soft-proxy ΔEAL={report["soft_eal_delta"]:.4f}',flush=True)

if __name__=='__main__':
    main()
