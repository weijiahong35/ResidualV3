"""Assigned-GPU numerical gate for vLLM target layer indexing and RMSNorm."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from .utils import ROOT, dump
from .vllm_hidden import OnlineHiddenProvider


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    p=json.loads(a.protocol.read_text());server=p['hidden_servers'][str(a.gpu)]
    torch.set_num_threads(4);torch.cuda.set_device(a.gpu);device=torch.device(f'cuda:{a.gpu}')
    from .live_trainer import load_examples
    example=load_examples(p)[0]
    example={'input_ids':example['input_ids'][:128]}
    model_dir=Path('/data/weijiahong/huggingface/models/Qwen3-8B')
    provider=OnlineHiddenProvider(server['endpoint'],server['transfer_root'],str(model_dir), prefetch=p['hidden_prefetch'])
    try:
        requests=[{'input_ids':example['input_ids'][:n]} for n in (128,120,112,104,96,88)]
        provider.begin_epoch(requests)
        batch=[provider(row) for row in requests]
        actual=batch[0].to(device)
        if provider.peak_inflight != p['hidden_prefetch'] or provider.requests != p['hidden_prefetch'] or provider.deleted != p['hidden_prefetch']:
            raise RuntimeError('six-request concurrency/deletion gate failed')
        del batch
    finally:provider.close()
    model=AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=torch.bfloat16,
                attn_implementation='sdpa',local_files_only=True).to(device).eval()
    ids=[2,10,18,26,34,36];hidden={};handles=[]
    for i in ids:
        def hook(_m,_args,out,key=i):hidden[key]=(out[0] if isinstance(out,tuple) else out).detach()
        handles.append(model.model.layers[i-1].register_forward_hook(hook))
    rows=[]
    with torch.no_grad():
        reference=model.model(input_ids=torch.tensor(example['input_ids'],device=device)[None],use_cache=False)
        for j,i in enumerate(ids):
            x,y=actual[:,j].float(),hidden[i][0].float()
            rel=float((x-y).norm()/y.norm().clamp_min(1e-8))
            cosine=float(torch.nn.functional.cosine_similarity(x,y,dim=-1).mean())
            rows.append(dict(layer=i,relative_l2=rel,mean_cosine=cosine,max_abs=float((x-y).abs().max())))
            if rel>.10 or cosine<.995:raise RuntimeError(f'target layer mismatch: {rows[-1]}')
        normalized=model.model.norm(actual[:,-1])
        rel_norm=float((normalized.float()-reference.last_hidden_state[0].float()).norm()/reference.last_hidden_state.float().norm())
        if rel_norm>.10:raise RuntimeError(f'final norm mismatch: {rel_norm}')
        logits=normalized[-32:]@model.lm_head.weight.T
        ref_logits=reference.last_hidden_state[0,-32:]@model.lm_head.weight.T
        agreement=float((logits.argmax(-1)==ref_logits.argmax(-1)).float().mean())
    for h in handles:h.remove()
    dump(a.output,dict(status='passed',gpu=a.gpu,tokens=len(example['input_ids']),layers=rows,
        normalized_relative_l2=rel_norm,last_32_argmax_agreement=agreement,
        thresholds={'relative_l2_max':.10,'mean_cosine_min':.995},
        hidden_transfer_files_remaining=len(list(provider.root.iterdir())),
        hidden_requests=provider.requests,hidden_deleted=provider.deleted,peak_inflight=provider.peak_inflight,
        scope='short target hidden layer/norm numerical sanity; not bitwise or native draft parity'))
    print(json.dumps(json.loads(a.output.read_text())),flush=True)

if __name__=='__main__':main()
