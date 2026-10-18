"""Freeze completed nonthinking data and derive the native-shared draft body."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parent
RUNTIME = FAMILY/'code/runtime_bundle'
sys.path.insert(0, str(RUNTIME))
from runtime.utils import dump, sha, now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output.resolve()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    model = Path('/data/weijiahong/huggingface/models/Qwen3-8B-speculator.dflash-nonthinking-20260916')
    target = Path('/data/weijiahong/huggingface/models/Qwen3-8B')
    body = FAMILY/'models/body_only_draft'
    config = json.loads((model/'config.json').read_text())
    proof_path=FAMILY/'models/model_provenance.json'
    if proof_path.exists():
        model_proof=json.loads(proof_path.read_text())
        assert sha(body/'model.safetensors')==model_proof['body_sha256']
        assert sha(body/'mask_embedding.pt')==model_proof['mask_sha256']
        assert sha(model/'model.safetensors')==model_proof['source_sha256']
    else:
        body.mkdir(exist_ok=False)
        config = json.loads((model/'config.json').read_text())
        assert config['block_size'] == 16 and not config['sample_from_anchor']
        for name in ['config.json', 'config.py']:
            shutil.copy2(model/name, body/name)
        with safe_open(model/'model.safetensors', framework='pt', device='cpu') as stream:
            keys = list(stream.keys())
            weights = {key:stream.get_tensor(key) for key in keys
                       if key not in ['embed_tokens.weight', 'lm_head.weight', 'verifier_lm_head.weight', 'verifier_norm.weight']}
            mask = stream.get_slice('embed_tokens.weight')[config['mask_token_id']:config['mask_token_id']+1].clone()
            save_file(weights, str(body/'model.safetensors'))
            torch.save(dict(mask_token_id=config['mask_token_id'], embedding=mask), body/'mask_embedding.pt')
        del weights
        model_proof = dict(source=str(model), source_resolved=str(model.resolve()),
            source_sha256=sha(model/'model.safetensors'), body_sha256=sha(body/'model.safetensors'),
            mask_sha256=sha(body/'mask_embedding.pt'), body_keys=[key for key in keys if key not in ['embed_tokens.weight','lm_head.weight']],
            excludes_original_embedding_and_lmhead=True, native_shares_original_target_tables=True,
            learned_mask_row_preserved=True, block_size=16, proposal_count=15,
            draft_layers=config['transformer_layer_config']['num_hidden_layers'])
        dump(FAMILY/'models/model_provenance.json', model_proof)
    generated = REPO/'regeneration_standalone/open_perfectblend_regen/pool_500k_seed42/regenerated_ctx4096/nonthinking'
    status = json.loads((REPO/'regeneration_standalone/open_perfectblend_regen/auto_scheduler_nonthinking_ctx4096_20261005/status.json').read_text())
    complete = {Path(row['path']).resolve() for row in status['progress']['shards'] if row['complete']}
    data = FAMILY/'data/train20k_20261006'
    data.mkdir(exist_ok=True)
    rows = []; groups = set(); rejected = {}; sources = []
    for path in sorted((generated/'train').glob('shard-*.jsonl')):
        if path.resolve() not in complete:
            continue
        used = 0
        with path.open() as stream:
            for line_index, line in enumerate(stream):
                row = json.loads(line); meta = row['metadata']
                ids, mask_values = row['input_ids'], row['loss_mask']
                group = meta['source']['pool']['group_sha256']
                reason = None
                if group in groups: reason = 'duplicate_group'
                elif not meta.get('conversation_complete'): reason = 'conversation_not_complete'
                elif meta.get('finish_reason') != 'stop': reason = 'truncated_or_other_finish'
                elif meta['sampling_params'].get('chat_template_kwargs',{}).get('enable_thinking') is not False: reason = 'not_nonthinking'
                elif len(ids) != len(mask_values) or not 17 <= len(ids) <= 4096: reason = 'length'
                elif any(type(token) is not int or not 0 <= token < 151936 for token in ids): reason = 'token_range'
                elif any(value not in [0,1] for value in mask_values) or 1 not in mask_values: reason = 'loss_mask'
                else:
                    first = mask_values.index(1)
                    if first == 0 or not all(mask_values[first:]) or sum(mask_values) < 17: reason = 'insufficient_contiguous_answer'
                if reason:
                    rejected[reason] = rejected.get(reason,0)+1
                    continue
                groups.add(group); used += 1
                rows.append(dict(input_ids=ids, loss_mask=mask_values, id=row['id'],
                    primary_id=row['primary_id'], group_sha256=group,
                    source_path=str(path), source_line=line_index,
                    source_row_sha256=hashlib.sha256(line.encode()).hexdigest()))
                if len(rows) == 20000:
                    break
        sources.append(dict(path=str(path), sha256=sha(path), selected=used))
        print(now(), 'DATA', len(rows), path.name, flush=True)
        if len(rows) == 20000:
            break
    assert len(rows) == len(groups) == 20000
    train = data/'train.jsonl'
    content=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
    if train.exists():assert train.read_text()==content
    else:train.write_text(content)
    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
    prompt_root = REPO/'regeneration_standalone/open_perfectblend_regen/pool_500k_seed42/prompts/train'
    panels = []; seen = set(groups)
    for path in sorted(prompt_root.glob('shard-003.jsonl')):
        with path.open() as stream:
            for line_index, line in enumerate(stream):
                row = json.loads(line); group = row['pool']['group_sha256']
                if group in seen: continue
                messages = row['conversations']
                if not messages or messages[-1]['role'] != 'user': continue
                tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                if hasattr(tokens, 'keys'): tokens = tokens['input_ids']
                if not 1 <= len(tokens) <= 2000: continue
                seen.add(group)
                panels.append(dict(index=len(panels), id=row['id'], group_key=group,
                    messages=messages, prompt_token_ids=tokens,
                    prompt_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
                    source_path=str(path), source_line=line_index,
                    scope='untrained train-pool shard003; online development use, not untouched final test'))
                if len(panels) == 2240: break
        if len(panels) == 2240: break
    assert len(panels) == 2240
    paths = dict(train=str(train))
    for name, values in [('warmup',panels[:40]),('final_C1',panels[40:240]),('final_C40',panels[240:2240])]:
        path=data/(name+'.jsonl');path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in values));paths[name]=str(path)
    m_path=RUNTIME/'assets/static_ids.json';m_path.write_text(json.dumps(list(range(32768)))+'\n')
    architecture=dict(hidden_size=4096,width=256,vocab_size=151936,basis_rank=512,depth=3,q_topk=256,dose_max=.8,dose_init=.01,use_hidden=True,use_soft=False,scale_mode='hidden_norm',fixed_scale=None)
    dump(out/'head_config.json', architecture)
    boot=json.loads((out/'bootstrap.json').read_text())
    p=dict(boot, runtime=str(RUNTIME), data_source=str(train), panel_paths=paths,
        head_config_path=str(out/'head_config.json'), architecture=architecture,
        experiment_seed=610006, smoke_seed=610007, stage1_lr=3e-4,dose_lr=3e-4,
        weight_decay=.01,clip_grad_norm=1.,warmup_fraction=.03,min_lr_fraction=.1,
        max_anchors=1536,head_chunk_candidates=[16,32,64],hidden_prefetch_candidates=[4,8,12,16],
        initial_hidden_prefetch=16,training_memory_fraction=.50,
        basis_fit=dict(examples=128,anchors=64,chunk_anchors=16,rank=512,
            recipe='uncentered second moment of normalized (p-q)W; every nonzero sampled mismatch direction; no reach/TV weights',seed=610008),
        head_initialization='fresh random no-soft3; never load thinking repair or basis',
        objective='OUTPUT teacher-prefix cumulative overlap proxy',
        m_ids_path=str(m_path),m_ids_sha256=sha(m_path),m_size=32768,
        native_C1_prompts=200,native_C40_prompts=2000,native_max_tokens=2096,native_max_model_len=4096,
        native_arms=['RAW','N_no_soft_3layer_nonthinking'],teacher_forced=True,
        checkpoint_interval=100,preserve_all_epochs=True,
        training_data_sha256=sha(train),model_provenance=model_proof,
        data_sources=sources,source_files={},asset_sha256={})
    assets=[train,model/'model.safetensors',body/'config.json',body/'model.safetensors',body/'mask_embedding.pt',out/'head_config.json',m_path]
    assets.extend(Path(path) for path in paths.values())
    p['asset_sha256']={str(path):sha(path) for path in assets}
    dump(out/'prepared_protocol.json',p)
    dump(data/'manifest.json',dict(status='complete',training_rows=20000,unique_training_groups=20000,
        max_prompt_plus_answer_tokens=max(len(row['input_ids']) for row in rows),
        min_answer_tokens=min(sum(row['loss_mask']) for row in rows),source_files=sources,
        filtering=rejected,paths=paths,sha256={name:sha(path) for name,path in paths.items()},
        validation_groups_disjoint=True,thinking=False,utc=now()))
    print(now(), 'PREPARED', str(out/'prepared_protocol.json'), flush=True)


if __name__ == '__main__':
    main()
