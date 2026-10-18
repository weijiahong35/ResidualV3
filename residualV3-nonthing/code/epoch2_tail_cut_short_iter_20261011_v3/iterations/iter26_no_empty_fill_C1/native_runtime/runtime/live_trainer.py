from __future__ import annotations
import sys,math,json
import torch
from pathlib import Path
from safetensors.torch import load_file
from .utils import HERE, ROOT


def load_examples(p):
    panel=[json.loads(s) for s in Path(p['panel_paths']['train']).read_text().splitlines()]
    wanted={r['source_index']:r for r in panel};found={}
    import hashlib
    from .data import validate_example
    with Path(p['data_source']).open() as f:
        for i,line in enumerate(f):
            if i not in wanted:continue
            if hashlib.sha256(line.encode()).hexdigest()!=wanted[i]['row_sha256']:
                raise RuntimeError('regenerated row hash changed')
            row=json.loads(line)
            if not validate_example(row):raise RuntimeError('invalid existing prompt+answer')
            found[i]=dict(input_ids=row['input_ids'],loss_mask=row['loss_mask'])
    return [found[r['source_index']] for r in panel]

def enable_compiled_flex_attention():
    """Fuse frozen draft attention; eager FlexAttention materializes Q*KV scores.

    ResidualV2 compiles its base forward. This online path calls layers directly,
    so compile just their shared attention operator, retaining the same masks.
    This changes only this training process, never installed/shared source files.
    """
    import speculators.models.attention as attention
    if not getattr(attention.flex_attention, '_online_training_compiled', False):
        compiled = torch.compile(attention.flex_attention, fullgraph=True, dynamic=True)
        compiled._online_training_compiled = True
        attention.flex_attention = compiled

def load_models(device, vllm_endpoint=None, hidden_state_root=None, hidden_prefetch=4):
    sys.path.insert(0,str(ROOT/'src'))
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from speculators.models.dflash.core import DFlashDraftModel
    from speculators.models.dflash.config import DFlashSpeculatorConfig
    enable_compiled_flex_attention()
    if vllm_endpoint is None:
        # Retained only for isolated CPU/reference checks, never training.
        target=AutoModelForCausalLM.from_pretrained(Path('/data/weijiahong/huggingface/models/Qwen3-8B'),
            torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to(device).eval()
    else:
        from .vllm_hidden import load_target_components
        target=load_target_components(Path('/data/weijiahong/huggingface/models/Qwen3-8B'), device,
                                      vllm_endpoint, hidden_state_root, prefetch=hidden_prefetch)
    config=DFlashSpeculatorConfig.from_pretrained(HERE/'body_only_draft',local_files_only=True)
    if config.sample_from_anchor or config.draft_vocab_size!=target.lm_head.weight.shape[0]:
        raise RuntimeError('expected anchored proposals with full original vocabulary')
    with torch.device('meta'):
        draft=DFlashDraftModel(config).to(dtype=torch.bfloat16)
    draft.to_empty(device=device)
    weights=load_file(str(HERE/'body_only_draft/model.safetensors'),device=str(device))
    missing,unexpected=draft.load_state_dict(weights,strict=False)
    if set(missing)!={'embed_tokens.weight','lm_head.weight','verifier_lm_head.weight','verifier_norm.weight'} or unexpected:
        raise RuntimeError(f'body keys changed: {missing}, {unexpected}')
    del weights
    draft.lm_head=target.lm_head
    draft.verifier_lm_head=target.lm_head
    draft.verifier_norm=target.model.norm
    # Native uses the original target embeddings except for the learned mask row.
    draft.embed_tokens=torch.nn.Embedding.from_pretrained(target.model.embed_tokens.weight.detach().clone(),freeze=True)
    mask=torch.load(HERE/'body_only_draft/mask_embedding.pt',map_location=device,weights_only=True)
    if isinstance(mask,dict):
        if int(mask['mask_token_id'])!=config.mask_token_id:raise RuntimeError('mask token ID changed')
        mask=mask['embedding']
    with torch.no_grad():draft.embed_tokens.weight[config.mask_token_id].copy_(mask.reshape(-1))
    draft.rotary_emb=Qwen3RotaryEmbedding(config.transformer_layer_config,device=device)
    draft.eval()
    for model in (target,draft):
        for param in model.parameters():param.requires_grad_(False)
    return target,draft

@torch.no_grad()
def online_states(target,draft,example,max_anchors,device):
    """One fresh target and draft forward. Return only small in-memory states.

    Mirrors ResidualV2 anchored block attention, but postpones the full-vocab
    projection until each head microbatch; never materializes 1536*8*V logits.
    """
    from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks,select_anchors
    ids=torch.tensor(example['input_ids'],device=device,dtype=torch.long)[None]
    mask=torch.tensor(example['loss_mask'],device=device,dtype=torch.bool)[None]
    n=ids.shape[1]; docs=torch.zeros_like(ids)
    # Select only anchors whose seven supervised positions all exist.
    block_size=draft.block_size
    eligible=mask.clone();eligible[:,-block_size:]=False
    count=min(max_anchors,int(eligible.sum()))
    if count<1:raise RuntimeError('no eligible answer anchors')
    anchors,valid=select_anchors(mask,count,block_size)
    if not bool(valid.all()):raise RuntimeError('padded anchors unexpectedly selected')
    if hasattr(target, 'hidden_provider'):
        packed = target.hidden_provider(example).to(device)
        aux = packed[:, :-1].flatten(1)[None].contiguous()
        # vLLM extraction saves the residual sum BEFORE final RMSNorm.
        last = target.model.norm(packed[:, -1])[None].contiguous()
        del packed
    else:
        taps={};hooks=[]
        for layer_id in draft.target_layer_ids:
            def capture(_m,_args,output,key=layer_id):
                taps[key]=(output[0] if isinstance(output,tuple) else output).detach()
            hooks.append(target.model.layers[layer_id-1].register_forward_hook(capture))
        try:
            output=target.model(input_ids=ids,use_cache=False,output_hidden_states=False)
            last=output.last_hidden_state.detach()  # Already final-normalized.
            del output
        finally:
            for handle in hooks:handle.remove()
        aux=torch.cat([taps[i] for i in draft.target_layer_ids],dim=-1)
        taps.clear()
    full=draft._create_attention_mask(docs,n,anchors,device) if draft.uses_full_attn else None
    sliding=draft._create_attention_mask(docs,n,anchors,device,sliding_window=draft.sliding_window,
        sliding_window_non_causal=draft.sliding_window_non_causal) if draft.uses_sliding_window_attn else None
    indices=get_base_indices_for_anchored_blocks(anchors,block_size)
    query=torch.full((1,count*block_size),draft.mask_token_id,device=device,dtype=torch.long)
    query[:,::block_size]=ids[:,anchors]
    noise=draft.embed_tokens(query)
    context=draft.hidden_norm(draft.fc(aux))
    positions=torch.cat((torch.arange(n,device=device),indices))[None]
    rotary=draft.rotary_emb(aux,positions)
    for i,layer in enumerate(draft.layers):
        noise=layer(hidden_states=noise,target_hidden=context,
            attention_mask=sliding if i in draft.sliding_window_indices else full,
            position_ids=positions,use_cache=False,position_embeddings=rotary)
    hidden=draft.norm(noise).reshape(count,block_size,-1)[:,1:].contiguous()
    # j=0 predicts answer[anchor+1] from target state[anchor].
    target_positions=anchors[:,None]+torch.arange(block_size-1,device=device)[None]
    answer_positions=target_positions+1
    target_hidden=last[0,target_positions].contiguous()
    answers=ids[0,answer_positions].contiguous()
    anchor_ids=ids[0,anchors].contiguous()
    return hidden,target_hidden,answers,anchor_ids

def schedule_scale(step,total,p):
    warm=max(1,int(total*p['warmup_fraction']))
    if step<=warm:return step/warm
    progress=(step-warm)/max(1,total-warm)
    return p['min_lr_fraction']+(1-p['min_lr_fraction'])*.5*(1+math.cos(math.pi*progress))
