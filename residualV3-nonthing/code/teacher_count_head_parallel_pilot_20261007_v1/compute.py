"""Same sampled anchors/loss; shard independent draft blocks and batch TF head."""
from contextlib import contextmanager
import torch
from runtime.precision import bf16_logit_update
from repair.model import project_candidates, output_loss
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks


@contextmanager
def reuse_context_projections(attention, context):
    # Frozen target context is identical across independent query shards.
    # Compute K/V once per layer, then use the normal decoder implementation.
    old_key, old_value = attention.k_proj.forward, attention.v_proj.forward
    key, value = old_key(context), old_value(context)
    attention.k_proj.forward = lambda x: key if x is context else old_key(x)
    attention.v_proj.forward = lambda x: value if x is context else old_value(x)
    try:
        yield
    finally:
        attention.k_proj.forward, attention.v_proj.forward = old_key, old_value


@torch.no_grad()
def sharded_states(target, draft, rows, chunks, budget, cap,
                   explicit_anchors=None, *, shard_anchors=512):
    assert shard_anchors in (256, 512, 1536)
    device = target.lm_head.weight.device
    from pipeline import DevicePack
    assert isinstance(chunks, DevicePack), 'prepacked asynchronous upload required'
    assert chunks.context.device == device
    ids, docs, positions, eligible = chunks.ids, chunks.docs, chunks.positions, chunks.eligible
    offset = chunks.offset
    if explicit_anchors is None:
        anchors = torch.sort(eligible[torch.randperm(len(eligible), device=device)[:cap]]).values
    else:
        anchors = torch.tensor(explicit_anchors, dtype=torch.long, device=device)
        assert torch.isin(anchors, eligible).all()
    index = anchors[:, None] + torch.arange(16, device=device)[None]
    assert torch.equal(docs[0, index], docs[0, anchors, None].expand(-1, 16))
    last, context = chunks.last, chunks.context
    states, full_masks, sliding_masks, rotary = [], [], [], []
    for begin in range(0, len(anchors), shard_anchors):
        local = anchors[begin:begin + shard_anchors]
        query = torch.full((1, len(local) * 16), draft.mask_token_id,
                           device=device, dtype=torch.long)
        query[:, ::16] = ids[:, local]
        states.append(draft.embed_tokens(query))
        full_masks.append(draft._create_attention_mask(docs, budget, local, device)
                          if draft.uses_full_attn else None)
        sliding_masks.append(draft._create_attention_mask(docs, budget, local, device,
            sliding_window=draft.sliding_window,
            sliding_window_non_causal=draft.sliding_window_non_causal)
            if draft.uses_sliding_window_attn else None)
        rotary_positions = torch.cat((positions,
            get_base_indices_for_anchored_blocks(positions[0, local], 16)[None]), 1)
        rotary.append(draft.rotary_emb(context, rotary_positions))
    for layer_index, layer in enumerate(draft.layers):
        with reuse_context_projections(layer.self_attn, context):
            for shard in range(len(states)):
                states[shard] = layer(hidden_states=states[shard], target_hidden=context,
                    attention_mask=sliding_masks[shard] if layer_index in draft.sliding_window_indices else full_masks[shard],
                    use_cache=False, position_embeddings=rotary[shard])
    hidden = torch.cat([draft.norm(state).reshape(-1, 16, 4096)[:, 1:]
                        for state in states], 0).contiguous()
    gold = last[index[:, :-1]].contiguous()
    answers = ids[0, index[:, 1:]].contiguous()
    return hidden, gold, answers, ids[0, anchors], anchors
