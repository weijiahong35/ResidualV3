"""External SpecMoLE execution helper for an unchanged DFlash decoder layer.

Stock DFlash never imports this module. ``ResidualDFlashModel`` calls the helper
only for a corrected layer with an actual residual control. The base layer
object and all of its Attention/Norm/MLP modules remain untouched.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import FlashAttentionKwargs
from typing_extensions import Unpack


def specmole_decoder_layer_forward(
    base_layer: nn.Module,
    hidden_states: torch.Tensor,
    target_hidden: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.LongTensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    residual_mole: nn.Module,
    residual_control: dict[str, torch.Tensor | float],
    *,
    past_key_value: Cache | None = None,
    output_attentions: bool | None = False,
    use_cache: bool | None = False,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> torch.Tensor:
    """Run the original layer math and add SpecMoLE at the second residual add."""
    residual = hidden_states
    attention_input = base_layer.input_layernorm(hidden_states)
    attention_output = base_layer.self_attn(
        hidden_states=attention_input,
        target_hidden=target_hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )[0]
    hidden_states = residual + attention_output

    residual = hidden_states
    mlp_input = base_layer.post_attention_layernorm(hidden_states)
    base_mlp = base_layer.mlp(mlp_input)
    correction = residual_mole(mlp_input, **residual_control)
    return residual + base_mlp + correction
