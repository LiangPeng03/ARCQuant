import torch
from torch import nn
from typing import Optional, Tuple

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from qLinearLayer import QLinearLayer
from quantize import fake_reorder_quantize_x

import sys
sys.path.append('kernels/build/')
import agemm


@torch.no_grad()
def quantize_int_group(x, nbits, group_size):
    """The optional KV-cache simulation used by the existing ARCQuant layers."""
    shape = x.shape
    x = x.reshape(-1, group_size)
    x_max = x.amax(dim=-1, keepdim=True)
    x_min = x.amin(dim=-1, keepdim=True)
    qmax = (2 ** nbits) - 1
    scales = (x_max - x_min).clamp(min=1e-5) / qmax
    zero = torch.round(-x_min / scales).clamp_(0, qmax)
    x = (torch.clamp(torch.round(x / scales) + zero, 0, qmax) - zero) * scales
    return x.reshape(shape)


def nvfp4_reorder_quantize_x(x, reorder_index, select_num):
    scale = torch.max(x.abs()).float() / (448.0 * 6.0)
    qx, scale_x = agemm.reorder_quantize_x(x / scale, reorder_index, select_num)
    return qx, scale_x, scale


def reorder_quantize_x(x, reorder_index, select_num, quant_type='NVFP4'):
    if quant_type == 'NVFP4':
        return nvfp4_reorder_quantize_x(x, reorder_index, select_num)
    index = reorder_index.to(torch.int32)
    return fake_reorder_quantize_x(
        torch.index_select(x, 1, index),
        torch.arange(x.shape[1], device=x.device),
        select_num,
        dtype=quant_type,
    )


class QQwen3RMSNorm(nn.Module):
    def __init__(self, original_norm: Qwen3RMSNorm):
        super().__init__()
        self.original_norm = original_norm

    @torch.no_grad()
    def forward(self, hidden_states):
        return self.original_norm(hidden_states)


class QQwen3Attention(nn.Module):
    """Qwen3 attention with ARCQuant projections and native Qwen3 Q/K norms."""

    def __init__(self, original_attn: Qwen3Attention, kv_cache, select_nums,
                 reorder_index, layer_idx, quant_type):
        super().__init__()
        self.config = original_attn.config
        self.layer_idx = layer_idx
        self.quant_type = quant_type
        self.q_kv_cache = kv_cache
        self.head_dim = original_attn.head_dim
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.scaling = original_attn.scaling
        self.attention_dropout = original_attn.attention_dropout

        key_template = 'layers.{}.{}.{}.{}'
        def linear(name, module):
            return QLinearLayer(
                module,
                select_num=select_nums[key_template.format(layer_idx, 'self_attn', name, 'input')],
                reorder_index=reorder_index[key_template.format(layer_idx, 'self_attn', name, 'input')],
                quant_type=quant_type,
            )

        self.q_proj = linear('q_proj', original_attn.q_proj)
        self.k_proj = linear('k_proj', original_attn.k_proj)
        self.v_proj = linear('v_proj', original_attn.v_proj)
        self.o_proj = linear('o_proj', original_attn.o_proj)
        # Qwen3 applies these after the Q/K projections and before RoPE.
        self.q_norm = original_attn.q_norm
        self.k_norm = original_attn.k_norm

        self.register_buffer(
            'q_reorder_index',
            reorder_index[key_template.format(layer_idx, 'self_attn', 'q_proj', 'input')].to(torch.int16),
        )
        self.register_buffer(
            'o_reorder_index',
            reorder_index[key_template.format(layer_idx, 'self_attn', 'o_proj', 'input')].to(torch.int16),
        )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        bsz, q_len = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        flat = hidden_states.reshape(bsz * q_len, -1).contiguous().detach()
        qx, scale_x, scale = reorder_quantize_x(
            flat, self.q_reorder_index, self.q_proj.select_num, self.quant_type
        )
        projections = (qx, scale_x, scale, bsz, q_len)
        query_states = self.q_proj(projections).view(hidden_shape)
        key_states = self.k_proj(projections).view(hidden_shape)
        value_states = self.v_proj(projections).view(hidden_shape)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.q_kv_cache:
            key_states = quantize_int_group(key_states, nbits=4, group_size=128)

        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if self.q_kv_cache:
            value_states = quantize_int_group(value_states, nbits=4, group_size=128)

        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]
        if query_states.device.type == 'cuda' and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=causal_mask is None and q_len > 1,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)

        flat = attn_output.reshape(bsz * q_len, -1).contiguous().detach()
        qx, scale_x, scale = reorder_quantize_x(
            flat, self.o_reorder_index, self.o_proj.select_num, self.quant_type
        )
        attn_output = self.o_proj((qx, scale_x, scale, bsz, q_len))
        return attn_output, None


class QQwen3MLP(nn.Module):
    def __init__(self, original_mlp: Qwen3MLP, select_nums, reorder_index,
                 layer_idx, quant_type):
        super().__init__()
        self.quant_type = quant_type
        key_template = 'layers.{}.{}.{}.{}'
        def linear(name, module):
            return QLinearLayer(
                module,
                select_num=select_nums[key_template.format(layer_idx, 'mlp', name, 'input')],
                reorder_index=reorder_index[key_template.format(layer_idx, 'mlp', name, 'input')],
                quant_type=quant_type,
            )

        self.gate_proj = linear('gate_proj', original_mlp.gate_proj)
        self.up_proj = linear('up_proj', original_mlp.up_proj)
        self.down_proj = linear('down_proj', original_mlp.down_proj)
        self.act_fn = original_mlp.act_fn
        self.register_buffer(
            'up_reorder_index',
            reorder_index[key_template.format(layer_idx, 'mlp', 'up_proj', 'input')].to(torch.int16),
        )
        self.register_buffer(
            'down_reorder_index',
            reorder_index[key_template.format(layer_idx, 'mlp', 'down_proj', 'input')].to(torch.int16),
        )

    @torch.no_grad()
    def forward(self, x):
        bsz, q_len, _ = x.shape
        flat = x.reshape(bsz * q_len, -1).contiguous().detach()
        qx, scale_x, scale = reorder_quantize_x(
            flat, self.up_reorder_index, self.up_proj.select_num, self.quant_type
        )
        projections = (qx, scale_x, scale, bsz, q_len)
        hidden = self.act_fn(self.gate_proj(projections)) * self.up_proj(projections)

        flat = hidden.reshape(bsz * q_len, -1).contiguous().detach()
        qx, scale_x, scale = reorder_quantize_x(
            flat, self.down_reorder_index, self.down_proj.select_num, self.quant_type
        )
        return self.down_proj((qx, scale_x, scale, bsz, q_len))


class QQwen3DecoderLayer(nn.Module):
    def __init__(self, original_layer: Qwen3DecoderLayer, kv_cache, select_nums,
                 reorder_index, layer_idx, quant_type):
        super().__init__()
        self.self_attn = QQwen3Attention(
            original_layer.self_attn, kv_cache, select_nums, reorder_index,
            layer_idx, quant_type,
        )
        self.mlp = QQwen3MLP(
            original_layer.mlp, select_nums, reorder_index, layer_idx, quant_type,
        )
        self.input_layernorm = QQwen3RMSNorm(original_layer.input_layernorm)
        self.post_attention_layernorm = QQwen3RMSNorm(
            original_layer.post_attention_layernorm
        )

    @torch.no_grad()
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
