"""
Individual mixture in PaliGemma format

Additional support for adaLN(-Zero) and (Q)LoRA

"""

from typing import Optional, Tuple

from internmanip.model.basemodel.openpi0.lora import get_layer
from internmanip.model.basemodel.openpi0.modules import AdaptiveLayerscale, AdaptiveRMSNorm
from internmanip.model.basemodel.openpi0.paligemma.modules import GemmaMLP, GemmaRMSNorm, GemmaRotaryEmbedding
from internmanip.model.basemodel.openpi0.utils import apply_rotary_pos_emb, repeat_kv
import torch
import torch.nn as nn

# from src.model.lora import get_layer
# from src.model.paligemma.modules import (
#     GemmaMLP,
#     GemmaRMSNorm,
#     GemmaRotaryEmbedding,
# )
# from src.model.utils import apply_rotary_pos_emb, repeat_kv
# from src.model.vla.modules import AdaptiveLayerscale, AdaptiveRMSNorm


class Mixture(nn.Module):
    def __init__(self, config, item_config):
        super().__init__()
        self.layers = nn.ModuleList(
            [MixtureDecoderLayer(config, item_config) for _ in range(config.num_hidden_layers)]
        )

        self.adaptive_mode = None
        if item_config.use_final_norm:
            self.adaptive_mode = None
            if self.adaptive_mode:
                self.norm = AdaptiveRMSNorm(
                    item_config.hidden_size,
                    config.time_hidden_size,
                    eps=config.rms_norm_eps,
                )
            else:
                self.norm = GemmaRMSNorm(
                    item_config.hidden_size,
                    eps=config.rms_norm_eps,
                )

    @property
    def head_dim(self) -> int:
        return self.layers[0].self_attn.head_dim

    def layer_func(
        self,
        method_name: str,
        layer_idx: int,
        *args,
    ) -> torch.FloatTensor:
        args = [arg for arg in args if arg is not None]
        return getattr(self.layers[layer_idx], method_name)(*args)

    def attn_func(
        self,
        method_name: str,
        layer_idx: int,
        *args,
    ) -> torch.FloatTensor:
        args = [arg for arg in args if arg is not None]
        return getattr(self.layers[layer_idx].self_attn, method_name)(*args)

    def forward_norm(
        self,
        x: torch.FloatTensor,
        cond: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        if hasattr(self, "norm"):
            args = [x] if self.adaptive_mode is None else [x, cond]
            return self.norm(*args)
        else:
            return None


class MixtureDecoderLayer(nn.Module):
    def __init__(self, config, item_config):
        super().__init__()
        self.self_attn = MixtureAttention(config, item_config)

        self.mlp = GemmaMLP(
            config, item_config, use_quantize=item_config.use_quantize, use_lora=item_config.use_lora
        )

        self.adaptive_mode = None
        if self.adaptive_mode:
            self.input_layernorm = AdaptiveRMSNorm(
                item_config.hidden_size,
                config.time_hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = AdaptiveRMSNorm(
                item_config.hidden_size,
                config.time_hidden_size,
                eps=config.rms_norm_eps,
            )
            if self.adaptive_mode == "adaLN-Zero":
                self.post_adaptive_scale = AdaptiveLayerscale(
                    item_config.hidden_size,
                    config.time_hidden_size,
                )
                self.final_adaptive_scale = AdaptiveLayerscale(
                    item_config.hidden_size,
                    config.time_hidden_size,
                )
        else:
            self.input_layernorm = GemmaRMSNorm(
                item_config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = GemmaRMSNorm(
                item_config.hidden_size,
                eps=config.rms_norm_eps,
            )

    def forward_norm(
        self,
        norm_name: str,
        x: torch.FloatTensor,
        cond: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        args = [x] if self.adaptive_mode is None else [x, cond]
        return getattr(self, norm_name)(*args)

    def forward_adaptive_scale(
        self,
        stage: str,
        x: torch.FloatTensor,
        cond: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        if self.adaptive_mode == "adaLN-Zero":
            if stage == "post_attn":
                return self.post_adaptive_scale(x, cond)
            elif stage == "final":
                return self.final_adaptive_scale(x, cond)
            else:
                raise ValueError(f"Invalid stage for adaptive scaling: {stage}!")
        return x


class MixtureAttention(nn.Module):
    """assume head_dim same for all blocks"""

    def __init__(self, config, item_config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        assert item_config.hidden_size % self.num_heads == 0

        layer = get_layer(
            item_config.use_quantize,
            item_config.use_lora,
            **config.lora if item_config.use_lora else {},
        )
        self.q_proj = layer(
            item_config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = layer(
            item_config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = layer(
            item_config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = layer(
            self.num_heads * self.head_dim,
            item_config.hidden_size,
            bias=config.attention_bias,
        )
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            base=item_config.rope_theta,
        )

    def forward_q_proj(self, x: torch.FloatTensor) -> torch.FloatTensor:
        bsz, q_len = x.shape[:2]
        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(x)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        return query_states

    def forward_k_proj(self, x: torch.FloatTensor) -> torch.FloatTensor:
        bsz, q_len = x.shape[:2]
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(x)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        return key_states

    def forward_v_proj(self, x: torch.FloatTensor) -> torch.FloatTensor:
        bsz, q_len = x.shape[:2]
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        value_states = self.v_proj(x)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        return value_states

    def forward_o_proj(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.o_proj(x)

    def forward_rotary_emb(
        self, x: torch.FloatTensor, position_ids: torch.LongTensor
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(x, position_ids)
        return cos, sin

    def forward_apply_rotary_emb(
        self,
        states: torch.FloatTensor,
        cos: torch.FloatTensor,
        sin: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # [Batch_Size, Num_Heads_Q / Num_Heads_KV, Seq_Len, Head_Dim]
        states = apply_rotary_pos_emb(states, cos, sin)
        return states

    def repeat_kv(
        self, key_states: torch.FloatTensor, value_states: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor]:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        return key_states, value_states
