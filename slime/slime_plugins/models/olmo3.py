"""Megatron model definition for dense OLMo 3 models.

OLMo 3 is not a pre-norm Llama variant. Its attention and MLP branches are
normalized immediately before the residual add, Q/K RMSNorm is applied over
the complete projection, and only every fourth layer uses YaRN RoPE. This
module keeps those details explicit instead of approximating the model with a
generic Llama spec.

The complete-projection Q/K norms are only local when tensor parallelism is
one. We deliberately reject TP > 1 rather than silently changing the model.
Data and pipeline parallelism remain supported for the training layout.
"""

from __future__ import annotations

import copy

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.common.embeddings import YarnRotaryEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


OLMO3_FULL_ATTENTION_INTERVAL = 4
OLMO3_ROPE_FACTOR = 8.0
OLMO3_ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS = 8192
OLMO3_ROPE_BETA_FAST = 32.0
OLMO3_ROPE_BETA_SLOW = 1.0
OLMO3_ROPE_MSCALE = 1.0
OLMO3_ROPE_MSCALE_ALL_DIM = 0.0


class Olmo3RMSNorm(torch.nn.Module):
    """RMSNorm matching ``transformers.models.olmo3.Olmo3RMSNorm``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class Olmo3SelfAttention(SelfAttention):
    """MCore attention with OLMo 3 projection norms and per-layer RoPE."""

    def __init__(self, config, submodules, layer_number, **kwargs):
        if config.tensor_model_parallel_size != 1:
            raise ValueError(
                "OLMo 3 actor training requires tensor model parallel size 1: Q/K RMSNorm spans the complete projection and cannot be applied independently to TP shards."
            )

        # Attention owns a shallow config copy so sliding layers can use ordinary
        # RoPE scaling while full-attention layers use YaRN scaling.
        attention_config = copy.copy(config)
        self.is_full_attention = layer_number % OLMO3_FULL_ATTENTION_INTERVAL == 0
        if not self.is_full_attention:
            for name in (
                "yarn_rotary_scaling_factor",
                "yarn_original_max_position_embeddings",
                "yarn_beta_fast",
                "yarn_beta_slow",
                "yarn_mscale",
                "yarn_mscale_all_dim",
                "yarn_correction_range_round_to_int",
            ):
                if hasattr(attention_config, name):
                    delattr(attention_config, name)

        super().__init__(
            config=attention_config,
            submodules=submodules,
            layer_number=layer_number,
            **kwargs,
        )
        self.q_norm = Olmo3RMSNorm(self.query_projection_size, config.layernorm_epsilon)
        self.k_norm = Olmo3RMSNorm(self.kv_projection_size, config.layernorm_epsilon)

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
        if output_gate:
            raise ValueError("OLMo 3 does not use gated attention output")
        if not split_qkv:
            raise ValueError("OLMo 3 complete-projection Q/K RMSNorm requires split QKV tensors")

        query, key, value = super().get_query_key_value_tensors(
            hidden_states,
            key_value_states=key_value_states,
            output_gate=False,
            split_qkv=True,
        )
        query_shape = query.shape
        key_shape = key.shape
        query = self.q_norm(query.flatten(start_dim=-2)).view(query_shape)
        key = self.k_norm(key.flatten(start_dim=-2)).view(key_shape)
        return query, key, value

    def forward(self, hidden_states, attention_mask, **kwargs):
        rotary_pos_emb = kwargs.get("rotary_pos_emb")
        if rotary_pos_emb is not None:
            # Pack sliding/full RoPE into one activation-checkpoint-safe tensor.
            # Indexing restores the standard [seq, 1, 1, head_dim] input.
            kwargs["rotary_pos_emb"] = rotary_pos_emb[int(self.is_full_attention)]
        return super().forward(hidden_states, attention_mask, **kwargs)


def _get_olmo3_layer_spec() -> ModuleSpec:
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=Olmo3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=None,
                    k_layernorm=None,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            post_self_attn_layernorm=TENorm,
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            post_mlp_layernorm=TENorm,
        ),
    )


class Olmo3GPTModel(GPTModel):
    """GPTModel with ordinary sliding-layer RoPE and full-layer YaRN RoPE."""

    def __init__(self, *args, **kwargs):
        # The base model must not create one shared RoPE module.
        kwargs["position_embedding_type"] = "none"
        super().__init__(*args, **kwargs)

        rotary_kwargs = {
            "kv_channels": self.config.kv_channels,
            "rotary_percent": self.rotary_percent,
            "rotary_interleaved": self.config.rotary_interleaved,
            "rotary_base": self.rotary_base,
            "use_cpu_initialization": self.config.use_cpu_initialization,
            "cp_group": self.pg_collection.cp,
        }
        self.sliding_rotary_pos_emb = RotaryEmbedding(**rotary_kwargs)
        self.full_rotary_pos_emb = YarnRotaryEmbedding(
            **rotary_kwargs,
            scaling_factor=OLMO3_ROPE_FACTOR,
            original_max_position_embeddings=OLMO3_ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            beta_fast=OLMO3_ROPE_BETA_FAST,
            beta_slow=OLMO3_ROPE_BETA_SLOW,
            mscale=OLMO3_ROPE_MSCALE,
            mscale_all_dim=OLMO3_ROPE_MSCALE_ALL_DIM,
            correction_range_round_to_int=True,
        )

    def _preprocess(
        self,
        input_ids,
        position_ids,
        decoder_input=None,
        inference_context=None,
        packed_seq_params=None,
    ):
        output = list(
            super()._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=decoder_input,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )
        )

        if packed_seq_params is not None:
            rotary_seq_len = max(packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv)
        elif inference_context is not None:
            rotary_seq_len = inference_context.max_sequence_length
        else:
            rotary_seq_len = output[0].size(0)
            if self.config.sequence_parallel:
                rotary_seq_len *= self.config.tensor_model_parallel_size
            rotary_seq_len *= self.config.context_parallel_size

        sliding_rotary = self.sliding_rotary_pos_emb(rotary_seq_len, packed_seq_params=packed_seq_params)
        full_rotary, _ = self.full_rotary_pos_emb(rotary_seq_len, packed_seq_params=packed_seq_params)
        # MCore checkpoints only tensors; stack per-layer RoPE variants as
        # [sliding, full, seq, 1, 1, dim] for exact selection in attention.
        output[1] = torch.stack((sliding_rotary, full_rotary), dim=0)
        return tuple(output)


def get_olmo3_spec(args, config, vp_stage):
    """Return a custom provider through Slime's ``--spec`` extension hook."""

    if args.tensor_model_parallel_size != 1:
        raise ValueError("OLMo 3 requires --tensor-model-parallel-size 1")

    # Full-attention layers use YaRN; custom attention copies and removes these
    # attributes for sliding layers before MCore applies RoPE.
    config.yarn_rotary_scaling_factor = OLMO3_ROPE_FACTOR
    config.yarn_original_max_position_embeddings = OLMO3_ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS
    config.yarn_beta_fast = OLMO3_ROPE_BETA_FAST
    config.yarn_beta_slow = OLMO3_ROPE_BETA_SLOW
    config.yarn_mscale = OLMO3_ROPE_MSCALE
    config.yarn_mscale_all_dim = OLMO3_ROPE_MSCALE_ALL_DIM
    config.yarn_correction_range_round_to_int = True

    layer_spec = _get_olmo3_layer_spec()

    def model_provider(pre_process=True, post_process=True, vp_stage=None):
        return Olmo3GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            vp_stage=vp_stage,
        )

    return model_provider
