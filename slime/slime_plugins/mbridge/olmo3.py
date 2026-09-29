"""HF/Megatron checkpoint mapping for OLMo 3."""

from mbridge.core import register_model
from mbridge.models import LLaMABridge


@register_model("olmo3")
class Olmo3Bridge(LLaMABridge):
    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": ["model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.q_norm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
        "self_attention.k_norm.weight": ["model.layers.{layer_number}.self_attn.k_norm.weight"],
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
        "post_mlp_layernorm.weight": ["model.layers.{layer_number}.post_feedforward_layernorm.weight"],
    }

    _OTHER_MAPPING = {
        "post_self_attn_layernorm.weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
    }
