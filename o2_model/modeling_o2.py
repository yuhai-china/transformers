# O2: routed dual-expert MoE FFN on a hybrid (linear/full attention)
# multimodal backbone.
#
# Every decoder layer keeps the shared hybrid attention and replaces the
# dense FFN with a routed MoE block: per-layer gate + N experts, no shared
# expert.
#
# Checkpoint layout:
#   model.language_model.layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight
#   model.language_model.layers.N.mlp.gate.weight
# packed on load via the "o2_text" checkpoint conversion mapping registered
# by this package.

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeExperts,
    Qwen3_5MoeTopKRouter,
)

from .configuration_o2 import O2Config, O2TextConfig


class O2TopKRouter(Qwen3_5MoeTopKRouter):
    """Per-layer router: Linear(hidden -> num_experts), softmax, top-k."""

    pass


class O2Experts(Qwen3_5MoeExperts):
    """Packed 3D expert weights: gate_up_proj [E, 2I, H], down_proj [E, H, I]."""

    pass


class O2SparseMoeBlock(nn.Module):
    """Routed MoE FFN without a shared expert."""

    def __init__(self, config: O2TextConfig):
        super().__init__()
        self.gate = O2TopKRouter(config)
        self.experts = O2Experts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)


class O2DecoderLayer(Qwen3_5DecoderLayer):
    """Hybrid decoder layer with the dense MLP swapped for the MoE block."""

    def __init__(self, config: O2TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = O2SparseMoeBlock(config)


class O2TextModel(Qwen3_5TextModel):
    config: O2TextConfig
    _no_split_modules = ["O2DecoderLayer"]
    _can_record_outputs = {"hidden_states": O2DecoderLayer}

    def __init__(self, config: O2TextConfig):
        super(Qwen3_5TextModel, self).__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [O2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class O2Model(Qwen3_5Model):
    config: O2Config
    _no_split_modules = ["O2DecoderLayer", "Qwen3_5VisionBlock"]
    _can_record_outputs = {"hidden_states": O2DecoderLayer}

    def __init__(self, config: O2Config):
        super(Qwen3_5Model, self).__init__(config)
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = O2TextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class O2ForCausalLM(Qwen3_5ForCausalLM):
    config: O2TextConfig
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]

    def __init__(self, config: O2TextConfig):
        super(Qwen3_5ForCausalLM, self).__init__(config)
        self.model = O2TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


class O2ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    config: O2Config

    def __init__(self, config: O2Config):
        super(Qwen3_5ForConditionalGeneration, self).__init__(config)
        self.model = O2Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


__all__ = [
    "O2TextModel",
    "O2Model",
    "O2ForCausalLM",
    "O2ForConditionalGeneration",
]
