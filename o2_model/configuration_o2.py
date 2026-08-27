# O2 model configuration — standalone implementation.
#
# Only depends on transformers core (`PreTrainedConfig`). No imports from
# `transformers.models.*`, so host-model refactors cannot break this package.
#
# Defaults follow the Qwen3.8-27B checkpoint layout (hybrid 3:1 linear/full
# attention, 5120 hidden, partial mRoPE) since O2 is merged from it.

from transformers.configuration_utils import PreTrainedConfig

# RoPE defaults of the O2 / Qwen3.8 text backbone.
_DEFAULT_ROPE_PARAMETERS = {
    "rope_type": "default",
    "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}


class O2VisionConfig(PreTrainedConfig):
    """Vision tower config (SigLIP-style ViT + patch merger)."""

    model_type = "o2_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=5120,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


class O2TextConfig(PreTrainedConfig):
    """Text backbone config: hybrid (linear/full attention) decoder with a
    per-layer routed MoE FFN.

    num_experts: number of routed FFN experts per decoder layer.
    num_experts_per_tok: experts each token is routed to.
    moe_intermediate_size: expert intermediate size (defaults to intermediate_size).
    norm_topk_prob: renormalize top-k routing probabilities (kept for checkpoint
        compatibility; the reference router always renormalizes).
    """

    model_type = "o2_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000000.0,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        partial_rotary_factor=0.25,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        layer_types=None,
        full_attention_interval=4,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=None,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.0,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        # Set RoPE attributes *before* super().__init__ so that the base
        # __post_init__ rope standardization sees them (hasattr check).
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_parameters = dict(_DEFAULT_ROPE_PARAMETERS)
        if rope_parameters:
            self.rope_parameters.update(rope_parameters)
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim

        # Linear attention (gated delta net) settings
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        self.full_attention_interval = full_attention_interval
        if layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % full_attention_interval) else "full_attention"
                for i in range(num_hidden_layers)
            ]
        else:
            self.layer_types = list(layer_types)
        for layer_type in self.layer_types:
            if layer_type not in ("linear_attention", "full_attention"):
                raise ValueError(f"Unknown layer type: {layer_type}")

        # MoE FFN settings
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = (
            moe_intermediate_size if moe_intermediate_size is not None else intermediate_size
        )
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef


class O2Config(PreTrainedConfig):
    """Composite O2 config: shared vision tower + hybrid text backbone with
    routed-expert FFN (`text_config` is an `O2TextConfig`)."""

    model_type = "o2"
    sub_configs = {"vision_config": O2VisionConfig, "text_config": O2TextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = O2TextConfig(**text_config)
        elif text_config is None:
            text_config = O2TextConfig()

        if isinstance(vision_config, dict):
            # Checkpoints may carry a stale/legacy model_type in the vision
            # section; the sub-config class decides its own model_type.
            vision_config = {k: v for k, v in vision_config.items() if k != "model_type"}
            vision_config = O2VisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = O2VisionConfig()

        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["O2Config", "O2TextConfig", "O2VisionConfig"]
