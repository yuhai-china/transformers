# O2 model package for transformers.
#
# Importing this package registers the O2 architecture (model_type "o2" /
# "o2_text") with the Auto classes and the checkpoint conversion machinery.
# No host package files are modified, and no `transformers.models.*` modules
# are imported — the modeling code in this package is self-contained.

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)
from transformers.conversion_mapping import (
    Concatenate,
    MergeModulelist,
    PrefixChange,
    WeightConverter,
    register_checkpoint_conversion_mapping,
)

from .configuration_o2 import O2Config, O2TextConfig, O2VisionConfig
from .modeling_o2 import (
    O2ForCausalLM,
    O2ForConditionalGeneration,
    O2Model,
    O2TextModel,
    O2VisionModel,
)

_registered = False

# Checkpoint layout bridge between mergekit-o2 (per-expert 2D tensors) and
# this package (packed 3D tensors), defined explicitly:
#   ...mlp.experts.E.{gate_proj,up_proj}.weight -> ...mlp.experts.gate_up_proj  [E, 2I, H]
#   ...mlp.experts.E.down_proj.weight          -> ...mlp.experts.down_proj      [E, H, I]
# The PrefixChange additionally lets the text-only O2ForCausalLM load directly
# from the composite checkpoint layout (strips the `language_model.` nesting).
# Save is the exact inverse, so `save_pretrained` writes the mergekit layout.
_O2_TEXT_CHECKPOINT_CONVERSION = [
    PrefixChange(prefix_to_remove="language_model", model_prefix="model"),
    WeightConverter(
        source_patterns=[
            "mlp.experts.*.gate_proj.weight",
            "mlp.experts.*.up_proj.weight",
        ],
        target_patterns="mlp.experts.gate_up_proj",
        operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
    ),
    WeightConverter(
        source_patterns="mlp.experts.*.down_proj.weight",
        target_patterns="mlp.experts.down_proj",
        operations=[MergeModulelist(dim=0)],
    ),
]


def register():
    global _registered
    if _registered:
        return
    AutoConfig.register("o2", O2Config, exist_ok=True)
    AutoConfig.register("o2_text", O2TextConfig, exist_ok=True)
    AutoModelForCausalLM.register(O2TextConfig, O2ForCausalLM, exist_ok=True)
    AutoModelForImageTextToText.register(
        O2Config, O2ForConditionalGeneration, exist_ok=True
    )
    try:
        from .tokenizer_o2 import register_processors

        register_processors(O2Config)
    except Exception:
        pass
    try:
        register_checkpoint_conversion_mapping("o2_text", list(_O2_TEXT_CHECKPOINT_CONVERSION))
    except ValueError:
        pass
    _registered = True


register()

__all__ = [
    "O2Config",
    "O2TextConfig",
    "O2VisionConfig",
    "O2VisionModel",
    "O2TextModel",
    "O2Model",
    "O2ForCausalLM",
    "O2ForConditionalGeneration",
    "register",
    "O2Tokenizer",
    "O2TokenizerFast",
    "O2ImageProcessor",
    "O2VideoProcessor",
    "O2Processor",
]

# Tokenizer/processor aliases are convenience shims over host classes; they
# must never block the modeling core (older transformers may lack the host
# processor modules entirely).
try:
    from .tokenizer_o2 import (  # noqa: E402
        O2ImageProcessor,
        O2Processor,
        O2Tokenizer,
        O2TokenizerFast,
        O2VideoProcessor,
    )
except Exception:  # noqa: BLE001
    O2ImageProcessor = O2Processor = O2Tokenizer = O2TokenizerFast = O2VideoProcessor = None
