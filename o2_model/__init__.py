# O2 model package for transformers.
#
# Importing this package registers the O2 architecture (model_type "o2" /
# "o2_text") with the Auto classes and the checkpoint conversion machinery.
# No host package files are modified.

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
)
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)

from .configuration_o2 import O2Config, O2TextConfig
from .modeling_o2 import (
    O2ForCausalLM,
    O2ForConditionalGeneration,
    O2Model,
    O2TextModel,
)

_registered = False


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
        from transformers.models.qwen3_vl import Qwen3VLProcessor

        AutoProcessor.register(O2Config, Qwen3VLProcessor, exist_ok=True)
    except Exception:
        pass
    # per-expert -> packed expert weight conversion, same rules as the host
    # MoE text backbone plus the language_model prefix shift
    mapping = list(get_checkpoint_conversion_mapping("qwen3_5_text")) + list(
        get_checkpoint_conversion_mapping("qwen2_moe")
    )
    try:
        register_checkpoint_conversion_mapping("o2_text", mapping)
    except ValueError:
        pass
    _registered = True


register()

__all__ = [
    "O2Config",
    "O2TextConfig",
    "O2Model",
    "O2TextModel",
    "O2ForCausalLM",
    "O2ForConditionalGeneration",
    "register",
]
