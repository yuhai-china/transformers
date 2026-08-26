# O2 tokenizer / processor aliases: O2-named subclasses of the host
# implementations, registered for the "o2" model_type so checkpoint configs
# never need to reference host class names.

from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    Qwen2Tokenizer,
    Qwen2TokenizerFast,
)
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor


class O2Tokenizer(Qwen2Tokenizer):
    pass


class O2TokenizerFast(Qwen2TokenizerFast):
    pass


class O2ImageProcessor(Qwen2VLImageProcessor):
    pass


class O2VideoProcessor(Qwen3VLVideoProcessor):
    pass


class O2Processor(Qwen3VLProcessor):
    pass


def register_processors(config_class):
    AutoTokenizer.register(
        config_class,
        slow_tokenizer_class=O2Tokenizer,
        fast_tokenizer_class=O2TokenizerFast,
        exist_ok=True,
    )
    AutoImageProcessor.register(O2ConfigProxy, O2ImageProcessor, exist_ok=True)
    AutoProcessor.register(config_class, O2Processor, exist_ok=True)


# AutoImageProcessor/AutoProcessor key on the top-level config class; the
# caller passes it in (avoids a circular import with configuration_o2).
from .configuration_o2 import O2Config as O2ConfigProxy  # noqa: E402

__all__ = [
    "O2Tokenizer",
    "O2TokenizerFast",
    "O2ImageProcessor",
    "O2VideoProcessor",
    "O2Processor",
    "register_processors",
]
