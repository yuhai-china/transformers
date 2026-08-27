# o2-transformers

O2 model support for `transformers` — a hybrid (linear/full attention)
multimodal backbone with a per-layer routed dual-expert MoE FFN.

**Self-contained**: the modeling code (`o2_model/modeling_o2.py`) imports only
transformers *core* APIs (`PreTrainedModel`, `cache_utils`, `masking_utils`,
`vision_utils`, the attention interface) — nothing from
`transformers.models.*`. Host-model refactors cannot break it. Numerics match
the Qwen3.5 reference implementation bit for bit (verified: identical weights
produce identical logits for prefill, cached decode, and image inputs, and
identical token ids on the real 83GB merged checkpoint).

The checkpoint conversion mapping (mergekit's per-expert 2D layout ↔ packed
3D experts) is likewise defined explicitly in `o2_model/__init__.py` instead of
reusing host-registered mappings.

## Install

```bash
pip install .
```

## Use

```python
import o2_model  # registers the "o2" architecture with the Auto classes
from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("/path/to/o2-checkpoint")
```

Checkpoint layout:

```
model.language_model.layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight
model.language_model.layers.N.mlp.gate.weight
```

Everything else (attention, norms, embeddings, `model.visual.*`) follows the
dense checkpoint layout of the base model. MTP tensors, if present, are
ignored on load.

## Package layout

| File | Contents |
|---|---|
| `configuration_o2.py` | `O2Config` / `O2TextConfig` / `O2VisionConfig` (plain `PreTrainedConfig`) |
| `modeling_o2.py` | RMSNorm, mRoPE rotary, full attention, gated delta net (pure-torch kernels), MoE router/experts, vision tower, composite model + LM heads |
| `tokenizer_o2.py` | O2-named tokenizer/processor aliases (host-class shims; optional, import-guarded) |
| `__init__.py` | Auto-class registration + explicit checkpoint conversion mapping |

## Compatibility notes

- Verified on transformers 5.12 and 5.14 (CPU fp32 and Apple MPS bf16). The
  cache-layer state API differs across releases (`conv_states` tensor vs
  per-slot mapping); both forms are handled.
- RoPE supports `rope_type="default"` (the O2 checkpoint setting). Long-context
  scaling (YaRN) is a serving-time concern handled by vLLM, not this package.
- The gated delta net uses the pure-torch reference kernels (chunked prefill,
  recurrent decode); the FLA/causal-conv1d fast paths are intentionally not
  wired in — this package is a reference implementation, serving runs on vLLM.
