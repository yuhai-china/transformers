# o2-transformers

O2 model support for `transformers` — a hybrid (linear/full attention)
multimodal backbone with a per-layer routed dual-expert MoE FFN.

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
host dense layout.
