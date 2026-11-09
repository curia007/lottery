---
language:
- en
license: mit
tags:
- lottery
- mlx
- apple-silicon
- pytorch
- transformers
- causal-lm
- sequence-prediction
- idaho-pick3
pipeline_tag: text-generation
---

# curia007/pick3_llm_model

Autoregressive Causal Transformer model trained on Idaho Pick 3 lottery sequences using Apple MLX and PyTorch.

## Architecture & Specifications

- **Vocabulary Size**: `17` (Digits `0-9`, `<PAD>`, `<BOS>`, `<EOS>`, `<DAY>`, `<NIGHT>`, `<SEP>`, `<PRED>`)
- **Embedding Dimension**: `64`
- **Attention Heads**: `4`
- **Transformer Layers**: `3`
- **Max Sequence Length**: `128`

## Model Configuration

```json
{
  "vocab_size": 17,
  "max_seq_len": 128,
  "n_embd": 64,
  "n_head": 4,
  "n_layer": 3,
  "mlp_ratio": 4,
  "dropout": 0.0
}
```

## Quickstart Usage

### 1. Apple MLX (Apple Silicon)

```python
from idaho.pick3.pick3_llm_ticket_model import Pick3LLM, Pick3Tokenizer

# Load checkpoint directly from Hugging Face Hub
model = Pick3LLM.from_pretrained("curia007/pick3_llm_model")
tokenizer = Pick3Tokenizer()

# Generate candidate tickets
tickets = model.generate_tickets(draw="Night", count=5, temperature=0.8)
print("Predicted Tickets:", tickets)
```

### 2. PyTorch & Hugging Face Transformers

```python
from idaho.pick3.pick3_hf_model import Pick3ForCausalLM, Pick3HFConfig

# Load causal language model
model = Pick3ForCausalLM.from_pretrained("curia007/pick3_llm_model")
model.eval()
```
