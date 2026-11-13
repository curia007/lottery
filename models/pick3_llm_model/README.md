---
language:
- en
license: mit
tags:
- lottery
- time-series
- transformer
- mlx
- apple-silicon
- safetensors
- causal-lm
- idaho-pick3
pipeline_tag: text-generation
---

# Idaho Pick 3 Transformer Model (`idaho-pick3-transformer`)

An autoregressive Causal Transformer trained on historical Idaho Pick 3 lottery draw sequences using Apple MLX and PyTorch.

## Model Summary

- **Architecture**: Autoregressive Causal Decoder Transformer
- **Vocabulary Size**: 75 (Digits `0-9`, `<PAD>`, `<BOS>`, `<EOS>`, `<DAY>`, `<NIGHT>`, `<SEP>`, `<PRED>`)
- **Embedding Dimension**: 64
- **Heads**: 4
- **Layers**: 3
- **Max Sequence Length**: 128
- **Parameters**: ~147.5k

## Evaluation Metrics

- **Validation Cross-Entropy Loss**: `N/A`
- **Token Accuracy**: `N/A`
- **Exact Draw Match Rate**: `N/A`

## Quickstart

### Loading with Apple MLX

```python
from idaho.pick3.pick3_llm_ticket_model import Pick3LLM, Pick3Tokenizer

# Load directly from Hugging Face Hub or local path
model = Pick3LLM.from_pretrained("idaho-pick3-transformer")
tokenizer = Pick3Tokenizer()
```

### Loading with PyTorch / Hugging Face Transformers

```python
from idaho.pick3.pick3_hf_model import Pick3ForCausalLM

model = Pick3ForCausalLM.from_pretrained("idaho-pick3-transformer")
model.eval()
```
