#!/usr/bin/env python3
"""
upload_to_hf_hub.py

Utility script and Python API to package model checkpoints, generate model cards,
and push assets to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import huggingface_hub
    from huggingface_hub import HfApi, create_repo
    HAS_HF_HUB = True
except ImportError:
    huggingface_hub = None
    HfApi = None
    create_repo = None
    HAS_HF_HUB = False

try:
    from idaho.pick3.pick3_hf_model import Pick3HFConfig, Pick3HFHub
except ImportError:
    # Fallback if executed directly or in standalone environments
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    try:
        from idaho.pick3.pick3_hf_model import Pick3HFConfig, Pick3HFHub
    except ImportError:
        Pick3HFConfig = None
        Pick3HFHub = None


def normalize_repo_id(repo_id: str) -> str:
    """Strips URI schemes from repository identifiers."""
    s = str(repo_id).strip()
    if s.startswith("hf://"):
        s = s[len("hf://"):]
    elif s.startswith("huggingface://"):
        s = s[len("huggingface://"):]
    return s.strip("/")


def create_model_card(
    repo_id: str,
    metrics: Optional[Dict[str, Any]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generates a complete markdown Model Card (README.md) with metadata for the Hugging Face Hub.
    """
    clean_repo = normalize_repo_id(repo_id)
    metrics = metrics or {}
    config_dict = config_dict or {}

    metrics_block = ""
    if metrics:
        metrics_block = "## Evaluation Metrics\n\n"
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                if "acc" in key.lower() or "rate" in key.lower():
                    metrics_block += f"- **{key}**: `{val * 100:.2f}%`\n"
                else:
                    metrics_block += f"- **{key}**: `{val:.4f}`\n"
            else:
                metrics_block += f"- **{key}**: `{val}`\n"
        metrics_block += "\n"

    cfg_block = ""
    if config_dict:
        cfg_block = "## Model Configuration\n\n```json\n" + json.dumps(config_dict, indent=2) + "\n```\n\n"

    vocab_size = config_dict.get("vocab_size", 75)
    hidden_size = config_dict.get("hidden_size", config_dict.get("n_embd", 64))
    num_heads = config_dict.get("num_attention_heads", config_dict.get("n_head", 4))
    num_layers = config_dict.get("num_hidden_layers", config_dict.get("n_layer", 3))
    max_seq_len = config_dict.get("max_position_embeddings", config_dict.get("max_seq_len", 128))

    return f"""---
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

# {clean_repo}

Autoregressive Causal Transformer model trained on Idaho Pick 3 lottery sequences using Apple MLX and PyTorch.

## Architecture & Specifications

- **Vocabulary Size**: `{vocab_size}` (Digits `0-9`, `<PAD>`, `<BOS>`, `<EOS>`, `<DAY>`, `<NIGHT>`, `<SEP>`, `<PRED>`)
- **Embedding Dimension**: `{hidden_size}`
- **Attention Heads**: `{num_heads}`
- **Transformer Layers**: `{num_layers}`
- **Max Sequence Length**: `{max_seq_len}`

{metrics_block}{cfg_block}## Quickstart Usage

### 1. Apple MLX (Apple Silicon)

```python
from idaho.pick3.pick3_llm_ticket_model import Pick3LLM, Pick3Tokenizer

# Load checkpoint directly from Hugging Face Hub
model = Pick3LLM.from_pretrained("{clean_repo}")
tokenizer = Pick3Tokenizer()

# Generate candidate tickets
tickets = model.generate_tickets(draw="Night", count=5, temperature=0.8)
print("Predicted Tickets:", tickets)
```

### 2. PyTorch & Hugging Face Transformers

```python
from idaho.pick3.pick3_hf_model import Pick3ForCausalLM, Pick3HFConfig

# Load causal language model
model = Pick3ForCausalLM.from_pretrained("{clean_repo}")
model.eval()
```
"""


DEFAULT_CHAT_TEMPLATE: str = (
    "{% for message in messages %}"
    "{% set content = (message['content'] if message['content'] is not none else '') | trim %}"
    "{% if loop.first and message['role'] != 'system' %}{{ '<BOS>' }}{% endif %}"
    "{% if message['role'] == 'system' %}{% if content %}{{ '<BOS>' ~ content ~ ' ' }}{% else %}{{ '<BOS>' }}{% endif %}"
    "{% elif message['role'] == 'user' %}{% if not loop.first and messages[loop.index0 - 1]['role'] == 'assistant' %}{{ ' ' }}{% endif %}{{ content ~ ' <PRED>' }}"
    "{% elif message['role'] == 'assistant' %}{% if content %}{{ ' ' ~ content ~ ' <EOS>' }}{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt and (messages[-1]['role'] != 'assistant' or not (messages[-1]['content'] if messages[-1]['content'] is not none else '')) %}{{ ' ' }}{% endif %}"
)


def ensure_hf_assets(
    model_directory: Union[str, Path],
    repo_id: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Ensures that all requisite Hugging Face metadata files exist in the model directory:
    - config.json
    - generation_config.json
    - tokenizer_config.json
    - vocab.json
    - special_tokens_map.json
    - README.md (model card)
    """
    src_dir = Path(model_directory).resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {src_dir}")

    config_path = src_dir / "config.json"
    config_data: Dict[str, Any] = {}
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception:
            pass

    # 1. Config JSON - ensure complete GPT-2 / Hugging Face configuration
    if Pick3HFConfig is not None:
        cfg = Pick3HFConfig.from_dict(config_data) if config_data else Pick3HFConfig()
        cfg.to_json_file(config_path)
        config_data = cfg.to_dict()
    else:
        default_cfg = {
            "model_type": "gpt2",
            "vocab_size": config_data.get("vocab_size", 17),
            "n_positions": config_data.get("n_positions", config_data.get("max_position_embeddings", 128)),
            "n_ctx": config_data.get("n_ctx", config_data.get("max_position_embeddings", 128)),
            "max_position_embeddings": config_data.get("max_position_embeddings", 128),
            "max_seq_len": config_data.get("max_seq_len", 128),
            "n_embd": config_data.get("n_embd", config_data.get("hidden_size", 64)),
            "hidden_size": config_data.get("hidden_size", 64),
            "n_head": config_data.get("n_head", config_data.get("num_attention_heads", 4)),
            "num_attention_heads": config_data.get("num_attention_heads", 4),
            "n_layer": config_data.get("n_layer", config_data.get("num_hidden_layers", 3)),
            "num_hidden_layers": config_data.get("num_hidden_layers", 3),
            "n_inner": config_data.get("n_inner", config_data.get("intermediate_size", 256)),
            "intermediate_size": config_data.get("intermediate_size", 256),
            "activation_function": "gelu",
            "layer_norm_epsilon": config_data.get("layer_norm_epsilon", 1e-5),
            "initializer_range": config_data.get("initializer_range", 0.02),
            "resid_pdrop": config_data.get("resid_pdrop", 0.0),
            "embd_pdrop": config_data.get("embd_pdrop", 0.0),
            "attn_pdrop": config_data.get("attn_pdrop", 0.0),
            "scale_attn_weights": config_data.get("scale_attn_weights", True),
            "use_cache": config_data.get("use_cache", True),
            "dropout": config_data.get("dropout", 0.0),
            "pad_token_id": config_data.get("pad_token_id", 10),
            "bos_token_id": config_data.get("bos_token_id", 11),
            "eos_token_id": config_data.get("eos_token_id", 12),
            "architectures": ["GPT2LMHeadModel", "Pick3ForCausalLM"],
        }
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(default_cfg, f, indent=2)
        config_data = default_cfg

    # 2. Generation Config
    gen_cfg_path = src_dir / "generation_config.json"
    if not gen_cfg_path.exists():
        gen_cfg = {
            "_from_model_config": True,
            "bos_token_id": config_data.get("bos_token_id", 11),
            "eos_token_id": config_data.get("eos_token_id", 12),
            "pad_token_id": config_data.get("pad_token_id", 10),
            "do_sample": True,
            "temperature": 0.8,
            "top_k": 5,
            "top_p": 0.9,
            "max_new_tokens": 4,
        }
        with gen_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(gen_cfg, f, indent=2)

    # 3. Tokenizer Configuration and Fast Tokenizer (tokenizer.json)
    tok_cfg_path = src_dir / "tokenizer_config.json"
    if not tok_cfg_path.exists():
        from idaho.pick3.pick3_llm_ticket_model import WORD_TOKENS

        max_len = config_data.get("max_position_embeddings", config_data.get("max_seq_len", 128))
        added_tokens_decoder = {
            "10": {"content": "<PAD>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "11": {"content": "<BOS>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "12": {"content": "<EOS>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "13": {"content": "<DAY>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": False},
            "14": {"content": "<NIGHT>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": False},
            "15": {"content": "<SEP>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": False},
            "16": {"content": "<PRED>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": False},
        }
        for idx, word in enumerate(WORD_TOKENS):
            token_id = 17 + idx
            added_tokens_decoder[str(token_id)] = {
                "content": word, "lstrip": False, "normalized": False,
                "rstrip": False, "single_word": True, "special": False
            }
        tok_cfg = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_type": "gpt2",
            "vocab_size": config_data.get("vocab_size", 75),
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "pad_token": "<PAD>",
            "unk_token": "[UNK]",
            "model_max_length": max_len,
            "clean_up_tokenization_spaces": False,
            "chat_template": DEFAULT_CHAT_TEMPLATE,
            "added_tokens_decoder": added_tokens_decoder,
            "special_tokens": ["<PAD>", "<BOS>", "<EOS>", "[UNK]", "<UNK>"],
        }
        with tok_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(tok_cfg, f, indent=2)

    with tok_cfg_path.open("r", encoding="utf-8") as f:
        tok_cfg = json.load(f)
    added_tokens_decoder = tok_cfg.setdefault("added_tokens_decoder", {})
    for token_id in ("13", "14", "15", "16"):
        token_data = added_tokens_decoder.get(token_id)
        if token_data is not None:
            token_data["special"] = False
    tok_cfg["special_tokens"] = ["<PAD>", "<BOS>", "<EOS>", "[UNK]", "<UNK>"]
    with tok_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2)

    tok_json_path = src_dir / "tokenizer.json"
    if not tok_json_path.exists():
        from idaho.pick3.pick3_llm_ticket_model import WORD_TOKENS

        vocab_dict = {str(i): i for i in range(10)}
        vocab_dict.update({
            "<PAD>": 10,
            "<BOS>": 11,
            "<EOS>": 12,
            "<DAY>": 13,
            "<NIGHT>": 14,
            "<SEP>": 15,
            "<PRED>": 16,
            "[UNK]": 10,
            "<UNK>": 10,
        })
        for idx, word in enumerate(WORD_TOKENS):
            vocab_dict[word] = 17 + idx
        added_tokens = [
            {"id": 10, "content": "<PAD>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 11, "content": "<BOS>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 12, "content": "<EOS>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 13, "content": "<DAY>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": False},
            {"id": 14, "content": "<NIGHT>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": False},
            {"id": 15, "content": "<SEP>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": False},
            {"id": 16, "content": "<PRED>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": False},
            {"id": 10, "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 10, "content": "<UNK>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        ]
        for idx, word in enumerate(WORD_TOKENS):
            added_tokens.append({
                "id": 17 + idx, "content": word, "single_word": True,
                "lstrip": False, "rstrip": False, "normalized": False, "special": False
            })
        tok_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": None,
            "pre_tokenizer": {
                "type": "WhitespaceSplit"
            },
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": vocab_dict,
                "unk_token": "[UNK]"
            }
        }
        with tok_json_path.open("w", encoding="utf-8") as f:
            json.dump(tok_json, f, indent=2)

    from idaho.pick3.pick3_hf_model import Pick3HFHub

    with tok_json_path.open("w", encoding="utf-8") as f:
        json.dump(Pick3HFHub.get_fast_tokenizer_dict(), f, indent=2)

    # 4. Vocab dictionary
    vocab_path = src_dir / "vocab.json"
    if not vocab_path.exists():
        from idaho.pick3.pick3_llm_ticket_model import WORD_TOKENS

        vocab_dict = {str(i): i for i in range(10)}
        vocab_dict.update({
            "<PAD>": 10,
            "<BOS>": 11,
            "<EOS>": 12,
            "<DAY>": 13,
            "<NIGHT>": 14,
            "<SEP>": 15,
            "<PRED>": 16,
            "[UNK]": 10,
            "<UNK>": 10,
        })
        for idx, word in enumerate(WORD_TOKENS):
            vocab_dict[word] = 17 + idx
        with vocab_path.open("w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2)

    # 5. Special tokens map
    tokens_map_path = src_dir / "special_tokens_map.json"
    if not tokens_map_path.exists():
        special_tokens_map = {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "pad_token": "<PAD>",
            "unk_token": "[UNK]",
            "additional_special_tokens": ["<UNK>"],
        }
        with tokens_map_path.open("w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, indent=2)

    with tokens_map_path.open("r", encoding="utf-8") as f:
        special_tokens_map = json.load(f)
    special_tokens_map["additional_special_tokens"] = [
        token
        for token in special_tokens_map.get("additional_special_tokens", [])
        if token not in {"<DAY>", "<NIGHT>", "<SEP>", "<PRED>"}
    ]
    with tokens_map_path.open("w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2)

    # 6. Model Card README.md
    readme_path = src_dir / "README.md"
    if not readme_path.exists():
        card_content = create_model_card(
            repo_id=repo_id,
            metrics=metrics,
            config_dict=config_data,
        )
        readme_path.write_text(card_content, encoding="utf-8")

    # 7. Model code file for dynamic auto class resolution
    py_target = src_dir / "pick3_hf_model.py"
    if not py_target.exists():
        potential_src = Path(__file__).parent / "pick3_hf_model.py"
        if not potential_src.exists():
            potential_src = Path("pick3_hf_model.py")
        if potential_src.exists():
            try:
                shutil.copy2(potential_src, py_target)
            except Exception:
                pass

    return src_dir


def upload_checkpoint_to_hub(
    model_directory: Union[str, Path],
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload Idaho Pick 3 Transformer checkpoint and artifacts",
    metrics: Optional[Dict[str, Any]] = None,
    create_assets: bool = True,
) -> str:
    """
    Validates, packages, and pushes a local model directory to the Hugging Face Hub.

    Args:
        model_directory: Path to local folder containing model weights and configs.
        repo_id: Target Hugging Face Hub repository identifier (e.g. 'username/idaho-pick3-transformer').
        token: Hugging Face authentication token (or reads HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars).
        private: Whether to create the Hub repository as private.
        commit_message: Git commit message for the Hub upload.
        metrics: Optional dictionary of evaluation metrics to include in the generated model card.
        create_assets: Whether to auto-generate missing HF metadata assets before upload.

    Returns:
        The URL of the uploaded Hugging Face Hub repository.
    """
    src_dir = Path(model_directory).resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {src_dir}")

    clean_repo = normalize_repo_id(repo_id)

    # Ensure metadata assets exist
    if create_assets:
        ensure_hf_assets(src_dir, repo_id=clean_repo, metrics=metrics)

    # Resolve authentication token
    auth_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not auth_token:
        print("[upload_to_hf_hub] Warning: No authentication token provided. Set HF_TOKEN environment variable.")

    if not HAS_HF_HUB or HfApi is None:
        raise RuntimeError(
            "The 'huggingface_hub' package is required for uploading to the Hub. "
            "Install it via: pip install huggingface_hub"
        )

    # Initialize Hub API client and upload folder
    print(f"Connecting to Hugging Face Hub to push artifacts from '{src_dir}' to '{clean_repo}'...")
    api = HfApi(token=auth_token)
    api.create_repo(repo_id=clean_repo, token=auth_token, private=private, exist_ok=True)

    api.upload_folder(
        folder_path=str(src_dir),
        repo_id=clean_repo,
        commit_message=commit_message,
        token=auth_token,
    )

    repo_url = f"https://huggingface.co/{clean_repo}"
    print(f"Successfully uploaded checkpoint to Hugging Face Hub: {repo_url}")
    return repo_url


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Upload lottery sequence model checkpoint and assets to the Hugging Face Hub."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/pick3_llm_model",
        help="Path to local model directory containing config and weights (default: 'models/pick3_llm_model')",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="curia007/pick3_llm_model",
        help="Hugging Face repository ID (default: 'curia007/pick3_llm_model')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face user access token (defaults to HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private",
    )
    parser.add_argument(
        "--commit-msg",
        type=str,
        default="Upload Idaho Pick 3 Transformer checkpoint and artifacts",
        help="Commit message for Hugging Face upload",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="JSON string or path to JSON file containing evaluation metrics for the model card",
    )

    args = parser.parse_args(argv)

    # Resolve model directory
    model_path = Path(args.model_dir)
    if not model_path.exists():
        fallback_path = Path("") / args.model_dir
        if fallback_path.exists():
            model_path = fallback_path

    # Parse metrics if provided
    metrics_dict: Optional[Dict[str, Any]] = None
    if args.metrics:
        metrics_file = Path(args.metrics)
        if metrics_file.exists():
            with metrics_file.open("r", encoding="utf-8") as f:
                metrics_dict = json.load(f)
        else:
            try:
                metrics_dict = json.loads(args.metrics)
            except Exception:
                print(f"Warning: Could not parse --metrics argument as JSON or file path: {args.metrics}")

    upload_checkpoint_to_hub(
        model_directory=model_path,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        commit_message=args.commit_msg,
        metrics=metrics_dict,
    )


if __name__ == "__main__":
    main()
