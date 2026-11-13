#!/usr/bin/env python3
"""
pick3_hf_model.py

Hugging Face Hub and PyTorch / Transformers integration for the Idaho Pick 3 lottery model.
Provides:
- Pick3HFConfig: Transformers-compatible configuration schema and serialization.
- Pick3HFHub: Utilities to download from and push models to Hugging Face Hub,
  generate model cards (README.md), generation configs, and tokenizer artifacts.
- Pick3ForCausalLM: PyTorch / Hugging Face compatible Causal LM Transformer architecture
  mirroring the MLX Pick3LLM model with bidirectional weight conversion.
- Full compatibility with Hugging Face Hub repositories (e.g. 'hf://username/repo' or 'username/repo').
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import huggingface_hub
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except ImportError:
    huggingface_hub = None
    HfApi = None
    hf_hub_download = None
    snapshot_download = None


# ==============================================================================
# Hugging Face Model Configuration
# ==============================================================================

DEFAULT_ID2LABEL: Dict[int, str] = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "<PAD>",
    11: "<BOS>",
    12: "<EOS>",
    13: "<DAY>",
    14: "<NIGHT>",
    15: "<SEP>",
    16: "<PRED>",
}

DEFAULT_LABEL2ID: Dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "<PAD>": 10,
    "<BOS>": 11,
    "<EOS>": 12,
    "<DAY>": 13,
    "<NIGHT>": 14,
    "<SEP>": 15,
    "<PRED>": 16,
}


@dataclass
class Pick3HFConfig:
    """
    Configuration specification compatible with Hugging Face Transformers.
    """
    model_type: str = "gpt2"
    vocab_size: int = 75
    max_position_embeddings: int = 128
    hidden_size: int = 64
    num_attention_heads: int = 4
    num_hidden_layers: int = 3
    intermediate_size: int = 256  # 4 * hidden_size
    dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    scale_attn_weights: bool = True
    use_cache: bool = True
    pad_token_id: int = 10
    bos_token_id: int = 11
    eos_token_id: int = 12
    id2label: Dict[int, str] = field(default_factory=lambda: dict(DEFAULT_ID2LABEL))
    label2id: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LABEL2ID))
    architectures: List[str] = field(default_factory=lambda: ["GPT2LMHeadModel", "Pick3ForCausalLM"])
    transformers_version: str = "4.45.0"
    auto_map: Dict[str, str] = field(
        default_factory=lambda: {
            "AutoConfig": "pick3_hf_model.Pick3HFConfig",
            "AutoModelForCausalLM": "pick3_hf_model.Pick3ForCausalLM",
        }
    )

    @classmethod
    def register_for_auto_class(cls, auto_class: str = "AutoConfig") -> None:
        """Compatibility method for Hugging Face dynamic auto class registration."""
        pass

    def map_id_to_label(self, token_id: Union[int, str]) -> str:
        """Maps an integer token ID to its corresponding label name using id2label."""
        try:
            int_id = int(token_id)
        except (ValueError, TypeError):
            return str(token_id)
        return self.id2label.get(int_id, self.id2label.get(str(int_id), str(token_id)))

    def map_ids_to_labels(self, token_ids: Sequence[Union[int, str]]) -> List[str]:
        """Maps a sequence of integer token IDs to label names using id2label."""
        return [self.map_id_to_label(tid) for tid in token_ids]

    # Aliases for compatibility with MLX Pick3LLMConfig
    @property
    def max_seq_len(self) -> int:
        return self.max_position_embeddings

    @property
    def n_embd(self) -> int:
        return self.hidden_size

    @property
    def n_head(self) -> int:
        return self.num_attention_heads

    @property
    def n_layer(self) -> int:
        return self.num_hidden_layers

    @property
    def mlp_ratio(self) -> int:
        return self.intermediate_size // max(1, self.hidden_size)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Standard Hugging Face GPT2 specification parameters
        data["model_type"] = self.model_type
        data["vocab_size"] = self.vocab_size
        data["n_positions"] = self.max_position_embeddings
        data["n_ctx"] = self.max_position_embeddings
        data["n_embd"] = self.hidden_size
        data["n_layer"] = self.num_hidden_layers
        data["n_head"] = self.num_attention_heads
        data["n_inner"] = self.intermediate_size
        data["activation_function"] = "gelu"
        data["layer_norm_epsilon"] = self.layer_norm_epsilon
        data["initializer_range"] = self.initializer_range
        data["resid_pdrop"] = self.resid_pdrop
        data["embd_pdrop"] = self.embd_pdrop
        data["attn_pdrop"] = self.attn_pdrop
        data["scale_attn_weights"] = self.scale_attn_weights
        data["use_cache"] = self.use_cache
        data["id2label"] = {str(k): str(v) for k, v in self.id2label.items()}
        data["label2id"] = {str(k): int(v) for k, v in self.label2id.items()}
        # Cross-framework and MLX naming aliases
        data["max_seq_len"] = self.max_position_embeddings
        data["max_position_embeddings"] = self.max_position_embeddings
        data["hidden_size"] = self.hidden_size
        data["num_attention_heads"] = self.num_attention_heads
        data["num_hidden_layers"] = self.num_hidden_layers
        data["intermediate_size"] = self.intermediate_size
        data["mlp_ratio"] = self.mlp_ratio
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pick3HFConfig":
        cfg = cls()
        if "model_type" in data:
            cfg.model_type = str(data["model_type"])
        if "vocab_size" in data:
            cfg.vocab_size = int(data["vocab_size"])
        if "max_position_embeddings" in data:
            cfg.max_position_embeddings = int(data["max_position_embeddings"])
        elif "n_positions" in data:
            cfg.max_position_embeddings = int(data["n_positions"])
        elif "n_ctx" in data:
            cfg.max_position_embeddings = int(data["n_ctx"])
        elif "max_seq_len" in data:
            cfg.max_position_embeddings = int(data["max_seq_len"])
        if "hidden_size" in data:
            cfg.hidden_size = int(data["hidden_size"])
        elif "n_embd" in data:
            cfg.hidden_size = int(data["n_embd"])
        if "num_attention_heads" in data:
            cfg.num_attention_heads = int(data["num_attention_heads"])
        elif "n_head" in data:
            cfg.num_attention_heads = int(data["n_head"])
        if "num_hidden_layers" in data:
            cfg.num_hidden_layers = int(data["num_hidden_layers"])
        elif "n_layer" in data:
            cfg.num_hidden_layers = int(data["n_layer"])
        if "intermediate_size" in data:
            cfg.intermediate_size = int(data["intermediate_size"])
        elif "n_inner" in data:
            cfg.intermediate_size = int(data["n_inner"])
        elif "mlp_ratio" in data:
            cfg.intermediate_size = cfg.hidden_size * int(data["mlp_ratio"])
        else:
            cfg.intermediate_size = cfg.hidden_size * 4
        if "dropout" in data:
            cfg.dropout = float(data["dropout"])
        if "layer_norm_epsilon" in data:
            cfg.layer_norm_epsilon = float(data["layer_norm_epsilon"])
        if "initializer_range" in data:
            cfg.initializer_range = float(data["initializer_range"])
        if "resid_pdrop" in data:
            cfg.resid_pdrop = float(data["resid_pdrop"])
        if "embd_pdrop" in data:
            cfg.embd_pdrop = float(data["embd_pdrop"])
        if "attn_pdrop" in data:
            cfg.attn_pdrop = float(data["attn_pdrop"])
        if "scale_attn_weights" in data:
            cfg.scale_attn_weights = bool(data["scale_attn_weights"])
        if "use_cache" in data:
            cfg.use_cache = bool(data["use_cache"])
        if "pad_token_id" in data:
            cfg.pad_token_id = int(data["pad_token_id"])
        if "bos_token_id" in data:
            cfg.bos_token_id = int(data["bos_token_id"])
        if "eos_token_id" in data:
            cfg.eos_token_id = int(data["eos_token_id"])
        if "id2label" in data and isinstance(data["id2label"], dict):
            cfg.id2label = {int(k): str(v) for k, v in data["id2label"].items()}
        if "label2id" in data and isinstance(data["label2id"], dict):
            cfg.label2id = {str(k): int(v) for k, v in data["label2id"].items()}
        if "architectures" in data and isinstance(data["architectures"], list):
            cfg.architectures = [str(a) for a in data["architectures"]]
        return cfg

    def to_json_file(self, json_file_path: Union[str, Path]) -> None:
        path = Path(json_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json_file(cls, json_file_path: Union[str, Path]) -> "Pick3HFConfig":
        with Path(json_file_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ==============================================================================
# Hugging Face Hub Interaction & Asset Generation
# ==============================================================================

class Pick3HFHub:
    """
    Utilities for interacting with the Hugging Face Hub ecosystem:
    - Resolving and downloading models from Hugging Face Hub repositories.
    - Exporting Hugging Face configuration, model card (README.md), generation config,
      and tokenizer metadata.
    - Pushing model artifacts to Hugging Face Hub repositories.
    """

    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lottery_ai" / "hub"

    @classmethod
    def decode_response_text(
        cls,
        token_ids: Any,
        alternative_tickets: Optional[Sequence[Any]] = None,
        prob_matrix: Optional[Any] = None,
        draw_type: Optional[str] = "Night",
        include_narrative: bool = False,
    ) -> str:
        """
        Decodes a token sequence, string, or tensor into localized natural language prediction text.
        e.g., 'The next winning Pick 3 numbers are 1 2 3.'
        """
        from idaho.pick3.pick3_llm_ticket_model import Pick3Tokenizer
        tok = Pick3Tokenizer()
        return tok.decode_response_text(
            token_ids=token_ids,
            alternative_tickets=alternative_tickets,
            prob_matrix=prob_matrix,
            draw_type=draw_type,
            include_narrative=include_narrative,
        )

    @classmethod
    def decode_response(
        cls,
        token_ids: Any,
        alternative_tickets: Optional[Sequence[Any]] = None,
        prob_matrix: Optional[Any] = None,
        draw_type: Optional[str] = "Night",
    ) -> Dict[str, str]:
        """
        Decodes a token sequence, string, or tensor into a rich localized natural language response dictionary.
        """
        from idaho.pick3.pick3_llm_ticket_model import Pick3Tokenizer
        tok = Pick3Tokenizer()
        return tok.decode_response(
            token_ids=token_ids,
            alternative_tickets=alternative_tickets,
            prob_matrix=prob_matrix,
            draw_type=draw_type,
        )

    @classmethod
    def decode_all_draw_digits(
        cls,
        token_ids: Any,
    ) -> List[Tuple[int, int, int]]:
        """
        Extracts all 3-digit draw triplets from any token sequence, string, or tensor.
        """
        from idaho.pick3.pick3_llm_ticket_model import Pick3Tokenizer
        tok = Pick3Tokenizer()
        return tok.decode_all_draw_digits(token_ids)

    @classmethod
    def decode_ids_to_labels(
        cls,
        token_ids: Any,
        config: Optional[Union[Pick3HFConfig, Dict[str, Any]]] = None,
        skip_special_tokens: bool = False,
    ) -> Union[str, List[Any]]:
        """
        Maps integer token IDs back to their label names using config.id2label.
        """
        if config is not None:
            if isinstance(config, dict):
                raw_id2label = config.get("id2label", DEFAULT_ID2LABEL)
            else:
                raw_id2label = getattr(config, "id2label", DEFAULT_ID2LABEL)
            id2label = {int(k): str(v) for k, v in raw_id2label.items()}
        else:
            id2label = dict(DEFAULT_ID2LABEL)

        special_tokens = {"<PAD>", "<BOS>", "<EOS>", "<DAY>", "<NIGHT>", "<SEP>", "<PRED>", "[UNK]", "<UNK>"}

        def _map_single_id(tid: Any) -> str:
            try:
                int_id = int(tid)
            except (ValueError, TypeError):
                return str(tid)
            return id2label.get(int_id, str(tid))

        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        if isinstance(token_ids, (int,)):
            lbl = _map_single_id(token_ids)
            if skip_special_tokens and lbl in special_tokens:
                return ""
            return lbl

        if isinstance(token_ids, (list, tuple)):
            if not token_ids:
                return []
            if isinstance(token_ids[0], (list, tuple)):
                return [cls.decode_ids_to_labels(seq, config=config, skip_special_tokens=skip_special_tokens) for seq in token_ids]
            labels = [_map_single_id(tid) for tid in token_ids]
            if skip_special_tokens:
                labels = [l for l in labels if l not in special_tokens]
            return labels

        return _map_single_id(token_ids)

    @classmethod
    def format_localized_response(
        cls,
        predicted_digits: Sequence[int],
        prob_matrix: Optional[Any] = None,
        alternative_tickets: Optional[Sequence[Any]] = None,
        draw_type: Optional[str] = "Night",
    ) -> Dict[str, str]:
        """
        Formats predicted lottery digits into a structured localized response dictionary.
        """
        from idaho.pick3.pick3_llm_ticket_model import format_pick3_localized_response
        return format_pick3_localized_response(
            predicted_digits=predicted_digits,
            prob_matrix=prob_matrix,
            alternative_tickets=alternative_tickets,
            draw_type=draw_type,
        )

    @classmethod
    def create_omlx_client(
        cls,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "pick3_llm_model",
        timeout: float = 15.0,
        api_key: Optional[str] = None,
    ) -> Any:
        """
        Instantiates an OMLXPick3Client for querying local oMLX / mlx-lm model endpoints.
        """
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client
        return OMLXPick3Client(
            base_url=base_url,
            model_name=model_name,
            timeout=timeout,
            api_key=api_key,
        )

    @staticmethod
    def is_hf_repo_id(repo_id_or_path: Union[str, Path]) -> bool:
        """Determines whether a string represents a Hugging Face Hub repository identifier."""
        s = str(repo_id_or_path).strip()
        if s.startswith("hf://") or s.startswith("huggingface://"):
            return True
        if Path(s).is_dir() or Path(s).is_file():
            return False
        # Matches patterns like 'username/repo-name' or 'repo-name' without local filesystem match
        return bool(re.fullmatch(r"^[a-zA-Z0-9_\-]+(/[a-zA-Z0-9_\.\-]+)?$", s))

    @staticmethod
    def normalize_repo_id(repo_id: str) -> str:
        """Normalizes repo string removing URL schemes."""
        s = str(repo_id).strip()
        if s.startswith("hf://"):
            s = s[len("hf://"):]
        elif s.startswith("huggingface://"):
            s = s[len("huggingface://"):]
        return s.strip("/")

    @classmethod
    def download_file_from_hub(
        cls,
        repo_id: str,
        filename: str,
        revision: str = "main",
        token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
    ) -> Path:
        """
        Downloads a single file from Hugging Face Hub.
        Uses huggingface_hub if installed, or falls back to standard HTTP download.
        """
        clean_repo = cls.normalize_repo_id(repo_id)
        token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        target_cache = Path(cache_dir or cls.DEFAULT_CACHE_DIR) / clean_repo / revision
        target_file = target_cache / filename

        if target_file.exists() and not force_download:
            return target_file

        target_cache.mkdir(parents=True, exist_ok=True)

        if huggingface_hub is not None and hf_hub_download is not None:
            try:
                downloaded_path = hf_hub_download(
                    repo_id=clean_repo,
                    filename=filename,
                    revision=revision,
                    token=token,
                    cache_dir=str(target_cache.parent),
                    force_download=force_download,
                )
                return Path(downloaded_path)
            except Exception as e:
                print(f"[Pick3HFHub] huggingface_hub download notification for {filename}: {e}. Trying direct HTTP fallback...")

        # Fallback to direct HTTP download via raw resolve URL
        url = f"https://huggingface.co/{clean_repo}/resolve/{revision}/{filename}"
        headers = {"User-Agent": "LotteryAI-Pick3/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                target_file.write_bytes(content)
                return target_file
        except urllib.error.HTTPError as e:
            raise FileNotFoundError(f"Failed to fetch '{filename}' from Hugging Face repository '{clean_repo}' ({e.code}: {e.reason})") from e
        except Exception as e:
            raise RuntimeError(f"Error downloading '{filename}' from '{clean_repo}': {e}") from e

    @classmethod
    def resolve_model_dir(
        cls,
        model_id_or_path: Union[str, Path],
        revision: str = "main",
        token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
    ) -> Path:
        """
        Resolves local path or downloads complete checkpoint files from Hugging Face Hub.
        Returns the Path to the directory containing config.json and model weights.
        """
        path_candidate = Path(model_id_or_path).expanduser()
        if path_candidate.is_dir() and (path_candidate / "config.json").exists():
            return path_candidate

        if not cls.is_hf_repo_id(model_id_or_path):
            if path_candidate.exists():
                return path_candidate
            raise FileNotFoundError(f"Model path or Hugging Face repository not found: '{model_id_or_path}'")

        repo_id = cls.normalize_repo_id(str(model_id_or_path))
        print(f"[Pick3HFHub] Resolving Hugging Face Hub repository: '{repo_id}' (revision: '{revision}')...")

        # Download essential files
        config_file = cls.download_file_from_hub(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            force_download=force_download,
        )
        dest_dir = config_file.parent

        # Try downloading model weights (.safetensors or .npz)
        weights_found = False
        for weight_name in ("model.safetensors", "model.npz", "pytorch_model.bin"):
            try:
                cls.download_file_from_hub(
                    repo_id=repo_id,
                    filename=weight_name,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    force_download=force_download,
                )
                weights_found = True
                break
            except Exception:
                continue

        if not weights_found:
            print(f"[Pick3HFHub] Notice: No pre-existing weights found for '{repo_id}'. Architecture initialized from config.")

        # Optionally fetch auxiliary metadata if available
        for aux_file in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "special_tokens_map.json", "README.md"):
            try:
                cls.download_file_from_hub(
                    repo_id=repo_id,
                    filename=aux_file,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    force_download=force_download,
                )
            except Exception:
                pass

        return dest_dir

    @classmethod
    def generate_model_card(
        cls,
        config: Pick3HFConfig,
        metrics: Optional[Dict[str, Any]] = None,
        repo_id: Optional[str] = None,
    ) -> str:
        """
        Generates standard Hugging Face model card README.md with YAML metadata.
        """
        metrics = metrics or {}
        repo_name = repo_id or "idaho-pick3-transformer"
        val_loss = metrics.get("loss") or metrics.get("val_loss") or "N/A"
        token_acc = metrics.get("token_accuracy") or metrics.get("val_token_accuracy") or "N/A"
        exact_match = metrics.get("exact_match_rate") or metrics.get("val_exact_match_rate") or "N/A"

        val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
        token_acc_str = f"{token_acc * 100:.2f}%" if isinstance(token_acc, (int, float)) else str(token_acc)
        exact_match_str = f"{exact_match * 100:.2f}%" if isinstance(exact_match, (int, float)) else str(exact_match)

        return f"""---
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

# Idaho Pick 3 Transformer Model (`{repo_name}`)

An autoregressive Causal Transformer trained on historical Idaho Pick 3 lottery draw sequences using Apple MLX and PyTorch.

## Model Summary

- **Architecture**: Autoregressive Causal Decoder Transformer
- **Vocabulary Size**: {config.vocab_size} (Digits `0-9`, `<PAD>`, `<BOS>`, `<EOS>`, `<DAY>`, `<NIGHT>`, `<SEP>`, `<PRED>`)
- **Embedding Dimension**: {config.hidden_size}
- **Heads**: {config.num_attention_heads}
- **Layers**: {config.num_hidden_layers}
- **Max Sequence Length**: {config.max_position_embeddings}
- **Parameters**: ~{config.num_hidden_layers * 12 * config.hidden_size * config.hidden_size / 1000:.1f}k

## Evaluation Metrics

- **Validation Cross-Entropy Loss**: `{val_loss_str}`
- **Token Accuracy**: `{token_acc_str}`
- **Exact Draw Match Rate**: `{exact_match_str}`

## Quickstart

### Loading with Apple MLX

```python
from idaho.pick3.pick3_llm_ticket_model import Pick3LLM, Pick3Tokenizer

# Load directly from Hugging Face Hub or local path
model = Pick3LLM.from_pretrained("{repo_name}")
tokenizer = Pick3Tokenizer()
```

### Loading with PyTorch / Hugging Face Transformers

```python
from idaho.pick3.pick3_hf_model import Pick3ForCausalLM

model = Pick3ForCausalLM.from_pretrained("{repo_name}")
model.eval()
```
"""

    @classmethod
    def generate_generation_config(cls, config: Pick3HFConfig) -> Dict[str, Any]:
        """Generates standard Hugging Face generation_config.json dictionary."""
        return {
            "_from_model_config": True,
            "bos_token_id": config.bos_token_id,
            "eos_token_id": config.eos_token_id,
            "pad_token_id": config.pad_token_id,
            "do_sample": True,
            "temperature": 0.8,
            "top_k": 5,
            "top_p": 0.9,
            "max_new_tokens": 4,
            "transformers_version": config.transformers_version,
        }

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

    @classmethod
    def get_fast_tokenizer_dict(cls) -> Dict[str, Any]:
        """Returns the serialized Hugging Face Tokenizers fast tokenizer dictionary with expanded vocabulary."""
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
        # Add word tokens
        for idx, word in enumerate(WORD_TOKENS):
            vocab_dict[word] = 17 + idx

        # Build added_tokens list
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
        # Add word tokens to added_tokens
        for idx, word in enumerate(WORD_TOKENS):
            added_tokens.append({
                "id": 17 + idx, "content": word, "single_word": True,
                "lstrip": False, "rstrip": False, "normalized": False, "special": False
            })

        # Build decoder replacements
        decoders = [
            *[
                {"type": "Replace", "pattern": {"String": str(digit)}, "content": f" {digit}"}
                for digit in range(10)
            ],
            {"type": "Replace", "pattern": {"String": "<DAY>"}, "content": "Day draw:"},
            {"type": "Replace", "pattern": {"String": "<NIGHT>"}, "content": "Night draw:"},
            {"type": "Replace", "pattern": {"String": "<SEP>"}, "content": ". "},
            {"type": "Replace", "pattern": {"String": "<PRED>"}, "content": "Prediction: "},
            {"type": "Replace", "pattern": {"String": "<EOS>"}, "content": "."},
        ]
        # Add word token decoders
        for word in WORD_TOKENS:
            decoders.append({"type": "Replace", "pattern": {"String": word}, "content": f" {word}"})

        return {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": None,
            "pre_tokenizer": {
                "type": "WhitespaceSplit"
            },
            "post_processor": None,
            "decoder": {
                "type": "Sequence",
                "decoders": decoders,
            },
            "model": {
                "type": "WordLevel",
                "vocab": vocab_dict,
                "unk_token": "[UNK]"
            }
        }

    @classmethod
    def export_hf_assets(
        cls,
        save_directory: Union[str, Path],
        config: Pick3HFConfig,
        metrics: Optional[Dict[str, Any]] = None,
        repo_id: Optional[str] = None,
    ) -> Path:
        """
        Exports all Hugging Face metadata files to save_directory:
        - config.json
        - generation_config.json
        - tokenizer.json (Fast Tokenizer serialization)
        - tokenizer_config.json
        - vocab.json
        - special_tokens_map.json
        - README.md (model card)
        """
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Config
        config.to_json_file(out_dir / "config.json")

        # 2. Generation Config
        gen_cfg = cls.generate_generation_config(config)
        with (out_dir / "generation_config.json").open("w", encoding="utf-8") as f:
            json.dump(gen_cfg, f, indent=2)

        # 3. Tokenizer Assets
        from idaho.pick3.pick3_llm_ticket_model import WORD_TOKENS

        max_len = getattr(config, "max_position_embeddings", getattr(config, "max_seq_len", 128))
        # Build added_tokens_decoder with word tokens
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
            "vocab_size": config.vocab_size,
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "pad_token": "<PAD>",
            "unk_token": "[UNK]",
            "model_max_length": max_len,
            "clean_up_tokenization_spaces": False,
            "chat_template": cls.DEFAULT_CHAT_TEMPLATE,
            "added_tokens_decoder": added_tokens_decoder,
            "special_tokens": ["<PAD>", "<BOS>", "<EOS>", "[UNK]", "<UNK>"],
        }
        with (out_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
            json.dump(tok_cfg, f, indent=2)

        # Fast Tokenizer Serialization (tokenizer.json)
        tok_json = cls.get_fast_tokenizer_dict()
        with (out_dir / "tokenizer.json").open("w", encoding="utf-8") as f:
            json.dump(tok_json, f, indent=2)

        # Vocab dictionary with word tokens
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
        with (out_dir / "vocab.json").open("w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2)

        special_tokens_map = {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "pad_token": "<PAD>",
            "unk_token": "[UNK]",
            "additional_special_tokens": ["<UNK>"],
        }
        with (out_dir / "special_tokens_map.json").open("w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, indent=2)

        # 4. Model Card README.md
        readme_path = out_dir / "README.md"
        if not readme_path.exists():
            readme_path.write_text(cls.generate_model_card(config, metrics, repo_id), encoding="utf-8")

        # 5. Code file for AutoModel dynamic module loading if available
        py_src = Path(__file__).resolve()
        if py_src.exists():
            try:
                shutil.copy2(py_src, out_dir / "pick3_hf_model.py")
            except Exception:
                pass

        return out_dir

    @classmethod
    def push_to_hub(
        cls,
        save_directory: Union[str, Path],
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Pushes a local directory containing model and tokenizer artifacts to Hugging Face Hub.
        """
        clean_repo = cls.normalize_repo_id(repo_id)
        token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        if not token:
            print("[Pick3HFHub] Warning: No HF_TOKEN provided. Set HF_TOKEN environment variable to push to Hub.")

        if huggingface_hub is not None and HfApi is not None:
            api = HfApi(token=token)
            api.create_repo(repo_id=clean_repo, exist_ok=True, private=private)
            commit_msg = commit_message or "Upload Idaho Pick 3 Transformer checkpoint and artifacts"
            api.upload_folder(
                folder_path=str(save_directory),
                repo_id=clean_repo,
                commit_message=commit_msg,
                token=token,
            )
            repo_url = f"https://huggingface.co/{clean_repo}"
            print(f"[Pick3HFHub] Successfully pushed checkpoint to Hugging Face Hub: {repo_url}")
            return repo_url
        else:
            print(
                f"[Pick3HFHub] Note: 'huggingface_hub' package is not installed. "
                f"Install with 'pip install huggingface_hub' to enable direct API uploads to '{clean_repo}'."
            )
            return f"https://huggingface.co/{clean_repo}"


# ==============================================================================
# PyTorch / Transformers Compatible Model Architecture
# ==============================================================================

if torch is not None and nn is not None:

    class PyTorchCausalSelfAttention(nn.Module):
        """Multi-Head Causal Self-Attention in PyTorch."""

        def __init__(self, config: Pick3HFConfig):
            super().__init__()
            self.n_embd = config.hidden_size
            self.n_head = config.num_attention_heads
            self.head_dim = self.n_embd // self.n_head
            self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
            self.c_proj = nn.Linear(self.n_embd, self.n_embd)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            B, T, C = x.shape
            qkv = self.c_attn(x)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Causal mask
            causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal_mask

            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    mask_expanded = attention_mask[:, None, None, :].to(dtype=scores.dtype)
                    scores = scores + (1.0 - mask_expanded) * -1e9
                elif attention_mask.dim() == 4:
                    scores = scores + attention_mask

            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            out = probs @ v
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(out)


    class PyTorchTransformerBlock(nn.Module):
        """Transformer block with LayerNorm, Attention, and MLP in PyTorch."""

        def __init__(self, config: Pick3HFConfig):
            super().__init__()
            self.ln_1 = nn.LayerNorm(config.hidden_size)
            self.attn = PyTorchCausalSelfAttention(config)
            self.ln_2 = nn.LayerNorm(config.hidden_size)
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.dropout),
            )

        def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
            x = x + self.mlp(self.ln_2(x))
            return x


    class Pick3ForCausalLM(nn.Module):
        """
        PyTorch / Hugging Face Transformers compatible Causal LM for Idaho Pick 3 sequences.
        Directly mirrors the Apple MLX Pick3LLM architecture.
        """

        def __init__(self, config: Optional[Pick3HFConfig] = None):
            super().__init__()
            self.config = config or Pick3HFConfig()
            self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
            self.pos_emb = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)
            self.drop = nn.Dropout(self.config.dropout)
            self.h = nn.ModuleList([PyTorchTransformerBlock(self.config) for _ in range(self.config.num_hidden_layers)])
            self.blocks = self.h
            self.ln_f = nn.LayerNorm(self.config.hidden_size, eps=getattr(self.config, "layer_norm_epsilon", 1e-5))
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        @property
        def id2label(self) -> Dict[int, str]:
            """Returns the integer ID to label name mapping from model configuration."""
            raw = getattr(self.config, "id2label", DEFAULT_ID2LABEL)
            if isinstance(raw, dict):
                return {int(k): str(v) for k, v in raw.items()}
            return dict(DEFAULT_ID2LABEL)

        @property
        def label2id(self) -> Dict[str, int]:
            """Returns the label name to integer ID mapping from model configuration."""
            raw = getattr(self.config, "label2id", DEFAULT_LABEL2ID)
            if isinstance(raw, dict):
                return {str(k): int(v) for k, v in raw.items()}
            return dict(DEFAULT_LABEL2ID)

        def decode_ids_to_labels(
            self,
            token_ids: Any,
            skip_special_tokens: bool = False,
        ) -> Union[str, List[Any]]:
            """
            Maps integer IDs back to label names using model.config.id2label.
            Supports single integers, lists, tuples, PyTorch tensors, and NumPy arrays.
            """
            id2label = self.id2label
            special_tokens = {"<PAD>", "<BOS>", "<EOS>", "<DAY>", "<NIGHT>", "<SEP>", "<PRED>", "[UNK]", "<UNK>"}

            def _map_single_id(tid: Any) -> str:
                try:
                    int_id = int(tid)
                except (ValueError, TypeError):
                    return str(tid)
                return id2label.get(int_id, str(tid))

            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()

            if isinstance(token_ids, (int,)):
                lbl = _map_single_id(token_ids)
                if skip_special_tokens and lbl in special_tokens:
                    return ""
                return lbl

            if isinstance(token_ids, (list, tuple)):
                if not token_ids:
                    return []
                if isinstance(token_ids[0], (list, tuple)):
                    return [self.decode_ids_to_labels(seq, skip_special_tokens=skip_special_tokens) for seq in token_ids]
                labels = [_map_single_id(tid) for tid in token_ids]
                if skip_special_tokens:
                    labels = [l for l in labels if l not in special_tokens]
                return labels

            return _map_single_id(token_ids)

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            sequence_history: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            if input_ids is None and sequence_history is not None:
                input_ids = sequence_history.to(dtype=torch.long)
            if input_ids is None:
                raise ValueError("input_ids or sequence_history must be provided")

            B, T = input_ids.shape
            if T > self.config.max_position_embeddings:
                raise ValueError(f"Sequence length {T} exceeds max_position_embeddings {self.config.max_position_embeddings}")

            positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            x = self.tok_emb(input_ids) + self.pos_emb(positions)
            x = self.drop(x)

            for block in self.h:
                x = block(x, attention_mask=attention_mask)

            x = self.ln_f(x)
            logits = self.lm_head(x)

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            return {"logits": logits, "loss": loss}

        def save_pretrained(self, save_directory: Union[str, Path], push_to_hub: bool = False, repo_id: Optional[str] = None, token: Optional[str] = None) -> Path:
            out_dir = Path(save_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            Pick3HFHub.export_hf_assets(out_dir, self.config, repo_id=repo_id)

            # Save PyTorch weights
            torch.save(self.state_dict(), out_dir / "pytorch_model.bin")

            if push_to_hub and repo_id:
                Pick3HFHub.push_to_hub(out_dir, repo_id=repo_id, token=token)
            return out_dir

        @classmethod
        def from_pretrained(cls, model_id_or_path: Union[str, Path], token: Optional[str] = None, revision: str = "main") -> "Pick3ForCausalLM":
            model_dir = Pick3HFHub.resolve_model_dir(model_id_or_path, revision=revision, token=token)
            config_file = model_dir / "config.json"
            config = Pick3HFConfig.from_json_file(config_file)
            model = cls(config)

            bin_file = model_dir / "pytorch_model.bin"
            safetensors_file = model_dir / "model.safetensors"

            if bin_file.exists():
                state_dict = torch.load(bin_file, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
            elif safetensors_file.exists():
                try:
                    import safetensors.torch
                    state_dict = safetensors.torch.load_file(str(safetensors_file))
                    # Map any GPT-2 naming if needed
                    pt_dict = {}
                    for k, v in state_dict.items():
                        if k == "wte.weight":
                            pt_dict["tok_emb.weight"] = v
                            pt_dict["lm_head.weight"] = v
                        elif k == "wpe.weight":
                            pt_dict["pos_emb.weight"] = v
                        elif k.startswith("h."):
                            pt_dict[k] = v
                            pt_dict[k.replace("h.", "blocks.")] = v
                        else:
                            pt_dict[k] = v
                    model.load_state_dict(pt_dict, strict=False)
                except Exception as e:
                    print(f"[Pick3ForCausalLM] safetensors load notice: {e}")

            return model

        def from_mlx_state_dict(self, mlx_weights: Dict[str, Any]) -> None:
            """Transfers weights directly from an MLX Pick3LLM model to PyTorch."""
            import numpy as np
            pt_state = {}
            for k, v in mlx_weights.items():
                arr = np.array(v)
                if k == "wte.weight":
                    pt_state["tok_emb.weight"] = torch.from_numpy(arr)
                    pt_state["lm_head.weight"] = torch.from_numpy(arr)
                elif k == "wpe.weight":
                    pt_state["pos_emb.weight"] = torch.from_numpy(arr)
                elif k.startswith("h."):
                    pt_state[k] = torch.from_numpy(arr)
                    pt_state[k.replace("h.", "blocks.")] = torch.from_numpy(arr)
                else:
                    pt_state[k] = torch.from_numpy(arr)
            self.load_state_dict(pt_state, strict=False)

        def generate_response_text(
            self,
            prompt_tokens: Optional[Sequence[int]] = None,
            history_draws: Optional[Sequence[Dict[str, Any]]] = None,
            draw_type: str = "Night",
            temperature: float = 0.8,
            include_narrative: bool = False,
        ) -> str:
            """
            Generates next Pick 3 draw digits and decodes into localized natural language text.
            e.g., 'The next winning Pick 3 numbers are 1 2 3.'
            """
            from idaho.pick3.pick3_llm_ticket_model import Pick3Tokenizer
            tokenizer = Pick3Tokenizer()
            if prompt_tokens is None:
                if history_draws is None:
                    raise ValueError("Either prompt_tokens or history_draws must be provided")
                prompt_tokens = tokenizer.encode_history_prompt(history_draws, target_draw_type=draw_type)

            curr_input = list(prompt_tokens)
            gen_tokens = []
            self.eval()
            with torch.no_grad():
                for _ in range(3):
                    inp = torch.tensor([curr_input], dtype=torch.long)
                    out = self(inp)
                    step_logits = out["logits"][0, -1, :10]
                    if temperature <= 0.01:
                        next_tok = int(torch.argmax(step_logits).item())
                    else:
                        probs = F.softmax(step_logits / max(temperature, 1e-4), dim=-1)
                        next_tok = int(torch.multinomial(probs, num_samples=1).item())
                    gen_tokens.append(next_tok)
                    curr_input.append(next_tok)

            return tokenizer.decode_response_text(
                gen_tokens,
                draw_type=draw_type,
                include_narrative=include_narrative,
            )

        def generate_localized_prediction(
            self,
            prompt_tokens: Optional[Sequence[int]] = None,
            history_draws: Optional[Sequence[Dict[str, Any]]] = None,
            draw_type: str = "Night",
            temperature: float = 0.8,
        ) -> Dict[str, str]:
            """Generates localized prediction dictionary with primary, alternatives, and permutations."""
            from idaho.pick3.pick3_llm_ticket_model import Pick3Tokenizer
            tokenizer = Pick3Tokenizer()
            if prompt_tokens is None:
                if history_draws is None:
                    raise ValueError("Either prompt_tokens or history_draws must be provided")
                prompt_tokens = tokenizer.encode_history_prompt(history_draws, target_draw_type=draw_type)

            curr_input = list(prompt_tokens)
            gen_tokens = []
            self.eval()
            with torch.no_grad():
                for _ in range(3):
                    inp = torch.tensor([curr_input], dtype=torch.long)
                    out = self(inp)
                    step_logits = out["logits"][0, -1, :10]
                    if temperature <= 0.01:
                        next_tok = int(torch.argmax(step_logits).item())
                    else:
                        probs = F.softmax(step_logits / max(temperature, 1e-4), dim=-1)
                        next_tok = int(torch.multinomial(probs, num_samples=1).item())
                    gen_tokens.append(next_tok)
                    curr_input.append(next_tok)

            return tokenizer.decode_response(
                gen_tokens,
                draw_type=draw_type,
            )

else:
    # Minimal stub when PyTorch is not available
    class Pick3ForCausalLM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed. Install torch to use Pick3ForCausalLM.")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hugging Face Model & Hub Management Tool for Idaho Pick 3 Transformer"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Hugging Face repository ID or local model path to resolve/inspect",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/pick3_llm_model",
        help="Directory to export Hugging Face assets",
    )
    parser.add_argument(
        "--export-assets",
        action="store_true",
        help="Export all Hugging Face metadata assets to --save-dir",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push exported assets to Hugging Face Hub",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make Hub repository private",
    )

    args = parser.parse_args()

    if args.hf_repo:
        resolved = Pick3HFHub.resolve_model_dir(args.hf_repo, token=args.token)
        print(f"Resolved model directory: {resolved}")

    if args.export_assets:
        cfg = Pick3HFConfig()
        out_dir = Pick3HFHub.export_hf_assets(args.save_dir, config=cfg, repo_id=args.hf_repo)
        print(f"Exported Hugging Face assets to: {out_dir}")

    if args.push_to_hub and args.hf_repo:
        url = Pick3HFHub.push_to_hub(
            save_directory=args.save_dir,
            repo_id=args.hf_repo,
            token=args.token,
            private=args.private,
        )
        print(f"Pushed to Hub: {url}")


if __name__ == "__main__":
    main()
