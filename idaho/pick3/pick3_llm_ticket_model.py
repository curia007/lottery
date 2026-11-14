#!/usr/bin/env python3
"""
pick3_llm_ticket_model.py

Create an LLM from a remote Lotto base model (or initialize/download a remote Lotto LLM),
generate and persist a local model, train/fine-tune it on Pick 3 lottery draw history,
and produce the next Pick 3 winning ticket candidates.

Features:
- Download / instantiate a remote Lotto Transformer LLM specification or weights.
- Generate and manage a local LLM with Causal Self-Attention, Multi-Head Attention,
  RMSNorm/LayerNorm, Positional Embeddings, and Autoregressive Head.
- Fine-tune/train the local LLM on Idaho Pick 3 draw sequences.
- Generate ranked winning ticket predictions using autoregressive token sequence probabilities
  and historical pattern scoring (frequency, overdue, pair dynamics).
- Supports Day, Night, and Both draw modes.
- Supports ticket types: exact (Straight), any (Box), 6-way, 3-way, straight_any.

Usage:
    python idaho/pick3/pick3_llm_ticket_model.py --draw Night --tickets 5
    python idaho/pick3/pick3_llm_ticket_model.py --remote-model remote://lottery-ai/base-pick3 --epochs 10
    python idaho/pick3/pick3_llm_ticket_model.py --csv idaho/pick3/data/idaho_pick3_history.csv --tickets 10 --ticket-type any
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1] if len(_SCRIPT_DIR.parents) >= 2 else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import pandas as pd

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as utils
except ImportError as exc:
    raise SystemExit(
        "MLX is not installed. Install it with:\n\n"
        "    pip install mlx\n\n"
        "MLX is optimized for Apple Silicon Macs."
    ) from exc

DrawType = Literal["Day", "Night", "Both", "both", "Combo", "combo", "Combined", "combined"]
TicketType = Literal["exact", "any", "straight_any", "6-way", "3-way"]

DEFAULT_CSV_PATH = "idaho/pick3/data/idaho_pick3_history.csv"
DEFAULT_MODEL_DIR = "models/pick3_llm_model"
DEFAULT_REMOTE_MODEL = "remote://lottery-ai/pick3-transformer-base"

# Word-level tokens for localized lottery text (expanded vocabulary)
# These enable the model to generate natural language responses like
# "The next winning Idaho Pick 3 lottery numbers are 1 2 3."
WORD_TOKENS = [
    "the", "next", "winning", "idaho", "pick", "lottery", "numbers",
    "are", "draw", "prediction", "day", "night", "is", "for",
    "of", "in", "and", "to", "a", "most", "probable", "combination",
    "combinations", "alternative", "first", "second", "third", "digit",
    "digits", "sequence", "sum", "odd", "even", "high", "low",
    "box", "play", "straight", "any", "order", "permutations",
    "confidence", "position", "positional", "breakdown", "coverage",
    "result", "results", "recommended", "based", "analysis",
    "Day", "Night", "Prediction",
    ".", ",", ":", "-",
]

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
# Add word tokens to id2label
for idx, word in enumerate(WORD_TOKENS):
    DEFAULT_ID2LABEL[17 + idx] = word

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
# Add word tokens to label2id
for idx, word in enumerate(WORD_TOKENS):
    DEFAULT_LABEL2ID[word] = 17 + idx


# ==============================================================================
# Tokenizer & Vocabulary
# ==============================================================================

class Pick3Tokenizer:
    """
    Tokenizer for lottery sequence modeling with expanded vocabulary.
    Tokens:
      0-9   : Digits '0' through '9'
      10    : '<PAD>' (Padding token)
      11    : '<BOS>' (Beginning of sequence)
      12    : '<EOS>' (End of sequence)
      13    : '<DAY>' (Day draw marker)
      14    : '<NIGHT>' (Night draw marker)
      15    : '<SEP>' (Separator between draws)
      16    : '<PRED>' (Target prediction prompt prefix)
      17+   : Word-level tokens for localized lottery text
              (e.g., 'the', 'next', 'winning', 'idaho', 'pick', 'lottery', 'numbers', etc.)
    """
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    DAY_TOKEN = "<DAY>"
    NIGHT_TOKEN = "<NIGHT>"
    SEP_TOKEN = "<SEP>"
    PRED_TOKEN = "<PRED>"

    def __init__(self):
        self.special_tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.DAY_TOKEN,
            self.NIGHT_TOKEN,
            self.SEP_TOKEN,
            self.PRED_TOKEN,
        ]
        self.digit_tokens = [str(i) for i in range(10)]
        self.word_tokens = list(WORD_TOKENS)
        self.vocab = self.digit_tokens + self.special_tokens + self.word_tokens
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.id2label = dict(DEFAULT_ID2LABEL)
        self.label2id = dict(DEFAULT_LABEL2ID)

    def decode_ids_to_labels(
        self,
        token_ids: Any,
        skip_special_tokens: bool = False,
    ) -> Union[str, List[Any]]:
        """
        Maps integer IDs back to label names using id2label.
        Supports single integers, lists, tuples, and NumPy / MLX arrays.
        """
        special_tokens = set(self.special_tokens) | {"[UNK]", "<UNK>"}

        def _map_single(tid: Any) -> str:
            try:
                int_id = int(tid)
            except (ValueError, TypeError):
                return str(tid)
            return self.id2label.get(int_id, str(tid))

        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        if isinstance(token_ids, (int, np.integer)):
            lbl = _map_single(token_ids)
            if skip_special_tokens and lbl in special_tokens:
                return ""
            return lbl

        if isinstance(token_ids, (list, tuple)):
            if not token_ids:
                return []
            if isinstance(token_ids[0], (list, tuple)):
                return [self.decode_ids_to_labels(seq, skip_special_tokens=skip_special_tokens) for seq in token_ids]
            labels = [_map_single(tid) for tid in token_ids]
            if skip_special_tokens:
                labels = [l for l in labels if l not in special_tokens]
            return labels

        return _map_single(token_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def day_id(self) -> int:
        return self.token_to_id[self.DAY_TOKEN]

    @property
    def night_id(self) -> int:
        return self.token_to_id[self.NIGHT_TOKEN]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[self.SEP_TOKEN]

    @property
    def pred_id(self) -> int:
        return self.token_to_id[self.PRED_TOKEN]

    def encode_draw(self, draw_type: str, d1: int, d2: int, d3: int) -> List[int]:
        draw_tag = self.day_id if draw_type.strip().lower().startswith("day") else self.night_id
        return [draw_tag, int(d1), int(d2), int(d3)]

    @staticmethod
    def tokens_per_draw() -> int:
        """Returns the number of tokens used to encode a single historical draw.
        
        Each draw encodes to: <DRAW_TAG> + d1 + d2 + d3 + <SEP> = 5 tokens.
        """
        return 5

    @staticmethod
    def prompt_overhead_tokens() -> int:
        """Returns the fixed overhead tokens for a prompt (BOS + PRED + target draw tag)."""
        return 3  # <BOS> + <PRED> + <DRAW_TAG>

    @staticmethod
    def target_tokens() -> int:
        """Returns the number of tokens for the target draw (3 digits + EOS)."""
        return 4  # d1 + d2 + d3 + <EOS>

    @staticmethod
    def max_draws_for_token_budget(max_seq_len: int, reserve_for_target: int = 4) -> int:
        """Calculates the maximum number of historical draws that fit within a token budget.
        
        Args:
            max_seq_len: Maximum sequence length in tokens (model's max_position_embeddings).
            reserve_for_target: Tokens reserved for the target draw (default 4: 3 digits + EOS).
        
        Returns:
            Maximum number of historical draws that fit within the budget.
        
        Formula:
            total_tokens = BOS + (draws * 6) + PRED + target_tag + target_digits + EOS
            total_tokens = 1 + (draws * 6) + 2 + 4 = 7 + (draws * 6)
            For input/target shift: total_tokens - 1 <= max_seq_len
            Therefore: draws <= (max_seq_len - 7) / 6
        """
        # Total sequence = BOS + (draws * 6) + PRED + target_tag + 3 digits + EOS
        # = 1 + (draws * 6) + 2 + 4 = 7 + (draws * 6)
        # For autoregressive shift, we need (total - 1) <= max_seq_len
        # So: draws <= (max_seq_len - 6) / 6
        overhead = 1 + 2  # BOS + PRED + target_tag
        tokens_per_draw = 5  # <DRAW_TAG> + d1 + d2 + d3 + <SEP>
        available = max_seq_len - overhead - reserve_for_target
        if available <= 0:
            return 0
        return available // tokens_per_draw

    def build_token_aware_window(
        self,
        records: Sequence[Dict[str, Any]],
        target_draw: Dict[str, Any],
        max_seq_len: int = 128,
        min_draws: int = 1,
    ) -> Tuple[List[int], int]:
        """Builds a token-aware sliding window prompt that maximizes historical context.
        
        Dynamically calculates how many historical draws fit within the token budget,
        ensuring the full sequence (prompt + target) stays within max_seq_len.
        
        Args:
            records: Chronologically sorted draw records (oldest first).
            target_draw: The target draw to predict.
            max_seq_len: Maximum sequence length in tokens.
            min_draws: Minimum number of historical draws to include.
        
        Returns:
            Tuple of (prompt_tokens, prompt_length) where prompt_length is the
            number of tokens in the prompt before the target digits.
        """
        max_draws = self.max_draws_for_token_budget(max_seq_len)
        # Use the available records, capped by max_draws and at least min_draws
        num_draws = max(min_draws, min(max_draws, len(records)))
        history = list(records)[-num_draws:] if num_draws < len(records) else list(records)

        # Encode the prompt
        prompt = self.encode_history_prompt(history, target_draw.get("draw", "Night"))
        return prompt, len(prompt)

    def count_tokens(self, text: str) -> int:
        """Counts the approximate number of tokens in a plain-text string.
        
        Uses the tokenizer's vocabulary to estimate token count by matching
        known tokens (digits, control tokens, word tokens) against the text.
        
        Args:
            text: Plain-text string to count tokens for.
        
        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        
        count = 0
        # Build a sorted list of all vocab tokens by length (longest first) for greedy matching
        all_tokens = sorted(self.vocab, key=len, reverse=True)
        remaining = text
        while remaining.strip():
            matched = False
            for token in all_tokens:
                # Check for control tokens (e.g., <DAY>, <BOS>) and their text equivalents
                if token in remaining:
                    count += 1
                    remaining = remaining.replace(token, " ", 1)
                    matched = True
                    break
            # Also check for text equivalents of control tokens
            if not matched:
                text_equivs = ["Day draw:", "Night draw:", "Day", "Night",
                               "Prediction:", "Prediction", "<BOS>", "<EOS>",
                               "<SEP>", "<PRED>", "<DAY>", "<NIGHT>"]
                for te in text_equivs:
                    if te in remaining:
                        count += 1
                        remaining = remaining.replace(te, " ", 1)
                        matched = True
                        break
            if not matched:
                # Count remaining characters as individual tokens (rough estimate)
                words = remaining.strip().split()
                count += len(words)
                break
        return count

    def truncate_to_fit(
        self,
        records: Sequence[Dict[str, Any]],
        target_draw: Dict[str, Any],
        max_seq_len: int = 128,
        reserve_for_target: int = 4,
        min_draws: int = 1,
    ) -> List[Dict[str, Any]]:
        """Drops the oldest draw records until the encoded prompt fits within max_seq_len.
        
        Iteratively removes the oldest (first) draw from the records list until
        the full sequence (prompt + target digits + EOS) fits safely below the
        server's maximum context length.
        
        Args:
            records: Chronologically sorted draw records (oldest first).
            target_draw: The target draw to predict (must have 'draw' key).
            max_seq_len: Maximum sequence length in tokens (server's limit).
            reserve_for_target: Tokens reserved for the target draw (default 4).
            min_draws: Minimum number of draws to keep even if budget is tight.
        
        Returns:
            A new list of draw records with the oldest entries removed so the
            encoded sequence fits within max_seq_len.
        """
        if not records:
            return []

        records_list = list(records)
        
        # Quick check: if the max_draws calculation already fits, return as-is
        max_allowed = self.max_draws_for_token_budget(max_seq_len, reserve_for_target)
        if len(records_list) <= max_allowed:
            return records_list

        # Iteratively drop oldest draws until the encoded sequence fits
        # Start with the maximum allowed and verify by encoding
        truncated = records_list[-max_allowed:]
        
        # Verify by actually encoding
        prompt = self.encode_history_prompt(truncated, target_draw.get("draw", "Night"))
        total_tokens = len(prompt) + reserve_for_target  # prompt + target digits + EOS
        
        # If still too long, keep dropping from the front (oldest)
        while total_tokens > max_seq_len and len(truncated) > min_draws:
            truncated = truncated[1:]  # Drop the oldest draw
            prompt = self.encode_history_prompt(truncated, target_draw.get("draw", "Night"))
            total_tokens = len(prompt) + reserve_for_target
        
        return truncated

    def truncate_history_summary(
        self,
        history_summary: str,
        max_tokens: int = 128,
        reserve_for_prompt: int = 20,
    ) -> str:
        """Truncates a plain-text history summary to fit within a token budget.
        
        Splits the summary by draw lines (each line representing a historical draw)
        and removes the oldest lines until the estimated token count fits.
        
        Args:
            history_summary: Plain-text summary of historical draws.
            max_tokens: Maximum total tokens allowed (server's context limit).
            reserve_for_prompt: Tokens reserved for the prediction prompt text.
        
        Returns:
            Truncated history summary string that fits within the token budget.
        """
        if not history_summary:
            return ""
        
        available = max_tokens - reserve_for_prompt
        if available <= 0:
            return ""
        
        # Split by lines, treating each line as a draw entry
        lines = history_summary.strip().split("\n")
        if not lines:
            return ""
        
        # Estimate tokens per line and build from the newest (end) backward
        truncated_lines = []
        current_tokens = 0
        
        # Process lines in reverse (newest first), then reverse back
        for line in reversed(lines):
            line_tokens = max(1, self.count_tokens(line))
            if current_tokens + line_tokens > available and truncated_lines:
                break
            truncated_lines.append(line)
            current_tokens += line_tokens
        
        # Reverse to restore chronological order (oldest first)
        truncated_lines.reverse()
        return "\n".join(truncated_lines)

    def encode_history_prompt(
        self,
        history_draws: Sequence[Dict[str, Any]],
        target_draw_type: str = "Night",
    ) -> List[int]:
        """Encodes historical context draws leading up to the target draw conditioning prompt."""
        tokens: List[int] = [self.bos_id]
        for d in history_draws:
            tokens.extend(self.encode_draw(d["draw"], d["num1"], d["num2"], d["num3"]))
            tokens.append(self.sep_id)

        target_tag = self.day_id if str(target_draw_type).strip().lower().startswith("day") else self.night_id
        tokens.extend([self.pred_id, target_tag])
        return tokens

    def _extract_digits_from_input(self, token_ids: Any) -> List[int]:
        """Extracts a flat list of integer digits 0-9 from lists, strings, tensors, or arrays."""
        if token_ids is None:
            return []
        if isinstance(token_ids, str):
            return [int(c) for c in token_ids if c.isdigit()]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, (int, np.integer)):
            if 0 <= int(token_ids) <= 9:
                return [int(token_ids)]
            return []

        digits: List[int] = []
        if isinstance(token_ids, (list, tuple)):
            for item in token_ids:
                if isinstance(item, (int, np.integer)) and 0 <= int(item) <= 9:
                    digits.append(int(item))
                elif isinstance(item, str) and item.isdigit() and len(item) == 1:
                    digits.append(int(item))
                elif isinstance(item, (list, tuple, np.ndarray)) or hasattr(item, "tolist"):
                    digits.extend(self._extract_digits_from_input(item))
                elif isinstance(item, str):
                    digits.extend([int(c) for c in item if c.isdigit()])
        return digits

    def decode_draw_digits(self, token_ids: Any) -> Optional[Tuple[int, int, int]]:
        """Extracts the first 3 consecutive numeric digit tokens from a sequence or string."""
        digits = self._extract_digits_from_input(token_ids)
        if len(digits) >= 3:
            return (digits[0], digits[1], digits[2])
        return None

    def decode_all_draw_digits(self, token_ids: Any) -> List[Tuple[int, int, int]]:
        """Extracts all non-overlapping 3-digit draw triplets from a sequence or string."""
        digits = self._extract_digits_from_input(token_ids)
        triplets: List[Tuple[int, int, int]] = []
        for i in range(0, len(digits) - 2, 3):
            triplets.append((digits[i], digits[i + 1], digits[i + 2]))
        return triplets

    def decode_response(
        self,
        token_ids: Any,
        alternative_tickets: Optional[Sequence[Any]] = None,
        prob_matrix: Optional[np.ndarray] = None,
        draw_type: Optional[str] = "Night",
    ) -> Dict[str, str]:
        """
        Decodes a token sequence or string into a rich localized natural language response dictionary.
        Automatically parses secondary 3-digit groups as alternative candidate tickets.
        """
        triplets = self.decode_all_draw_digits(token_ids)
        if not triplets:
            raw_text = self.decode(token_ids, skip_special_tokens=True) if isinstance(token_ids, (list, tuple)) else str(token_ids)
            return {
                "primary": f"Prediction sequence: {raw_text}",
                "primary_dash": f"Prediction sequence: {raw_text}",
                "alternatives": "No alternative candidates identified.",
                "positional_summary": f"Decoded tokens: {raw_text}",
                "box_permutations": "N/A",
                "profile": "N/A",
                "narrative": f"Prediction sequence: {raw_text}",
                "draw": str(draw_type).capitalize() if draw_type else "Upcoming",
            }
        primary_digits = triplets[0]
        if alternative_tickets is None and len(triplets) > 1:
            alternative_tickets = triplets[1:]

        return format_pick3_localized_response(
            predicted_digits=primary_digits,
            prob_matrix=prob_matrix,
            alternative_tickets=alternative_tickets,
            draw_type=draw_type,
        )

    def decode_response_text(
        self,
        token_ids: Any,
        alternative_tickets: Optional[Sequence[Any]] = None,
        prob_matrix: Optional[np.ndarray] = None,
        draw_type: Optional[str] = "Night",
        include_narrative: bool = False,
    ) -> str:
        """
        Decodes a token sequence or string directly into localized natural language prediction text.
        e.g., 'The next winning Pick 3 numbers are 1 2 3.'
        """
        resp = self.decode_response(
            token_ids=token_ids,
            alternative_tickets=alternative_tickets,
            prob_matrix=prob_matrix,
            draw_type=draw_type,
        )
        return resp["narrative"] if include_narrative else resp["primary"]

    def decode(
        self,
        token_ids: Any,
        skip_special_tokens: bool = False,
        localize: bool = False,
        alternative_tickets: Optional[Sequence[Any]] = None,
        prob_matrix: Optional[np.ndarray] = None,
        draw_type: Optional[str] = "Night",
        **kwargs,
    ) -> str:
        """Converts integer token IDs back to a readable string or localized response text."""
        if localize:
            return self.decode_response_text(
                token_ids=token_ids,
                alternative_tickets=alternative_tickets,
                prob_matrix=prob_matrix,
                draw_type=draw_type,
            )
        if isinstance(token_ids, str):
            return token_ids
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        result = []
        for tid in token_ids:
            tok = self.id_to_token.get(int(tid), "<UNK>")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            result.append(tok)
        return " ".join(result)

    def save_pretrained(self, save_directory: Union[str, Path]) -> Path:
        """Saves tokenizer configurations and fast tokenizer serialization compatible with Hugging Face."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        from idaho.pick3.pick3_hf_model import Pick3HFHub

        # Build added_tokens_decoder dynamically with all tokens
        added_tokens_decoder = {
            "10": {"content": "<PAD>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "11": {"content": "<BOS>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "12": {"content": "<EOS>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "13": {"content": "<DAY>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": False},
            "14": {"content": "<NIGHT>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": False},
            "15": {"content": "<SEP>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": False},
            "16": {"content": "<PRED>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": False},
        }
        # Add word tokens
        for idx, word in enumerate(self.word_tokens):
            token_id = 17 + idx
            added_tokens_decoder[str(token_id)] = {
                "content": word, "lstrip": False, "normalized": False,
                "rstrip": False, "single_word": True, "special": False
            }
        tok_cfg = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_type": "gpt2",
            "vocab_size": self.vocab_size,
            "bos_token": self.BOS_TOKEN,
            "eos_token": self.EOS_TOKEN,
            "pad_token": self.PAD_TOKEN,
            "unk_token": "[UNK]",
            "model_max_length": 128,
            "clean_up_tokenization_spaces": False,
            "chat_template": Pick3HFHub.DEFAULT_CHAT_TEMPLATE,
            "added_tokens_decoder": added_tokens_decoder,
            "special_tokens": [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, "[UNK]", "<UNK>"],
        }
        with (out_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
            json.dump(tok_cfg, f, indent=2)

        # Fast Tokenizer Serialization (tokenizer.json)
        from idaho.pick3.pick3_hf_model import Pick3HFHub
        tok_json = Pick3HFHub.get_fast_tokenizer_dict()
        with (out_dir / "tokenizer.json").open("w", encoding="utf-8") as f:
            json.dump(tok_json, f, indent=2)

        vocab_dict = dict(self.token_to_id)
        vocab_dict["[UNK]"] = 10
        vocab_dict["<UNK>"] = 10
        with (out_dir / "vocab.json").open("w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2)
        special_map = {
            "bos_token": self.BOS_TOKEN,
            "eos_token": self.EOS_TOKEN,
            "pad_token": self.PAD_TOKEN,
            "unk_token": "[UNK]",
            "additional_special_tokens": ["<UNK>"],
        }
        with (out_dir / "special_tokens_map.json").open("w", encoding="utf-8") as f:
            json.dump(special_map, f, indent=2)
        return out_dir

    @classmethod
    def from_pretrained(cls, load_directory_or_repo: Union[str, Path], token: Optional[str] = None) -> "Pick3Tokenizer":
        """Loads tokenizer from local directory or Hugging Face Hub."""
        from idaho.pick3.pick3_hf_model import Pick3HFHub
        model_dir = Pick3HFHub.resolve_model_dir(load_directory_or_repo, token=token)
        return cls()


# ==============================================================================
# Model Architecture (Transformer LLM)
# ==============================================================================

@dataclass
class Pick3LLMConfig:
    vocab_size: int = 75
    max_seq_len: int = 128
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 3
    mlp_ratio: int = 4
    dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5
    id2label: Dict[int, str] = field(default_factory=lambda: dict(DEFAULT_ID2LABEL))
    label2id: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LABEL2ID))

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

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["model_type"] = "gpt2"
        data["n_positions"] = self.max_seq_len
        data["n_ctx"] = self.max_seq_len
        data["n_embd"] = self.n_embd
        data["n_layer"] = self.n_layer
        data["n_head"] = self.n_head
        data["n_inner"] = self.n_embd * self.mlp_ratio
        data["activation_function"] = "gelu"
        data["layer_norm_epsilon"] = self.layer_norm_epsilon
        data["initializer_range"] = 0.02
        data["pad_token_id"] = 10
        data["bos_token_id"] = 11
        data["eos_token_id"] = 12
        data["id2label"] = {str(k): str(v) for k, v in self.id2label.items()}
        data["label2id"] = {str(k): int(v) for k, v in self.label2id.items()}
        data["architectures"] = ["GPT2LMHeadModel", "Pick3ForCausalLM"]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pick3LLMConfig":
        cfg = cls()
        if "vocab_size" in data:
            cfg.vocab_size = int(data["vocab_size"])
        if "max_seq_len" in data:
            cfg.max_seq_len = int(data["max_seq_len"])
        elif "max_position_embeddings" in data:
            cfg.max_seq_len = int(data["max_position_embeddings"])
        elif "n_positions" in data:
            cfg.max_seq_len = int(data["n_positions"])
        elif "n_ctx" in data:
            cfg.max_seq_len = int(data["n_ctx"])
        if "n_embd" in data:
            cfg.n_embd = int(data["n_embd"])
        elif "hidden_size" in data:
            cfg.n_embd = int(data["hidden_size"])
        if "n_head" in data:
            cfg.n_head = int(data["n_head"])
        elif "num_attention_heads" in data:
            cfg.n_head = int(data["num_attention_heads"])
        if "n_layer" in data:
            cfg.n_layer = int(data["n_layer"])
        elif "num_hidden_layers" in data:
            cfg.n_layer = int(data["num_hidden_layers"])
        if "mlp_ratio" in data:
            cfg.mlp_ratio = int(data["mlp_ratio"])
        elif "intermediate_size" in data and cfg.n_embd > 0:
            cfg.mlp_ratio = int(data["intermediate_size"]) // cfg.n_embd
        elif "n_inner" in data and cfg.n_embd > 0:
            cfg.mlp_ratio = int(data["n_inner"]) // cfg.n_embd
        if "dropout" in data:
            cfg.dropout = float(data["dropout"])
        if "layer_norm_epsilon" in data:
            cfg.layer_norm_epsilon = float(data["layer_norm_epsilon"])
        if "id2label" in data and isinstance(data["id2label"], dict):
            cfg.id2label = {int(k): str(v) for k, v in data["id2label"].items()}
        if "label2id" in data and isinstance(data["label2id"], dict):
            cfg.label2id = {str(k): int(v) for k, v in data["label2id"].items()}
        return cfg


class GPT2Attention(nn.Module):
    """Causal Self-Attention block following canonical GPT-2 architecture."""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        probs = mx.softmax(scores, axis=-1)
        out = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(out)


class GPT2MLP(nn.Module):
    """Feedforward network following canonical GPT-2 architecture."""

    def __init__(self, n_embd: int, mlp_ratio: int = 4):
        super().__init__()
        mlp_dim = n_embd * mlp_ratio
        self.c_fc = nn.Linear(n_embd, mlp_dim, bias=True)
        self.c_proj = nn.Linear(mlp_dim, n_embd, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.gelu(self.c_fc(x)))


class TransformerBlock(nn.Module):
    """Causal Transformer Decoder Block matching GPT-2 specification."""

    def __init__(self, n_embd: int, n_head: int, mlp_ratio: int = 4, eps: float = 1e-5):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=eps)
        self.attn = GPT2Attention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd, eps=eps)
        self.mlp = GPT2MLP(n_embd, mlp_ratio)

    @property
    def ln1(self):
        return self.ln_1

    @property
    def ln2(self):
        return self.ln_2

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Pick3LLM(nn.Module):
    """
    Autoregressive Causal Language Model for Pick 3 Lottery Sequence Generation.
    Follows canonical GPT-2 architecture for Apple MLX, mlx-lm, and oMLX compatibility.
    """

    def __init__(self, config: Pick3LLMConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_seq_len, config.n_embd)
        self.h = [
            TransformerBlock(
                config.n_embd,
                config.n_head,
                config.mlp_ratio,
                getattr(config, "layer_norm_epsilon", 1e-5),
            )
            for _ in range(config.n_layer)
        ]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=getattr(config, "layer_norm_epsilon", 1e-5))

    @property
    def blocks(self):
        """Backward compatibility alias for self.h."""
        return self.h

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
        Supports single integers, lists, tuples, NumPy arrays, and MLX arrays.
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

        if isinstance(token_ids, (int, np.integer)):
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

    def __call__(self, idx: mx.array) -> mx.array:
        B, T = idx.shape
        if T > self.config.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}")

        positions = mx.arange(0, T)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(positions)
        x = tok_emb + pos_emb

        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

        for block in self.h:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.wte.as_linear(x)
        return logits

    def sanitize_weights_for_load(self, raw_weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Sanitizes raw weights from checkpoints, handling legacy block names,
        fused attention projections, prefix stripping, Conv1D/Linear transpose,
        and vocabulary size mismatches (e.g., old checkpoints with vocab_size=17
        loaded into models with expanded vocab_size=75).
        """
        cleaned = {}
        for k, v in raw_weights.items():
            if k.startswith("model."):
                k = k[6:]
            elif k.startswith("transformer."):
                k = k[12:]
            if k == "lm_head.weight":
                continue
            k = k.replace("blocks.", "h.").replace(".ln1.", ".ln_1.").replace(".ln2.", ".ln_2.")
            k = k.replace(".mlp.layers.0.", ".mlp.c_fc.").replace(".mlp.layers.2.", ".mlp.c_proj.")
            cleaned[k] = v

        # Fuse legacy q, k, v if present
        layers = {
            int(k.split(".")[1])
            for k in cleaned.keys()
            if k.startswith("h.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit()
        }
        for i in sorted(layers):
            qk = f"h.{i}.attn.query_proj.weight"
            kk = f"h.{i}.attn.key_proj.weight"
            vk = f"h.{i}.attn.value_proj.weight"
            if qk in cleaned and kk in cleaned and vk in cleaned:
                qw = cleaned.pop(qk)
                kw = cleaned.pop(kk)
                vw = cleaned.pop(vk)
                cleaned[f"h.{i}.attn.c_attn.weight"] = mx.concatenate([qw, kw, vw], axis=0)
            outk = f"h.{i}.attn.out_proj.weight"
            if outk in cleaned:
                cleaned[f"h.{i}.attn.c_proj.weight"] = cleaned.pop(outk)

        # Transpose Conv1D weights to Linear weights if needed
        C = self.config.n_embd
        final = {}
        for k, v in cleaned.items():
            if k.endswith(".attn.c_attn.weight") and v.shape == (C, 3 * C):
                final[k] = v.transpose(1, 0)
            elif k.endswith(".attn.c_proj.weight") and v.shape == (C, C):
                final[k] = v.transpose(1, 0)
            elif k.endswith(".mlp.c_fc.weight") and v.shape == (C, 4 * C):
                final[k] = v.transpose(1, 0)
            elif k.endswith(".mlp.c_proj.weight") and v.shape == (4 * C, C):
                final[k] = v.transpose(1, 0)
            else:
                final[k] = v

        # Handle vocabulary size mismatch: resize wte.weight if saved vocab differs from config
        if "wte.weight" in final:
            saved_vocab, saved_embd = final["wte.weight"].shape
            expected_vocab = self.config.vocab_size
            if saved_vocab != expected_vocab:
                print(f"[Pick3LLM] Resizing embedding wte.weight from ({saved_vocab}, {saved_embd}) "
                      f"to ({expected_vocab}, {saved_embd}) to match config vocab_size.")
                extra = expected_vocab - saved_vocab
                padding = mx.zeros((extra, saved_embd), dtype=final["wte.weight"].dtype)
                final["wte.weight"] = mx.concatenate([final["wte.weight"], padding], axis=0)

        # Handle sequence length mismatch: resize wpe.weight if saved max_seq_len differs from config
        if "wpe.weight" in final:
            saved_seq, saved_embd = final["wpe.weight"].shape
            expected_seq = self.config.max_seq_len
            if saved_seq != expected_seq:
                print(f"[Pick3LLM] Resizing position embedding wpe.weight from ({saved_seq}, {saved_embd}) "
                      f"to ({expected_seq}, {saved_embd}) to match config max_seq_len.")
                extra = expected_seq - saved_seq
                padding = mx.zeros((extra, saved_embd), dtype=final["wpe.weight"].dtype)
                final["wpe.weight"] = mx.concatenate([final["wpe.weight"], padding], axis=0)

        return final

    def export_hf_weights(self) -> Dict[str, mx.array]:
        """
        Exports model parameters in standard Hugging Face GPT-2 Conv1D format
        for out-of-the-box compatibility with mlx-lm, oMLX, and Transformers.
        """
        params = utils.tree_flatten(self.parameters())
        hf_weights = {}
        C = self.config.n_embd
        for k, v in params:
            if k.endswith(".attn.c_attn.weight") and v.shape == (3 * C, C):
                hf_weights[k] = v.transpose(1, 0)
            elif k.endswith(".attn.c_proj.weight") and v.shape == (C, C):
                hf_weights[k] = v.transpose(1, 0)
            elif k.endswith(".mlp.c_fc.weight") and v.shape == (4 * C, C):
                hf_weights[k] = v.transpose(1, 0)
            elif k.endswith(".mlp.c_proj.weight") and v.shape == (C, 4 * C):
                hf_weights[k] = v.transpose(1, 0)
            else:
                hf_weights[k] = v
        return hf_weights

    def save_weights(self, file_path: Union[str, Path], *args, **kwargs) -> None:
        """Saves model weights in standard Hugging Face / MLX safetensors format."""
        p = str(file_path)
        weights = self.export_hf_weights()
        if p.endswith(".safetensors"):
            mx.save_safetensors(p, weights)
        elif p.endswith(".npz"):
            mx.savez(p, **weights)
        else:
            mx.save_safetensors(p + ".safetensors", weights)

    def load_weights(
        self,
        file_path: Union[str, Path, Dict[str, mx.array], List[Tuple[str, mx.array]]],
        *args,
        **kwargs,
    ) -> Any:
        """Loads and sanitizes weights from file, dictionary, or list."""
        if isinstance(file_path, (str, Path)):
            raw = mx.load(str(file_path))
        elif isinstance(file_path, list):
            raw = dict(file_path)
        elif isinstance(file_path, dict):
            raw = file_path
        else:
            raise TypeError(f"Unsupported file_path type: {type(file_path)}")

        sanitized = self.sanitize_weights_for_load(raw)
        return super().load_weights(list(sanitized.items()), *args, **kwargs)

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        private: bool = False,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves local model architecture configuration, weights, and Hugging Face assets."""
        from idaho.pick3.pick3_hf_model import Pick3HFConfig, Pick3HFHub

        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = out_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save weights using MLX native weight serialization (.safetensors / .npz)
        weights_path = out_dir / "model.safetensors"
        self.save_weights(str(weights_path))

        # Export Hugging Face metadata & tokenizer assets
        hf_config = Pick3HFConfig.from_dict(self.config.to_dict())
        Pick3HFHub.export_hf_assets(out_dir, hf_config, metrics=metrics, repo_id=repo_id)

        if push_to_hub and repo_id:
            Pick3HFHub.push_to_hub(out_dir, repo_id=repo_id, token=token, private=private)

        return out_dir

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
        save_directory: Optional[Union[str, Path]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Saves checkpoint artifacts and pushes to Hugging Face Hub."""
        import tempfile
        from idaho.pick3.pick3_hf_model import Pick3HFHub

        target_dir = Path(save_directory) if save_directory else Path(tempfile.mkdtemp(prefix="hf_pick3_"))
        self.save_pretrained(target_dir, push_to_hub=False, repo_id=repo_id, metrics=metrics)
        return Pick3HFHub.push_to_hub(
            save_directory=target_dir,
            repo_id=repo_id,
            token=token,
            private=private,
            commit_message=commit_message,
        )

    @classmethod
    def from_pretrained(
        cls,
        load_directory_or_repo: Union[str, Path],
        token: Optional[str] = None,
        revision: str = "main",
        force_download: bool = False,
    ) -> "Pick3LLM":
        """Loads a model from a local saved directory or Hugging Face Hub repository."""
        from idaho.pick3.pick3_hf_model import Pick3HFHub

        in_dir = Pick3HFHub.resolve_model_dir(
            model_id_or_path=load_directory_or_repo,
            revision=revision,
            token=token,
            force_download=force_download,
        )
        config_path = in_dir / "config.json"
        safetensors_path = in_dir / "model.safetensors"
        npz_path = in_dir / "model.npz"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        config = Pick3LLMConfig.from_dict(cfg_dict)
        model = cls(config)

        if safetensors_path.exists():
            model.load_weights(str(safetensors_path))
            mx.eval(model.parameters())
        elif npz_path.exists():
            try:
                model.load_weights(str(npz_path))
                mx.eval(model.parameters())
            except Exception as e:
                print(f"[Warning] Could not load weights from {npz_path}: {e}")

        return model

    def generate_response_text(
        self,
        prompt_tokens: Optional[Sequence[int]] = None,
        history_draws: Optional[Sequence[Dict[str, Any]]] = None,
        draw_type: str = "Night",
        temperature: float = 0.8,
        include_narrative: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[int]] = None,
    ) -> str:
        """
        Generates and decodes the model response directly into localized natural language text.
        e.g. 'The next winning Pick 3 numbers are 1 2 3.'
        
        Args:
            prompt_tokens: Input token sequence.
            history_draws: Historical draw records for prompt construction.
            draw_type: Day or Night draw.
            temperature: Sampling temperature (lower = more deterministic).
            include_narrative: Whether to include full narrative response.
            frequency_penalty: Penalizes tokens based on their frequency (0.0-2.0).
            presence_penalty: Penalizes tokens based on whether they appear (0.0-2.0).
            stop: List of token IDs that stop generation when encountered.
        """
        tokenizer = Pick3Tokenizer()
        if prompt_tokens is None:
            if history_draws is None:
                raise ValueError("Either prompt_tokens or history_draws must be provided")
            prompt_tokens = tokenizer.encode_history_prompt(history_draws, target_draw_type=draw_type)

        stop_ids = set(stop) if stop else set()
        curr_input = list(prompt_tokens)
        gen_tokens: List[int] = []
        token_counts: Counter = Counter()

        for _ in range(10):  # Max generation length
            arr = mx.array([curr_input], dtype=mx.int32)
            step_logits = self(arr)[:, -1, :10]

            # Apply frequency and presence penalties
            if frequency_penalty > 0.0 or presence_penalty > 0.0:
                freq_penalties = mx.array([float(token_counts.get(int(t), 0)) for t in range(10)])
                pres_penalties = mx.array([1.0 if token_counts.get(int(t), 0) > 0 else 0.0 for t in range(10)])
                step_logits = step_logits - frequency_penalty * freq_penalties - presence_penalty * pres_penalties

            if temperature <= 0.01:
                next_tok = int(mx.argmax(step_logits, axis=-1).item())
            else:
                scaled = step_logits / max(temperature, 1e-4)
                probs = mx.softmax(scaled, axis=-1)
                next_tok = int(mx.random.categorical(scaled).item())

            gen_tokens.append(next_tok)
            token_counts[next_tok] += 1
            curr_input.append(next_tok)

            # Check stop condition
            if next_tok in stop_ids:
                break

            # Stop after 3 digits if no stop tokens configured
            if len(gen_tokens) == 3 and not stop_ids:
                break

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
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[int]] = None,
    ) -> Dict[str, str]:
        """Generates localized prediction dictionary with primary, alternatives, and permutations.
        
        Args:
            prompt_tokens: Input token sequence.
            history_draws: Historical draw records for prompt construction.
            draw_type: Day or Night draw.
            temperature: Sampling temperature (lower = more deterministic).
            frequency_penalty: Penalizes tokens based on their frequency (0.0-2.0).
            presence_penalty: Penalizes tokens based on whether they appear (0.0-2.0).
            stop: List of token IDs that stop generation when encountered.
        """
        tokenizer = Pick3Tokenizer()
        if prompt_tokens is None:
            if history_draws is None:
                raise ValueError("Either prompt_tokens or history_draws must be provided")
            prompt_tokens = tokenizer.encode_history_prompt(history_draws, target_draw_type=draw_type)

        stop_ids = set(stop) if stop else set()
        curr_input = list(prompt_tokens)
        gen_tokens: List[int] = []
        token_counts: Counter = Counter()

        for _ in range(10):  # Max generation length
            arr = mx.array([curr_input], dtype=mx.int32)
            step_logits = self(arr)[0, -1, :10]

            # Apply frequency and presence penalties
            if frequency_penalty > 0.0 or presence_penalty > 0.0:
                freq_penalties = mx.array([float(token_counts.get(int(t), 0)) for t in range(10)])
                pres_penalties = mx.array([1.0 if token_counts.get(int(t), 0) > 0 else 0.0 for t in range(10)])
                step_logits = step_logits - frequency_penalty * freq_penalties - presence_penalty * pres_penalties

            if temperature <= 0.01:
                next_tok = int(mx.argmax(step_logits, axis=-1).item())
            else:
                scaled = step_logits / max(temperature, 1e-4)
                next_tok = int(mx.random.categorical(scaled).item())

            gen_tokens.append(next_tok)
            token_counts[next_tok] += 1
            curr_input.append(next_tok)

            # Check stop condition
            if next_tok in stop_ids:
                break

            # Stop after 3 digits if no stop tokens configured
            if len(gen_tokens) == 3 and not stop_ids:
                break

        return tokenizer.decode_response(
            gen_tokens,
            draw_type=draw_type,
        )


# ==============================================================================
# Remote Model Loader & Local Model Generator
# ==============================================================================

class RemoteLottoModelLoader:
    """
    Handles fetching or instantiating a remote Lotto model specification and
    instantiating the local model instance.
    """

    @staticmethod
    def fetch_or_create_model(
        remote_model_uri: str,
        local_model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
        config: Optional[Pick3LLMConfig] = None,
        force_download: bool = False,
    ) -> Pick3LLM:
        """
        Creates an LLM from a remote Lotto model reference or URL.
        Generates the local model and caches weights locally.
        """
        local_path = Path(local_model_dir)
        local_config_file = local_path / "config.json"
        local_weights_exist = (local_path / "model.safetensors").exists() or (local_path / "model.npz").exists()

        # Check if local model already exists and we're not forcing refresh
        if local_config_file.exists() and local_weights_exist and not force_download:
            print(f"[RemoteModelLoader] Loading existing local model from '{local_path}'")
            return Pick3LLM.from_pretrained(local_path)

        print(f"[RemoteModelLoader] Resolving remote Lotto model: '{remote_model_uri}'")

        # Handle Hugging Face Hub models (e.g. hf://username/repo or username/repo)
        if (
            remote_model_uri.startswith("hf://")
            or remote_model_uri.startswith("huggingface://")
            or ("/" in remote_model_uri and not remote_model_uri.startswith("remote://") and not remote_model_uri.startswith("http"))
        ):
            print(f"[RemoteModelLoader] Fetching model from Hugging Face Hub: '{remote_model_uri}'...")
            try:
                hf_model = Pick3LLM.from_pretrained(remote_model_uri, force_download=force_download)
                hf_model.save_pretrained(local_path)
                return hf_model
            except Exception as e:
                print(f"[RemoteModelLoader] Hugging Face Hub resolution notice: ({e}). Initializing architecture.")

        # Handle HTTP / HTTPS remote models
        if remote_model_uri.startswith("http://") or remote_model_uri.startswith("https://"):
            try:
                print(f"[RemoteModelLoader] Fetching remote model config from {remote_model_uri}...")
                req = urllib.request.Request(
                    remote_model_uri,
                    headers={"User-Agent": "LottoLLM/1.0 (Pick3-Client)"}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    remote_data = json.loads(response.read().decode("utf-8"))
                    config = Pick3LLMConfig.from_dict(remote_data.get("config", remote_data))
                    print("[RemoteModelLoader] Remote config downloaded successfully.")
            except Exception as e:
                print(f"[RemoteModelLoader] Remote download notice: ({e}). Utilizing base remote architecture.")
                if config is None:
                    config = Pick3LLMConfig()
        else:
            # Remote URI schema e.g. "remote://lottery-ai/pick3-transformer-base"
            print(f"[RemoteModelLoader] Initializing base Lotto Transformer from remote spec: {remote_model_uri}")
            if config is None:
                config = Pick3LLMConfig(
                    vocab_size=75,
                    max_seq_len=128,
                    n_embd=64,
                    n_head=4,
                    n_layer=3,
                    mlp_ratio=4,
                )

        print(f"[RemoteModelLoader] Generating local Pick3LLM with {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dims...")
        model = Pick3LLM(config)
        mx.eval(model.parameters())

        # Save local generated model
        saved_dir = model.save_pretrained(local_path)
        print(f"[RemoteModelLoader] Local model generated and saved to '{saved_dir}'")
        return model


# ==============================================================================
# oMLX Client & Server Integration
# ==============================================================================

class OMLXPick3Client:
    """
    Client for interacting with local oMLX and mlx-lm inference servers.
    Formats chat and completion requests, handles raw token/digit stream outputs,
    and decodes them directly into localized natural language lottery predictions.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "pick3_llm_model",
        timeout: float = 15.0,
        api_key: Optional[str] = None,
        max_context_tokens: int = 128,
        max_history: Optional[int] = None,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
        stop_sequences: Optional[List[str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.tokenizer = Pick3Tokenizer()
        self.max_context_tokens = max_context_tokens
        self.max_history = max_history
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop_sequences = stop_sequences if stop_sequences is not None else ["<EOS>", "[EOS]"]

    def sliding_window_messages(
        self,
        messages: List[Dict[str, str]],
        max_history: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Apply a message-count sliding window, keeping system instructions intact.

        Preserves all messages with role 'system' and keeps only the last
        `max_history` non-system messages (e.g., the previous turn + the new
        user prompt). If `max_history` is None, returns the messages unchanged.

        Args:
            messages: Full conversation message list.
            max_history: Maximum number of non-system messages to keep.

        Returns:
            A new message list with system messages preserved and non-system
            messages truncated to the last `max_history` entries.
        """
        limit = max_history if max_history is not None else self.max_history
        if limit is None or limit <= 0:
            return list(messages)

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]

        if len(non_system_msgs) <= limit:
            return list(messages)

        # Keep only the last `limit` non-system messages
        non_system_msgs = non_system_msgs[-limit:]
        return system_msgs + non_system_msgs

    def reset_session(self) -> None:
        """Reset the session by clearing all accumulated state.
        
        Clears conversation history, resets penalty accumulators, and prepares
        a fresh state for new predictions. This helps avoid poisoned responses
        from previous conversation turns.
        """
        # Reset generation parameters to defaults
        self.frequency_penalty = 0.5
        self.presence_penalty = 0.5
        self.stop_sequences = ["<EOS>", "[EOS]"]

    def extract_delta_content(self, chunk: Any) -> str:
        """
        Safely extracts text delta from streaming chunk without NoneType TypeError.
        Coalesces None or missing values to empty string.
        """
        if not isinstance(chunk, dict):
            return ""
        choices = chunk.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            return ""
        choice = choices[0]
        delta = choice.get("delta", {})
        if isinstance(delta, dict):
            content = delta.get("content")
            return str(content) if content is not None else ""
        elif "text" in choice:
            content = choice.get("text")
            return str(content) if content is not None else ""
        return ""

    def parse_response(
        self,
        raw_output: Any,
        draw_type: str = "Night",
    ) -> Dict[str, str]:
        """
        Parses raw text or JSON response from oMLX into a localized prediction dictionary.
        Extracts primary winning numbers and multi-draw candidate combinations.
        """
        raw_text = ""
        if isinstance(raw_output, dict):
            # Handle OpenAI-compatible response format
            choices = raw_output.get("choices", [])
            if choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    if "message" in first_choice and isinstance(first_choice["message"], dict):
                        content = first_choice["message"].get("content")
                        raw_text = str(content) if content is not None else ""
                    elif "delta" in first_choice and isinstance(first_choice["delta"], dict):
                        content = first_choice["delta"].get("content")
                        raw_text = str(content) if content is not None else ""
                    elif "text" in first_choice:
                        content = first_choice.get("text")
                        raw_text = str(content) if content is not None else ""
                    else:
                        raw_text = str(first_choice)
                else:
                    raw_text = str(first_choice)
            else:
                raw_text = str(raw_output)
        elif raw_output is not None:
            raw_text = str(raw_output)

        return self.tokenizer.decode_response(raw_text, draw_type=draw_type)

    def parse_response_text(
        self,
        raw_output: Any,
        draw_type: str = "Night",
        include_narrative: bool = False,
    ) -> str:
        """
        Parses raw text or JSON response from oMLX directly into localized prediction text.
        e.g., 'The next winning Pick 3 numbers are 6 5 2.'
        """
        resp = self.parse_response(raw_output, draw_type=draw_type)
        return resp["narrative"] if include_narrative else resp["primary"]

    def create_chat_prompt(
        self,
        draw_type: str = "Night",
        history_summary: Optional[str] = None,
        history_records: Optional[Sequence[Dict[str, Any]]] = None,
        max_context_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_history: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Constructs an OpenAI-compatible messages list conditioned for Pick 3 prediction.

        Automatically truncates oversized history to fit within the server's maximum
        context length by dropping the oldest entries first.
        If `messages` is provided, applies a message-count sliding window to keep
        only the last `max_history` non-system messages while preserving system instructions.

        Args:
            draw_type: Day or Night draw.
            history_summary: Plain-text history summary (truncated if too long).
            history_records: Structured draw records (truncated if too long).
            max_context_tokens: Override the client's max_context_tokens limit.
            messages: Existing conversation messages list (applies sliding window).
            max_history: Max non-system messages to keep (sliding window).
        """
        budget = max_context_tokens if max_context_tokens is not None else self.max_context_tokens

        # If an existing messages list is provided, apply the sliding window
        if messages:
            return self.sliding_window_messages(messages, max_history)

        # Otherwise, build a new single-turn prompt
        # Truncate structured records if provided
        if history_records:
            target = {"draw": draw_type}
            truncated = self.tokenizer.truncate_to_fit(
                history_records, target, max_seq_len=budget
            )
            # Build summary from truncated records
            if truncated:
                lines = []
                for r in truncated:
                    dt = r.get("draw", "Night")
                    lines.append(f"{dt}: {r['num1']} {r['num2']} {r['num3']}")
                history_summary = "\n".join(lines)

        # Truncate plain-text summary if too long
        if history_summary:
            history_summary = self.tokenizer.truncate_history_summary(
                history_summary, max_tokens=budget
            )

        content = f"Predict the next Pick 3 {draw_type} draw numbers."
        if history_summary:
            content = f"{history_summary}\n{content}"
        return [
            {"role": "user", "content": content}
        ]

    def create_completion_prompt(self, draw_type: str = "Night") -> str:
        """Constructs raw token prompt string formatted with control tokens."""
        tag = "<DAY>" if str(draw_type).strip().lower().startswith("day") else "<NIGHT>"
        return f"<BOS> <PRED> {tag}"

    def stream_chat(
        self,
        draw_type: str = "Night",
        temperature: float = 0.8,
        max_tokens: int = 16,
        history_summary: Optional[str] = None,
        history_records: Optional[Sequence[Dict[str, Any]]] = None,
        max_context_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_history: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Any:
        """
        Streams chat completion delta tokens from oMLX / mlx-lm endpoint via Server-Sent Events (SSE).
        Safely handles None delta contents and yields text deltas in real time.
        Automatically truncates history to fit within the server's context limit.
        If `messages` is provided, applies a message-count sliding window.
        Supports frequency/presence penalties and custom stop sequences.
        """
        freq_pen = frequency_penalty if frequency_penalty is not None else self.frequency_penalty
        pres_pen = presence_penalty if presence_penalty is not None else self.presence_penalty
        stops = stop_sequences if stop_sequences is not None else self.stop_sequences
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": self.create_chat_prompt(
                draw_type=draw_type,
                history_summary=history_summary,
                history_records=history_records,
                max_context_tokens=max_context_tokens,
                messages=messages,
                max_history=max_history,
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stop": stops,
            "frequency_penalty": freq_pen,
            "presence_penalty": pres_pen,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str or line_str.startswith(":"):
                    continue
                if line_str == "data: [DONE]":
                    break
                if line_str.startswith("data: "):
                    data_json = line_str[6:].strip()
                    try:
                        chunk = json.loads(data_json)
                        delta_text = self.extract_delta_content(chunk)
                        if delta_text:
                            yield delta_text
                    except json.JSONDecodeError:
                        continue

    def stream_predict(
        self,
        draw_type: str = "Night",
        temperature: float = 0.8,
        max_tokens: int = 16,
        history_summary: Optional[str] = None,
        history_records: Optional[Sequence[Dict[str, Any]]] = None,
        max_context_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_history: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Consumes streaming tokens safely and returns both the stream tokens and final parsed localized prediction.
        Automatically truncates history to fit within the server's context limit.
        If `messages` is provided, applies a message-count sliding window.
        Supports frequency/presence penalties and custom stop sequences.
        """
        tokens: List[str] = []
        for token in self.stream_chat(
            draw_type=draw_type,
            temperature=temperature,
            max_tokens=max_tokens,
            history_summary=history_summary,
            history_records=history_records,
            max_context_tokens=max_context_tokens,
            messages=messages,
            max_history=max_history,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
        ):
            if token:
                tokens.append(token)
        full_text = "".join(tokens)
        parsed = self.parse_response(full_text, draw_type=draw_type)
        return tokens, parsed

    def predict(
        self,
        draw_type: str = "Night",
        temperature: float = 0.8,
        max_tokens: int = 16,
        use_chat: bool = True,
        history_summary: Optional[str] = None,
        history_records: Optional[Sequence[Dict[str, Any]]] = None,
        max_context_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_history: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Queries the running oMLX server and returns the parsed, localized prediction dictionary.
        Automatically truncates history to fit within the server's context limit.
        If `messages` is provided, applies a message-count sliding window.
        Supports frequency/presence penalties and custom stop sequences.
        """
        freq_pen = frequency_penalty if frequency_penalty is not None else self.frequency_penalty
        pres_pen = presence_penalty if presence_penalty is not None else self.presence_penalty
        stops = stop_sequences if stop_sequences is not None else self.stop_sequences
        endpoint = f"{self.base_url}/chat/completions" if use_chat else f"{self.base_url}/completions"
        if use_chat:
            payload = {
                "model": self.model_name,
                "messages": self.create_chat_prompt(
                    draw_type=draw_type,
                    history_summary=history_summary,
                    history_records=history_records,
                    max_context_tokens=max_context_tokens,
                    messages=messages,
                    max_history=max_history,
                ),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stops,
                "frequency_penalty": freq_pen,
                "presence_penalty": pres_pen,
            }
        else:
            payload = {
                "model": self.model_name,
                "prompt": self.create_completion_prompt(draw_type=draw_type),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stops,
                "frequency_penalty": freq_pen,
                "presence_penalty": pres_pen,
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return self.parse_response(data, draw_type=draw_type)


# ==============================================================================
# Data Loading & Preprocessing
# ==============================================================================

@dataclass
class RankedTicket:
    rank: int
    ticket: str
    score: float
    llm_prob: float
    freq_score: float
    overdue_score: float
    ticket_type: str
    draw: str


def normalize_draw(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("day"):
        return "Day"
    if text.startswith("night"):
        return "Night"
    if text in ("combo", "comb", "combined"):
        return "Combo"
    if text in ("both", "all"):
        return "Both"
    raise ValueError(f"Unknown draw type: {value!r}")


def load_pick3_history(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Loads and standardizes the Idaho Pick 3 history CSV."""
    path = Path(csv_path)
    if not path.exists():
        # Check relative to script dir or project root
        script_dir = Path(__file__).resolve().parent
        alt_path = script_dir / "data" / "idaho_pick3_history.csv"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Pick 3 CSV file not found at '{path}' or '{alt_path}'")

    df = pd.read_csv(path)
    required = {"Date", "Draw", "Num1", "Num2", "Num3"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Draw"] = df["Draw"].apply(normalize_draw)

    for col in ["Num1", "Num2", "Num3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Num1", "Num2", "Num3"])

    for col in ["Num1", "Num2", "Num3"]:
        df[col] = df[col].astype(int)
        if not df[col].between(0, 9).all():
            raise ValueError(f"Column {col} has values outside 0-9")

    # Sort chronologically (oldest draw first)
    df["Draw_Order"] = df["Draw"].apply(lambda d: 0 if d == "Day" else 1)
    df = df.sort_values(by=["Date", "Draw_Order"], ascending=[True, True]).reset_index(drop=True)
    return df


def build_training_sequences(
    df: pd.DataFrame,
    tokenizer: Pick3Tokenizer,
    history_window: int = 10,
    max_seq_len: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts historical draws into autoregressive token sequences.
    Each training example is a sliding window of past draws followed by <PRED> and target draw.
    """
    draw_records = []
    for _, row in df.iterrows():
        draw_records.append((
            row["Draw"],
            int(row["Num1"]),
            int(row["Num2"]),
            int(row["Num3"]),
        ))

    input_sequences = []
    target_sequences = []

    for i in range(history_window, len(draw_records)):
        history = draw_records[i - history_window : i]
        target_draw_type, t1, t2, t3 = draw_records[i]

        # Construct prompt tokens
        tokens = [tokenizer.bos_id]
        for d_type, d1, d2, d3 in history:
            tokens.extend(tokenizer.encode_draw(d_type, d1, d2, d3))
            tokens.append(tokenizer.sep_id)

        tokens.append(tokenizer.pred_id)
        target_draw_tag = tokenizer.day_id if target_draw_type == "Day" else tokenizer.night_id
        tokens.append(target_draw_tag)

        # Append target digits and EOS
        full_seq = tokens + [t1, t2, t3, tokenizer.eos_id]

        if len(full_seq) > max_seq_len:
            # Truncate oldest history keeping BOS
            excess = len(full_seq) - max_seq_len
            tokens_to_keep = [tokenizer.bos_id] + full_seq[1 + excess :]
            full_seq = tokens_to_keep

        # Standard autoregressive inputs and targets (shifted by 1)
        inp = full_seq[:-1]
        tgt = full_seq[1:]

        # Pad to max_seq_len - 1 if necessary
        seq_len = max_seq_len - 1
        if len(inp) < seq_len:
            pad_amount = seq_len - len(inp)
            inp = inp + [tokenizer.pad_id] * pad_amount
            tgt = tgt + [tokenizer.pad_id] * pad_amount

        input_sequences.append(inp[:seq_len])
        target_sequences.append(tgt[:seq_len])

    return np.array(input_sequences, dtype=np.int32), np.array(target_sequences, dtype=np.int32)


# ==============================================================================
# Model Training / Fine-tuning
# ==============================================================================

def train_pick3_llm(
    model: Pick3LLM,
    inputs: np.ndarray,
    targets: np.ndarray,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    verbose: bool = True,
) -> Pick3LLM:
    """
    Trains / fine-tunes the local Pick 3 LLM on draw history sequences.
    """
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    pad_id = Pick3Tokenizer().pad_id
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(m: Pick3LLM, x: mx.array, y: mx.array) -> mx.array:
        logits = m(x)  # (B, T, vocab_size)
        loss = nn.losses.cross_entropy(logits, y, reduction="none")
        mask = (y != pad_id).astype(mx.float32)
        masked_loss = mx.sum(loss * mask) / mx.maximum(mx.sum(mask), 1.0)
        return masked_loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    num_samples = len(inputs)
    indices = np.arange(num_samples)

    if verbose:
        print(f"\n[Training] Fine-tuning Pick3LLM over {epochs} epochs on {num_samples} draw sequences (batch_size={batch_size})...")

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        total_loss = 0.0
        batches = 0

        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            xb = mx.array(inputs[batch_idx])
            yb = mx.array(targets[batch_idx])

            loss, grads = loss_and_grad_fn(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += float(loss.item())
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        if verbose:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Causal LM Loss: {avg_loss:.4f} | Perplexity: {math.exp(min(avg_loss, 20.0)):.2f}")

    return model


# ==============================================================================
# Ticket Prediction & Ranking
# ==============================================================================

def compute_frequency_and_overdue_scores(df: pd.DataFrame) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Calculates historical digit frequency and overdue recency scores."""
    digits = df[["Num1", "Num2", "Num3"]].values.flatten()
    total_digits = len(digits)

    # Frequency
    counts = Counter(digits)
    freq_scores = {d: (counts[d] + 1) / (total_digits + 10) for d in range(10)}

    # Overdue
    overdue_scores = {}
    total_draws = len(df)
    for d in range(10):
        # find last appearance in reverse
        matches = np.where(digits[::-1] == d)[0]
        if len(matches) > 0:
            gap = matches[0] // 3
        else:
            gap = total_draws
        # Normalized overdue score
        overdue_scores[d] = min(gap / 30.0, 1.0)

    return freq_scores, overdue_scores


def calculate_ticket_probabilities(
    model: Pick3LLM,
    df: pd.DataFrame,
    target_draw: str,
    tokenizer: Pick3Tokenizer,
    history_window: int = 10,
) -> np.ndarray:
    """
    Computes exact joint probabilities P(d1, d2, d3 | prompt) for all 1,000 combinations (000 - 999).
    If target_draw is 'Combo' / 'Combined', averages the conditional joint distribution of Day and Night draws.
    """
    norm_target = normalize_draw(target_draw)
    if norm_target in ("Combo", "Both"):
        # Combine predictions by evaluating joint probabilities under both Day and Night draw conditions
        probs_day = calculate_ticket_probabilities(model, df, "Day", tokenizer, history_window)
        probs_night = calculate_ticket_probabilities(model, df, "Night", tokenizer, history_window)
        combined_probs = 0.5 * probs_day + 0.5 * probs_night
        comb_sum = np.sum(combined_probs)
        return combined_probs / comb_sum if comb_sum > 0 else combined_probs

    recent_draws = df.tail(history_window)
    tokens = [tokenizer.bos_id]
    for _, row in recent_draws.iterrows():
        tokens.extend(tokenizer.encode_draw(row["Draw"], row["Num1"], row["Num2"], row["Num3"]))
        tokens.append(tokenizer.sep_id)

    tokens.append(tokenizer.pred_id)
    target_draw_tag = tokenizer.day_id if norm_target == "Day" else tokenizer.night_id
    tokens.append(target_draw_tag)

    prompt_len = len(tokens)

    # We will evaluate 1,000 combinations
    all_combos = []
    for d1 in range(10):
        for d2 in range(10):
            for d3 in range(10):
                all_combos.append([d1, d2, d3])

    all_combos_arr = np.array(all_combos, dtype=np.int32)
    num_combos = len(all_combos_arr)

    # Build sequence matrix (1000, prompt_len + 3)
    seq_matrix = np.tile(tokens, (num_combos, 1))
    seq_matrix = np.concatenate([seq_matrix, all_combos_arr], axis=1)

    # Autoregressive forward pass in batches
    probs = np.zeros(num_combos, dtype=np.float64)
    batch_size = 250

    for i in range(0, num_combos, batch_size):
        batch_seqs = mx.array(seq_matrix[i : i + batch_size])
        logits = model(batch_seqs)  # (B, T, vocab_size)
        logits_np = np.array(logits)

        for b in range(len(batch_seqs)):
            # Logits for d1 at position (prompt_len - 1)
            logits_d1 = logits_np[b, prompt_len - 1, :10]
            exp_d1 = np.exp(logits_d1 - np.max(logits_d1))
            p_d1 = exp_d1 / np.sum(exp_d1)

            # Logits for d2 at position (prompt_len)
            logits_d2 = logits_np[b, prompt_len, :10]
            exp_d2 = np.exp(logits_d2 - np.max(logits_d2))
            p_d2 = exp_d2 / np.sum(exp_d2)

            # Logits for d3 at position (prompt_len + 1)
            logits_d3 = logits_np[b, prompt_len + 1, :10]
            exp_d3 = np.exp(logits_d3 - np.max(logits_d3))
            p_d3 = exp_d3 / np.sum(exp_d3)

            d1, d2, d3 = all_combos_arr[i + b]
            probs[i + b] = p_d1[d1] * p_d2[d2] * p_d3[d3]

    # Normalize joint distribution over the 1,000 tickets
    probs_sum = np.sum(probs)
    if probs_sum > 0:
        probs = probs / probs_sum

    return probs


def ticket_shape(d1: int, d2: int, d3: int) -> Tuple[int, ...]:
    counts = Counter([d1, d2, d3])
    return tuple(sorted(counts.values(), reverse=True))


def matches_ticket_type(d1: int, d2: int, d3: int, ticket_type: TicketType) -> bool:
    shape = ticket_shape(d1, d2, d3)
    if ticket_type in ("exact", "any", "straight_any"):
        return True
    if ticket_type == "6-way":
        return shape == (1, 1, 1)
    if ticket_type == "3-way":
        return shape == (2, 1)
    return False


def generate_pick3_winning_tickets(
    model: Pick3LLM,
    df: pd.DataFrame,
    draw_type: str = "Night",
    num_tickets: int = 5,
    ticket_type: TicketType = "exact",
    history_window: int = 10,
) -> List[RankedTicket]:
    """
    Produces the top ranked Pick 3 winning ticket candidates using LLM token probabilities
    combined with statistical lottery indicators.
    """
    tokenizer = Pick3Tokenizer()
    target_draw = normalize_draw(draw_type)
    probs = calculate_ticket_probabilities(model, df, target_draw, tokenizer, history_window)
    freq_scores, overdue_scores = compute_frequency_and_overdue_scores(df)

    candidates: List[RankedTicket] = []
    seen_box = set()

    for idx, (d1, d2, d3) in enumerate(
        (d1, d2, d3) for d1 in range(10) for d2 in range(10) for d3 in range(10)
    ):
        if not matches_ticket_type(d1, d2, d3, ticket_type):
            continue

        ticket_str = f"{d1}{d2}{d3}"

        # If any / box ticket type, optionally aggregate canonical box combinations
        if ticket_type in ("any", "6-way", "3-way"):
            box_key = "".join(sorted(ticket_str))
            if box_key in seen_box:
                continue
            seen_box.add(box_key)

        llm_p = float(probs[idx])
        f_score = float((freq_scores[d1] + freq_scores[d2] + freq_scores[d3]) / 3.0)
        o_score = float((overdue_scores[d1] + overdue_scores[d2] + overdue_scores[d3]) / 3.0)

        # Composite score weighting LLM confidence and statistical signals
        composite_score = (0.60 * llm_p * 1000.0) + (0.25 * f_score * 10.0) + (0.15 * o_score * 10.0)

        candidates.append(
            RankedTicket(
                rank=0,
                ticket=ticket_str,
                score=composite_score,
                llm_prob=llm_p,
                freq_score=f_score,
                overdue_score=o_score,
                ticket_type=ticket_type,
                draw=target_draw,
            )
        )

    # Sort candidates by composite score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    # Assign ranks and return top N
    results = []
    for rank, cand in enumerate(candidates[:num_tickets], start=1):
        cand.rank = rank
        results.append(cand)

    return results


def format_pick3_localized_response(
    predicted_digits: Sequence[int],
    prob_matrix: Optional[np.ndarray] = None,
    alternative_tickets: Optional[Sequence[Union[str, Sequence[int], RankedTicket]]] = None,
    draw_type: Optional[str] = "Night",
) -> Dict[str, str]:
    """
    Generates localized natural language text responses for Idaho Pick 3 predictions.

    Returns a dictionary of standardized response formats:
      - 'primary': e.g., "The next winning Pick 3 numbers are 1 2 3."
      - 'primary_dash': e.g., "The next winning Pick 3 numbers are 1-2-3."
      - 'alternatives': localized text listing other likely candidate draws.
      - 'positional_summary': breakdown of leading digits and probabilities per ball position.
      - 'box_permutations': explanation of Box/Any order permutations for the primary draw.
      - 'profile': parity, sum, and high/low statistical characteristics.
      - 'narrative': an integrated executive narrative summary of all predictions.
    """
    d1, d2, d3 = int(predicted_digits[0]), int(predicted_digits[1]), int(predicted_digits[2])
    draw_label = str(draw_type).capitalize() if draw_type else "Upcoming"

    # 1. Primary Natural Language Prediction
    primary_text = f"The next winning Pick 3 numbers are {d1} {d2} {d3}."
    primary_dash = f"The next winning Pick 3 numbers are {d1}-{d2}-{d3}."

    # 2. Alternative Likely Combinations
    alt_combos: List[str] = []
    if alternative_tickets:
        for t in alternative_tickets:
            if isinstance(t, RankedTicket):
                toks = [int(c) for c in t.ticket]
            elif isinstance(t, str):
                toks = [int(c) for c in t if c.isdigit()]
            else:
                toks = [int(c) for c in t]
            if len(toks) == 3 and (toks[0], toks[1], toks[2]) != (d1, d2, d3):
                formatted = f"{toks[0]} {toks[1]} {toks[2]}"
                if formatted not in alt_combos:
                    alt_combos.append(formatted)
            if len(alt_combos) >= 3:
                break
    elif prob_matrix is not None:
        p_mat = np.asarray(prob_matrix, dtype=np.float64)
        if p_mat.shape == (3, 10):
            # Normalize row probabilities if in 0..100 scale
            if np.max(p_mat) > 1.0:
                p_mat = p_mat / 100.0
            combos = []
            for c1 in range(10):
                for c2 in range(10):
                    for c3 in range(10):
                        if (c1, c2, c3) != (d1, d2, d3):
                            joint_p = p_mat[0, c1] * p_mat[1, c2] * p_mat[2, c3]
                            combos.append((joint_p, f"{c1} {c2} {c3}"))
            combos.sort(key=lambda x: x[0], reverse=True)
            alt_combos = [c[1] for c in combos[:3]]

    if alt_combos:
        if len(alt_combos) == 1:
            alternatives_text = f"The next most probable Pick 3 number combination is {alt_combos[0]}."
        elif len(alt_combos) == 2:
            alternatives_text = f"Alternative probable Pick 3 number combinations are {alt_combos[0]} and {alt_combos[1]}."
        else:
            alternatives_text = f"Alternative probable Pick 3 number combinations are {alt_combos[0]}, {alt_combos[1]}, and {alt_combos[2]}."
    else:
        alternatives_text = "No secondary alternative combinations meet the probability threshold."

    # 3. Positional Probability Breakdown
    if prob_matrix is not None:
        p_mat = np.asarray(prob_matrix, dtype=np.float64)
        if np.max(p_mat) <= 1.0:
            p_mat = p_mat * 100.0
        p1 = p_mat[0, d1]
        p2 = p_mat[1, d2]
        p3 = p_mat[2, d3]
        pos_summary = (
            f"Positional breakdown: First digit {d1} ({p1:.1f}% confidence), "
            f"second digit {d2} ({p2:.1f}% confidence), third digit {d3} ({p3:.1f}% confidence)."
        )
    else:
        pos_summary = f"Positional sequence: First digit is {d1}, second digit is {d2}, and third digit is {d3}."

    # 4. Box / Any-Order Permutations
    unique_perms = sorted({"".join(map(str, p)) for p in itertools.permutations([d1, d2, d3])})
    if len(unique_perms) == 6:
        box_type = "6-way Box"
        perms_str = ", ".join(f"{p[0]}-{p[1]}-{p[2]}" for p in unique_perms)
        box_text = f"For a 6-way Box (any order) play, numbers {d1} {d2} {d3} cover: {perms_str}."
    elif len(unique_perms) == 3:
        box_type = "3-way Box"
        perms_str = ", ".join(f"{p[0]}-{p[1]}-{p[2]}" for p in unique_perms)
        box_text = f"For a 3-way Box (any order) play with double digits, numbers {d1} {d2} {d3} cover: {perms_str}."
    else:
        box_type = "Straight Only (Triple)"
        box_text = f"All three digits are identical ({d1}-{d2}-{d3}), which plays strictly as a Straight wager (Straight Only)."

    # 5. Statistical Profile (Sum, Parity, High/Low)
    seq_sum = d1 + d2 + d3
    parities = ["Odd" if x % 2 == 1 else "Even" for x in (d1, d2, d3)]
    odd_count = sum(1 for p in parities if p == "Odd")
    high_low = ["High" if x >= 5 else "Low" for x in (d1, d2, d3)]
    high_count = sum(1 for hl in high_low if hl == "High")
    profile_text = (
        f"Sequence sum is {seq_sum} ({odd_count} Odd, {3 - odd_count} Even; "
        f"{high_count} High [5-9], {3 - high_count} Low [0-4])."
    )

    # 6. Integrated Executive Narrative
    narrative_text = (
        f"{primary_text} {pos_summary} {alternatives_text} {box_text} {profile_text}"
    )

    return {
        "primary": primary_text,
        "primary_dash": primary_dash,
        "alternatives": alternatives_text,
        "positional_summary": pos_summary,
        "box_permutations": box_text,
        "profile": profile_text,
        "narrative": narrative_text,
        "draw": draw_label,
    }


def print_tickets_table(tickets: List[RankedTicket], draw_label: str):
    """Nicely formats and prints ticket predictions to the console."""
    print(f"\n=======================================================================")
    print(f"       TOP PREDICTED PICK 3 WINNING TICKETS ({draw_label.upper()} DRAW)       ")
    print(f"=======================================================================")
    if tickets:
        d_top = [int(c) for c in tickets[0].ticket]
        localized = format_pick3_localized_response(
            predicted_digits=d_top,
            alternative_tickets=tickets[1:],
            draw_type=draw_label,
        )
        print(f"▶ LOCALIZED MODEL PREDICTION:")
        print(f"  {localized['primary']}")
        print(f"  {localized['alternatives']}")
        print(f"  {localized['box_permutations']}")
        print(f"-----------------------------------------------------------------------")
    print(f"{'Rank':<5} {'Ticket':<8} {'Composite Score':<17} {'LLM Prob':<12} {'Freq Score':<12} {'Type':<10}")
    print(f"-----------------------------------------------------------------------")
    for t in tickets:
        print(
            f"#{t.rank:<4} {t.ticket:<8} {t.score:<17.4f} {t.llm_prob:<12.5f} {t.freq_score:<12.4f} {t.ticket_type:<10}"
        )
    print(f"=======================================================================\n")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create an LLM from a remote Lotto model, train local model, and produce Pick 3 tickets."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f"Path to Idaho Pick 3 history CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--remote-model",
        type=str,
        default=DEFAULT_REMOTE_MODEL,
        help=f"Remote Lotto model URI or URL (default: {DEFAULT_REMOTE_MODEL})",
    )
    parser.add_argument(
        "--local-model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Local directory to save/load the local LLM model (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--draw",
        type=str,
        choices=["Day", "Night", "both", "Both", "combo", "Combo", "combined", "Combined"],
        default="combo",
        help="Target draw type: 'combo' (combine both night and day draws), 'Day', 'Night', or 'both' (default: combo)",
    )
    parser.add_argument(
        "--tickets",
        type=int,
        default=5,
        help="Number of winning ticket candidates to generate (default: 5)",
    )
    parser.add_argument(
        "--ticket-type",
        type=str,
        choices=["exact", "any", "straight_any", "6-way", "3-way"],
        default="exact",
        help="Ticket bet type (default: exact)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of training/fine-tuning epochs for local LLM (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Number of recent historical draws per LLM sequence window (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save generated predictions as CSV",
    )

    args = parser.parse_args()

    print("=======================================================================")
    print("           PICK 3 LOTTO LLM GENERATOR & PREDICTION PIPELINE           ")
    print("=======================================================================")

    # 1. Step 1: Create LLM from Remote Lotto model & generate local model
    print("\n[Step 1/3] Loading / Creating LLM from remote model...")
    local_model = RemoteLottoModelLoader.fetch_or_create_model(
        remote_model_uri=args.remote_model,
        local_model_dir=args.local_model_dir,
    )

    # 2. Step 2: Load historical data and train/fine-tune local LLM
    print(f"\n[Step 2/3] Loading history from '{args.csv}' and training local LLM...")
    df = load_pick3_history(args.csv)
    print(f"  Loaded {len(df)} total draws from history.")

    tokenizer = Pick3Tokenizer()
    inputs, targets = build_training_sequences(
        df=df,
        tokenizer=tokenizer,
        history_window=args.window,
        max_seq_len=local_model.config.max_seq_len,
    )

    trained_model = train_pick3_llm(
        model=local_model,
        inputs=inputs,
        targets=targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    # Persist updated local model weights
    trained_model.save_pretrained(args.local_model_dir)
    print(f"[Model Saved] Local LLM updated and saved to '{args.local_model_dir}'")

    # 3. Step 3: Produce the next Pick 3 winning tickets
    print(f"\n[Step 3/3] Generating next Pick 3 winning tickets (type={args.ticket_type})...")
    draws_to_run = ["Day", "Night"] if args.draw.lower() == "both" else [args.draw]

    all_results = []
    for draw_name in draws_to_run:
        tickets = generate_pick3_winning_tickets(
            model=trained_model,
            df=df,
            draw_type=draw_name,
            num_tickets=args.tickets,
            ticket_type=args.ticket_type,
            history_window=args.window,
        )
        print_tickets_table(tickets, draw_label=draw_name)
        all_results.extend(tickets)

    if args.output:
        out_df = pd.DataFrame([asdict(t) for t in all_results])
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved {len(all_results)} ticket predictions to '{out_path}'")


if __name__ == "__main__":
    main()
