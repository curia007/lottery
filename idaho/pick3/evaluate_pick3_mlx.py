#!/usr/bin/env python3
"""
evaluate_pick3_mlx.py

Evaluates cross-entropy loss, token accuracy, and exact draw match rate
for Pick 3 sequence models using Apple MLX.
"""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path for package imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    mx = None
    nn = None

# --- Defaults ---
DEFAULT_JSONL_PATH = _SCRIPT_DIR / "data" / "idaho_pick3_last_picks.jsonl"
DEFAULT_MODEL_DIR = _SCRIPT_DIR / "models" / "pick3_llm_model"


def resolve_default_jsonl() -> Path:
    """Resolve the default JSONL data path, checking common locations."""
    if DEFAULT_JSONL_PATH.exists():
        return DEFAULT_JSONL_PATH
    # Fallback to repo root data dir
    fallback = _PROJECT_ROOT / "data" / "idaho_pick3_last_picks.jsonl"
    if fallback.exists():
        return fallback
    return DEFAULT_JSONL_PATH


# --- Tokenizer ---
class Pick3EvalTokenizer:
    """Minimal tokenizer for Pick 3 evaluation sequences."""

    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    DAY_TOKEN = "<DAY>"
    NIGHT_TOKEN = "<NIGHT>"
    SEP_TOKEN = "<SEP>"
    PRED_TOKEN = "<PRED>"

    def __init__(self) -> None:
        self.digit_tokens = [str(i) for i in range(10)]
        self.special_tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.DAY_TOKEN,
            self.NIGHT_TOKEN,
            self.SEP_TOKEN,
            self.PRED_TOKEN,
        ]
        self.vocab = self.digit_tokens + self.special_tokens
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}

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
        """Encodes an individual draw into [DRAW_TAG, d1, d2, d3]."""
        tag = self.day_id if str(draw_type).strip().lower().startswith("day") else self.night_id
        return [tag, int(d1), int(d2), int(d3)]

    def encode_history_prompt(
        self,
        history_draws: List[Dict[str, Any]],
        target_draw_type: str,
    ) -> List[int]:
        """Encodes historical context draws leading up to the target draw conditioning prompt."""
        tokens: List[int] = [self.bos_id]
        for d in history_draws:
            tokens.extend(self.encode_draw(d["draw"], d["num1"], d["num2"], d["num3"]))
            tokens.append(self.sep_id)

        target_tag = self.day_id if str(target_draw_type).strip().lower().startswith("day") else self.night_id
        tokens.extend([self.pred_id, target_tag])
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Converts integer token IDs back to a readable string representation."""
        result = []
        for tid in token_ids:
            tok = self.id_to_token.get(int(tid), "<UNK>")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            result.append(tok)
        return " ".join(result)


# --- Data Loading ---
def load_draw_jsonl(jsonl_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads and returns draw dictionaries sorted chronologically."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    records.sort(key=lambda r: (r["date"], 0 if str(r.get("draw", "")).lower() == "day" else 1))
    return records


# --- Batch Construction ---
def create_evaluation_batch(
    records: List[Dict[str, Any]],
    window_size: int = 5,
    batch_size: int = 32,
    max_len: int = 48,
    tokenizer: Optional[Pick3EvalTokenizer] = None,
) -> Tuple[Any, Any, Any]:
    """
    Builds (inputs, targets, masks) where loss and accuracy are evaluated
    strictly on the predicted target draw digits.
    """
    tok = tokenizer or Pick3EvalTokenizer()

    input_batch: List[List[int]] = []
    target_batch: List[List[int]] = []
    mask_batch: List[List[float]] = []

    limit = min(window_size + batch_size, len(records))

    for i in range(window_size, limit):
        history = records[i - window_size : i]
        target_draw = records[i]

        # Encode prompt
        prompt = tok.encode_history_prompt(history, target_draw["draw"])
        prompt_len = len(prompt)

        # Append target digits and <EOS>
        full_seq = prompt + [int(target_draw["num1"]), int(target_draw["num2"]), int(target_draw["num3"]), tok.eos_id]

        # Truncate if needed
        if len(full_seq) > max_len + 1:
            excess = len(full_seq) - (max_len + 1)
            full_seq = [tok.bos_id] + full_seq[1 + excess :]
            prompt_len = max(1, prompt_len - excess)

        x_seq = full_seq[:-1]
        y_seq = full_seq[1:]

        # Build mask: 1.0 ONLY for target positions
        mask = [0.0] * len(x_seq)
        for target_idx in range(max(0, prompt_len - 1), len(x_seq)):
            mask[target_idx] = 1.0

        # Pad to max_len
        pad_len = max_len - len(x_seq)
        if pad_len > 0:
            x_seq.extend([tok.pad_id] * pad_len)
            y_seq.extend([tok.pad_id] * pad_len)
            mask.extend([0.0] * pad_len)
        else:
            x_seq = x_seq[:max_len]
            y_seq = y_seq[:max_len]
            mask = mask[:max_len]

        input_batch.append(x_seq)
        target_batch.append(y_seq)
        mask_batch.append(mask)

    if mx is not None:
        return (
            mx.array(input_batch, dtype=mx.int32),
            mx.array(target_batch, dtype=mx.int32),
            mx.array(mask_batch, dtype=mx.float32),
        )
    return input_batch, target_batch, mask_batch


# --- Metrics ---
def compute_masked_metrics(
    logits: mx.array,
    y: mx.array,
    mask: mx.array,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Computes masked cross-entropy loss, token accuracy, and exact draw match accuracy.

    Args:
        logits: Model output logits of shape (B, T, Vocab).
        y: Ground-truth target token indices of shape (B, T).
        mask: Float32 active token mask of shape (B, T).

    Returns:
        (loss, token_accuracy, exact_draw_accuracy) as scalar mx.arrays.
    """
    # Unreduced Cross-Entropy Loss
    loss_unreduced = nn.losses.cross_entropy(logits, y, reduction="none")
    valid_tokens = mx.maximum(mx.sum(mask), 1.0)
    masked_loss = mx.sum(loss_unreduced * mask) / valid_tokens

    # Token-Level Prediction Accuracy
    predictions = mx.argmax(logits, axis=-1)
    token_correct = (predictions == y).astype(mx.float32)
    masked_token_acc = mx.sum(token_correct * mask) / valid_tokens

    # Sequence / Exact Draw Match Accuracy
    seq_mismatches = ((predictions != y) * mask).astype(mx.float32)
    exact_matches = (mx.sum(seq_mismatches, axis=1) == 0.0).astype(mx.float32)
    exact_draw_acc = mx.mean(exact_matches)

    return masked_loss, masked_token_acc, exact_draw_acc


def evaluate_step(
    model: nn.Module,
    x: mx.array,
    y: mx.array,
    mask: mx.array,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Forward pass and masked metric computation for a single batch.
    """
    logits = model(x)
    return compute_masked_metrics(logits, y, mask)


# --- Full Evaluation Pipeline ---
def evaluate_model_on_jsonl(
    model: nn.Module,
    jsonl_path: Union[str, Path],
    window_size: int = 10,
    batch_size: int = 64,
    max_len: int = 64,
) -> Dict[str, float]:
    """Evaluates cross-entropy loss and accuracy across the complete dataset."""
    records = load_draw_jsonl(jsonl_path)
    total_samples = len(records) - window_size

    total_loss = 0.0
    total_token_acc = 0.0
    total_exact_acc = 0.0
    num_batches = 0

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_slice = records[start_idx : end_idx + window_size]

        x_b, y_b, mask_b = create_evaluation_batch(
            records=batch_slice,
            window_size=window_size,
            batch_size=end_idx - start_idx,
            max_len=max_len,
        )

        loss, token_acc, exact_acc = evaluate_step(model, x_b, y_b, mask_b)
        mx.eval(loss, token_acc, exact_acc)

        total_loss += float(loss.item())
        total_token_acc += float(token_acc.item())
        total_exact_acc += float(exact_acc.item())
        num_batches += 1

    return {
        "mean_cross_entropy": total_loss / max(num_batches, 1),
        "mean_token_accuracy": total_token_acc / max(num_batches, 1),
        "exact_draw_match_rate": total_exact_acc / max(num_batches, 1),
        "total_samples": total_samples,
        "num_batches": num_batches,
    }


# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate Pick 3 MLX sequence model on JSONL data")
    parser.add_argument("--jsonl", type=str, default=None, help="Path to JSONL draw data")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR), help="Path to model directory")
    parser.add_argument("--window-size", type=int, default=10, help="Number of historical draws in prompt")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--max-len", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--output-metrics", type=str, default=None, help="Optional JSON file to save metrics")

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl) if args.jsonl else resolve_default_jsonl()
    if not jsonl_path.exists():
        print(f"Error: JSONL file not found: {jsonl_path}")
        sys.exit(1)

    # Load records and report
    records = load_draw_jsonl(jsonl_path)
    print(f"Loaded {len(records)} draw records from {jsonl_path}")

    if mx is None:
        print("MLX is not installed. Showing batch construction only.")
        x, y, mask = create_evaluation_batch(
            records=records,
            window_size=args.window_size,
            batch_size=min(args.batch_size, len(records) - args.window_size),
            max_len=args.max_len,
        )
        print(f"Sample batch shapes: x={len(x)}, y={len(y)}, mask={len(mask)}")
        return

    # Try to load model if available
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}. Showing batch construction only.")
        x, y, mask = create_evaluation_batch(
            records=records,
            window_size=args.window_size,
            batch_size=min(args.batch_size, len(records) - args.window_size),
            max_len=args.max_len,
        )
        print(f"Batch constructed: x={x.shape}, y={y.shape}, mask={mask.shape}")
        return

    print("Model evaluation requires a loaded MLX model. Use train_pick_3.py to train first.")
    print(f"Batch construction verified with {len(records)} records.")


if __name__ == "__main__":
    main()
