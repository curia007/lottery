#!/usr/bin/env python3
"""
evaluate_pick3_mlx.py

Batch evaluation and metric calculation for autoregressive Pick 3 lottery
sequence models using Apple MLX (mlx.core and mlx.nn).

Features:
- Ingests historical lottery draw records from JSONL (idaho_pick3_last_picks.jsonl).
- Constructs structured evaluation inputs (x), shifted targets (y), and binary active-target masks.
- Computes masked cross-entropy loss, token prediction accuracy, and exact draw match rate.
- Provides a clean CLI and reusable API for model evaluation across batches.

Usage:
    python idaho/pick3/evaluate_pick3_mlx.py
    python idaho/pick3/evaluate_pick3_mlx.py --jsonl idaho/pick3/data/idaho_pick3_last_picks.jsonl --batch-size 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    mx = None
    nn = None

DEFAULT_JSONL_PATH = "idaho/pick3/data/idaho_pick3_last_picks.jsonl"
DEFAULT_MODEL_DIR = "models/pick3_llm_model"


# ==============================================================================
# Tokenizer Specification
# ==============================================================================

class Pick3EvalTokenizer:
    """
    Tokenizer for lottery sequence evaluation matching Pick3LLM vocabulary.
    
    Tokens:
      0-9   : Digits '0' through '9'
      10    : '<PAD>' (Padding token)
      11    : '<BOS>' (Beginning of sequence)
      12    : '<EOS>' (End of sequence)
      13    : '<DAY>' (Day draw marker)
      14    : '<NIGHT>' (Night draw marker)
      15    : '<SEP>' (Separator between draws)
      16    : '<PRED>' (Target prediction prompt prefix)
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
        draw_tag = self.day_id if str(draw_type).strip().lower().startswith("day") else self.night_id
        return [draw_tag, int(d1), int(d2), int(d3)]


# ==============================================================================
# Data Ingestion from JSONL
# ==============================================================================

def load_draw_jsonl(jsonl_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Loads and returns draw dictionaries from a JSONL file, sorted chronologically.
    
    Args:
        jsonl_path: Path to the JSONL file.
        
    Returns:
        List of draw dictionaries with keys: date, draw, num1, num2, num3.
    """
    path = Path(jsonl_path)
    if not path.exists():
        script_dir = Path(__file__).resolve().parent
        alt_path = script_dir / "data" / "idaho_pick3_last_picks.jsonl"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"JSONL file not found at '{path}' or '{alt_path}'")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                rec = json.loads(line_str)
                records.append({
                    "date": rec["date"],
                    "draw": str(rec["draw"]).capitalize(),
                    "num1": int(rec["num1"]),
                    "num2": int(rec["num2"]),
                    "num3": int(rec["num3"]),
                })

    # Sort chronological (oldest draw first, Day before Night on same date)
    records.sort(
        key=lambda r: (
            r["date"],
            0 if r["draw"].lower() == "day" else 1,
        )
    )
    return records


# ==============================================================================
# Batch Construction & Mask Tensor Generation
# ==============================================================================

def create_evaluation_batch(
    records: List[Dict[str, Any]],
    window_size: int = 5,
    batch_size: int = 32,
    max_len: int = 48,
    tokenizer: Optional[Pick3EvalTokenizer] = None,
) -> Tuple[Any, Any, Any]:
    """
    Builds evaluation batch tensors (inputs, targets, masks).
    
    The mask tensor is 1.0 ONLY for target draw digit and EOS positions,
    and 0.0 for prompt context and padding tokens.
    
    Args:
        records: Chronologically sorted draw dictionaries.
        window_size: Number of past draws to use as prompt context.
        batch_size: Number of evaluation samples in the batch.
        max_len: Maximum token length of padded sequences.
        tokenizer: Optional tokenizer instance.
        
    Returns:
        (x, y, mask) as MLX arrays (or raw lists if MLX is not installed).
    """
    tok = tokenizer or Pick3EvalTokenizer()
    input_batch: List[List[int]] = []
    target_batch: List[List[int]] = []
    mask_batch: List[List[float]] = []

    limit = min(window_size + batch_size, len(records))

    for i in range(window_size, limit):
        history = records[i - window_size : i]
        target_draw = records[i]

        # 1. Encode prompt: [history draws] + <PRED> + [target draw type]
        seq: List[int] = [tok.bos_id]
        for d in history:
            seq.extend(tok.encode_draw(d["draw"], d["num1"], d["num2"], d["num3"]))
            seq.append(tok.sep_id)

        target_draw_tag = tok.day_id if target_draw["draw"].lower() == "day" else tok.night_id
        seq.extend([tok.pred_id, target_draw_tag])
        prompt_len = len(seq)

        # 2. Append target digits and <EOS>
        target_digits = [int(target_draw["num1"]), int(target_draw["num2"]), int(target_draw["num3"]), tok.eos_id]
        seq.extend(target_digits)

        # 3. Truncate if exceeding max_len + 1
        if len(seq) > max_len + 1:
            excess = len(seq) - (max_len + 1)
            seq = [tok.bos_id] + seq[1 + excess :]
            prompt_len = max(1, prompt_len - excess)

        # 4. Create input tokens (0..T-2) and target tokens (1..T-1)
        x_tokens = seq[:-1]
        y_tokens = seq[1:]

        # 5. Construct mask: 1.0 ONLY for target prediction positions
        # Prompt positions are 0.0, target positions are 1.0
        mask = [0.0] * len(x_tokens)
        for target_pos in range(max(0, prompt_len - 1), len(x_tokens)):
            mask[target_pos] = 1.0

        # 6. Pad to max_len
        pad_len = max_len - len(x_tokens)
        if pad_len > 0:
            x_tokens.extend([tok.pad_id] * pad_len)
            y_tokens.extend([tok.pad_id] * pad_len)
            mask.extend([0.0] * pad_len)
        else:
            x_tokens = x_tokens[:max_len]
            y_tokens = y_tokens[:max_len]
            mask = mask[:max_len]

        input_batch.append(x_tokens)
        target_batch.append(y_tokens)
        mask_batch.append(mask)

    if mx is not None:
        return (
            mx.array(input_batch, dtype=mx.int32),
            mx.array(target_batch, dtype=mx.int32),
            mx.array(mask_batch, dtype=mx.float32),
        )
    return input_batch, target_batch, mask_batch


# ==============================================================================
# Masked Loss and Accuracy Computation in MLX
# ==============================================================================

def compute_masked_metrics(
    logits: Any,
    y: Any,
    mask: Any,
) -> Tuple[Any, Any, Any]:
    """
    Computes masked cross-entropy loss, token accuracy, and exact draw match accuracy
    directly from logits, targets, and masks.
    
    Args:
        logits: Model output logits of shape (Batch, Seq_Len, Vocab_Size).
        y: Ground-truth target token indices of shape (Batch, Seq_Len).
        mask: Float32 active token mask of shape (Batch, Seq_Len).
        
    Returns:
        (loss, token_accuracy, exact_draw_accuracy) as scalar MLX arrays.
    """
    if mx is None or nn is None:
        raise RuntimeError("MLX is required to execute compute_masked_metrics.")

    # 1. Unreduced Cross-Entropy Loss
    loss_unreduced = nn.losses.cross_entropy(logits, y, reduction="none")  # (B, T)
    valid_tokens = mx.maximum(mx.sum(mask), 1.0)
    masked_loss = mx.sum(loss_unreduced * mask) / valid_tokens

    # 2. Token-Level Prediction Accuracy
    predictions = mx.argmax(logits, axis=-1)  # (B, T)
    token_correct = (predictions == y).astype(mx.float32)
    masked_token_acc = mx.sum(token_correct * mask) / valid_tokens

    # 3. Sequence / Exact Draw Match Accuracy (all target positions correct)
    seq_mismatches = ((predictions != y) * mask).astype(mx.float32)
    exact_matches = (mx.sum(seq_mismatches, axis=1) == 0.0).astype(mx.float32)
    exact_draw_acc = mx.mean(exact_matches)

    return masked_loss, masked_token_acc, exact_draw_acc


def evaluate_step(
    model: Any,
    x: Any,
    y: Any,
    mask: Any,
) -> Tuple[Any, Any, Any]:
    """
    Performs forward pass and computes evaluation metrics for a single batch.
    
    Args:
        model: MLX neural sequence model yielding logits (B, T, Vocab).
        x: Input token indices of shape (B, T).
        y: Ground-truth target token indices of shape (B, T).
        mask: Float32 active token mask of shape (B, T).
        
    Returns:
        (loss, token_accuracy, exact_draw_accuracy) as scalar MLX arrays.
    """
    logits = model(x)
    return compute_masked_metrics(logits, y, mask)


# ==============================================================================
# Full Evaluation Pipeline Across Batches
# ==============================================================================

def evaluate_model_on_jsonl(
    model: Any,
    jsonl_path: Union[str, Path],
    window_size: int = 10,
    batch_size: int = 64,
    max_len: int = 64,
    tokenizer: Optional[Pick3EvalTokenizer] = None,
) -> Dict[str, Any]:
    """
    Evaluates cross-entropy loss and accuracy across the complete JSONL dataset.
    
    Args:
        model: Pretrained MLX sequence model.
        jsonl_path: Path to the JSONL dataset.
        window_size: Historical draw window size.
        batch_size: Batch size for evaluation.
        max_len: Sequence length.
        tokenizer: Optional tokenizer instance.
        
    Returns:
        Dictionary containing aggregate evaluation metrics.
    """
    if mx is None:
        raise RuntimeError("MLX is required to execute evaluate_model_on_jsonl.")

    records = load_draw_jsonl(jsonl_path)
    total_samples = len(records) - window_size
    if total_samples <= 0:
        raise ValueError(
            f"Not enough draw records ({len(records)}) for window_size {window_size}."
        )

    tok = tokenizer or Pick3EvalTokenizer()
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
            tokenizer=tok,
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


def resolve_default_jsonl() -> Path:
    """Finds default JSONL file based on execution context."""
    p_root = Path("idaho/pick3/data/idaho_pick3_last_picks.jsonl")
    if p_root.exists():
        return p_root
    p_local = Path("data/idaho_pick3_last_picks.jsonl")
    if p_local.exists():
        return p_local
    script_dir = Path(__file__).resolve().parent
    return script_dir / "data" / "idaho_pick3_last_picks.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Pick 3 MLX model cross-entropy loss and accuracy on JSONL data."
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help=f"Path to JSONL draw history. Default: {DEFAULT_JSONL_PATH}",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing saved Pick3LLM weights. Default: {DEFAULT_MODEL_DIR}",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Number of historical draws per sequence prompt. Default: 10",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size. Default: 64",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=64,
        help="Max sequence token length. Default: 64",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save evaluation metrics as JSON.",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl) if args.jsonl else resolve_default_jsonl()
    print(f"[Evaluation] Loading Pick 3 draw records from: {jsonl_path}")

    records = load_draw_jsonl(jsonl_path)
    print(f"[Evaluation] Loaded {len(records):,} total historical draws.")

    if mx is None:
        print("[Evaluation] MLX is not installed. Testing batch and mask generation only.")
        x, y, mask = create_evaluation_batch(
            records=records,
            window_size=args.window_size,
            batch_size=min(args.batch_size, len(records) - args.window_size),
            max_len=args.max_len,
        )
        print(f"[Evaluation] Successfully constructed batch of {len(x)} sequences.")
        return

    # Attempt to load model from idaho.pick3.pick3_llm_ticket_model
    try:
        from idaho.pick3.pick3_llm_ticket_model import Pick3LLM, Pick3LLMConfig, RemoteLottoModelLoader
        model_path = Path(args.model_dir)
        if (model_path / "config.json").exists():
            print(f"[Evaluation] Loading model from '{model_path}'...")
            model = Pick3LLM.from_pretrained(model_path)
        else:
            print(f"[Evaluation] Model checkpoint not found at '{model_path}'. Initializing base model...")
            model = RemoteLottoModelLoader.fetch_or_create_model(
                remote_model_uri="remote://lottery-ai/pick3-transformer-base",
                local_model_dir=model_path,
                config=Pick3LLMConfig(max_seq_len=args.max_len),
            )
    except Exception as e:
        print(f"[Evaluation] Notice loading model: {e}")
        return

    metrics = evaluate_model_on_jsonl(
        model=model,
        jsonl_path=jsonl_path,
        window_size=args.window_size,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )

    print("\n" + "=" * 55)
    print("      Pick 3 MLX Model Evaluation Results")
    print("=" * 55)
    print(f"  Evaluated Sequences   : {metrics['total_samples']:,}")
    print(f"  Total Batches         : {metrics['num_batches']:,}")
    print(f"  Mean Cross-Entropy    : {metrics['mean_cross_entropy']:.4f}")
    print(f"  Mean Token Accuracy   : {metrics['mean_token_accuracy'] * 100:.2f}%")
    print(f"  Exact Draw Match Rate : {metrics['exact_draw_match_rate'] * 100:.2f}%")
    print("=" * 55 + "\n")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Evaluation] Saved evaluation metrics to: {out_p}")


if __name__ == "__main__":
    main()
