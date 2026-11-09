#!/usr/bin/env python3
"""
train_pick_3.py

MLX training pipeline for Idaho Pick 3 lottery sequence modeling.
Ingests historical draw records from JSONL (or CSV), tokenizes prompt-target
sequences, trains an autoregressive Transformer LLM on Apple Silicon,
tracks validation loss and exact draw accuracy, and persists model checkpoints.

Usage:
    python idaho/pick3/train_pick_3.py
    python idaho/pick3/train_pick_3.py --epochs 10 --batch-size 32 --lr 1e-3
    python idaho/pick3/train_pick_3.py --jsonl idaho/pick3/data/idaho_pick3_last_picks.jsonl --predict --tickets 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError as exc:
    mx = None
    nn = None
    optim = None

from idaho.pick3.pick3_llm_ticket_model import (
    Pick3LLM,
    Pick3LLMConfig,
    Pick3Tokenizer,
    RemoteLottoModelLoader,
    calculate_ticket_probabilities,
    generate_pick3_winning_tickets,
)

DEFAULT_JSONL_PATH = "idaho/pick3/data/idaho_pick3_last_picks.jsonl"
DEFAULT_CSV_PATH = "idaho/pick3/data/idaho_pick3_history.csv"
DEFAULT_MODEL_DIR = "models/pick3_llm_model"
DEFAULT_REMOTE_MODEL = "remote://lottery-ai/pick3-transformer-base"


# ==============================================================================
# Draw Record Ingestion & Parsing
# ==============================================================================

@dataclass(frozen=True)
class Pick3DrawRecord:
    date: str
    draw: str
    num1: int
    num2: int
    num3: int

    @property
    def draw_normalized(self) -> str:
        return "Day" if self.draw.strip().lower().startswith("day") else "Night"

    @property
    def digits(self) -> Tuple[int, int, int]:
        return self.num1, self.num2, self.num3


def parse_draw_date(value: str) -> datetime:
    """Parses date string for chronological ordering."""
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {value}")


def load_draw_records(source_path: Union[str, Path]) -> List[Pick3DrawRecord]:
    """
    Ingests draw records from either a JSONL or CSV file, returning records
    sorted in strict chronological order (oldest first, Day before Night).
    """
    path = Path(source_path)
    if not path.exists():
        script_dir = Path(__file__).resolve().parent
        alt_path = script_dir / path.name
        if alt_path.exists():
            path = alt_path
        else:
            alt_data_path = script_dir / "data" / path.name
            if alt_data_path.exists():
                path = alt_data_path
            else:
                raise FileNotFoundError(f"Draw history file not found: {source_path}")

    records: List[Pick3DrawRecord] = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    item = json.loads(line_str)
                    records.append(
                        Pick3DrawRecord(
                            date=str(item["date"]).strip(),
                            draw=str(item.get("draw", "Day")).strip().capitalize(),
                            num1=int(item["num1"]),
                            num2=int(item["num2"]),
                            num3=int(item["num3"]),
                        )
                    )
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            field_map = {k.lower(): k for k in (reader.fieldnames or [])}
            d_col = field_map.get("date", "Date")
            dr_col = field_map.get("draw", "Draw")
            n1_col = field_map.get("num1", "Num1")
            n2_col = field_map.get("num2", "Num2")
            n3_col = field_map.get("num3", "Num3")

            for row in reader:
                records.append(
                    Pick3DrawRecord(
                        date=str(row[d_col]).strip(),
                        draw=str(row[dr_col]).strip().capitalize(),
                        num1=int(row[n1_col]),
                        num2=int(row[n2_col]),
                        num3=int(row[n3_col]),
                    )
                )

    # Sort chronological: oldest first, Day before Night
    records.sort(
        key=lambda r: (
            parse_draw_date(r.date),
            0 if r.draw_normalized == "Day" else 1,
        )
    )
    return records


# ==============================================================================
# Dataset Construction & Mask Alignment
# ==============================================================================

def build_training_and_validation_datasets(
    records: Sequence[Pick3DrawRecord],
    tokenizer: Optional[Pick3Tokenizer] = None,
    window_size: int = 10,
    max_seq_len: int = 64,
    val_split: float = 0.1,
    target_loss_only: bool = False,
) -> Tuple[Tuple[Any, Any, Any], Optional[Tuple[Any, Any, Any]]]:
    """
    Constructs MLX input (x), shifted target (y), and loss mask (mask) arrays
    from draw history sequences, maintaining chronological separation between
    train and validation splits to prevent lookahead bias.
    """
    tok = tokenizer or Pick3Tokenizer()
    total_draws = len(records)
    total_sequences = total_draws - window_size

    if total_sequences <= 0:
        raise ValueError(
            f"Insufficient records ({total_draws}) for history window {window_size}."
        )

    val_size = int(total_sequences * val_split) if (0.0 < val_split < 1.0) else 0
    train_size = total_sequences - val_size

    def create_split_arrays(start_idx: int, end_idx: int) -> Tuple[Any, Any, Any]:
        x_list: List[List[int]] = []
        y_list: List[List[int]] = []
        mask_list: List[List[float]] = []

        for i in range(start_idx, end_idx):
            history = records[i - window_size : i]
            target_draw = records[i]

            # 1. Encode prompt
            seq: List[int] = [tok.bos_id]
            for d in history:
                draw_tag = tok.day_id if d.draw_normalized == "Day" else tok.night_id
                seq.extend([draw_tag, d.num1, d.num2, d.num3, tok.sep_id])

            target_tag = tok.day_id if target_draw.draw_normalized == "Day" else tok.night_id
            seq.extend([tok.pred_id, target_tag])
            prompt_len = len(seq)

            # 2. Append target digits and <EOS>
            target_payload = [target_draw.num1, target_draw.num2, target_draw.num3, tok.eos_id]
            seq.extend(target_payload)

            # 3. Truncate if sequence exceeds max_seq_len
            if len(seq) > max_seq_len:
                excess = len(seq) - max_seq_len
                seq = [tok.bos_id] + seq[1 + excess :]
                prompt_len = max(1, prompt_len - excess)

            # 4. Inputs (0..T-2) and shifted targets (1..T-1)
            x_tokens = seq[:-1]
            y_tokens = seq[1:]

            # 5. Build mask
            if target_loss_only:
                # Mask 1.0 strictly for target digits and EOS
                mask = [0.0] * len(x_tokens)
                for t_pos in range(max(0, prompt_len - 1), len(x_tokens)):
                    mask[t_pos] = 1.0
            else:
                # Mask 1.0 for all active target tokens
                mask = [1.0] * len(x_tokens)

            # 6. Pad to max_seq_len - 1
            pad_len = (max_seq_len - 1) - len(x_tokens)
            if pad_len > 0:
                x_tokens.extend([tok.pad_id] * pad_len)
                y_tokens.extend([tok.pad_id] * pad_len)
                mask.extend([0.0] * pad_len)
            else:
                x_tokens = x_tokens[: max_seq_len - 1]
                y_tokens = y_tokens[: max_seq_len - 1]
                mask = mask[: max_seq_len - 1]

            x_list.append(x_tokens)
            y_list.append(y_tokens)
            mask_list.append(mask)

        if mx is not None:
            return (
                mx.array(x_list, dtype=mx.int32),
                mx.array(y_list, dtype=mx.int32),
                mx.array(mask_list, dtype=mx.float32),
            )
        return x_list, y_list, mask_list

    train_data = create_split_arrays(window_size, window_size + train_size)
    val_data = (
        create_split_arrays(window_size + train_size, total_draws)
        if val_size > 0
        else None
    )

    return train_data, val_data


# ==============================================================================
# Model Evaluation Helpers
# ==============================================================================

def evaluate_model_metrics(
    model: nn.Module,
    x: mx.array,
    y: mx.array,
    mask: mx.array,
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Evaluates cross-entropy loss, token prediction accuracy, and exact draw match
    rate across a dataset split using masked reductions.
    """
    if mx is None or nn is None:
        raise RuntimeError("MLX is required to evaluate model metrics.")

    total_samples = len(x)
    if total_samples == 0:
        return {"loss": 0.0, "token_accuracy": 0.0, "exact_match_rate": 0.0, "perplexity": 1.0}

    total_loss = 0.0
    total_token_acc = 0.0
    total_exact_acc = 0.0
    batches = 0

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        xb = x[start_idx:end_idx]
        yb = y[start_idx:end_idx]
        maskb = mask[start_idx:end_idx]

        logits = model(xb)
        loss_unreduced = nn.losses.cross_entropy(logits, yb, reduction="none")
        valid_tokens = mx.maximum(mx.sum(maskb), 1.0)
        batch_loss = mx.sum(loss_unreduced * maskb) / valid_tokens

        predictions = mx.argmax(logits, axis=-1)
        token_correct = (predictions == yb).astype(mx.float32)
        batch_token_acc = mx.sum(token_correct * maskb) / valid_tokens

        seq_mismatches = ((predictions != yb) * maskb).astype(mx.float32)
        exact_matches = (mx.sum(seq_mismatches, axis=1) == 0.0).astype(mx.float32)
        batch_exact_acc = mx.mean(exact_matches)

        mx.eval(batch_loss, batch_token_acc, batch_exact_acc)

        total_loss += float(batch_loss.item())
        total_token_acc += float(batch_token_acc.item())
        total_exact_acc += float(batch_exact_acc.item())
        batches += 1

    mean_loss = total_loss / max(batches, 1)
    mean_token_acc = total_token_acc / max(batches, 1)
    mean_exact_acc = total_exact_acc / max(batches, 1)
    perplexity = math.exp(min(mean_loss, 20.0))

    return {
        "loss": mean_loss,
        "token_accuracy": mean_token_acc,
        "exact_match_rate": mean_exact_acc,
        "perplexity": perplexity,
    }


# ==============================================================================
# MLX Training Pipeline
# ==============================================================================

@dataclass
class TrainingHistory:
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    train_perplexity: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_token_accuracy: List[float] = field(default_factory=list)
    val_exact_match_rate: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def train_pick_3_model(
    model: Pick3LLM,
    train_data: Tuple[mx.array, mx.array, mx.array],
    val_data: Optional[Tuple[mx.array, mx.array, mx.array]] = None,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    model_dir: Optional[Union[str, Path]] = None,
    save_best: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Pick3LLM, TrainingHistory]:
    """
    Executes the MLX training loop on Pick 3 sequences, optimizing causal
    cross-entropy loss with AdamW, computing epoch validation metrics,
    and persisting model checkpoints.
    """
    if mx is None or nn is None or optim is None:
        raise RuntimeError("MLX is required to execute train_pick_3_model.")

    random.seed(seed)
    mx.random.seed(seed)

    train_x, train_y, train_mask = train_data
    num_train_samples = len(train_x)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(m: Pick3LLM, x: mx.array, y: mx.array, mask: mx.array) -> mx.array:
        logits = m(x)
        loss_unreduced = nn.losses.cross_entropy(logits, y, reduction="none")
        valid_tokens = mx.maximum(mx.sum(mask), 1.0)
        return mx.sum(loss_unreduced * mask) / valid_tokens

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    history = TrainingHistory()

    best_val_loss = float("inf")
    best_weights_path: Optional[Path] = None
    save_dir = Path(model_dir) if model_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 65)
        print("          Idaho Pick 3 MLX Training Pipeline")
        print("=" * 65)
        print(f"  Training Sequences    : {num_train_samples:,}")
        if val_data is not None:
            print(f"  Validation Sequences  : {len(val_data[0]):,}")
        print(f"  Total Epochs          : {epochs}")
        print(f"  Batch Size            : {batch_size}")
        print(f"  Initial Learning Rate : {lr}")
        print(f"  Weight Decay          : {weight_decay}")
        if save_dir:
            print(f"  Checkpoint Directory  : {save_dir}")
        print("=" * 65 + "\n")

    indices = list(range(num_train_samples))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        random.shuffle(indices)
        total_epoch_loss = 0.0
        batches = 0

        for start_idx in range(0, num_train_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            xb = train_x[batch_indices]
            yb = train_y[batch_indices]
            maskb = train_mask[batch_indices]

            loss, grads = loss_and_grad_fn(model, xb, yb, maskb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_epoch_loss += float(loss.item())
            batches += 1

        epoch_time = time.time() - t0
        avg_train_loss = total_epoch_loss / max(batches, 1)
        train_ppl = math.exp(min(avg_train_loss, 20.0))

        history.epochs.append(epoch)
        history.train_loss.append(avg_train_loss)
        history.train_perplexity.append(train_ppl)
        history.epoch_times.append(epoch_time)

        # Validation evaluation
        if val_data is not None:
            val_x, val_y, val_mask = val_data
            val_metrics = evaluate_model_metrics(model, val_x, val_y, val_mask, batch_size=batch_size)
            v_loss = val_metrics["loss"]
            v_acc = val_metrics["token_accuracy"]
            v_exact = val_metrics["exact_match_rate"]

            history.val_loss.append(v_loss)
            history.val_token_accuracy.append(v_acc)
            history.val_exact_match_rate.append(v_exact)

            if verbose:
                print(
                    f"  Epoch {epoch:2d}/{epochs:2d} [{epoch_time:.2f}s] | "
                    f"Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f}) | "
                    f"Val Loss: {v_loss:.4f} | "
                    f"Val Token Acc: {v_acc * 100:.2f}% | "
                    f"Val Exact Draw: {v_exact * 100:.2f}%"
                )

            # Save best checkpoint
            if save_best and save_dir and v_loss < best_val_loss:
                best_val_loss = v_loss
                model.save_pretrained(save_dir)
                if verbose:
                    print(f"    -> Saved improved model checkpoint to: {save_dir}")
        else:
            if verbose:
                print(
                    f"  Epoch {epoch:2d}/{epochs:2d} [{epoch_time:.2f}s] | "
                    f"Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f})"
                )
            if save_dir:
                model.save_pretrained(save_dir)

    # Final checkpoint save if not saved during validation or if required
    if save_dir and (val_data is None or not save_best):
        model.save_pretrained(save_dir)
        if verbose:
            print(f"\n[Training] Final model weights saved to: {save_dir}")

    return model, history


# ==============================================================================
# Helper & CLI Entry Point
# ==============================================================================

def resolve_default_data_source() -> Path:
    """Finds best default draw history file (JSONL preferred, CSV fallback)."""
    p_jsonl_root = Path("idaho/pick3/data/idaho_pick3_last_picks.jsonl")
    if p_jsonl_root.exists():
        return p_jsonl_root
    p_jsonl_local = Path("data/idaho_pick3_last_picks.jsonl")
    if p_jsonl_local.exists():
        return p_jsonl_local
    p_csv_root = Path("idaho/pick3/data/idaho_pick3_history.csv")
    if p_csv_root.exists():
        return p_csv_root
    p_csv_local = Path("data/idaho_pick3_history.csv")
    if p_csv_local.exists():
        return p_csv_local
    script_dir = Path(__file__).resolve().parent
    return script_dir / "data" / "idaho_pick3_last_picks.jsonl"


def run_training_pipeline(
    data_path: Union[str, Path],
    model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
    remote_model_uri: str = DEFAULT_REMOTE_MODEL,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    window_size: int = 10,
    max_seq_len: int = 64,
    val_split: float = 0.1,
    target_loss_only: bool = False,
    n_embd: int = 64,
    n_head: int = 4,
    n_layer: int = 3,
    seed: int = 42,
    predict: bool = False,
    num_tickets: int = 5,
    draw_type: str = "Night",
    output_tickets_csv: Optional[Union[str, Path]] = None,
    history_json: Optional[Union[str, Path]] = None,
    push_to_hub: bool = False,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[Pick3LLM, TrainingHistory]:
    """
    Full pipeline function to load data, initialize model, execute training,
    save artifacts, and optionally generate predictions.
    """
    records = load_draw_records(data_path)
    if verbose:
        print(f"[Pipeline] Ingested {len(records):,} historical draw records from: {data_path}")

    tokenizer = Pick3Tokenizer()
    train_data, val_data = build_training_and_validation_datasets(
        records=records,
        tokenizer=tokenizer,
        window_size=window_size,
        max_seq_len=max_seq_len,
        val_split=val_split,
        target_loss_only=target_loss_only,
    )

    if mx is None:
        raise RuntimeError("MLX is not installed. Please run on Apple Silicon with MLX.")

    # Initialize or load base model
    model_path = Path(model_dir)
    config = Pick3LLMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=max_seq_len,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
    )

    if (model_path / "config.json").exists():
        if verbose:
            print(f"[Pipeline] Loading existing checkpoint from '{model_path}'...")
        model = Pick3LLM.from_pretrained(model_path)
    else:
        if verbose:
            print(f"[Pipeline] Initializing model from base specification '{remote_model_uri}'...")
        model = RemoteLottoModelLoader.fetch_or_create_model(
            remote_model_uri=remote_model_uri,
            local_model_dir=model_path,
            config=config,
        )

    # Train model
    trained_model, history = train_pick_3_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        model_dir=model_path,
        save_best=True,
        seed=seed,
        verbose=verbose,
    )

    # Push to Hugging Face Hub if requested
    if push_to_hub and hf_repo:
        val_metrics = None
        if history.val_loss:
            val_metrics = {
                "val_loss": history.val_loss[-1],
                "val_token_accuracy": history.val_token_accuracy[-1],
                "val_exact_match_rate": history.val_exact_match_rate[-1],
            }
        trained_model.push_to_hub(
            repo_id=hf_repo,
            token=hf_token,
            save_directory=model_path,
            metrics=val_metrics,
        )

    # Save history if requested
    if history_json:
        h_path = Path(history_json)
        h_path.parent.mkdir(parents=True, exist_ok=True)
        with h_path.open("w", encoding="utf-8") as f:
            json.dump(history.to_dict(), f, indent=2)
        if verbose:
            print(f"[Pipeline] Saved training history metrics to: {h_path}")

    # Generate predictions if requested
    if predict:
        import pandas as pd
        # Convert records to DataFrame for ticket generator
        df = pd.DataFrame([
            {
                "Date": r.date,
                "Draw": r.draw_normalized,
                "Num1": r.num1,
                "Num2": r.num2,
                "Num3": r.num3,
            }
            for r in records
        ])
        tickets = generate_pick3_winning_tickets(
            model=trained_model,
            df=df,
            draw_type=draw_type,
            num_tickets=num_tickets,
            ticket_type="exact",
            history_window=window_size,
        )
        if output_tickets_csv:
            out_p = Path(output_tickets_csv)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            ticket_rows = [
                {
                    "rank": t.rank,
                    "ticket": t.ticket,
                    "score": t.score,
                    "llm_prob": t.llm_prob,
                    "freq_score": t.freq_score,
                    "overdue_score": t.overdue_score,
                    "ticket_type": t.ticket_type,
                    "draw": t.draw,
                }
                for t in tickets
            ]
            pd.DataFrame(ticket_rows).to_csv(out_p, index=False)
            if verbose:
                print(f"[Pipeline] Exported {len(tickets)} predicted tickets to: {out_p}")
        if verbose:
            print("\n" + "=" * 55)
            print(f"  Generated {len(tickets)} Top Next-Draw Candidate Tickets ({draw_type})")
            print("=" * 55)
            if tickets:
                from idaho.pick3.pick3_llm_ticket_model import format_pick3_localized_response
                d_top = [int(c) for c in tickets[0].ticket]
                localized = format_pick3_localized_response(
                    predicted_digits=d_top,
                    alternative_tickets=tickets[1:],
                    draw_type=draw_type,
                )
                print(f"  ▶ Localized Prediction: {localized['primary']}")
                print(f"  ▶ Alternative Candidates: {localized['alternatives']}")
                print(f"  ▶ Box Permutations: {localized['box_permutations']}")
                print("-" * 55)
            for idx, t in enumerate(tickets, 1):
                d_str = "".join(map(str, t.ticket))
                print(f"  #{idx:02d} | Ticket: {d_str} (Digits: {t.ticket}) | Score: {t.score:.4e} | Mode: {t.draw}")
            print("=" * 55 + "\n")

    return trained_model, history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an MLX Transformer model on Idaho Pick 3 lottery sequences."
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help=f"Path to JSONL draw history. Default: {DEFAULT_JSONL_PATH}",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to CSV draw history (alternative to JSONL).",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory to save model weights and config. Default: {DEFAULT_MODEL_DIR}",
    )
    parser.add_argument(
        "--remote-model",
        default=DEFAULT_REMOTE_MODEL,
        help=f"Remote base model URI. Default: {DEFAULT_REMOTE_MODEL}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs. Default: 5",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size. Default: 32",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate. Default: 1e-3",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW. Default: 1e-4",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Historical draw context window size. Default: 10",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=64,
        help="Maximum sequence length. Default: 64",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (0.0 to disable). Default: 0.1",
    )
    parser.add_argument(
        "--target-loss-only",
        action="store_true",
        help="Calculate loss strictly on target prediction digits instead of full sequence.",
    )
    parser.add_argument(
        "--n-embd",
        type=int,
        default=64,
        help="Embedding dimension. Default: 64",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=4,
        help="Number of attention heads. Default: 4",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=3,
        help="Number of transformer layers. Default: 3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Generate top winning ticket predictions after training.",
    )
    parser.add_argument(
        "--tickets",
        type=int,
        default=5,
        help="Number of tickets to generate if --predict is set. Default: 5",
    )
    parser.add_argument(
        "--draw",
        default="Night",
        choices=["Day", "Night", "Both", "both", "Combo", "combo"],
        help="Target draw session for prediction. Default: Night",
    )
    parser.add_argument(
        "--output-tickets",
        default=None,
        help="Optional path to export predicted tickets as CSV.",
    )
    parser.add_argument(
        "--history-json",
        default=None,
        help="Optional path to export training history metrics as JSON.",
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="Hugging Face Hub repository ID (e.g. 'username/idaho-pick3-transformer').",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained model and Hugging Face assets to Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face authentication token (or set HF_TOKEN env var).",
    )

    args = parser.parse_args()

    data_path = Path(args.csv) if args.csv else (Path(args.jsonl) if args.jsonl else resolve_default_data_source())

    run_training_pipeline(
        data_path=data_path,
        model_dir=args.model_dir,
        remote_model_uri=args.remote_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        window_size=args.window_size,
        max_seq_len=args.max_seq_len,
        val_split=args.val_split,
        target_loss_only=args.target_loss_only,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        seed=args.seed,
        predict=args.predict,
        num_tickets=args.tickets,
        draw_type=args.draw,
        output_tickets_csv=args.output_tickets,
        history_json=args.history_json,
        push_to_hub=args.push_to_hub,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        verbose=True,
    )


if __name__ == "__main__":
    main()
