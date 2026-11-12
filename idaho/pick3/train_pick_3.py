#!/usr/bin/env python3
"""
train_pick_3.py

Sequence training pipeline for the Idaho Pick 3 Transformer using Apple MLX.
Ingests historical draw data (JSONL or CSV), constructs masked autoregressive
sequences, optimizes causal Transformer models, tracks validation metrics,
persists checkpoints, and optionally publishes to Hugging Face Hub.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import argparse
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
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
    import mlx.optimizers as optim
except ImportError:
    mx = None
    nn = None
    optim = None

# --- Defaults ---
DEFAULT_CSV_PATH = "idaho/pick3/data/idaho_pick3_history.csv"
DEFAULT_JSONL_PATH = _SCRIPT_DIR / "data" / "idaho_pick3_last_picks.jsonl"
DEFAULT_MODEL_DIR = _SCRIPT_DIR / "models" / "pick3_llm_model"
DEFAULT_REMOTE_MODEL = "remote://lottery-ai/pick3-transformer-base"


# --- Data Structures ---
@dataclass
class Pick3DrawRecord:
    """A single Pick 3 draw record."""
    date: str
    draw: str
    num1: int
    num2: int
    num3: int

    @property
    def draw_normalized(self) -> str:
        return "Day" if str(self.draw).strip().lower().startswith("day") else "Night"

    @property
    def digits(self) -> Tuple[int, int, int]:
        return (self.num1, self.num2, self.num3)


@dataclass
class TrainingHistory:
    """Training history metrics."""
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    train_perplexity: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_perplexity: List[float] = field(default_factory=list)
    val_token_accuracy: List[float] = field(default_factory=list)
    val_exact_match_rate: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    learning_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --- Utility Functions ---
def parse_draw_date(date_str: str) -> datetime:
    """Parse a date string (YYYY-MM-DD) into a datetime object."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def load_draw_records(data_path: Union[str, Path]) -> List[Pick3DrawRecord]:
    """
    Load draw records from JSONL or CSV file, sorted chronologically.
    """
    path = Path(data_path)
    records = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records.append(Pick3DrawRecord(
                        date=rec["date"],
                        draw=rec.get("draw", "Day"),
                        num1=int(rec["num1"]),
                        num2=int(rec["num2"]),
                        num3=int(rec["num3"]),
                    ))
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(Pick3DrawRecord(
                    date=row["Date"],
                    draw=row.get("Draw", "Day"),
                    num1=int(row["Num1"]),
                    num2=int(row["Num2"]),
                    num3=int(row["Num3"]),
                ))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Sort chronologically: oldest first, Day before Night within same date
    records.sort(key=lambda r: (parse_draw_date(r.date), 0 if r.draw_normalized == "Day" else 1))
    return records


# --- Dataset Construction ---
def build_training_and_validation_datasets(
    records: List[Pick3DrawRecord],
    tokenizer,
    window_size: int = 10,
    max_seq_len: int = 64,
    val_split: float = 0.2,
    target_loss_only: bool = True,
) -> Tuple[Tuple[mx.array, mx.array, mx.array], Optional[Tuple[mx.array, mx.array, mx.array]]]:
    """
    Build training and validation datasets from draw records.
    Returns (train_data, val_data) where each is (x, y, mask) tuple.
    """
    input_list: List[List[int]] = []
    target_list: List[List[int]] = []
    mask_list: List[List[float]] = []

    total = len(records)
    val_start = int(total * (1.0 - val_split))

    for i in range(window_size, total):
        history = records[i - window_size:i]
        target = records[i]

        # Encode prompt
        prompt = tokenizer.encode_history_prompt(
            [{"draw": d.draw_normalized, "num1": d.num1, "num2": d.num2, "num3": d.num3} for d in history],
            target.draw_normalized,
        )
        prompt_len = len(prompt)

        # Append target digits and EOS
        full_seq = prompt + [int(target.num1), int(target.num2), int(target.num3), tokenizer.eos_id]

        # Truncate if needed
        if len(full_seq) > max_seq_len + 1:
            excess = len(full_seq) - (max_seq_len + 1)
            full_seq = [tokenizer.bos_id] + full_seq[1 + excess:]
            prompt_len = max(1, prompt_len - excess)

        x_seq = full_seq[:-1]
        y_seq = full_seq[1:]

        # Build mask: 1.0 only for target positions
        mask = [0.0] * len(x_seq)
        for target_idx in range(max(0, prompt_len - 1), len(x_seq)):
            mask[target_idx] = 1.0

        # Pad to max_seq_len - 1
        seq_len = max_seq_len - 1
        pad_len = seq_len - len(x_seq)
        if pad_len > 0:
            x_seq.extend([tokenizer.pad_id] * pad_len)
            y_seq.extend([tokenizer.pad_id] * pad_len)
            mask.extend([0.0] * pad_len)
        else:
            x_seq = x_seq[:seq_len]
            y_seq = y_seq[:seq_len]
            mask = mask[:seq_len]

        input_list.append(x_seq)
        target_list.append(y_seq)
        mask_list.append(mask)

    # Split into train and validation
    # val_start is the index in records where validation begins
    # The first usable sample is at index window_size, so we need to adjust
    total_samples = total - window_size
    val_start_sample = int(total_samples * val_split)
    train_end_sample = total_samples - val_start_sample

    train_input = input_list[:train_end_sample]
    train_target = target_list[:train_end_sample]
    train_mask = mask_list[:train_end_sample]

    val_input = input_list[train_end_sample:]
    val_target = target_list[train_end_sample:]
    val_mask = mask_list[train_end_sample:]

    train_data = (
        mx.array(train_input, dtype=mx.int32),
        mx.array(train_target, dtype=mx.int32),
        mx.array(train_mask, dtype=mx.float32),
    )

    val_data = None
    if val_input:
        val_data = (
            mx.array(val_input, dtype=mx.int32),
            mx.array(val_target, dtype=mx.int32),
            mx.array(val_mask, dtype=mx.float32),
        )

    return train_data, val_data


# --- Training ---
def train_pick_3_model(
    model,
    train_data: Tuple[mx.array, mx.array, mx.array],
    val_data: Optional[Tuple[mx.array, mx.array, mx.array]],
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    model_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[Any, TrainingHistory]:
    """
    Train a Pick 3 MLX model and return the trained model and history.
    """
    train_x, train_y, train_mask = train_data
    history = TrainingHistory(learning_rate=lr)

    optimizer = optim.AdamW(learning_rate=lr)
    model.train(True)

    def loss_fn(params, x, y, mask):
        model.update(params)
        logits = model(x)
        loss_unreduced = nn.losses.cross_entropy(logits, y, reduction="none")
        valid_tokens = mx.maximum(mx.sum(mask), 1.0)
        return mx.sum(loss_unreduced * mask) / valid_tokens

    train_step = nn.value_and_grad(model, loss_fn)

    for epoch in range(epochs):
        # Training
        total_loss = 0.0
        num_batches = 0
        n = len(train_x)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bx = train_x[start:end]
            by = train_y[start:end]
            bm = train_mask[start:end]

            loss, grads = train_step(model.parameters(), bx, by, bm)
            model.update(grads)
            mx.eval(model, loss)

            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = float(mx.exp(mx.array(avg_loss)).item())

        history.epochs.append(epoch + 1)
        history.train_loss.append(avg_loss)
        history.train_perplexity.append(perplexity)

        # Validation
        val_loss_val = float("inf")
        val_token_acc_val = 0.0
        val_exact_acc_val = 0.0

        if val_data is not None:
            val_x, val_y, val_mask = val_data
            model.train(False)
            val_total_loss = 0.0
            val_total_token_acc = 0.0
            val_total_exact_acc = 0.0
            val_num_batches = 0

            for start in range(0, len(val_x), batch_size):
                end = min(start + batch_size, len(val_x))
                vx = val_x[start:end]
                vy = val_y[start:end]
                vm = val_mask[start:end]

                logits = model(vx)
                loss_unreduced = nn.losses.cross_entropy(logits, vy, reduction="none")
                valid_tokens = mx.maximum(mx.sum(vm), 1.0)
                batch_loss = mx.sum(loss_unreduced * vm) / valid_tokens

                predictions = mx.argmax(logits, axis=-1)
                token_correct = (predictions == vy).astype(mx.float32)
                batch_token_acc = mx.sum(token_correct * vm) / valid_tokens

                seq_mismatches = ((predictions != vy) * vm).astype(mx.float32)
                exact_matches = (mx.sum(seq_mismatches, axis=1) == 0.0).astype(mx.float32)
                batch_exact_acc = mx.mean(exact_matches)

                mx.eval(batch_loss, batch_token_acc, batch_exact_acc)

                val_total_loss += float(batch_loss.item())
                val_total_token_acc += float(batch_token_acc.item())
                val_total_exact_acc += float(batch_exact_acc.item())
                val_num_batches += 1

            val_loss_val = val_total_loss / max(val_num_batches, 1)
            val_token_acc_val = val_total_token_acc / max(val_num_batches, 1)
            val_exact_acc_val = val_total_exact_acc / max(val_num_batches, 1)
            model.train(True)

        history.val_loss.append(val_loss_val)
        history.val_perplexity.append(float(mx.exp(mx.array(val_loss_val)).item()) if val_loss_val != float("inf") else float("inf"))
        history.val_token_accuracy.append(val_token_acc_val)
        history.val_exact_match_rate.append(val_exact_acc_val)

        if val_loss_val < history.best_val_loss:
            history.best_val_loss = val_loss_val

        if verbose:
            msg = f"Epoch {epoch+1}/{epochs} - train_loss: {avg_loss:.4f}, train_ppl: {perplexity:.2f}"
            if val_data is not None:
                msg += f" - val_loss: {val_loss_val:.4f}, val_token_acc: {val_token_acc_val:.4f}, val_exact: {val_exact_acc_val:.4f}"
            print(msg)

    # Save model
    if model_dir:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_dir))

    return model, history


def evaluate_model_metrics(
    model,
    val_x: mx.array,
    val_y: mx.array,
    val_mask: mx.array,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluate model metrics on validation data."""
    model.train(False)
    total_loss = 0.0
    total_token_acc = 0.0
    total_exact_acc = 0.0
    num_batches = 0

    for start in range(0, len(val_x), batch_size):
        end = min(start + batch_size, len(val_x))
        vx = val_x[start:end]
        vy = val_y[start:end]
        vm = val_mask[start:end]

        logits = model(vx)
        loss_unreduced = nn.losses.cross_entropy(logits, vy, reduction="none")
        valid_tokens = mx.maximum(mx.sum(vm), 1.0)
        batch_loss = mx.sum(loss_unreduced * vm) / valid_tokens

        predictions = mx.argmax(logits, axis=-1)
        token_correct = (predictions == vy).astype(mx.float32)
        batch_token_acc = mx.sum(token_correct * vm) / valid_tokens

        seq_mismatches = ((predictions != vy) * vm).astype(mx.float32)
        exact_matches = (mx.sum(seq_mismatches, axis=1) == 0.0).astype(mx.float32)
        batch_exact_acc = mx.mean(exact_matches)

        mx.eval(batch_loss, batch_token_acc, batch_exact_acc)

        total_loss += float(batch_loss.item())
        total_token_acc += float(batch_token_acc.item())
        total_exact_acc += float(batch_exact_acc.item())
        num_batches += 1

    model.train(True)
    return {
        "loss": total_loss / max(num_batches, 1),
        "token_accuracy": total_token_acc / max(num_batches, 1),
        "exact_match_rate": total_exact_acc / max(num_batches, 1),
    }


# --- Full Pipeline ---
def run_training_pipeline(
    data_path: Union[str, Path],
    model_dir: Union[str, Path],
    epochs: int = 10,
    batch_size: int = 32,
    window_size: int = 10,
    max_seq_len: int = 64,
    val_split: float = 0.2,
    n_embd: int = 64,
    n_head: int = 4,
    n_layer: int = 3,
    lr: float = 1e-3,
    predict: bool = False,
    num_tickets: int = 5,
    draw_type: str = "Night",
    output_tickets_csv: Optional[Union[str, Path]] = None,
    history_json: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    hf_repo: Optional[str] = None,
    push_to_hub: bool = False,
    hf_token: Optional[str] = None,
) -> Tuple[Any, TrainingHistory]:
    """
    Run the full training pipeline: load data, build datasets, train, evaluate, and optionally predict.
    """
    from idaho.pick3.pick3_llm_ticket_model import Pick3LLM, Pick3LLMConfig, Pick3Tokenizer

    # Load records
    records = load_draw_records(data_path)
    if verbose:
        print(f"Loaded {len(records)} draw records from {data_path}")

    # Tokenizer
    tokenizer = Pick3Tokenizer()

    # Build datasets
    train_data, val_data = build_training_and_validation_datasets(
        records=records,
        tokenizer=tokenizer,
        window_size=window_size,
        max_seq_len=max_seq_len,
        val_split=val_split,
    )

    if verbose:
        print(f"Training samples: {len(train_data[0])}, Validation samples: {len(val_data[0]) if val_data else 0}")

    # Model config
    config = Pick3LLMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=max_seq_len,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
    )
    model = Pick3LLM(config)

    # Train
    model_dir_path = Path(model_dir)
    trained_model, history = train_pick_3_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        model_dir=model_dir_path,
        verbose=verbose,
    )

    # Save history
    if history_json:
        hj_path = Path(history_json)
        hj_path.write_text(json.dumps(history.to_dict(), indent=2), encoding="utf-8")
        if verbose:
            print(f"Training history saved to {hj_path}")

    # Predict
    if predict:
        from idaho.pick3.pick3_llm_ticket_model import Pick3Tokenizer
        tokenizer = Pick3Tokenizer()
        # Build a simple history prompt from the last few records
        history_draws = [
            {"draw": r.draw_normalized, "num1": r.num1, "num2": r.num2, "num3": r.num3}
            for r in records[-window_size:]
        ]
        tickets = []
        for _ in range(num_tickets):
            response = trained_model.generate_localized_prediction(
                history_draws=history_draws,
                draw_type=draw_type,
                temperature=0.8,
            )
            digits = tokenizer.decode_response_text(response.get("primary", ""))
            if digits and len(digits) == 3:
                tickets.append(digits)
            elif len(tickets) == 0:
                # Fallback: generate random digits
                import random
                tickets.append(tuple(random.randint(0, 9) for _ in range(3)))

        if verbose:
            print(f"\nPredicted tickets for {draw_type} draw:")
            for t in tickets:
                print(f"  {' '.join(str(d) for d in t)}")

        if output_tickets_csv:
            oc_path = Path(output_tickets_csv)
            with oc_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["num1", "num2", "num3", "draw"])
                for t in tickets:
                    writer.writerow([t[0], t[1], t[2], draw_type])
            if verbose:
                print(f"Tickets saved to {oc_path}")

    # Push to Hub
    if push_to_hub and hf_repo:
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            try:
                from idaho.pick3.upload_to_hf_hub import upload_checkpoint_to_hub
                upload_checkpoint_to_hub(
                    model_directory=model_dir_path,
                    repo_id=hf_repo,
                    token=token,
                    metrics={
                        "best_val_loss": history.best_val_loss,
                        "final_train_loss": history.train_loss[-1] if history.train_loss else 0.0,
                    },
                )
            except Exception as e:
                if verbose:
                    print(f"Hub upload failed: {e}")

    return trained_model, history


# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="Train Pick 3 MLX sequence model")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV draw data")
    parser.add_argument("--jsonl", type=str, default=None, help="Path to JSONL draw data")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR), help="Model output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--window-size", type=int, default=10, help="Historical draws in prompt")
    parser.add_argument("--max-seq-len", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--n-embd", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layer", type=int, default=3, help="Number of Transformer layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--predict", action="store_true", help="Generate predictions after training")
    parser.add_argument("--tickets", type=int, default=5, help="Number of prediction tickets")
    parser.add_argument("--draw", type=str, default="Night", help="Draw type for prediction")
    parser.add_argument("--output-tickets", type=str, default=None, help="CSV file for prediction tickets")
    parser.add_argument("--history-json", type=str, default=None, help="JSON file for training history")
    parser.add_argument("--push-to-hub", action="store_true", help="Push best checkpoint to Hugging Face Hub")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repository ID")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token")

    args = parser.parse_args()

    # Resolve data path
    if args.jsonl:
        data_path = Path(args.jsonl)
    elif args.csv:
        data_path = Path(args.csv)
    else:
        # Try default JSONL
        if DEFAULT_JSONL_PATH.exists():
            data_path = DEFAULT_JSONL_PATH
        else:
            fallback = _PROJECT_ROOT / "data" / "idaho_pick3_last_picks.jsonl"
            if fallback.exists():
                data_path = fallback
            else:
                print(f"Error: No data file found. Specify --jsonl or --csv.")
                sys.exit(1)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    run_training_pipeline(
        data_path=data_path,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        window_size=args.window_size,
        max_seq_len=args.max_seq_len,
        val_split=args.val_split,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        lr=args.lr,
        predict=args.predict,
        num_tickets=args.tickets,
        draw_type=args.draw,
        output_tickets_csv=args.output_tickets,
        history_json=args.history_json,
        push_to_hub=args.push_to_hub,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
