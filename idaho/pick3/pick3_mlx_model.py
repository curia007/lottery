#!/usr/bin/env python3
"""
Train a small MLX model from Idaho Pick 3 CSV history and suggest next numbers.

Expected CSV columns from scrape_idaho_pick3.py:
    Date,Draw,Num1,Num2,Num3

Default input file:
    idaho_pick3_history.csv

Examples:
    python pick3_mlx_model.py
    python pick3_mlx_model.py --csv data/idaho_pick3_history.csv --draw Night
    python pick3_mlx_model.py --csv data/idaho_pick3_history.csv --epochs 300 --window 30

Important:
    Lottery drawings are random. This script creates a statistical/ML-style
    suggestion from historical data, not a guaranteed prediction.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError as exc:
    raise SystemExit(
        "MLX is not installed. Install it with:\n\n"
        "    pip install mlx\n\n"
        "MLX is designed primarily for Apple silicon Macs."
    ) from exc

DEFAULT_CSV = "data/idaho_pick3_history.csv"


@dataclass(frozen=True)
class Pick3Row:
    date: str
    draw: str
    num1: int
    num2: int
    num3: int

    @property
    def draw_flag(self) -> float:
        """Day = 0.0, Night = 1.0."""
        return 1.0 if self.draw.strip().lower() == "night" else 0.0

    @property
    def digits(self) -> Tuple[int, int, int]:
        return self.num1, self.num2, self.num3


def parse_date(value: str) -> datetime:
    """Parse common date formats. The scraper emits YYYY-MM-DD."""
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value}")


def load_rows(csv_path: str | Path) -> List[Pick3Row]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            "Run scrape_idaho_pick3.py first, or pass --csv path/to/file.csv"
        )

    rows: List[Pick3Row] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"Date", "Draw", "Num1", "Num2", "Num3"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

        for raw in reader:
            try:
                row = Pick3Row(
                    date=str(raw["Date"]).strip(),
                    draw=str(raw["Draw"]).strip().capitalize(),
                    num1=int(raw["Num1"]),
                    num2=int(raw["Num2"]),
                    num3=int(raw["Num3"]),
                )
            except Exception as exc:
                raise ValueError(f"Bad CSV row: {raw}") from exc

            for digit in row.digits:
                if digit < 0 or digit > 9:
                    raise ValueError(f"Pick 3 digit must be 0-9. Bad row: {row}")
            rows.append(row)

    # Train chronologically: oldest first. Day before Night on the same date.
    rows.sort(key=lambda r: (parse_date(r.date), 1 if r.draw.lower() == "night" else 0))
    return rows


def row_to_features(row: Pick3Row) -> List[float]:
    """
    Convert one draw into numeric features.

    Each draw contributes:
      - three digits normalized to 0.0-1.0
      - draw flag: Day=0, Night=1
      - sum normalized to 0.0-1.0
      - duplicate count normalized to 0.0-1.0
    """
    d1, d2, d3 = row.digits
    total = d1 + d2 + d3
    unique_count = len({d1, d2, d3})
    duplicate_strength = (3 - unique_count) / 2.0
    return [
        d1 / 9.0,
        d2 / 9.0,
        d3 / 9.0,
        row.draw_flag,
        total / 27.0,
        duplicate_strength,
    ]


def make_dataset(rows: Sequence[Pick3Row], window: int) -> Tuple[mx.array, mx.array]:
    if len(rows) <= window:
        raise ValueError(
            f"Not enough rows to train. Need more than window={window}; got {len(rows)} rows."
        )

    x_values: List[List[float]] = []
    y_values: List[List[int]] = []

    for i in range(window, len(rows)):
        window_rows = rows[i - window : i]
        features: List[float] = []
        for row in window_rows:
            features.extend(row_to_features(row))
        x_values.append(features)
        y_values.append(list(rows[i].digits))

    return mx.array(x_values, dtype=mx.float32), mx.array(y_values, dtype=mx.int32)


class Pick3MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 30),  # 3 digit positions x 10 possible digits
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


def loss_fn(model: Pick3MLP, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    loss1 = nn.losses.cross_entropy(logits[:, 0:10], y[:, 0], reduction="mean")
    loss2 = nn.losses.cross_entropy(logits[:, 10:20], y[:, 1], reduction="mean")
    loss3 = nn.losses.cross_entropy(logits[:, 20:30], y[:, 2], reduction="mean")
    return (loss1 + loss2 + loss3) / 3.0


def batch_indices(n: int, batch_size: int) -> Iterable[List[int]]:
    indices = list(range(n))
    random.shuffle(indices)
    for start in range(0, n, batch_size):
        yield indices[start : start + batch_size]


def take_rows(x: mx.array, y: mx.array, idx: Sequence[int]) -> Tuple[mx.array, mx.array]:
    idx_array = mx.array(idx, dtype=mx.int32)
    return x[idx_array], y[idx_array]


def train_model(
    x: mx.array,
    y: mx.array,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    seed: int,
) -> Pick3MLP:
    random.seed(seed)
    mx.random.seed(seed)

    input_dim = int(x.shape[1])
    model = Pick3MLP(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    n = int(x.shape[0])
    for epoch in range(1, epochs + 1):
        epoch_losses: List[float] = []
        for idx in batch_indices(n, batch_size):
            xb, yb = take_rows(x, y, idx)
            loss, grads = loss_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_losses.append(float(loss))

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            print(f"epoch={epoch:4d} loss={avg_loss:.4f}")

    return model


def softmax_list(values: mx.array) -> List[float]:
    probs = mx.softmax(values)
    return [float(v) for v in probs.tolist()]


def top_digits(logits: mx.array, top_k: int = 3) -> List[List[Tuple[int, float]]]:
    results: List[List[Tuple[int, float]]] = []
    for start in (0, 10, 20):
        probs = softmax_list(logits[start : start + 10])
        ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
        results.append(ranked[:top_k])
    return results


def build_latest_feature(rows: Sequence[Pick3Row], window: int, preferred_draw: str | None) -> mx.array:
    latest = list(rows[-window:])

    # Small hint to the model about the desired next draw type. This does not change
    # historical data; it only sets the draw flag in the most recent feature position.
    if preferred_draw:
        preferred_draw = preferred_draw.strip().capitalize()
        last = latest[-1]
        latest[-1] = Pick3Row(
            date=last.date,
            draw=preferred_draw,
            num1=last.num1,
            num2=last.num2,
            num3=last.num3,
        )

    features: List[float] = []
    for row in latest:
        features.extend(row_to_features(row))
    return mx.array([features], dtype=mx.float32)


def format_prediction(top: List[List[Tuple[int, float]]]) -> Tuple[str, List[str]]:
    best_digits = [str(position[0][0]) for position in top]
    best_number = "".join(best_digits)

    detail_lines = []
    for i, choices in enumerate(top, start=1):
        formatted = ", ".join(f"{digit} ({prob:.1%})" for digit, prob in choices)
        detail_lines.append(f"Position {i}: {formatted}")

    return best_number, detail_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an MLX model from Idaho Pick 3 CSV history and suggest next digits."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help=f"Input CSV. Default: {DEFAULT_CSV}")
    parser.add_argument("--window", type=int, default=25, help="Number of past draws used as context.")
    parser.add_argument("--epochs", type=int, default=250, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size.")
    parser.add_argument("--draw", choices=["Day", "Night", "day", "night"], default=None, help="Optional next draw type hint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable training.")
    parser.add_argument("--top-k", type=int, default=3, help="Top digit choices to show per position.")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    print(f"Loaded {len(rows):,} Pick 3 rows from {args.csv}")

    x, y = make_dataset(rows, window=args.window)
    print(f"Training samples: {int(x.shape[0]):,}; input features: {int(x.shape[1]):,}")

    model = train_model(
        x=x,
        y=y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )

    latest_x = build_latest_feature(rows, window=args.window, preferred_draw=args.draw)
    logits = model(latest_x)[0]
    mx.eval(logits)

    top = top_digits(logits, top_k=args.top_k)
    prediction, detail_lines = format_prediction(top)

    print("\n=== MLX Pick 3 Suggestion ===")
    if args.draw:
        print(f"Draw hint: {args.draw.capitalize()}")
    print(f"Suggested next number: {prediction}")
    print("\nTop digit probabilities by position:")
    for line in detail_lines:
        print(f"  {line}")

    print(
        "\nReminder: Pick 3 drawings are random. This is a historical-pattern model, "
        "not a guaranteed winning-number predictor."
    )


if __name__ == "__main__":
    main()
