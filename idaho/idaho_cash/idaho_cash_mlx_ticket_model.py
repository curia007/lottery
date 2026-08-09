#!/usr/bin/env python3
"""
idaho_cash_mlx_ticket_model.py

Train a small MLX model from Idaho Cash CSV history and generate ranked ticket
candidates for the next draw.

Expected CSV format from scrape_idaho_cash.py:

    Date,Num1,Num2,Num3,Num4,Num5
    2026-02-24,4,13,24,33,43

Important:
    Lottery drawings are random. This script ranks tickets from historical
    patterns; it does not guarantee or truly predict a winning result.

Install:
    pip install mlx pandas numpy

Examples:
    python idaho_cash_mlx_ticket_model.py --csv idaho_cash_history.csv --tickets 5

    python idaho_cash_mlx_ticket_model.py --csv idaho_cash_history.csv --tickets 10 --ticket-type balanced

    python idaho_cash_mlx_ticket_model.py --csv idaho_cash_history.csv --tickets 10 --ticket-type hot

    python idaho_cash_mlx_ticket_model.py --csv idaho_cash_history.csv --tickets 10 --ticket-type overdue

    python idaho_cash_mlx_ticket_model.py --csv idaho_cash_history.csv --tickets 5 --exclude-recent 14 --output idaho_cash_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError as exc:
    raise SystemExit(
        "MLX is not installed. Install it with:\n\n"
        "    pip install mlx\n\n"
        "MLX works best on Apple Silicon Macs."
    ) from exc


TicketType = Literal["model", "balanced", "hot", "overdue", "hot_overdue"]


@dataclass
class RankedTicket:
    rank: int
    ticket: str
    score: float
    model_score: float
    frequency_score: float
    overdue_score: float
    pair_score: float
    balance_score: float
    ticket_type: str


class IdahoCashModel(nn.Module):
    """
    Small neural network that predicts a 45-number multi-label vector.

    Input:
        Features from recent Idaho Cash draws.

    Output:
        45 logits, one for each Idaho Cash number 1-45.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 45),
        )

    def __call__(self, x):
        return self.net(x)


def load_idaho_cash_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    required = {"Date", "Num1", "Num2", "Num3", "Num4", "Num5"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["Num1", "Num2", "Num3", "Num4", "Num5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Num1", "Num2", "Num3", "Num4", "Num5"])

    for col in ["Num1", "Num2", "Num3", "Num4", "Num5"]:
        df[col] = df[col].astype(int)
        bad = ~df[col].between(1, 45)
        if bad.any():
            raise ValueError(f"Column {col} contains values outside Idaho Cash range 1-45.")

    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 25:
        raise ValueError(f"Need at least 25 rows to train. Found {len(df)}.")

    return df


def number_rows(df: pd.DataFrame) -> np.ndarray:
    values = df[["Num1", "Num2", "Num3", "Num4", "Num5"]].to_numpy(dtype=np.int64)
    return np.sort(values, axis=1)


def to_multihot(row: Iterable[int]) -> np.ndarray:
    y = np.zeros(45, dtype=np.float32)
    for n in row:
        y[int(n) - 1] = 1.0
    return y


def make_features_and_labels(numbers: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build training rows.

    Features:
      - Recent draw numbers flattened and scaled 0-1.
      - Frequency for numbers 1-45 in the recent window.
      - Gap / overdue value for numbers 1-45 in the recent window.
      - Last draw multi-hot vector.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for i in range(window, len(numbers)):
        recent = numbers[i - window : i]
        target = numbers[i]

        flattened = recent.flatten().astype(np.float32) / 45.0

        freq = np.zeros(45, dtype=np.float32)
        for n in recent.flatten():
            freq[int(n) - 1] += 1.0
        freq = freq / max(1, recent.size)

        flat_recent = recent.flatten()
        gaps = np.ones(45, dtype=np.float32)
        for n in range(1, 46):
            locations = np.where(flat_recent == n)[0]
            if len(locations) == 0:
                gaps[n - 1] = 1.0
            else:
                gaps[n - 1] = (len(flat_recent) - 1 - locations[-1]) / max(1, len(flat_recent))

        last_draw = to_multihot(recent[-1])

        x = np.concatenate([flattened, freq, gaps, last_draw]).astype(np.float32)
        y = to_multihot(target)

        xs.append(x)
        ys.append(y)

    return np.stack(xs), np.stack(ys)


def binary_cross_entropy_with_logits(logits, targets):
    """
    Stable BCE loss for multi-label classification.
    """
    return mx.mean(mx.maximum(logits, 0) - logits * targets + mx.log1p(mx.exp(-mx.abs(logits))))


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> IdahoCashModel:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    model = IdahoCashModel(input_dim=x_train.shape[1])
    optimizer = optim.Adam(learning_rate=learning_rate)

    x_mx = mx.array(x_train)
    y_mx = mx.array(y_train)

    def loss_fn(model, xb, yb):
        logits = model(xb)
        return binary_cross_entropy_with_logits(logits, yb)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for _ in range(epochs):
        loss, grads = loss_and_grad_fn(model, x_mx, y_mx)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    return model


def build_next_feature(numbers: np.ndarray, window: int) -> np.ndarray:
    if len(numbers) < window:
        window = len(numbers)

    recent = numbers[-window:]
    flattened = recent.flatten().astype(np.float32) / 45.0

    freq = np.zeros(45, dtype=np.float32)
    for n in recent.flatten():
        freq[int(n) - 1] += 1.0
    freq = freq / max(1, recent.size)

    flat_recent = recent.flatten()
    gaps = np.ones(45, dtype=np.float32)
    for n in range(1, 46):
        locations = np.where(flat_recent == n)[0]
        if len(locations) == 0:
            gaps[n - 1] = 1.0
        else:
            gaps[n - 1] = (len(flat_recent) - 1 - locations[-1]) / max(1, len(flat_recent))

    last_draw = to_multihot(recent[-1])

    return np.concatenate([flattened, freq, gaps, last_draw]).astype(np.float32)


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def model_number_probabilities(model: IdahoCashModel, next_feature: np.ndarray) -> np.ndarray:
    logits = model(mx.array(next_feature.reshape(1, -1)))
    logits_np = np.array(logits).reshape(45)
    return sigmoid_np(logits_np)


def frequency_scores(numbers: np.ndarray) -> dict[int, float]:
    # Try to load from number_counts.csv if it exists
    counts_file = Path(__file__).parent / "data" / "number_counts.csv"
    if counts_file.exists():
        try:
            df_counts = pd.read_csv(counts_file)
            # number_counts.csv columns: Number, Count
            # We want to normalize these counts for scoring
            counts_dict = dict(zip(df_counts["Number"], df_counts["Count"]))
            
            # Map to 1-45 range
            all_counts = np.array([float(counts_dict.get(n, 0)) for n in range(1, 46)])
            all_counts += 1.0  # Laplace smoothing
            all_counts /= all_counts.sum()
            
            return {n: float(all_counts[n - 1]) for n in range(1, 46)}
        except Exception as e:
            print(f"Warning: Could not read number_counts.csv: {e}. Falling back to historical data.")

    counts = np.zeros(45, dtype=np.float64)
    for n in numbers.flatten():
        counts[int(n) - 1] += 1.0

    counts += 1.0
    counts = counts / counts.sum()

    return {n: float(counts[n - 1]) for n in range(1, 46)}


def overdue_scores(numbers: np.ndarray) -> dict[int, float]:
    flat = numbers.flatten()
    raw = {}

    for n in range(1, 46):
        positions = np.where(flat == n)[0]
        if len(positions) == 0:
            gap = len(flat)
        else:
            gap = len(flat) - 1 - positions[-1]
        raw[n] = float(gap)

    max_gap = max(raw.values()) or 1.0
    return {n: raw[n] / max_gap for n in range(1, 46)}


def pair_scores(numbers: np.ndarray) -> dict[tuple[int, int], float]:
    pairs: dict[tuple[int, int], int] = {}

    for row in numbers:
        for pair in itertools.combinations(sorted(map(int, row)), 2):
            pairs[pair] = pairs.get(pair, 0) + 1

    max_count = max(pairs.values()) if pairs else 1

    scores = {}
    for a in range(1, 46):
        for b in range(a + 1, 46):
            scores[(a, b)] = pairs.get((a, b), 0) / max_count

    return scores


def ticket_to_str(ticket: Iterable[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(ticket))


def recent_ticket_keys(numbers: np.ndarray, recent_count: int) -> set[tuple[int, ...]]:
    if recent_count <= 0:
        return set()

    recent = numbers[-recent_count:]
    return {tuple(sorted(map(int, row))) for row in recent}


def balance_score(ticket: tuple[int, int, int, int, int]) -> float:
    """
    Rough balance score:
      - Prefer a mix of low/high numbers.
      - Prefer odd/even split around 2/3 or 3/2.
      - Prefer a sum near the middle of possible sums.
      - Slightly discourage highly clustered tickets.
    """
    nums = sorted(ticket)

    low_count = sum(1 for n in nums if n <= 22)
    odd_count = sum(1 for n in nums if n % 2 == 1)

    low_high = 1.0 - abs(low_count - 2.5) / 2.5
    odd_even = 1.0 - abs(odd_count - 2.5) / 2.5

    total = sum(nums)
    # Idaho Cash 5-number sum usually lives between 15 and 215.
    sum_center = 115
    sum_score = max(0.0, 1.0 - abs(total - sum_center) / 100.0)

    spread = nums[-1] - nums[0]
    spread_score = min(1.0, spread / 30.0)

    return float((low_high + odd_even + sum_score + spread_score) / 4.0)


def get_weights(ticket_type: TicketType, custom_weights: tuple[float, float, float, float, float] | None):
    if custom_weights:
        return custom_weights

    defaults = {
        "model": (0.60, 0.15, 0.10, 0.10, 0.05),
        "balanced": (0.35, 0.15, 0.15, 0.10, 0.25),
        "hot": (0.25, 0.45, 0.05, 0.20, 0.05),
        "overdue": (0.25, 0.05, 0.50, 0.10, 0.10),
        "hot_overdue": (0.35, 0.25, 0.25, 0.10, 0.05),
    }

    return defaults[ticket_type]


def parse_weights(text: str) -> tuple[float, float, float, float, float]:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "Weights must be five comma-separated numbers: model,frequency,overdue,pair,balance"
        )

    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("Weights must sum to a positive number.")

    return tuple(x / total for x in parts)  # type: ignore[return-value]


def candidate_pool(
    model_probs: np.ndarray,
    freq: dict[int, float],
    overdue: dict[int, float],
    pool_size: int,
    ticket_type: TicketType,
) -> list[int]:
    """
    Reduce combinations to a smart candidate pool.

    Full 45 choose 5 = 1,221,759 tickets. That is possible but slower.
    This script builds a ranked candidate number pool, then scores combinations
    from that pool.
    """
    base = []

    for n in range(1, 46):
        if ticket_type == "hot":
            score = 0.30 * model_probs[n - 1] + 0.60 * freq[n] + 0.10 * overdue[n]
        elif ticket_type == "overdue":
            score = 0.30 * model_probs[n - 1] + 0.10 * freq[n] + 0.60 * overdue[n]
        elif ticket_type == "hot_overdue":
            score = 0.40 * model_probs[n - 1] + 0.30 * freq[n] + 0.30 * overdue[n]
        elif ticket_type == "balanced":
            score = 0.55 * model_probs[n - 1] + 0.20 * freq[n] + 0.25 * overdue[n]
        else:
            score = 0.70 * model_probs[n - 1] + 0.20 * freq[n] + 0.10 * overdue[n]

        base.append((n, score))

    base.sort(key=lambda x: x[1], reverse=True)
    pool_size = max(5, min(pool_size, 45))

    return [n for n, _ in base[:pool_size]]


def rank_tickets(
    model_probs: np.ndarray,
    numbers: np.ndarray,
    ticket_type: TicketType,
    tickets: int,
    exclude_recent: int,
    pool_size: int,
    custom_weights: tuple[float, float, float, float, float] | None,
) -> list[RankedTicket]:
    freq = frequency_scores(numbers)
    overdue = overdue_scores(numbers)
    pairs = pair_scores(numbers)
    excluded = recent_ticket_keys(numbers, exclude_recent)

    model_w, freq_w, overdue_w, pair_w, balance_w = get_weights(ticket_type, custom_weights)

    pool = candidate_pool(
        model_probs=model_probs,
        freq=freq,
        overdue=overdue,
        pool_size=pool_size,
        ticket_type=ticket_type,
    )

    ranked: list[RankedTicket] = []

    for ticket in itertools.combinations(sorted(pool), 5):
        ticket_key = tuple(sorted(ticket))

        if ticket_key in excluded:
            continue

        model_score = float(np.mean([model_probs[n - 1] for n in ticket]))
        frequency_score = float(np.mean([freq[n] for n in ticket]))
        overdue_score = float(np.mean([overdue[n] for n in ticket]))

        ticket_pairs = list(itertools.combinations(sorted(ticket), 2))
        pair_score = float(np.mean([pairs.get(pair, 0.0) for pair in ticket_pairs]))

        bal_score = balance_score(ticket)

        score = (
            model_w * model_score
            + freq_w * frequency_score
            + overdue_w * overdue_score
            + pair_w * pair_score
            + balance_w * bal_score
        )

        ranked.append(
            RankedTicket(
                rank=0,
                ticket=ticket_to_str(ticket),
                score=score,
                model_score=model_score,
                frequency_score=frequency_score,
                overdue_score=overdue_score,
                pair_score=pair_score,
                balance_score=bal_score,
                ticket_type=ticket_type,
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)

    for idx, item in enumerate(ranked[:tickets], start=1):
        item.rank = idx

    return ranked[:tickets]


def save_ranked_csv(path: str | Path, rows: list[RankedTicket]) -> None:
    fieldnames = [
        "rank",
        "ticket_type",
        "ticket",
        "score",
        "model_score",
        "frequency_score",
        "overdue_score",
        "pair_score",
        "balance_score",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            writer.writerow(
                {
                    "rank": r.rank,
                    "ticket_type": r.ticket_type,
                    "ticket": r.ticket,
                    "score": f"{r.score:.8f}",
                    "model_score": f"{r.model_score:.8f}",
                    "frequency_score": f"{r.frequency_score:.8f}",
                    "overdue_score": f"{r.overdue_score:.8f}",
                    "pair_score": f"{r.pair_score:.8f}",
                    "balance_score": f"{r.balance_score:.8f}",
                }
            )


def print_ranked(rows: list[RankedTicket]) -> None:
    print()
    print("Ranked Idaho Cash ticket candidates")
    print("-" * 92)
    print(
        f"{'Rank':<6}"
        f"{'Type':<14}"
        f"{'Ticket':<20}"
        f"{'Score':<14}"
        f"{'Model':<14}"
        f"{'Overdue':<14}"
    )
    print("-" * 92)

    for r in rows:
        print(
            f"{r.rank:<6}"
            f"{r.ticket_type:<14}"
            f"{r.ticket:<20}"
            f"{r.score:<14.8f}"
            f"{r.model_score:<14.8f}"
            f"{r.overdue_score:<14.8f}"
        )

    print("-" * 92)
    print("Reminder: this ranks historical-pattern candidates; lottery results are random.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an MLX Idaho Cash model and produce ranked ticket candidates."
    )

    parser.add_argument(
        "--csv",
        default="data/idaho_cash_history.csv",
        help="CSV output from scrape_idaho_cash.py. Default: idaho_cash_history.csv",
    )

    parser.add_argument(
        "--tickets",
        "--top",
        dest="tickets",
        type=int,
        default=5,
        help="Number of ranked tickets to return. Default: 5",
    )

    parser.add_argument(
        "--ticket-type",
        choices=["model", "balanced", "hot", "overdue", "hot_overdue"],
        default="balanced",
        help=(
            "Ticket style: model, balanced, hot, overdue, hot_overdue. "
            "Default: model"
        ),
    )

    parser.add_argument(
        "--window",
        type=int,
        default=40,
        help="Number of past rows used as model context. Default: 40",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Training epochs. Default: 300",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.005,
        help="MLX Adam learning rate. Default: 0.005",
    )

    parser.add_argument(
        "--exclude-recent",
        type=int,
        default=0,
        help="Exclude tickets that exactly appeared in the last N draws. Default: 0",
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=26,
        help=(
            "How many top candidate numbers to combine into tickets. "
            "Higher is broader but slower. Default: 26"
        ),
    )

    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=None,
        help=(
            "Optional custom weights as model,frequency,overdue,pair,balance. "
            "Example: --weights 0.50,0.20,0.15,0.10,0.05"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training. Default: 42",
    )

    parser.add_argument(
        "--output",
        help="Optional CSV file to save ranked output.",
    )

    args = parser.parse_args()

    if args.tickets < 1:
        raise SystemExit("--tickets / --top must be at least 1.")

    if args.pool_size < 5:
        raise SystemExit("--pool-size must be at least 5.")

    df = load_idaho_cash_csv(args.csv)
    numbers = number_rows(df)

    window = min(args.window, max(5, len(numbers) // 3))
    x_train, y_train = make_features_and_labels(numbers, window=window)

    model = train_model(
        x_train=x_train,
        y_train=y_train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    next_feature = build_next_feature(numbers, window=window)
    model_probs = model_number_probabilities(model, next_feature)

    ranked = rank_tickets(
        model_probs=model_probs,
        numbers=numbers,
        ticket_type=args.ticket_type,
        tickets=args.tickets,
        exclude_recent=args.exclude_recent,
        pool_size=args.pool_size,
        custom_weights=args.weights,
    )

    print_ranked(ranked)

    if args.output:
        save_ranked_csv(args.output, ranked)
        print(f"\nSaved ranked tickets to: {args.output}")


if __name__ == "__main__":
    main()
