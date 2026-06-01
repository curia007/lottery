#!/usr/bin/env python3
"""
pick4_mlx_number_select_model.py

Train a small MLX model from Idaho Pick 4 CSV history and generate ranked ticket
candidates for the next Day or Night draw.

Expected CSV format from scrape_idaho_pick4.py:

    Date,Draw,Num1,Num2,Num3,Num4
    2026-02-17,Night,1,2,3,4
    2026-02-18,Day,4,5,6,7

Important:
    Lottery drawings are random. This script ranks tickets from historical
    patterns; it does not guarantee or truly predict a winning number.

Install:
    pip install mlx pandas numpy

Examples:
    python pick4_mlx_number_select_model.py --csv idaho_pick4_history.csv --draw Night --tickets 5

    python pick4_mlx_number_select_model.py --csv idaho_pick4_history.csv --draw Day --tickets 10 --number-select exact --ticket-type balanced

    python pick4_mlx_number_select_model.py --csv idaho_pick4_history.csv --draw Night --tickets 10 --number-select any --ticket-type hot_overdue

    python pick4_mlx_number_select_model.py --csv idaho_pick4_history.csv --draw both --tickets 5 --number-select exact --ticket-type model

Arguments:
    --number-select:
        exact   = straight exact order, e.g. 1234
        any     = any-order / boxed grouping, e.g. 1234 covers 1234, 1243, 1324, etc.
        boxed   = alias for any

    --ticket-type:
        model       = strongest ML model weighting
        balanced    = balanced digit mix
        hot         = favors frequent digits
        overdue     = favors overdue digits
        hot_overdue = hybrid hot + overdue strategy
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
NumberSelect = Literal["exact", "any", "boxed"]


@dataclass
class RankedTicket:
    rank: int
    draw: str
    ticket_type: str
    number_select: str
    ticket: str
    score: float
    model_score: float
    frequency_score: float
    overdue_score: float
    pair_score: float
    balance_score: float


class Pick4Model(nn.Module):
    """
    Neural network that predicts the next Pick 4 digits.

    Output:
        40 logits total:
        - 10 logits for position 1
        - 10 logits for position 2
        - 10 logits for position 3
        - 10 logits for position 4
    """

    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 40),
        )

    def __call__(self, x):
        return self.net(x)


def normalize_draw(value: str) -> str:
    text = str(value).strip().lower()

    if text.startswith("day") or "midday" in text:
        return "Day"

    if text.startswith("night") or "evening" in text:
        return "Night"

    return ""


def load_pick4_csv(csv_path: str | Path, draw: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    required = {"Date", "Draw", "Num1", "Num2", "Num3", "Num4"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df["Draw"] = df["Draw"].apply(normalize_draw)

    for col in ["Num1", "Num2", "Num3", "Num4"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Num1", "Num2", "Num3", "Num4"])

    for col in ["Num1", "Num2", "Num3", "Num4"]:
        df[col] = df[col].astype(int)
        bad = ~df[col].between(0, 9)

        if bad.any():
            raise ValueError(f"Column {col} contains values outside 0-9.")

    df = df.sort_values("Date").reset_index(drop=True)

    if draw.lower() != "both":
        draw_name = normalize_draw(draw)
        df = df[df["Draw"] == draw_name].reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            f"Not enough rows after filtering draw={draw!r}. "
            f"Need at least 20 rows, found {len(df)}."
        )

    return df


def digit_rows(df: pd.DataFrame) -> np.ndarray:
    return df[["Num1", "Num2", "Num3", "Num4"]].to_numpy(dtype=np.int64)


def make_features_and_labels(digits: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for i in range(window, len(digits)):
        recent = digits[i - window : i]
        target = digits[i]

        flattened = recent.flatten().astype(np.float32) / 9.0

        freq = np.zeros(10, dtype=np.float32)
        for d in recent.flatten():
            freq[int(d)] += 1
        freq = freq / max(1, recent.size)

        gaps = np.ones(10, dtype=np.float32)
        flat_recent = recent.flatten()

        for d in range(10):
            locations = np.where(flat_recent == d)[0]
            if len(locations) == 0:
                gaps[d] = 1.0
            else:
                gaps[d] = (len(flat_recent) - 1 - locations[-1]) / max(1, len(flat_recent))

        last_draw = recent[-1].astype(np.float32) / 9.0

        x = np.concatenate([flattened, freq, gaps, last_draw]).astype(np.float32)

        xs.append(x)
        ys.append(target.astype(np.int64))

    return np.stack(xs), np.stack(ys)


def cross_entropy_for_four_positions(logits, y):
    logits = logits.reshape((-1, 4, 10))

    losses = [
        nn.losses.cross_entropy(logits[:, pos, :], y[:, pos], reduction="mean")
        for pos in range(4)
    ]

    return sum(losses) / 4.0


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> Pick4Model:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    model = Pick4Model(input_dim=x_train.shape[1])
    optimizer = optim.Adam(learning_rate=learning_rate)

    x_mx = mx.array(x_train)
    y_mx = mx.array(y_train)

    def loss_fn(model, xb, yb):
        logits = model(xb)
        return cross_entropy_for_four_positions(logits, yb)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for _ in range(epochs):
        loss, grads = loss_and_grad_fn(model, x_mx, y_mx)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    return model


def build_next_feature(digits: np.ndarray, window: int) -> np.ndarray:
    if len(digits) < window:
        window = len(digits)

    recent = digits[-window:]
    flattened = recent.flatten().astype(np.float32) / 9.0

    freq = np.zeros(10, dtype=np.float32)
    for d in recent.flatten():
        freq[int(d)] += 1
    freq = freq / max(1, recent.size)

    gaps = np.ones(10, dtype=np.float32)
    flat_recent = recent.flatten()

    for d in range(10):
        locations = np.where(flat_recent == d)[0]
        if len(locations) == 0:
            gaps[d] = 1.0
        else:
            gaps[d] = (len(flat_recent) - 1 - locations[-1]) / max(1, len(flat_recent))

    last_draw = recent[-1].astype(np.float32) / 9.0

    return np.concatenate([flattened, freq, gaps, last_draw]).astype(np.float32)


def softmax_np(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)


def model_digit_probabilities(model: Pick4Model, next_feature: np.ndarray) -> np.ndarray:
    logits = model(mx.array(next_feature.reshape(1, -1)))
    logits_np = np.array(logits).reshape(4, 10)
    return np.stack([softmax_np(logits_np[i]) for i in range(4)])


def ticket_to_str(ticket: Iterable[int]) -> str:
    return "".join(str(int(x)) for x in ticket)


def ticket_key(ticket: tuple[int, int, int, int], number_select: str) -> str:
    if number_select in {"any", "boxed"}:
        return "".join(sorted(ticket_to_str(ticket)))

    return ticket_to_str(ticket)


def exact_model_score(ticket: tuple[int, int, int, int], probs: np.ndarray) -> float:
    return float(
        probs[0, ticket[0]]
        * probs[1, ticket[1]]
        * probs[2, ticket[2]]
        * probs[3, ticket[3]]
    )


def frequency_scores(digits: np.ndarray) -> dict[int, float]:
    counts = np.zeros(10, dtype=np.float64)

    for d in digits.flatten():
        counts[int(d)] += 1

    counts += 1.0
    counts = counts / counts.sum()

    return {i: float(counts[i]) for i in range(10)}


def overdue_scores(digits: np.ndarray) -> dict[int, float]:
    flat = digits.flatten()
    raw = {}

    for d in range(10):
        positions = np.where(flat == d)[0]

        if len(positions) == 0:
            gap = len(flat)
        else:
            gap = len(flat) - 1 - positions[-1]

        raw[d] = float(gap)

    max_gap = max(raw.values()) or 1.0

    return {d: raw[d] / max_gap for d in range(10)}


def pair_scores(digits: np.ndarray) -> dict[tuple[int, int], float]:
    pairs = {}

    for row in digits:
        for pair in itertools.combinations(map(int, row), 2):
            key = tuple(sorted(pair))
            pairs[key] = pairs.get(key, 0) + 1

    max_count = max(pairs.values()) if pairs else 1

    scores = {}
    for a in range(10):
        for b in range(10):
            key = tuple(sorted((a, b)))
            scores[key] = pairs.get(key, 0) / max_count

    return scores


def balance_score(ticket: tuple[int, int, int, int]) -> float:
    digits = list(ticket)

    odd_count = sum(1 for d in digits if d % 2 == 1)
    high_count = sum(1 for d in digits if d >= 5)
    unique_count = len(set(digits))
    total = sum(digits)

    odd_even_score = 1.0 - abs(odd_count - 2.0) / 2.0
    high_low_score = 1.0 - abs(high_count - 2.0) / 2.0
    unique_score = unique_count / 4.0
    sum_score = max(0.0, 1.0 - abs(total - 18.0) / 18.0)

    return float((odd_even_score + high_low_score + unique_score + sum_score) / 4.0)


def recent_ticket_keys(
    digits: np.ndarray,
    recent_count: int,
    number_select: str,
) -> set[str]:
    if recent_count <= 0:
        return set()

    recent = digits[-recent_count:]
    return {ticket_key(tuple(map(int, row)), number_select) for row in recent}


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


def get_weights(
    ticket_type: TicketType,
    custom_weights: tuple[float, float, float, float, float] | None,
):
    if custom_weights:
        return custom_weights

    defaults = {
        "model": (0.65, 0.12, 0.08, 0.08, 0.07),
        "balanced": (0.38, 0.15, 0.15, 0.07, 0.25),
        "hot": (0.28, 0.45, 0.05, 0.17, 0.05),
        "overdue": (0.28, 0.05, 0.47, 0.10, 0.10),
        "hot_overdue": (0.36, 0.25, 0.25, 0.09, 0.05),
    }

    return defaults[ticket_type]


def rank_tickets(
    probs: np.ndarray,
    digits: np.ndarray,
    draw: str,
    ticket_type: TicketType,
    number_select: NumberSelect,
    tickets: int,
    exclude_recent: int,
    weights: tuple[float, float, float, float, float] | None,
) -> list[RankedTicket]:
    freq = frequency_scores(digits)
    overdue = overdue_scores(digits)
    pairs = pair_scores(digits)
    excluded = recent_ticket_keys(digits, exclude_recent, number_select)

    model_w, freq_w, overdue_w, pair_w, balance_w = get_weights(ticket_type, weights)

    ranked: list[RankedTicket] = []
    seen_keys: set[str] = set()

    for raw_ticket in itertools.product(range(10), repeat=4):
        ticket = tuple(map(int, raw_ticket))
        key = ticket_key(ticket, number_select)

        if key in excluded:
            continue

        if number_select in {"any", "boxed"}:
            if key in seen_keys:
                continue

            seen_keys.add(key)

            permutations = set(itertools.permutations(ticket, 4))
            model_score = max(exact_model_score(p, probs) for p in permutations)
            display_ticket = key
        else:
            model_score = exact_model_score(ticket, probs)
            display_ticket = ticket_to_str(ticket)

        frequency_score = float(np.mean([freq[d] for d in ticket]))
        overdue_score = float(np.mean([overdue[d] for d in ticket]))

        ticket_pairs = [tuple(sorted(pair)) for pair in itertools.combinations(ticket, 2)]
        pair_score = float(np.mean([pairs[p] for p in ticket_pairs]))

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
                draw=draw,
                ticket_type=ticket_type,
                number_select=number_select,
                ticket=display_ticket,
                score=score,
                model_score=model_score,
                frequency_score=frequency_score,
                overdue_score=overdue_score,
                pair_score=pair_score,
                balance_score=bal_score,
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)

    for idx, item in enumerate(ranked[:tickets], start=1):
        item.rank = idx

    return ranked[:tickets]


def save_ranked_csv(path: str | Path, rows: list[RankedTicket]) -> None:
    fieldnames = [
        "rank",
        "draw",
        "ticket_type",
        "number_select",
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
                    "draw": r.draw,
                    "ticket_type": r.ticket_type,
                    "number_select": r.number_select,
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
    print("Ranked Idaho Pick 4 ticket candidates")
    print("-" * 106)
    print(
        f"{'Rank':<6}"
        f"{'Draw':<8}"
        f"{'Strategy':<14}"
        f"{'Select':<10}"
        f"{'Ticket':<12}"
        f"{'Score':<14}"
        f"{'Model':<14}"
    )
    print("-" * 106)

    for r in rows:
        print(
            f"{r.rank:<6}"
            f"{r.draw:<8}"
            f"{r.ticket_type:<14}"
            f"{r.number_select:<10}"
            f"{r.ticket:<12}"
            f"{r.score:<14.8f}"
            f"{r.model_score:<14.8f}"
        )

    print("-" * 106)
    print("Reminder: this ranks historical-pattern candidates; lottery results are random.")


def run_for_draw(args, draw_name: str) -> list[RankedTicket]:
    df = load_pick4_csv(args.csv, draw_name)
    digits = digit_rows(df)

    window = min(args.window, max(5, len(digits) // 3))
    x_train, y_train = make_features_and_labels(digits, window=window)

    model = train_model(
        x_train=x_train,
        y_train=y_train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    next_feature = build_next_feature(digits, window=window)
    probs = model_digit_probabilities(model, next_feature)

    return rank_tickets(
        probs=probs,
        digits=digits,
        draw=draw_name,
        ticket_type=args.ticket_type,
        number_select=args.number_select,
        tickets=args.tickets,
        exclude_recent=args.exclude_recent,
        weights=args.weights,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an MLX Pick 4 model and produce ranked ticket candidates."
    )

    parser.add_argument(
        "--csv",
        default="idaho_pick4_history.csv",
        help="CSV output from scrape_idaho_pick4.py. Default: idaho_pick4_history.csv",
    )

    parser.add_argument(
        "--draw",
        choices=["Day", "Night", "both"],
        default="Night",
        help="Draw type to train against. Use 'both' to generate separate Day and Night picks.",
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
        help="Scoring strategy: model, balanced, hot, overdue, hot_overdue.",
    )

    parser.add_argument(
        "--number-select",
        choices=["exact", "any", "boxed"],
        default="exact",
        help="Number selection type: exact, any, boxed. boxed is an alias for any-order grouping.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=40,
        help="Number of past same-draw rows used as model context. Default: 40",
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
        default=0.008,
        help="MLX Adam learning rate. Default: 0.008",
    )

    parser.add_argument(
        "--exclude-recent",
        type=int,
        default=0,
        help="Exclude tickets appearing in the last N filtered draws. Default: 0",
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

    draws = ["Day", "Night"] if args.draw == "both" else [args.draw]

    all_ranked: list[RankedTicket] = []

    for draw_name in draws:
        ranked = run_for_draw(args, draw_name)
        all_ranked.extend(ranked)

    print_ranked(all_ranked)

    if args.output:
        save_ranked_csv(args.output, all_ranked)
        print(f"\nSaved ranked tickets to: {args.output}")


if __name__ == "__main__":
    main()
