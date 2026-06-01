#!/usr/bin/env python3
"""
pick3_mlx_ticket_model.py

Train a small MLX model from Idaho Pick 3 CSV history and generate ranked ticket
candidates for the next Day or Night draw.

Expected CSV format from scrape_idaho_pick3.py:

    Date,Draw,Num1,Num2,Num3
    2026-02-17,Night,1,2,3
    2026-02-18,Day,4,5,6

Important:
    Lottery drawings are random. This script ranks tickets from historical
    patterns; it does not guarantee or truly predict a winning number.

Install:
    pip install mlx pandas numpy

Examples:
    python pick3_mlx_ticket_model.py --csv idaho_pick3_history.csv --draw Night --tickets 5

    python pick3_mlx_ticket_model.py --csv idaho_pick3_history.csv --draw Day --tickets 10 --ticket-type any

    python pick3_mlx_ticket_model.py --csv idaho_pick3_history.csv --draw Night --tickets 5 --ticket-type exact --exclude-recent 30

    python pick3_mlx_ticket_model.py --csv idaho_pick3_history.csv --draw both --tickets 5
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
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


DrawType = Literal["Day", "Night"]
TicketType = Literal["exact", "any", "straight_any"]


@dataclass
class RankedTicket:
    rank: int
    ticket: str
    score: float
    model_score: float
    frequency_score: float
    overdue_score: float
    pair_score: float
    ticket_type: str
    draw: str


class Pick3Model(nn.Module):
    """
    Small neural network that predicts next draw digits from prior window features.

    The model outputs 30 logits:
        - 10 logits for digit position 1
        - 10 logits for digit position 2
        - 10 logits for digit position 3
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 30),
        )

    def __call__(self, x):
        return self.net(x)


def normalize_draw(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("day"):
        return "Day"
    if text.startswith("night"):
        return "Night"
    raise ValueError(f"Unknown draw type: {value!r}")


def load_pick3_csv(csv_path: str | Path, draw: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    required = {"Date", "Draw", "Num1", "Num2", "Num3"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df["Draw"] = df["Draw"].apply(normalize_draw)

    for col in ["Num1", "Num2", "Num3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Num1", "Num2", "Num3"])

    for col in ["Num1", "Num2", "Num3"]:
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
    return df[["Num1", "Num2", "Num3"]].to_numpy(dtype=np.int64)


def make_features_and_labels(digits: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build training samples.

    Feature vector contains:
      - flattened prior digits over the window, scaled 0-1
      - frequency of digits 0-9 in recent window
      - last seen gap for digits 0-9 in recent window
    """
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

        x = np.concatenate([flattened, freq, gaps]).astype(np.float32)
        xs.append(x)
        ys.append(target.astype(np.int64))

    return np.stack(xs), np.stack(ys)


def cross_entropy_for_three_positions(logits, y):
    logits = logits.reshape((-1, 3, 10))
    loss1 = nn.losses.cross_entropy(logits[:, 0, :], y[:, 0], reduction="mean")
    loss2 = nn.losses.cross_entropy(logits[:, 1, :], y[:, 1], reduction="mean")
    loss3 = nn.losses.cross_entropy(logits[:, 2, :], y[:, 2], reduction="mean")
    return (loss1 + loss2 + loss3) / 3.0


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> Pick3Model:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    model = Pick3Model(input_dim=x_train.shape[1])
    optimizer = optim.Adam(learning_rate=learning_rate)

    x_mx = mx.array(x_train)
    y_mx = mx.array(y_train)

    def loss_fn(model, xb, yb):
        logits = model(xb)
        return cross_entropy_for_three_positions(logits, yb)

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

    return np.concatenate([flattened, freq, gaps]).astype(np.float32)


def softmax_np(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)


def model_digit_probabilities(model: Pick3Model, next_feature: np.ndarray) -> np.ndarray:
    logits = model(mx.array(next_feature.reshape(1, -1)))
    logits_np = np.array(logits).reshape(3, 10)
    return np.stack([softmax_np(logits_np[i]) for i in range(3)])


def ticket_to_str(ticket: Iterable[int]) -> str:
    return "".join(str(int(x)) for x in ticket)


def combo_key(ticket: tuple[int, int, int], ticket_type: str) -> str:
    if ticket_type == "any":
        return "".join(sorted(ticket_to_str(ticket)))
    return ticket_to_str(ticket)


def exact_model_score(ticket: tuple[int, int, int], probs: np.ndarray) -> float:
    return float(probs[0, ticket[0]] * probs[1, ticket[1]] * probs[2, ticket[2]])


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
        for pair in [(row[0], row[1]), (row[0], row[2]), (row[1], row[2])]:
            key = tuple(sorted((int(pair[0]), int(pair[1]))))
            pairs[key] = pairs.get(key, 0) + 1

    max_count = max(pairs.values()) if pairs else 1
    scores = {}
    for a in range(10):
        for b in range(10):
            key = tuple(sorted((a, b)))
            scores[key] = pairs.get(key, 0) / max_count
    return scores


def recent_ticket_keys(digits: np.ndarray, recent_count: int, ticket_type: str) -> set[str]:
    if recent_count <= 0:
        return set()
    recent = digits[-recent_count:]
    return {combo_key(tuple(map(int, row)), ticket_type) for row in recent}


def rank_tickets(
    probs: np.ndarray,
    digits: np.ndarray,
    draw: str,
    ticket_type: TicketType,
    tickets: int,
    exclude_recent: int,
    weights: tuple[float, float, float, float],
) -> list[RankedTicket]:
    freq = frequency_scores(digits)
    overdue = overdue_scores(digits)
    pairs = pair_scores(digits)
    excluded = recent_ticket_keys(digits, exclude_recent, ticket_type)

    model_w, freq_w, overdue_w, pair_w = weights

    ranked: list[RankedTicket] = []
    seen_keys: set[str] = set()

    all_tickets = itertools.product(range(10), repeat=3)

    for raw_ticket in all_tickets:
        ticket = tuple(map(int, raw_ticket))
        key = combo_key(ticket, ticket_type)

        if key in excluded:
            continue

        if ticket_type == "any":
            # Deduplicate permutations, e.g. 123, 132, 213 are one Any ticket group.
            if key in seen_keys:
                continue
            seen_keys.add(key)

            permutations = set(itertools.permutations(ticket, 3))
            model_score = max(exact_model_score(p, probs) for p in permutations)
            display_ticket = key
        else:
            model_score = exact_model_score(ticket, probs)
            display_ticket = ticket_to_str(ticket)

        frequency_score = float(np.mean([freq[d] for d in ticket]))
        overdue_score = float(np.mean([overdue[d] for d in ticket]))

        ticket_pairs = [
            tuple(sorted((ticket[0], ticket[1]))),
            tuple(sorted((ticket[0], ticket[2]))),
            tuple(sorted((ticket[1], ticket[2]))),
        ]
        pair_score = float(np.mean([pairs[p] for p in ticket_pairs]))

        score = (
            model_w * model_score
            + freq_w * frequency_score
            + overdue_w * overdue_score
            + pair_w * pair_score
        )

        ranked.append(
            RankedTicket(
                rank=0,
                ticket=display_ticket,
                score=score,
                model_score=model_score,
                frequency_score=frequency_score,
                overdue_score=overdue_score,
                pair_score=pair_score,
                ticket_type=ticket_type,
                draw=draw,
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
        "ticket",
        "score",
        "model_score",
        "frequency_score",
        "overdue_score",
        "pair_score",
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
                    "ticket": r.ticket,
                    "score": f"{r.score:.8f}",
                    "model_score": f"{r.model_score:.8f}",
                    "frequency_score": f"{r.frequency_score:.8f}",
                    "overdue_score": f"{r.overdue_score:.8f}",
                    "pair_score": f"{r.pair_score:.8f}",
                }
            )


def print_ranked(rows: list[RankedTicket]) -> None:
    print()
    print("Ranked ticket candidates")
    print("-" * 76)
    print(f"{'Rank':<6}{'Draw':<8}{'Type':<14}{'Ticket':<10}{'Score':<14}{'Model':<14}")
    print("-" * 76)

    for r in rows:
        print(
            f"{r.rank:<6}"
            f"{r.draw:<8}"
            f"{r.ticket_type:<14}"
            f"{r.ticket:<10}"
            f"{r.score:<14.8f}"
            f"{r.model_score:<14.8f}"
        )

    print("-" * 76)
    print("Reminder: this ranks historical-pattern candidates; lottery results are random.")


def parse_weights(text: str) -> tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Weights must be four comma-separated numbers.")

    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("Weights must sum to a positive number.")

    return tuple(x / total for x in parts)  # type: ignore[return-value]


def run_for_draw(args, draw_name: str) -> list[RankedTicket]:
    df = load_pick3_csv(args.csv, draw_name)
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

    ranked = rank_tickets(
        probs=probs,
        digits=digits,
        draw=draw_name,
        ticket_type=args.ticket_type,
        tickets=args.tickets,
        exclude_recent=args.exclude_recent,
        weights=args.weights,
    )

    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an MLX Pick 3 model and produce ranked ticket candidates."
    )

    parser.add_argument(
        "--csv",
        default="idaho_pick3_history.csv",
        help="CSV output from scrape_idaho_pick3.py. Default: idaho_pick3_history.csv",
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
        choices=["exact", "any", "straight_any"],
        default="exact",
        help=(
            "Ticket style. exact = exact order. any = deduplicated any-order groups. "
            "straight_any = exact-order ranking but labeled for Straight/Any play."
        ),
    )

    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Number of past same-draw rows used as model context. Default: 30",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Training epochs. Default: 250",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="MLX Adam learning rate. Default: 0.01",
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
        default=parse_weights("0.55,0.20,0.15,0.10"),
        help=(
            "Scoring weights as model,frequency,overdue,pair. "
            "Default: 0.55,0.20,0.15,0.10"
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
