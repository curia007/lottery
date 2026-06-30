#!/usr/bin/env python3
"""
mega_millions_mlx_ticket_model.py

Train a small MLX model from Mega Millions CSV history and generate ranked ticket
candidates for the next draw.

Expected CSV format from scrape_mega_millions.py:

    Date,Num1,Num2,Num3,Num4,Num5,MegaBall

Important:
    Lottery drawings are random. This script ranks tickets from historical
    patterns; it does not guarantee or truly predict a winning result.

Install:
    pip install mlx pandas numpy

Examples:
    python mega_millions_mlx_ticket_model.py --csv data/mega_millions_history.csv --tickets 5

    python mega_millions_mlx_ticket_model.py --csv data/mega_millions_history.csv --tickets 10 --ticket-type balanced

    python mega_millions_mlx_ticket_model.py --csv data/mega_millions_history.csv --tickets 10 --ticket-type hot

    python mega_millions_mlx_ticket_model.py --csv data/mega_millions_history.csv --tickets 10 --ticket-type overdue

    python mega_millions_mlx_ticket_model.py --csv data/mega_millions_history.csv --tickets 5 --exclude-recent 20 --output mega_millions_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import re
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


MAIN_MIN = 1
MAIN_MAX = 70
MEGA_MIN = 1
MEGA_MAX = 25
MAIN_COUNT = 5

TicketType = Literal["model", "balanced", "hot", "overdue", "hot_overdue"]

DEFAULT_CSV = "data/mega_millions_history.csv"
DEFAULT_TICKETS = 5
DEFAULT_TICKET_TYPE: TicketType = "balanced"
DEFAULT_WINDOW = 45
DEFAULT_EPOCHS = 350
DEFAULT_LEARNING_RATE = 0.004
DEFAULT_EXCLUDE_RECENT = 0
DEFAULT_POOL_SIZE = 30
DEFAULT_MEGA_POOL_SIZE = 6
DEFAULT_SEED = 42


@dataclass
class RankedTicket:
    rank: int
    ticket: str
    main_numbers: str
    mega_ball: int
    score: float
    model_score: float
    frequency_score: float
    overdue_score: float
    pair_score: float
    balance_score: float
    mega_score: float
    ticket_type: str


class MegaMillionsModel(nn.Module):
    """
    Small neural network that predicts:
      - 70 main-number logits, one for each main number 1-70
      - 25 Mega Ball logits, one for each Mega Ball number 1-25
    """

    def __init__(self, input_dim: int, hidden_dim: int = 160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, MAIN_MAX + MEGA_MAX),
        )

    def __call__(self, x):
        return self.net(x)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept likely CSV formats from the scraper and convert to:
        Date, Num1, Num2, Num3, Num4, Num5, MegaBall
    """
    df = df.copy()
    cols_lower = {c.lower().strip().replace(" ", "").replace("_", ""): c for c in df.columns}

    preferred = ["Date", "Num1", "Num2", "Num3", "Num4", "Num5", "MegaBall"]
    if all(c in df.columns for c in preferred):
        return df[preferred].copy()

    date_col = cols_lower.get("date") or cols_lower.get("drawdate")
    mega_col = (
        cols_lower.get("megaball")
        or cols_lower.get("mega")
        or cols_lower.get("meganumber")
        or cols_lower.get("megaballnumber")
        or cols_lower.get("mb")
    )

    num_cols = []
    for i in range(1, 6):
        col = (
            cols_lower.get(f"num{i}")
            or cols_lower.get(f"n{i}")
            or cols_lower.get(f"number{i}")
            or cols_lower.get(f"ball{i}")
        )
        if col:
            num_cols.append(col)

    if date_col and mega_col and len(num_cols) == 5:
        out = df[[date_col] + num_cols + [mega_col]].copy()
        out.columns = preferred
        return out

    winning_col = (
        cols_lower.get("winningnumbers")
        or cols_lower.get("numbers")
        or cols_lower.get("winningnumber")
        or cols_lower.get("result")
        or cols_lower.get("results")
    )

    if date_col and winning_col and mega_col:
        rows = []
        for _, row in df.iterrows():
            main_nums = extract_ints(row[winning_col])
            mega_nums = extract_ints(row[mega_col])

            if len(main_nums) >= 5 and len(mega_nums) >= 1:
                rows.append(
                    {
                        "Date": row[date_col],
                        "Num1": main_nums[0],
                        "Num2": main_nums[1],
                        "Num3": main_nums[2],
                        "Num4": main_nums[3],
                        "Num5": main_nums[4],
                        "MegaBall": mega_nums[0],
                    }
                )

        return pd.DataFrame(rows, columns=preferred)

    if date_col:
        rows = []
        for _, row in df.iterrows():
            values = []
            for c in df.columns:
                if c == date_col:
                    continue
                values.extend(extract_ints(row[c]))

            if len(values) >= 6:
                rows.append(
                    {
                        "Date": row[date_col],
                        "Num1": values[0],
                        "Num2": values[1],
                        "Num3": values[2],
                        "Num4": values[3],
                        "Num5": values[4],
                        "MegaBall": values[5],
                    }
                )

        if rows:
            return pd.DataFrame(rows, columns=preferred)

    raise ValueError(
        "Could not identify Mega Millions columns. Expected either "
        "Date,Num1,Num2,Num3,Num4,Num5,MegaBall or a similar format."
    )


def extract_ints(value) -> list[int]:
    return [int(x) for x in re.findall(r"\b\d{1,2}\b", str(value))]


def load_mega_millions_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    raw = pd.read_csv(path)
    df = normalize_columns(raw)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["Num1", "Num2", "Num3", "Num4", "Num5", "MegaBall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Num1", "Num2", "Num3", "Num4", "Num5", "MegaBall"])

    for col in ["Num1", "Num2", "Num3", "Num4", "Num5"]:
        df[col] = df[col].astype(int)
        bad = ~df[col].between(MAIN_MIN, MAIN_MAX)
        if bad.any():
            raise ValueError(f"Column {col} contains values outside {MAIN_MIN}-{MAIN_MAX}.")

    df["MegaBall"] = df["MegaBall"].astype(int)
    bad_mega = ~df["MegaBall"].between(MEGA_MIN, MEGA_MAX)
    if bad_mega.any():
        raise ValueError(f"MegaBall contains values outside {MEGA_MIN}-{MEGA_MAX}.")

    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 25:
        raise ValueError(f"Need at least 25 rows to train. Found {len(df)}.")

    return df


def main_rows(df: pd.DataFrame) -> np.ndarray:
    values = df[["Num1", "Num2", "Num3", "Num4", "Num5"]].to_numpy(dtype=np.int64)
    return np.sort(values, axis=1)


def mega_rows(df: pd.DataFrame) -> np.ndarray:
    return df["MegaBall"].to_numpy(dtype=np.int64)


def to_main_multihot(row: Iterable[int]) -> np.ndarray:
    y = np.zeros(MAIN_MAX, dtype=np.float32)
    for n in row:
        y[int(n) - 1] = 1.0
    return y


def to_mega_onehot(mega_ball: int) -> np.ndarray:
    y = np.zeros(MEGA_MAX, dtype=np.float32)
    y[int(mega_ball) - 1] = 1.0
    return y


def make_features_and_labels(
    main_numbers: np.ndarray,
    mega_balls: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Features:
      - Recent main numbers flattened and scaled.
      - Recent Mega Balls scaled.
      - Main number frequency over window.
      - Main number gap / overdue over window.
      - Mega Ball frequency over window.
      - Mega Ball gap / overdue over window.
      - Last draw main multi-hot.
      - Last Mega Ball one-hot.

    Labels:
      - Main numbers as 70-way multi-hot.
      - Mega Ball as 25-way one-hot.
    """
    xs: list[np.ndarray] = []
    y_main: list[np.ndarray] = []
    y_mega: list[np.ndarray] = []

    for i in range(window, len(main_numbers)):
        recent_main = main_numbers[i - window : i]
        recent_mega = mega_balls[i - window : i]
        target_main = main_numbers[i]
        target_mega = mega_balls[i]

        flattened_main = recent_main.flatten().astype(np.float32) / MAIN_MAX
        flattened_mega = recent_mega.astype(np.float32) / MEGA_MAX

        main_freq = np.zeros(MAIN_MAX, dtype=np.float32)
        for n in recent_main.flatten():
            main_freq[int(n) - 1] += 1.0
        main_freq = main_freq / max(1, recent_main.size)

        flat_recent_main = recent_main.flatten()
        main_gaps = np.ones(MAIN_MAX, dtype=np.float32)
        for n in range(1, MAIN_MAX + 1):
            locations = np.where(flat_recent_main == n)[0]
            if len(locations) == 0:
                main_gaps[n - 1] = 1.0
            else:
                main_gaps[n - 1] = (len(flat_recent_main) - 1 - locations[-1]) / max(1, len(flat_recent_main))

        mega_freq = np.zeros(MEGA_MAX, dtype=np.float32)
        for m in recent_mega:
            mega_freq[int(m) - 1] += 1.0
        mega_freq = mega_freq / max(1, len(recent_mega))

        mega_gaps = np.ones(MEGA_MAX, dtype=np.float32)
        for m in range(1, MEGA_MAX + 1):
            locations = np.where(recent_mega == m)[0]
            if len(locations) == 0:
                mega_gaps[m - 1] = 1.0
            else:
                mega_gaps[m - 1] = (len(recent_mega) - 1 - locations[-1]) / max(1, len(recent_mega))

        last_main = to_main_multihot(recent_main[-1])
        last_mega = to_mega_onehot(int(recent_mega[-1]))

        x = np.concatenate(
            [
                flattened_main,
                flattened_mega,
                main_freq,
                main_gaps,
                mega_freq,
                mega_gaps,
                last_main,
                last_mega,
            ]
        ).astype(np.float32)

        xs.append(x)
        y_main.append(to_main_multihot(target_main))
        y_mega.append(to_mega_onehot(int(target_mega)))

    return np.stack(xs), np.stack(y_main), np.stack(y_mega)


def binary_cross_entropy_with_logits(logits, targets):
    return mx.mean(mx.maximum(logits, 0) - logits * targets + mx.log1p(mx.exp(-mx.abs(logits))))


def softmax_cross_entropy(logits, targets):
    return nn.losses.cross_entropy(logits, mx.argmax(targets, axis=1), reduction="mean")


def train_model(
    x_train: np.ndarray,
    y_main_train: np.ndarray,
    y_mega_train: np.ndarray,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> MegaMillionsModel:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    model = MegaMillionsModel(input_dim=x_train.shape[1])
    optimizer = optim.Adam(learning_rate=learning_rate)

    x_mx = mx.array(x_train)
    y_main_mx = mx.array(y_main_train)
    y_mega_mx = mx.array(y_mega_train)

    def loss_fn(model, xb, yb_main, yb_mega):
        logits = model(xb)
        main_logits = logits[:, :MAIN_MAX]
        mega_logits = logits[:, MAIN_MAX:]
        main_loss = binary_cross_entropy_with_logits(main_logits, yb_main)
        mega_loss = softmax_cross_entropy(mega_logits, yb_mega)
        return main_loss + 0.35 * mega_loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for _ in range(epochs):
        loss, grads = loss_and_grad_fn(model, x_mx, y_main_mx, y_mega_mx)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    return model


def build_next_feature(main_numbers: np.ndarray, mega_balls: np.ndarray, window: int) -> np.ndarray:
    if len(main_numbers) < window:
        window = len(main_numbers)

    recent_main = main_numbers[-window:]
    recent_mega = mega_balls[-window:]

    flattened_main = recent_main.flatten().astype(np.float32) / MAIN_MAX
    flattened_mega = recent_mega.astype(np.float32) / MEGA_MAX

    main_freq = np.zeros(MAIN_MAX, dtype=np.float32)
    for n in recent_main.flatten():
        main_freq[int(n) - 1] += 1.0
    main_freq = main_freq / max(1, recent_main.size)

    flat_recent_main = recent_main.flatten()
    main_gaps = np.ones(MAIN_MAX, dtype=np.float32)
    for n in range(1, MAIN_MAX + 1):
        locations = np.where(flat_recent_main == n)[0]
        if len(locations) == 0:
            main_gaps[n - 1] = 1.0
        else:
            main_gaps[n - 1] = (len(flat_recent_main) - 1 - locations[-1]) / max(1, len(flat_recent_main))

    mega_freq = np.zeros(MEGA_MAX, dtype=np.float32)
    for m in recent_mega:
        mega_freq[int(m) - 1] += 1.0
    mega_freq = mega_freq / max(1, len(recent_mega))

    mega_gaps = np.ones(MEGA_MAX, dtype=np.float32)
    for m in range(1, MEGA_MAX + 1):
        locations = np.where(recent_mega == m)[0]
        if len(locations) == 0:
            mega_gaps[m - 1] = 1.0
        else:
            mega_gaps[m - 1] = (len(recent_mega) - 1 - locations[-1]) / max(1, len(recent_mega))

    last_main = to_main_multihot(recent_main[-1])
    last_mega = to_mega_onehot(int(recent_mega[-1]))

    return np.concatenate(
        [
            flattened_main,
            flattened_mega,
            main_freq,
            main_gaps,
            mega_freq,
            mega_gaps,
            last_main,
            last_mega,
        ]
    ).astype(np.float32)


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def softmax_np(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)


def model_probabilities(model: MegaMillionsModel, next_feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = model(mx.array(next_feature.reshape(1, -1)))
    logits_np = np.array(logits).reshape(MAIN_MAX + MEGA_MAX)
    main_probs = sigmoid_np(logits_np[:MAIN_MAX])
    mega_probs = softmax_np(logits_np[MAIN_MAX:])
    return main_probs, mega_probs


def frequency_scores_main(main_numbers: np.ndarray) -> dict[int, float]:
    counts = np.zeros(MAIN_MAX, dtype=np.float64)
    for n in main_numbers.flatten():
        counts[int(n) - 1] += 1.0
    counts += 1.0
    counts = counts / counts.sum()
    return {n: float(counts[n - 1]) for n in range(1, MAIN_MAX + 1)}


def frequency_scores_mega(mega_balls: np.ndarray) -> dict[int, float]:
    counts = np.zeros(MEGA_MAX, dtype=np.float64)
    for m in mega_balls:
        counts[int(m) - 1] += 1.0
    counts += 1.0
    counts = counts / counts.sum()
    return {m: float(counts[m - 1]) for m in range(1, MEGA_MAX + 1)}


def overdue_scores_main(main_numbers: np.ndarray) -> dict[int, float]:
    flat = main_numbers.flatten()
    raw = {}
    for n in range(1, MAIN_MAX + 1):
        positions = np.where(flat == n)[0]
        gap = len(flat) if len(positions) == 0 else len(flat) - 1 - positions[-1]
        raw[n] = float(gap)
    max_gap = max(raw.values()) or 1.0
    return {n: raw[n] / max_gap for n in range(1, MAIN_MAX + 1)}


def overdue_scores_mega(mega_balls: np.ndarray) -> dict[int, float]:
    raw = {}
    for m in range(1, MEGA_MAX + 1):
        positions = np.where(mega_balls == m)[0]
        gap = len(mega_balls) if len(positions) == 0 else len(mega_balls) - 1 - positions[-1]
        raw[m] = float(gap)
    max_gap = max(raw.values()) or 1.0
    return {m: raw[m] / max_gap for m in range(1, MEGA_MAX + 1)}


def pair_scores(main_numbers: np.ndarray) -> dict[tuple[int, int], float]:
    pairs: dict[tuple[int, int], int] = {}
    for row in main_numbers:
        for pair in itertools.combinations(sorted(map(int, row)), 2):
            pairs[pair] = pairs.get(pair, 0) + 1
    max_count = max(pairs.values()) if pairs else 1

    scores = {}
    for a in range(1, MAIN_MAX + 1):
        for b in range(a + 1, MAIN_MAX + 1):
            scores[(a, b)] = pairs.get((a, b), 0) / max_count

    return scores


def main_ticket_to_str(ticket: Iterable[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(ticket))


def full_ticket_to_str(main_ticket: Iterable[int], mega_ball: int) -> str:
    return f"{main_ticket_to_str(main_ticket)} | Mega Ball {int(mega_ball):02d}"


def recent_ticket_keys(main_numbers: np.ndarray, mega_balls: np.ndarray, recent_count: int) -> set[tuple[tuple[int, ...], int]]:
    if recent_count <= 0:
        return set()

    recent_main = main_numbers[-recent_count:]
    recent_mega = mega_balls[-recent_count:]
    return {
        (tuple(sorted(map(int, row))), int(mega))
        for row, mega in zip(recent_main, recent_mega)
    }


def balance_score(ticket: tuple[int, int, int, int, int]) -> float:
    nums = sorted(ticket)

    low_count = sum(1 for n in nums if n <= 35)
    odd_count = sum(1 for n in nums if n % 2 == 1)

    low_high = 1.0 - abs(low_count - 2.5) / 2.5
    odd_even = 1.0 - abs(odd_count - 2.5) / 2.5

    total = sum(nums)
    sum_center = 177
    sum_score = max(0.0, 1.0 - abs(total - sum_center) / 160.0)

    spread = nums[-1] - nums[0]
    spread_score = min(1.0, spread / 50.0)

    return float((low_high + odd_even + sum_score + spread_score) / 4.0)


def parse_weights(text: str) -> tuple[float, float, float, float, float, float]:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "Weights must be six comma-separated numbers: model,frequency,overdue,pair,balance,mega"
        )

    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("Weights must sum to a positive number.")

    return tuple(x / total for x in parts)  # type: ignore[return-value]


def get_weights(ticket_type: TicketType, custom_weights: tuple[float, float, float, float, float, float] | None):
    if custom_weights:
        return custom_weights

    defaults = {
        "model": (0.58, 0.12, 0.08, 0.08, 0.04, 0.10),
        "balanced": (0.35, 0.12, 0.13, 0.08, 0.22, 0.10),
        "hot": (0.25, 0.42, 0.04, 0.17, 0.02, 0.10),
        "overdue": (0.25, 0.04, 0.45, 0.06, 0.10, 0.10),
        "hot_overdue": (0.34, 0.24, 0.24, 0.08, 0.02, 0.08),
    }

    return defaults[ticket_type]


def candidate_pool(
    main_probs: np.ndarray,
    freq: dict[int, float],
    overdue: dict[int, float],
    pool_size: int,
    ticket_type: TicketType,
) -> list[int]:
    base = []

    for n in range(1, MAIN_MAX + 1):
        if ticket_type == "hot":
            score = 0.30 * main_probs[n - 1] + 0.60 * freq[n] + 0.10 * overdue[n]
        elif ticket_type == "overdue":
            score = 0.30 * main_probs[n - 1] + 0.10 * freq[n] + 0.60 * overdue[n]
        elif ticket_type == "hot_overdue":
            score = 0.40 * main_probs[n - 1] + 0.30 * freq[n] + 0.30 * overdue[n]
        elif ticket_type == "balanced":
            score = 0.55 * main_probs[n - 1] + 0.20 * freq[n] + 0.25 * overdue[n]
        else:
            score = 0.70 * main_probs[n - 1] + 0.20 * freq[n] + 0.10 * overdue[n]

        base.append((n, score))

    base.sort(key=lambda x: x[1], reverse=True)
    pool_size = max(MAIN_COUNT, min(pool_size, MAIN_MAX))
    return [n for n, _ in base[:pool_size]]


def candidate_mega_balls(
    mega_probs: np.ndarray,
    mega_freq: dict[int, float],
    mega_overdue: dict[int, float],
    count: int,
    ticket_type: TicketType,
) -> list[int]:
    base = []

    for m in range(1, MEGA_MAX + 1):
        if ticket_type == "hot":
            score = 0.35 * mega_probs[m - 1] + 0.55 * mega_freq[m] + 0.10 * mega_overdue[m]
        elif ticket_type == "overdue":
            score = 0.35 * mega_probs[m - 1] + 0.10 * mega_freq[m] + 0.55 * mega_overdue[m]
        elif ticket_type == "hot_overdue":
            score = 0.40 * mega_probs[m - 1] + 0.30 * mega_freq[m] + 0.30 * mega_overdue[m]
        else:
            score = 0.60 * mega_probs[m - 1] + 0.20 * mega_freq[m] + 0.20 * mega_overdue[m]

        base.append((m, score))

    base.sort(key=lambda x: x[1], reverse=True)
    count = max(1, min(count, MEGA_MAX))
    return [m for m, _ in base[:count]]


def rank_tickets(
    main_probs: np.ndarray,
    mega_probs: np.ndarray,
    main_numbers: np.ndarray,
    mega_balls: np.ndarray,
    ticket_type: TicketType,
    tickets: int,
    exclude_recent: int,
    pool_size: int,
    mega_pool_size: int,
    custom_weights: tuple[float, float, float, float, float, float] | None,
) -> list[RankedTicket]:
    freq = frequency_scores_main(main_numbers)
    overdue = overdue_scores_main(main_numbers)
    mega_freq = frequency_scores_mega(mega_balls)
    mega_overdue = overdue_scores_mega(mega_balls)
    pairs = pair_scores(main_numbers)
    excluded = recent_ticket_keys(main_numbers, mega_balls, exclude_recent)

    model_w, freq_w, overdue_w, pair_w, balance_w, mega_w = get_weights(ticket_type, custom_weights)

    pool = candidate_pool(
        main_probs=main_probs,
        freq=freq,
        overdue=overdue,
        pool_size=pool_size,
        ticket_type=ticket_type,
    )

    mega_pool = candidate_mega_balls(
        mega_probs=mega_probs,
        mega_freq=mega_freq,
        mega_overdue=mega_overdue,
        count=mega_pool_size,
        ticket_type=ticket_type,
    )

    ranked: list[RankedTicket] = []

    for main_ticket in itertools.combinations(sorted(pool), MAIN_COUNT):
        main_key = tuple(sorted(main_ticket))

        model_score = float(np.mean([main_probs[n - 1] for n in main_ticket]))
        frequency_score = float(np.mean([freq[n] for n in main_ticket]))
        overdue_score = float(np.mean([overdue[n] for n in main_ticket]))

        ticket_pairs = list(itertools.combinations(sorted(main_ticket), 2))
        pair_score = float(np.mean([pairs.get(pair, 0.0) for pair in ticket_pairs]))

        bal_score = balance_score(main_ticket)

        for mega_ball in mega_pool:
            ticket_key = (main_key, int(mega_ball))
            if ticket_key in excluded:
                continue

            mega_score = float(
                0.60 * mega_probs[mega_ball - 1]
                + 0.20 * mega_freq[mega_ball]
                + 0.20 * mega_overdue[mega_ball]
            )

            score = (
                model_w * model_score
                + freq_w * frequency_score
                + overdue_w * overdue_score
                + pair_w * pair_score
                + balance_w * bal_score
                + mega_w * mega_score
            )

            ranked.append(
                RankedTicket(
                    rank=0,
                    ticket=full_ticket_to_str(main_ticket, mega_ball),
                    main_numbers=main_ticket_to_str(main_ticket),
                    mega_ball=int(mega_ball),
                    score=score,
                    model_score=model_score,
                    frequency_score=frequency_score,
                    overdue_score=overdue_score,
                    pair_score=pair_score,
                    balance_score=bal_score,
                    mega_score=mega_score,
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
        "main_numbers",
        "mega_ball",
        "score",
        "model_score",
        "frequency_score",
        "overdue_score",
        "pair_score",
        "balance_score",
        "mega_score",
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
                    "main_numbers": r.main_numbers,
                    "mega_ball": r.mega_ball,
                    "score": f"{r.score:.8f}",
                    "model_score": f"{r.model_score:.8f}",
                    "frequency_score": f"{r.frequency_score:.8f}",
                    "overdue_score": f"{r.overdue_score:.8f}",
                    "pair_score": f"{r.pair_score:.8f}",
                    "balance_score": f"{r.balance_score:.8f}",
                    "mega_score": f"{r.mega_score:.8f}",
                }
            )


def print_ranked(rows: list[RankedTicket]) -> None:
    print()
    print("Ranked Mega Millions ticket candidates")
    print("-" * 104)
    print(
        f"{'Rank':<6}"
        f"{'Type':<14}"
        f"{'Ticket':<34}"
        f"{'Score':<14}"
        f"{'Model':<14}"
        f"{'Mega':<14}"
    )
    print("-" * 104)

    for r in rows:
        print(
            f"{r.rank:<6}"
            f"{r.ticket_type:<14}"
            f"{r.ticket:<34}"
            f"{r.score:<14.8f}"
            f"{r.model_score:<14.8f}"
            f"{r.mega_score:<14.8f}"
        )

    print("-" * 104)
    print("Reminder: this ranks historical-pattern candidates; lottery results are random.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an MLX Mega Millions model and produce ranked ticket candidates."
    )

    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="CSV output from scrape_mega_millions.py. Default: data/mega_millions_history.csv",
    )

    parser.add_argument(
        "--tickets",
        "--top",
        dest="tickets",
        type=int,
        default=DEFAULT_TICKETS,
        help="Number of ranked tickets to return. Default: 5",
    )

    parser.add_argument(
        "--ticket-type",
        choices=["model", "balanced", "hot", "overdue", "hot_overdue"],
        default=DEFAULT_TICKET_TYPE,
        help="Ticket style: model, balanced, hot, overdue, hot_overdue. Default: balanced",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Number of past rows used as model context. Default: 45",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Training epochs. Default: 350",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="MLX Adam learning rate. Default: 0.004",
    )

    parser.add_argument(
        "--exclude-recent",
        type=int,
        default=DEFAULT_EXCLUDE_RECENT,
        help="Exclude tickets that exactly appeared in the last N draws. Default: 0",
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help=(
            "How many top candidate main numbers to combine into tickets. "
            "Higher is broader but slower. Default: 30"
        ),
    )

    parser.add_argument(
        "--mega-pool-size",
        type=int,
        default=DEFAULT_MEGA_POOL_SIZE,
        help="How many top Mega Ball candidates to combine with main tickets. Default: 6",
    )

    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=None,
        help=(
            "Optional custom weights as model,frequency,overdue,pair,balance,mega. "
            "Example: --weights 0.50,0.15,0.10,0.10,0.05,0.10"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible training. Default: 42",
    )

    parser.add_argument(
        "--output",
        help="Optional CSV file to save ranked output.",
    )

    args = parser.parse_args()

    if args.tickets < 1:
        raise SystemExit("--tickets / --top must be at least 1.")

    if args.pool_size < MAIN_COUNT:
        raise SystemExit(f"--pool-size must be at least {MAIN_COUNT}.")

    if args.mega_pool_size < 1:
        raise SystemExit("--mega-pool-size must be at least 1.")

    df = load_mega_millions_csv(args.csv)
    mains = main_rows(df)
    megas = mega_rows(df)

    window = min(args.window, max(5, len(mains) // 3))
    x_train, y_main_train, y_mega_train = make_features_and_labels(
        main_numbers=mains,
        mega_balls=megas,
        window=window,
    )

    model = train_model(
        x_train=x_train,
        y_main_train=y_main_train,
        y_mega_train=y_mega_train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    next_feature = build_next_feature(mains, megas, window=window)
    main_probs, mega_probs = model_probabilities(model, next_feature)

    ranked = rank_tickets(
        main_probs=main_probs,
        mega_probs=mega_probs,
        main_numbers=mains,
        mega_balls=megas,
        ticket_type=args.ticket_type,
        tickets=args.tickets,
        exclude_recent=args.exclude_recent,
        pool_size=args.pool_size,
        mega_pool_size=args.mega_pool_size,
        custom_weights=args.weights,
    )

    print_ranked(ranked)

    if args.output:
        save_ranked_csv(args.output, ranked)
        print(f"\nSaved ranked tickets to: {args.output}")


if __name__ == "__main__":
    main()
