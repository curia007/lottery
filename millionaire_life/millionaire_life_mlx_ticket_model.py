#!/usr/bin/env python3
"""
millionaire_life_mlx_ticket_model.py

Train a small MLX model from Millionaire for Life CSV history and generate
ranked ticket candidates for the next draw.

Expected CSV format from scrape_millionaire_life.py:

    Date,Num1,Num2,Num3,Num4,Num5,Extra,WinningNumbers

The script supports flexible game shapes:
    - Main numbers are read from Num1..Num5 by default.
    - Extra is optional and used if present.
    - Number ranges are inferred from the CSV unless supplied by arguments.

Important:
    Lottery drawings are random. This script ranks tickets from historical
    patterns; it does not guarantee or truly predict a winning result.

Install:
    pip install mlx pandas numpy

Examples:
    python millionaire_life_mlx_ticket_model.py --csv millionaire_life_history.csv --tickets 5

    python millionaire_life_mlx_ticket_model.py --csv millionaire_life_history.csv --tickets 10 --ticket-type balanced

    python millionaire_life_mlx_ticket_model.py --csv millionaire_life_history.csv --tickets 10 --ticket-type hot

    python millionaire_life_mlx_ticket_model.py --csv millionaire_life_history.csv --tickets 10 --ticket-type overdue

    python millionaire_life_mlx_ticket_model.py --csv millionaire_life_history.csv --tickets 5 --include-extra

    python millionaire_life_mlx_ticket_model.py --csv millionaire_life_history.csv --tickets 20 --output millionaire_life_predictions.csv
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


TicketType = Literal["model", "balanced", "hot", "overdue", "hot_overdue"]


@dataclass
class GameConfig:
    main_min: int
    main_max: int
    main_count: int
    extra_min: int | None
    extra_max: int | None
    include_extra: bool


@dataclass
class RankedTicket:
    rank: int
    ticket: str
    main_numbers: str
    extra: int | None
    score: float
    model_score: float
    frequency_score: float
    overdue_score: float
    pair_score: float
    balance_score: float
    extra_score: float
    ticket_type: str


class MillionaireLifeModel(nn.Module):
    """
    Small neural network that predicts:
      - main-number logits, one for each number in main_min..main_max
      - optional Extra logits, one for each number in extra_min..extra_max
    """

    def __init__(self, input_dim: int, main_dim: int, extra_dim: int = 0, hidden_dim: int = 144):
        super().__init__()
        self.main_dim = main_dim
        self.extra_dim = extra_dim
        self.output_dim = main_dim + extra_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def __call__(self, x):
        return self.net(x)


def extract_ints(value) -> list[int]:
    return [int(x) for x in re.findall(r"\b\d{1,3}\b", str(value))]


def infer_main_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["Num1", "Num2", "Num3", "Num4", "Num5"]
    existing = [c for c in preferred if c in df.columns]

    if existing:
        return existing

    lower_map = {c.lower().strip().replace(" ", "").replace("_", ""): c for c in df.columns}
    cols = []
    for i in range(1, 11):
        c = (
            lower_map.get(f"num{i}")
            or lower_map.get(f"n{i}")
            or lower_map.get(f"number{i}")
            or lower_map.get(f"ball{i}")
        )
        if c:
            cols.append(c)

    if cols:
        return cols[:5]

    if "WinningNumbers" in df.columns:
        return []

    raise ValueError("Could not find Num1..Num5 columns or WinningNumbers column.")


def load_millionaire_life_csv(
    csv_path: str | Path,
    include_extra: bool,
    main_min: int | None,
    main_max: int | None,
    extra_min: int | None,
    extra_max: int | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, GameConfig]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if "Date" not in df.columns:
        raise ValueError("CSV must contain a Date column.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    main_cols = infer_main_columns(df)

    rows_main: list[list[int]] = []
    rows_extra: list[int | None] = []

    if main_cols:
        for col in main_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        maybe_extra_col = "Extra" if "Extra" in df.columns else None
        if maybe_extra_col:
            df[maybe_extra_col] = pd.to_numeric(df[maybe_extra_col], errors="coerce")

        for _, row in df.iterrows():
            nums = []
            for col in main_cols:
                if pd.notna(row[col]):
                    nums.append(int(row[col]))

            if len(nums) < 2:
                continue

            # Keep first five main columns as the main draw.
            nums = nums[:5]
            rows_main.append(sorted(nums))

            if include_extra and maybe_extra_col and pd.notna(row[maybe_extra_col]):
                rows_extra.append(int(row[maybe_extra_col]))
            else:
                rows_extra.append(None)

    elif "WinningNumbers" in df.columns:
        for _, row in df.iterrows():
            nums = extract_ints(row["WinningNumbers"])
            if len(nums) < 2:
                continue

            rows_main.append(sorted(nums[:5]))

            if include_extra and len(nums) >= 6:
                rows_extra.append(nums[5])
            else:
                rows_extra.append(None)

    if not rows_main:
        raise ValueError("No usable winning number rows found.")

    main_count = max(len(r) for r in rows_main)
    rows_main = [r[:main_count] for r in rows_main if len(r) >= main_count]

    main_array = np.array(rows_main, dtype=np.int64)

    if main_min is None:
        main_min = int(np.min(main_array))
    if main_max is None:
        main_max = int(np.max(main_array))

    if main_min < 0:
        raise ValueError("Main number minimum cannot be negative.")

    bad_main = (main_array < main_min) | (main_array > main_max)
    if bad_main.any():
        raise ValueError(f"Main numbers outside configured range {main_min}-{main_max}.")

    extra_array: np.ndarray | None = None

    usable_extras = [e for e in rows_extra if e is not None]
    if include_extra and usable_extras:
        extra_array = np.array([e if e is not None else usable_extras[-1] for e in rows_extra], dtype=np.int64)

        if extra_min is None:
            extra_min = int(np.min(extra_array))
        if extra_max is None:
            extra_max = int(np.max(extra_array))

        bad_extra = (extra_array < extra_min) | (extra_array > extra_max)
        if bad_extra.any():
            raise ValueError(f"Extra numbers outside configured range {extra_min}-{extra_max}.")
    else:
        include_extra = False
        extra_min = None
        extra_max = None

    df = df.sort_values("Date").reset_index(drop=True)

    if len(main_array) < 25:
        raise ValueError(f"Need at least 25 rows to train. Found {len(main_array)}.")

    config = GameConfig(
        main_min=main_min,
        main_max=main_max,
        main_count=main_count,
        extra_min=extra_min,
        extra_max=extra_max,
        include_extra=include_extra,
    )

    return df, main_array, extra_array, config


def main_dim(config: GameConfig) -> int:
    return config.main_max - config.main_min + 1


def extra_dim(config: GameConfig) -> int:
    if not config.include_extra or config.extra_min is None or config.extra_max is None:
        return 0
    return config.extra_max - config.extra_min + 1


def main_index(n: int, config: GameConfig) -> int:
    return int(n) - config.main_min


def extra_index(n: int, config: GameConfig) -> int:
    if config.extra_min is None:
        raise ValueError("Extra range is not configured.")
    return int(n) - config.extra_min


def to_main_multihot(row: Iterable[int], config: GameConfig) -> np.ndarray:
    y = np.zeros(main_dim(config), dtype=np.float32)
    for n in row:
        y[main_index(int(n), config)] = 1.0
    return y


def to_extra_onehot(extra: int, config: GameConfig) -> np.ndarray:
    y = np.zeros(extra_dim(config), dtype=np.float32)
    y[extra_index(extra, config)] = 1.0
    return y


def make_features_and_labels(
    numbers: np.ndarray,
    extras: np.ndarray | None,
    window: int,
    config: GameConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    xs: list[np.ndarray] = []
    y_main: list[np.ndarray] = []
    y_extra: list[np.ndarray] = []

    m_dim = main_dim(config)
    e_dim = extra_dim(config)

    for i in range(window, len(numbers)):
        recent = numbers[i - window : i]
        target = numbers[i]

        flattened = recent.flatten().astype(np.float32) / max(1, config.main_max)

        freq = np.zeros(m_dim, dtype=np.float32)
        for n in recent.flatten():
            freq[main_index(int(n), config)] += 1.0
        freq = freq / max(1, recent.size)

        flat_recent = recent.flatten()
        gaps = np.ones(m_dim, dtype=np.float32)
        for n in range(config.main_min, config.main_max + 1):
            locations = np.where(flat_recent == n)[0]
            idx = main_index(n, config)
            if len(locations) == 0:
                gaps[idx] = 1.0
            else:
                gaps[idx] = (len(flat_recent) - 1 - locations[-1]) / max(1, len(flat_recent))

        last_draw = to_main_multihot(recent[-1], config)

        feature_parts = [flattened, freq, gaps, last_draw]

        if config.include_extra and extras is not None and e_dim > 0:
            recent_extra = extras[i - window : i]
            flat_extra = recent_extra.astype(np.float32) / max(1, config.extra_max or 1)

            extra_freq = np.zeros(e_dim, dtype=np.float32)
            for e in recent_extra:
                extra_freq[extra_index(int(e), config)] += 1.0
            extra_freq = extra_freq / max(1, len(recent_extra))

            extra_gaps = np.ones(e_dim, dtype=np.float32)
            for e in range(config.extra_min or 1, (config.extra_max or 1) + 1):
                locations = np.where(recent_extra == e)[0]
                idx = extra_index(e, config)
                if len(locations) == 0:
                    extra_gaps[idx] = 1.0
                else:
                    extra_gaps[idx] = (len(recent_extra) - 1 - locations[-1]) / max(1, len(recent_extra))

            last_extra = to_extra_onehot(int(recent_extra[-1]), config)
            feature_parts.extend([flat_extra, extra_freq, extra_gaps, last_extra])

            y_extra.append(to_extra_onehot(int(extras[i]), config))

        x = np.concatenate(feature_parts).astype(np.float32)
        xs.append(x)
        y_main.append(to_main_multihot(target, config))

    if config.include_extra and extras is not None and e_dim > 0:
        return np.stack(xs), np.stack(y_main), np.stack(y_extra)

    return np.stack(xs), np.stack(y_main), None


def binary_cross_entropy_with_logits(logits, targets):
    return mx.mean(mx.maximum(logits, 0) - logits * targets + mx.log1p(mx.exp(-mx.abs(logits))))


def softmax_cross_entropy(logits, targets):
    return nn.losses.cross_entropy(logits, mx.argmax(targets, axis=1), reduction="mean")


def train_model(
    x_train: np.ndarray,
    y_main_train: np.ndarray,
    y_extra_train: np.ndarray | None,
    config: GameConfig,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> MillionaireLifeModel:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    model = MillionaireLifeModel(
        input_dim=x_train.shape[1],
        main_dim=main_dim(config),
        extra_dim=extra_dim(config),
    )
    optimizer = optim.Adam(learning_rate=learning_rate)

    x_mx = mx.array(x_train)
    y_main_mx = mx.array(y_main_train)
    y_extra_mx = mx.array(y_extra_train) if y_extra_train is not None else None

    def loss_fn(model, xb, yb_main, yb_extra):
        logits = model(xb)
        main_logits = logits[:, : main_dim(config)]
        main_loss = binary_cross_entropy_with_logits(main_logits, yb_main)

        if config.include_extra and yb_extra is not None and extra_dim(config) > 0:
            extra_logits = logits[:, main_dim(config) :]
            extra_loss = softmax_cross_entropy(extra_logits, yb_extra)
            return main_loss + 0.35 * extra_loss

        return main_loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for _ in range(epochs):
        loss, grads = loss_and_grad_fn(model, x_mx, y_main_mx, y_extra_mx)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    return model


def build_next_feature(
    numbers: np.ndarray,
    extras: np.ndarray | None,
    window: int,
    config: GameConfig,
) -> np.ndarray:
    if len(numbers) < window:
        window = len(numbers)

    recent = numbers[-window:]
    m_dim = main_dim(config)
    e_dim = extra_dim(config)

    flattened = recent.flatten().astype(np.float32) / max(1, config.main_max)

    freq = np.zeros(m_dim, dtype=np.float32)
    for n in recent.flatten():
        freq[main_index(int(n), config)] += 1.0
    freq = freq / max(1, recent.size)

    flat_recent = recent.flatten()
    gaps = np.ones(m_dim, dtype=np.float32)
    for n in range(config.main_min, config.main_max + 1):
        locations = np.where(flat_recent == n)[0]
        idx = main_index(n, config)
        if len(locations) == 0:
            gaps[idx] = 1.0
        else:
            gaps[idx] = (len(flat_recent) - 1 - locations[-1]) / max(1, len(flat_recent))

    last_draw = to_main_multihot(recent[-1], config)

    feature_parts = [flattened, freq, gaps, last_draw]

    if config.include_extra and extras is not None and e_dim > 0:
        recent_extra = extras[-window:]
        flat_extra = recent_extra.astype(np.float32) / max(1, config.extra_max or 1)

        extra_freq = np.zeros(e_dim, dtype=np.float32)
        for e in recent_extra:
            extra_freq[extra_index(int(e), config)] += 1.0
        extra_freq = extra_freq / max(1, len(recent_extra))

        extra_gaps = np.ones(e_dim, dtype=np.float32)
        for e in range(config.extra_min or 1, (config.extra_max or 1) + 1):
            locations = np.where(recent_extra == e)[0]
            idx = extra_index(e, config)
            if len(locations) == 0:
                extra_gaps[idx] = 1.0
            else:
                extra_gaps[idx] = (len(recent_extra) - 1 - locations[-1]) / max(1, len(recent_extra))

        last_extra = to_extra_onehot(int(recent_extra[-1]), config)
        feature_parts.extend([flat_extra, extra_freq, extra_gaps, last_extra])

    return np.concatenate(feature_parts).astype(np.float32)


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def softmax_np(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)


def model_probabilities(
    model: MillionaireLifeModel,
    next_feature: np.ndarray,
    config: GameConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    logits = model(mx.array(next_feature.reshape(1, -1)))
    logits_np = np.array(logits).reshape(main_dim(config) + extra_dim(config))

    main_probs = sigmoid_np(logits_np[: main_dim(config)])

    if config.include_extra and extra_dim(config) > 0:
        extra_probs = softmax_np(logits_np[main_dim(config) :])
    else:
        extra_probs = None

    return main_probs, extra_probs


def frequency_scores(numbers: np.ndarray, config: GameConfig) -> dict[int, float]:
    counts = np.zeros(main_dim(config), dtype=np.float64)
    for n in numbers.flatten():
        counts[main_index(int(n), config)] += 1.0
    counts += 1.0
    counts = counts / counts.sum()
    return {n: float(counts[main_index(n, config)]) for n in range(config.main_min, config.main_max + 1)}


def overdue_scores(numbers: np.ndarray, config: GameConfig) -> dict[int, float]:
    flat = numbers.flatten()
    raw = {}
    for n in range(config.main_min, config.main_max + 1):
        positions = np.where(flat == n)[0]
        gap = len(flat) if len(positions) == 0 else len(flat) - 1 - positions[-1]
        raw[n] = float(gap)
    max_gap = max(raw.values()) or 1.0
    return {n: raw[n] / max_gap for n in range(config.main_min, config.main_max + 1)}


def extra_frequency_scores(extras: np.ndarray, config: GameConfig) -> dict[int, float]:
    if config.extra_min is None or config.extra_max is None:
        return {}
    counts = np.zeros(extra_dim(config), dtype=np.float64)
    for e in extras:
        counts[extra_index(int(e), config)] += 1.0
    counts += 1.0
    counts = counts / counts.sum()
    return {e: float(counts[extra_index(e, config)]) for e in range(config.extra_min, config.extra_max + 1)}


def extra_overdue_scores(extras: np.ndarray, config: GameConfig) -> dict[int, float]:
    if config.extra_min is None or config.extra_max is None:
        return {}
    raw = {}
    for e in range(config.extra_min, config.extra_max + 1):
        positions = np.where(extras == e)[0]
        gap = len(extras) if len(positions) == 0 else len(extras) - 1 - positions[-1]
        raw[e] = float(gap)
    max_gap = max(raw.values()) or 1.0
    return {e: raw[e] / max_gap for e in range(config.extra_min, config.extra_max + 1)}


def pair_scores(numbers: np.ndarray) -> dict[tuple[int, int], float]:
    pairs: dict[tuple[int, int], int] = {}
    for row in numbers:
        for pair in itertools.combinations(sorted(map(int, row)), 2):
            pairs[pair] = pairs.get(pair, 0) + 1
    max_count = max(pairs.values()) if pairs else 1
    return {pair: count / max_count for pair, count in pairs.items()}


def balance_score(ticket: tuple[int, ...], config: GameConfig) -> float:
    nums = sorted(ticket)

    midpoint = (config.main_min + config.main_max) / 2
    low_count = sum(1 for n in nums if n <= midpoint)
    odd_count = sum(1 for n in nums if n % 2 == 1)

    target_split = len(nums) / 2
    low_high = 1.0 - abs(low_count - target_split) / max(1.0, target_split)
    odd_even = 1.0 - abs(odd_count - target_split) / max(1.0, target_split)

    min_sum = sum(range(config.main_min, config.main_min + len(nums)))
    max_sum = sum(range(config.main_max - len(nums) + 1, config.main_max + 1))
    center_sum = (min_sum + max_sum) / 2
    sum_score = max(0.0, 1.0 - abs(sum(nums) - center_sum) / max(1.0, center_sum))

    spread = nums[-1] - nums[0]
    max_spread = config.main_max - config.main_min
    spread_score = min(1.0, spread / max(1.0, max_spread * 0.65))

    return float((low_high + odd_even + sum_score + spread_score) / 4.0)


def ticket_to_str(ticket: Iterable[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(ticket))


def full_ticket_to_str(ticket: Iterable[int], extra: int | None) -> str:
    main = ticket_to_str(ticket)
    if extra is None:
        return main
    return f"{main} | Extra {int(extra):02d}"


def recent_ticket_keys(
    numbers: np.ndarray,
    extras: np.ndarray | None,
    recent_count: int,
) -> set[tuple[tuple[int, ...], int | None]]:
    if recent_count <= 0:
        return set()

    recent_numbers = numbers[-recent_count:]
    if extras is not None:
        recent_extras = extras[-recent_count:]
        return {
            (tuple(sorted(map(int, row))), int(extra))
            for row, extra in zip(recent_numbers, recent_extras)
        }

    return {(tuple(sorted(map(int, row))), None) for row in recent_numbers}


def parse_weights(text: str) -> tuple[float, float, float, float, float, float]:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "Weights must be six comma-separated numbers: model,frequency,overdue,pair,balance,extra"
        )
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("Weights must sum to a positive number.")
    return tuple(x / total for x in parts)  # type: ignore[return-value]


def get_weights(ticket_type: TicketType, custom_weights: tuple[float, float, float, float, float, float] | None):
    if custom_weights:
        return custom_weights

    defaults = {
        "model": (0.60, 0.13, 0.09, 0.08, 0.04, 0.06),
        "balanced": (0.36, 0.13, 0.13, 0.08, 0.24, 0.06),
        "hot": (0.25, 0.45, 0.05, 0.17, 0.02, 0.06),
        "overdue": (0.25, 0.05, 0.47, 0.07, 0.10, 0.06),
        "hot_overdue": (0.35, 0.25, 0.25, 0.08, 0.02, 0.05),
    }
    return defaults[ticket_type]


def candidate_pool(
    main_probs: np.ndarray,
    freq: dict[int, float],
    overdue: dict[int, float],
    config: GameConfig,
    pool_size: int,
    ticket_type: TicketType,
) -> list[int]:
    base = []

    for n in range(config.main_min, config.main_max + 1):
        p = main_probs[main_index(n, config)]

        if ticket_type == "hot":
            score = 0.30 * p + 0.60 * freq[n] + 0.10 * overdue[n]
        elif ticket_type == "overdue":
            score = 0.30 * p + 0.10 * freq[n] + 0.60 * overdue[n]
        elif ticket_type == "hot_overdue":
            score = 0.40 * p + 0.30 * freq[n] + 0.30 * overdue[n]
        elif ticket_type == "balanced":
            score = 0.55 * p + 0.20 * freq[n] + 0.25 * overdue[n]
        else:
            score = 0.70 * p + 0.20 * freq[n] + 0.10 * overdue[n]

        base.append((n, score))

    base.sort(key=lambda x: x[1], reverse=True)
    pool_size = max(config.main_count, min(pool_size, len(base)))
    return [n for n, _ in base[:pool_size]]


def candidate_extras(
    extra_probs: np.ndarray | None,
    extras: np.ndarray | None,
    config: GameConfig,
    count: int,
    ticket_type: TicketType,
) -> list[int | None]:
    if not config.include_extra or extra_probs is None or extras is None:
        return [None]

    freq = extra_frequency_scores(extras, config)
    overdue = extra_overdue_scores(extras, config)

    base = []
    for e in range(config.extra_min or 1, (config.extra_max or 1) + 1):
        p = extra_probs[extra_index(e, config)]

        if ticket_type == "hot":
            score = 0.35 * p + 0.55 * freq[e] + 0.10 * overdue[e]
        elif ticket_type == "overdue":
            score = 0.35 * p + 0.10 * freq[e] + 0.55 * overdue[e]
        elif ticket_type == "hot_overdue":
            score = 0.40 * p + 0.30 * freq[e] + 0.30 * overdue[e]
        else:
            score = 0.60 * p + 0.20 * freq[e] + 0.20 * overdue[e]

        base.append((e, score))

    base.sort(key=lambda x: x[1], reverse=True)
    count = max(1, min(count, len(base)))
    return [e for e, _ in base[:count]]


def rank_tickets(
    main_probs: np.ndarray,
    extra_probs: np.ndarray | None,
    numbers: np.ndarray,
    extras: np.ndarray | None,
    config: GameConfig,
    ticket_type: TicketType,
    tickets: int,
    exclude_recent: int,
    pool_size: int,
    extra_pool_size: int,
    custom_weights: tuple[float, float, float, float, float, float] | None,
) -> list[RankedTicket]:
    freq = frequency_scores(numbers, config)
    overdue = overdue_scores(numbers, config)
    pairs = pair_scores(numbers)
    excluded = recent_ticket_keys(numbers, extras, exclude_recent)

    model_w, freq_w, overdue_w, pair_w, balance_w, extra_w = get_weights(ticket_type, custom_weights)

    pool = candidate_pool(main_probs, freq, overdue, config, pool_size, ticket_type)
    extra_pool = candidate_extras(extra_probs, extras, config, extra_pool_size, ticket_type)

    extra_freq = extra_frequency_scores(extras, config) if extras is not None and config.include_extra else {}
    extra_overdue = extra_overdue_scores(extras, config) if extras is not None and config.include_extra else {}

    ranked: list[RankedTicket] = []

    for main_ticket in itertools.combinations(sorted(pool), config.main_count):
        main_key = tuple(sorted(main_ticket))

        model_score = float(np.mean([main_probs[main_index(n, config)] for n in main_ticket]))
        frequency_score = float(np.mean([freq[n] for n in main_ticket]))
        overdue_score = float(np.mean([overdue[n] for n in main_ticket]))

        ticket_pairs = list(itertools.combinations(sorted(main_ticket), 2))
        pair_score = float(np.mean([pairs.get(pair, 0.0) for pair in ticket_pairs]))

        bal_score = balance_score(main_ticket, config)

        for extra in extra_pool:
            if (main_key, extra) in excluded:
                continue

            if extra is not None and extra_probs is not None:
                e_score = float(
                    0.60 * extra_probs[extra_index(extra, config)]
                    + 0.20 * extra_freq.get(extra, 0.0)
                    + 0.20 * extra_overdue.get(extra, 0.0)
                )
            else:
                e_score = 0.0

            score = (
                model_w * model_score
                + freq_w * frequency_score
                + overdue_w * overdue_score
                + pair_w * pair_score
                + balance_w * bal_score
                + extra_w * e_score
            )

            ranked.append(
                RankedTicket(
                    rank=0,
                    ticket=full_ticket_to_str(main_ticket, extra),
                    main_numbers=ticket_to_str(main_ticket),
                    extra=extra,
                    score=score,
                    model_score=model_score,
                    frequency_score=frequency_score,
                    overdue_score=overdue_score,
                    pair_score=pair_score,
                    balance_score=bal_score,
                    extra_score=e_score,
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
        "extra",
        "score",
        "model_score",
        "frequency_score",
        "overdue_score",
        "pair_score",
        "balance_score",
        "extra_score",
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
                    "extra": "" if r.extra is None else r.extra,
                    "score": f"{r.score:.8f}",
                    "model_score": f"{r.model_score:.8f}",
                    "frequency_score": f"{r.frequency_score:.8f}",
                    "overdue_score": f"{r.overdue_score:.8f}",
                    "pair_score": f"{r.pair_score:.8f}",
                    "balance_score": f"{r.balance_score:.8f}",
                    "extra_score": f"{r.extra_score:.8f}",
                }
            )


def print_ranked(rows: list[RankedTicket], config: GameConfig) -> None:
    print()
    print("Ranked Millionaire for Life ticket candidates")
    print("-" * 108)
    print(
        f"{'Rank':<6}"
        f"{'Type':<14}"
        f"{'Ticket':<36}"
        f"{'Score':<14}"
        f"{'Model':<14}"
        f"{'Overdue':<14}"
    )
    print("-" * 108)

    for r in rows:
        print(
            f"{r.rank:<6}"
            f"{r.ticket_type:<14}"
            f"{r.ticket:<36}"
            f"{r.score:<14.8f}"
            f"{r.model_score:<14.8f}"
            f"{r.overdue_score:<14.8f}"
        )

    print("-" * 108)
    print(
        f"Detected game range: main {config.main_min}-{config.main_max}, "
        f"{config.main_count} main numbers"
        + (
            f", extra {config.extra_min}-{config.extra_max}"
            if config.include_extra
            else ", no extra number modeled"
        )
    )
    print("Reminder: this ranks historical-pattern candidates; lottery results are random.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an MLX Millionaire for Life model and produce ranked ticket candidates."
    )

    parser.add_argument(
        "--csv",
        default="data/millionaire_life_history.csv",
        help="CSV output from scrape_millionaire_life.py. Default: millionaire_life_history.csv",
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
        help="Ticket style: model, balanced, hot, overdue, hot_overdue. Default: balanced",
    )

    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="Model the Extra column as a separate number if present.",
    )

    parser.add_argument("--main-min", type=int, default=None, help="Optional main number minimum.")
    parser.add_argument("--main-max", type=int, default=None, help="Optional main number maximum.")
    parser.add_argument("--extra-min", type=int, default=None, help="Optional extra number minimum.")
    parser.add_argument("--extra-max", type=int, default=None, help="Optional extra number maximum.")

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
        default=28,
        help="How many top candidate main numbers to combine into tickets. Default: 28",
    )

    parser.add_argument(
        "--extra-pool-size",
        type=int,
        default=5,
        help="How many top Extra candidates to combine with main tickets. Default: 5",
    )

    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=None,
        help=(
            "Optional custom weights as model,frequency,overdue,pair,balance,extra. "
            "Example: --weights 0.50,0.15,0.10,0.10,0.05,0.10"
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

    _, numbers, extras, config = load_millionaire_life_csv(
        csv_path=args.csv,
        include_extra=args.include_extra,
        main_min=args.main_min,
        main_max=args.main_max,
        extra_min=args.extra_min,
        extra_max=args.extra_max,
    )

    if args.pool_size < config.main_count:
        raise SystemExit(f"--pool-size must be at least {config.main_count}.")

    if args.extra_pool_size < 1:
        raise SystemExit("--extra-pool-size must be at least 1.")

    window = min(args.window, max(5, len(numbers) // 3))

    x_train, y_main_train, y_extra_train = make_features_and_labels(
        numbers=numbers,
        extras=extras,
        window=window,
        config=config,
    )

    model = train_model(
        x_train=x_train,
        y_main_train=y_main_train,
        y_extra_train=y_extra_train,
        config=config,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    next_feature = build_next_feature(numbers, extras, window=window, config=config)
    main_probs, extra_probs = model_probabilities(model, next_feature, config)

    ranked = rank_tickets(
        main_probs=main_probs,
        extra_probs=extra_probs,
        numbers=numbers,
        extras=extras,
        config=config,
        ticket_type=args.ticket_type,
        tickets=args.tickets,
        exclude_recent=args.exclude_recent,
        pool_size=args.pool_size,
        extra_pool_size=args.extra_pool_size,
        custom_weights=args.weights,
    )

    print_ranked(ranked, config)

    if args.output:
        save_ranked_csv(args.output, ranked)
        print(f"\nSaved ranked tickets to: {args.output}")


if __name__ == "__main__":
    main()
