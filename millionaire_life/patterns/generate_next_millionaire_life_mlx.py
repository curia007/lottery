#!/usr/bin/env python3
"""
generate_next_millionaire_life_mlx.py

Train an Apple MLX deep learning model on past Millionaire for Life lottery
drawings and compute the next winning ticket candidates.

Key Features:
- Multi-task MLX neural network predicting 5 White Balls (1..58) and 1 Millionaire Ball (1..5).
- Strictly enforces NO duplicate numbers within any winning ticket (5 unique white balls).
- Ensures distinct, deduplicated winning ticket candidates across all selected strategies.
- Multiple ticket generation strategies: Model, Balanced, Hot, Overdue, and Mixed.
- Comprehensive statistics (odd/even, high/low, sum balance, overdue gap analysis).
- CSV export and full CLI options.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError as exc:
    raise SystemExit(
        "MLX is not installed. Please install with 'pip install mlx'.\n"
        "MLX requires macOS with Apple Silicon."
    ) from exc

# ---------------------------------------------------------------------------
# Millionaire for Life Game Constants
# ---------------------------------------------------------------------------
MAIN_MIN = 1
MAIN_MAX = 58
MAIN_PICK = 5
EXTRA_MIN = 1
EXTRA_MAX = 5
EXTRA_PICK = 1

DEFAULT_WINDOW = 20
DEFAULT_EPOCHS = 150
DEFAULT_LR = 0.01
DEFAULT_SEED = 42


def find_default_history_csv() -> Path:
    """Find the default historical CSV file location across standard paths."""
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "millionaire_life_history.csv",
        Path(__file__).resolve().parent / "data" / "millionaire_life_history.csv",
        Path("millionaire_life/data/millionaire_life_history.csv"),
        Path("data/millionaire_life_history.csv"),
        Path("../data/millionaire_life_history.csv"),
        Path("../../data/millionaire_life_history.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def find_default_counts_csv() -> Optional[Path]:
    """Find the default number_counts.csv file location if available."""
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "number_counts.csv",
        Path(__file__).resolve().parent / "data" / "number_counts.csv",
        Path("millionaire_life/data/number_counts.csv"),
        Path("data/number_counts.csv"),
        Path("../data/number_counts.csv"),
        Path("../../data/number_counts.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class RankedTicket:
    rank: int
    ticket_str: str
    main_numbers: List[int]
    extra: int
    score: float
    model_score: float
    frequency_score: float
    overdue_score: float
    balance_score: float
    strategy: str

    def __post_init__(self):
        # Strict validation: Ensure no duplicate numbers in ticket
        if len(self.main_numbers) != MAIN_PICK:
            raise ValueError(f"Ticket must contain exactly {MAIN_PICK} white balls, got {len(self.main_numbers)}")
        if len(set(self.main_numbers)) != MAIN_PICK:
            raise ValueError(f"Duplicate white ball numbers detected in ticket: {self.main_numbers}")
        for n in self.main_numbers:
            if not (MAIN_MIN <= n <= MAIN_MAX):
                raise ValueError(f"White ball number {n} out of range ({MAIN_MIN}..{MAIN_MAX})")
        if not (EXTRA_MIN <= self.extra <= EXTRA_MAX):
            raise ValueError(f"Millionaire Ball {self.extra} out of range ({EXTRA_MIN}..{EXTRA_MAX})")


# ---------------------------------------------------------------------------
# Data Loading & Processing
# ---------------------------------------------------------------------------
def load_data(
    history_path: str | Path,
    counts_path: Optional[str | Path] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], pd.DataFrame]:
    """
    Loads and cleans Millionaire for Life past drawings.
    Returns:
        main_history: (N, 5) ndarray of white balls in chronological order.
        extra_history: (N,) ndarray of millionaire extra balls.
        counts_dict: Frequency count mapping {number: count}.
        df: Cleaned dataframe.
    """
    h_path = Path(history_path)
    if not h_path.exists():
        raise FileNotFoundError(f"History file not found: {h_path}")

    df = pd.read_csv(h_path)
    main_cols = ["Num1", "Num2", "Num3", "Num4", "Num5"]

    for col in main_cols + ["Extra"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=main_cols + ["Extra"]).copy()
    for col in main_cols + ["Extra"]:
        df[col] = df[col].astype(int)

    # Sort chronologically (oldest to newest)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    else:
        # Default assume top row is newest, reverse
        df = df.iloc[::-1].reset_index(drop=True)

    main_history = df[main_cols].values.astype(int)
    extra_history = df["Extra"].values.astype(int)

    # Calculate frequency counts
    counts_dict: Dict[int, int] = {}
    if counts_path and Path(counts_path).exists():
        cdf = pd.read_csv(counts_path)
        if "Number" in cdf.columns and "Count" in cdf.columns:
            counts_dict = dict(zip(cdf["Number"].astype(int), cdf["Count"].astype(int)))

    if not counts_dict:
        for num in range(MAIN_MIN, MAIN_MAX + 1):
            counts_dict[num] = int(np.sum(main_history == num))

    return main_history, extra_history, counts_dict, df


def to_multihot_white(numbers: Sequence[int]) -> np.ndarray:
    """Encodes white ball numbers to a 58-element multi-hot binary vector."""
    vec = np.zeros(MAIN_MAX, dtype=np.float32)
    for n in numbers:
        if MAIN_MIN <= n <= MAIN_MAX:
            vec[n - MAIN_MIN] = 1.0
    return vec


def to_onehot_extra(extra_num: int) -> np.ndarray:
    """Encodes Millionaire extra ball (1..5) to a 5-element one-hot vector."""
    vec = np.zeros(EXTRA_MAX, dtype=np.float32)
    if EXTRA_MIN <= extra_num <= EXTRA_MAX:
        vec[extra_num - EXTRA_MIN] = 1.0
    return vec


def extract_features(
    main_window: np.ndarray,
    extra_window: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Extracts numerical feature vector from a rolling history window:
    - Flattened normalized main numbers and extra numbers
    - Window frequency distributions
    - Overdue gap values
    - Multi-hot encoding of the most recent draw
    - Consecutive draw repeat dynamics
    - Statistical balance ratios (Odd/Even, High/Low)
    """
    w_len = len(main_window)
    main_dim = MAIN_MAX
    extra_dim = EXTRA_MAX

    # 1. Flattened normalized draw sequences
    flat_main = (main_window.flatten().astype(np.float32) - MAIN_MIN) / (MAIN_MAX - MAIN_MIN)
    flat_extra = (extra_window.astype(np.float32) - EXTRA_MIN) / max(1.0, float(EXTRA_MAX - EXTRA_MIN))

    # 2. White ball frequencies in window
    main_freq = np.zeros(main_dim, dtype=np.float32)
    for row in main_window:
        for b in row:
            if MAIN_MIN <= b <= MAIN_MAX:
                main_freq[b - MAIN_MIN] += 1.0
    main_freq /= max(1.0, float(w_len * MAIN_PICK))

    # 3. Extra ball frequencies in window
    extra_freq = np.zeros(extra_dim, dtype=np.float32)
    for e in extra_window:
        if EXTRA_MIN <= e <= EXTRA_MAX:
            extra_freq[e - EXTRA_MIN] += 1.0
    extra_freq /= max(1.0, float(w_len))

    # 4. White ball gaps (0 = appeared in last draw, 1 = max overdue)
    main_gaps = np.ones(main_dim, dtype=np.float32)
    for b in range(MAIN_MIN, MAIN_MAX + 1):
        for dist, row in enumerate(reversed(main_window)):
            if b in row:
                main_gaps[b - MAIN_MIN] = float(dist) / float(window_size)
                break

    # 5. Extra ball gaps
    extra_gaps = np.ones(extra_dim, dtype=np.float32)
    for e in range(EXTRA_MIN, EXTRA_MAX + 1):
        for dist, val in enumerate(reversed(extra_window)):
            if e == val:
                extra_gaps[e - EXTRA_MIN] = float(dist) / float(window_size)
                break

    # 6. Last draw multi-hot / one-hot
    last_main_hot = to_multihot_white(main_window[-1])
    last_extra_hot = to_onehot_extra(extra_window[-1])

    # 7. Consecutive repeat signals
    if w_len >= 2:
        main_consec = to_multihot_white(main_window[-1]) * to_multihot_white(main_window[-2])
        extra_consec = to_onehot_extra(extra_window[-1]) * to_onehot_extra(extra_window[-2])
    else:
        main_consec = np.zeros(main_dim, dtype=np.float32)
        extra_consec = np.zeros(extra_dim, dtype=np.float32)

    # 8. High/Low and Odd/Even balance ratios
    odd_count = sum(1 for row in main_window for b in row if b % 2 == 1)
    high_count = sum(1 for row in main_window for b in row if b > MAIN_MAX // 2)
    odd_ratio = np.array([odd_count / (w_len * MAIN_PICK)], dtype=np.float32)
    high_ratio = np.array([high_count / (w_len * MAIN_PICK)], dtype=np.float32)

    return np.concatenate([
        flat_main,
        flat_extra,
        main_freq,
        extra_freq,
        main_gaps,
        extra_gaps,
        last_main_hot,
        last_extra_hot,
        main_consec,
        extra_consec,
        odd_ratio,
        high_ratio,
    ])


def make_dataset(
    main_history: np.ndarray,
    extra_history: np.ndarray,
    window: int = DEFAULT_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constructs training feature vectors and target labels."""
    xs, ys_main, ys_extra = [], [], []
    n_draws = len(main_history)

    for i in range(window, n_draws):
        m_win = main_history[i - window : i]
        e_win = extra_history[i - window : i]

        feat = extract_features(m_win, e_win, window)
        y_m = to_multihot_white(main_history[i])
        y_e = to_onehot_extra(extra_history[i])

        xs.append(feat)
        ys_main.append(y_m)
        ys_extra.append(y_e)

    return np.stack(xs), np.stack(ys_main), np.stack(ys_extra)


# ---------------------------------------------------------------------------
# MLX Deep Learning Model
# ---------------------------------------------------------------------------
class MillionaireLifeMLXModel(nn.Module):
    """
    MLX Multi-Task Neural Network that computes:
      - Main White Ball logits (58 outputs)
      - Millionaire Ball logits (5 outputs)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.main_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, MAIN_MAX),
        )
        self.extra_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, EXTRA_MAX),
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        rep = self.shared(x)
        main_logits = self.main_head(rep)
        extra_logits = self.extra_head(rep)
        return main_logits, extra_logits


def train_model(
    x_train: np.ndarray,
    y_main: np.ndarray,
    y_extra: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    verbose: bool = False,
) -> MillionaireLifeMLXModel:
    """Trains the MLX neural network on past Millionaire for Life draws."""
    input_dim = x_train.shape[1]
    model = MillionaireLifeMLXModel(input_dim)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=lr)

    x_arr = mx.array(x_train)
    ym_arr = mx.array(y_main)
    ye_arr = mx.array(y_extra)

    def loss_fn(model_ref, x, ym, ye):
        logits_m, logits_e = model_ref(x)

        # Multi-label binary cross entropy with sigmoid for 5 White Balls
        sig = mx.sigmoid(logits_m)
        eps = 1e-7
        bce = -mx.mean(ym * mx.log(sig + eps) + (1.0 - ym) * mx.log(1.0 - sig + eps))

        # Categorical cross entropy with softmax for Millionaire Extra Ball
        ce_extra = mx.mean(nn.losses.cross_entropy(logits_e, mx.argmax(ye, axis=1)))

        return bce + 1.2 * ce_extra

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(epochs):
        loss_val, grads = loss_and_grad(model, x_arr, ym_arr, ye_arr)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if verbose and (epoch + 1) % 25 == 0:
            print(f"  [Epoch {epoch+1:03d}/{epochs}] Joint Training Loss: {loss_val.item():.4f}")

    return model


def compute_model_probabilities(
    model: MillionaireLifeMLXModel,
    main_history: np.ndarray,
    extra_history: np.ndarray,
    window: int = DEFAULT_WINDOW,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes predicted probabilities for White Balls and Millionaire Extra Balls for the next draw."""
    m_win = main_history[-window:]
    e_win = extra_history[-window:]
    feat = extract_features(m_win, e_win, window)

    x_next = mx.array(feat[None, :])
    logits_m, logits_e = model(x_next)

    main_probs = np.array(mx.sigmoid(logits_m))[0]
    extra_probs = np.array(mx.softmax(logits_e, axis=1))[0]

    return main_probs, extra_probs


# ---------------------------------------------------------------------------
# Statistical & Pattern Metrics
# ---------------------------------------------------------------------------
def compute_overdue_scores(main_history: np.ndarray) -> Dict[int, float]:
    """Calculates overdue normalized gap scores for all White Balls (1..58)."""
    gaps: Dict[int, int] = {}
    for num in range(MAIN_MIN, MAIN_MAX + 1):
        gaps[num] = len(main_history)  # default if not found
        for dist, row in enumerate(reversed(main_history)):
            if num in row:
                gaps[num] = dist
                break

    max_gap = max(gaps.values()) if gaps else 1
    return {num: float(gap) / float(max(1, max_gap)) for num, gap in gaps.items()}


def compute_balance_score(numbers: Sequence[int]) -> float:
    """
    Evaluates balance of a 5-number combination:
      - Odd / Even distribution (ideal is 2:3 or 3:2 split)
      - High / Low distribution (ideal is 2:3 or 3:2 split relative to 29)
      - Sum total distribution (Millionaire Life expected sum center ~147.5)
    """
    unique_nums = sorted(set(numbers))
    if len(unique_nums) != MAIN_PICK:
        return 0.0

    low_count = sum(1 for n in unique_nums if n <= MAIN_MAX // 2)
    odd_count = sum(1 for n in unique_nums if n % 2 == 1)

    low_high_score = 1.0 - abs(low_count - 2.5) / 2.5
    odd_even_score = 1.0 - abs(odd_count - 2.5) / 2.5

    sum_total = sum(unique_nums)
    expected_sum = (MAIN_MIN + MAIN_MAX) * MAIN_PICK / 2.0  # 59 * 2.5 = 147.5
    sum_score = max(0.0, 1.0 - abs(sum_total - expected_sum) / 100.0)

    return float((low_high_score + odd_even_score + sum_score) / 3.0)


# ---------------------------------------------------------------------------
# Winning Ticket Generation & Duplicate Elimination
# ---------------------------------------------------------------------------
def generate_winning_tickets(
    model_probs: np.ndarray,
    extra_probs: np.ndarray,
    counts_dict: Dict[int, int],
    overdue_scores: Dict[int, float],
    main_history: np.ndarray,
    num_tickets: int = 5,
    ticket_type: str = "mixed",
    seed: int = DEFAULT_SEED,
) -> List[RankedTicket]:
    """
    Generates and ranks winning tickets with NO duplicate numbers within any ticket,
    and distinct ticket combinations across all generated tickets.

    Supported strategies:
      - 'model': Highest neural network probability combinations
      - 'balanced': Optimal odd/even, high/low, and sum distribution
      - 'hot': High-frequency momentum numbers
      - 'overdue': Longest overdue numbers due for rebound
      - 'mixed': Blend across multiple strategies
    """
    rng = random.Random(seed)

    # Frequency normalization
    max_c = max(counts_dict.values()) if counts_dict and any(counts_dict.values()) else 1
    freq_scores = {n: float(counts_dict.get(n, 0)) / float(max_c) for n in range(MAIN_MIN, MAIN_MAX + 1)}

    # Extra Ball selection: rank by probability
    ranked_extras = sorted(
        [(e + EXTRA_MIN, float(extra_probs[e])) for e in range(EXTRA_MAX)],
        key=lambda x: x[1],
        reverse=True,
    )
    best_extra = ranked_extras[0][0]

    # Calculate overall number scores per strategy
    def get_number_score(n: int, strat: str) -> float:
        m_s = float(model_probs[n - MAIN_MIN])
        f_s = freq_scores.get(n, 0.0)
        o_s = overdue_scores.get(n, 0.0)

        if strat == "hot":
            return f_s * 0.70 + m_s * 0.30
        elif strat == "overdue":
            return o_s * 0.70 + m_s * 0.30
        elif strat == "balanced":
            return m_s * 0.40 + f_s * 0.30 + o_s * 0.30
        elif strat == "model":
            return m_s
        else:  # mixed
            return m_s * 0.45 + f_s * 0.30 + o_s * 0.25

    strategies_to_run = (
        ["model", "balanced", "hot", "overdue", "mixed"]
        if ticket_type == "mixed"
        else [ticket_type]
    )

    generated_tickets: List[RankedTicket] = []
    seen_combinations: Set[Tuple[int, ...]] = set()

    for idx in range(num_tickets):
        current_strat = strategies_to_run[idx % len(strategies_to_run)]

        # Score numbers for current strategy
        scores = [(n, get_number_score(n, current_strat)) for n in range(MAIN_MIN, MAIN_MAX + 1)]
        scores.sort(key=lambda x: x[1], reverse=True)

        # Pool of top candidates (top 15-20 numbers)
        pool = [n for n, s in scores[:18]]

        # Evaluate combinations from pool
        candidate_combos = []
        for combo in itertools.combinations(pool, MAIN_PICK):
            combo_tuple = tuple(sorted(combo))
            if combo_tuple in seen_combinations:
                continue

            # Check that combo has NO duplicates (guaranteed by itertools.combinations)
            assert len(set(combo_tuple)) == MAIN_PICK

            m_score = float(np.mean([model_probs[n - MAIN_MIN] for n in combo_tuple]))
            f_score = float(np.mean([freq_scores.get(n, 0.0) for n in combo_tuple]))
            o_score = float(np.mean([overdue_scores.get(n, 0.0) for n in combo_tuple]))
            b_score = compute_balance_score(combo_tuple)

            if current_strat == "model":
                final_score = m_score * 0.70 + b_score * 0.30
            elif current_strat == "hot":
                final_score = f_score * 0.50 + m_score * 0.30 + b_score * 0.20
            elif current_strat == "overdue":
                final_score = o_score * 0.50 + m_score * 0.30 + b_score * 0.20
            elif current_strat == "balanced":
                final_score = b_score * 0.45 + m_score * 0.35 + f_score * 0.20
            else:  # mixed
                final_score = m_score * 0.35 + b_score * 0.25 + f_score * 0.20 + o_score * 0.20

            candidate_combos.append((combo_tuple, final_score, m_score, f_score, o_score, b_score))

        if candidate_combos:
            candidate_combos.sort(key=lambda x: x[1], reverse=True)
            chosen_combo, score, m_s, f_s, o_s, b_s = candidate_combos[0]
        else:
            # Fallback: select top 5 distinct numbers from scores
            chosen_list = [n for n, _ in scores[:MAIN_PICK]]
            chosen_combo = tuple(sorted(set(chosen_list)))
            score = float(np.mean([s for _, s in scores[:MAIN_PICK]]))
            m_s = float(np.mean([model_probs[n - MAIN_MIN] for n in chosen_combo]))
            f_s = float(np.mean([freq_scores.get(n, 0.0) for n in chosen_combo]))
            o_s = float(np.mean([overdue_scores.get(n, 0.0) for n in chosen_combo]))
            b_s = compute_balance_score(chosen_combo)

        seen_combinations.add(chosen_combo)

        # Distribute Millionaire Ball across top predicted candidates
        if idx < 3:
            assigned_extra = ranked_extras[0][0]
        else:
            assigned_extra = ranked_extras[idx % min(3, len(ranked_extras))][0]

        ticket_numbers_list = list(chosen_combo)
        # Double check: no duplicate numbers
        assert len(ticket_numbers_list) == len(set(ticket_numbers_list)) == MAIN_PICK

        ticket_str = f"{' '.join(f'{n:02d}' for n in ticket_numbers_list)} [MB: {assigned_extra}]"

        ranked_ticket = RankedTicket(
            rank=idx + 1,
            ticket_str=ticket_str,
            main_numbers=ticket_numbers_list,
            extra=assigned_extra,
            score=score,
            model_score=m_s,
            frequency_score=f_s,
            overdue_score=o_s,
            balance_score=b_s,
            strategy=current_strat.capitalize(),
        )
        generated_tickets.append(ranked_ticket)

    # Sort final tickets by total score
    generated_tickets.sort(key=lambda t: t.score, reverse=True)
    for i, t in enumerate(generated_tickets):
        t.rank = i + 1

    return generated_tickets


# ---------------------------------------------------------------------------
# Presentation & CSV Export
# ---------------------------------------------------------------------------
def print_winning_tickets_report(
    tickets: List[RankedTicket],
    top_extras: List[Tuple[int, float]],
    top_white: List[Tuple[int, float]],
    draw_count: int,
):
    """Displays formatted winning ticket candidates and MLX probabilities."""
    print("=" * 80)
    print(" MILLIONAIRE FOR LIFE - MLX PREDICTED WINNING TICKETS")
    print("=" * 80)
    print(f" Historical Draws Analyzed: {draw_count}")
    print(f" White Ball Range: {MAIN_MIN} - {MAIN_MAX} (Pick {MAIN_PICK} distinct numbers, NO duplicates)")
    print(f" Millionaire Ball Range: {EXTRA_MIN} - {EXTRA_MAX} (Pick {EXTRA_PICK})")
    print("-" * 80)

    print("\n[1] MLX PREDICTED MILLIONAIRE BALLS (EXTRA 1 - 5):")
    print(f"  {'Rank':<6} {'Millionaire Ball':<20} {'MLX Probability':<18}")
    print("  " + "-" * 46)
    for rank, (extra_num, prob) in enumerate(top_extras, start=1):
        marker = " <== TOP PICK" if rank == 1 else ""
        print(f"  #{rank:<5} Ball {extra_num:<15} {prob * 100:6.2f}%{marker}")

    print("\n[2] TOP MLX PREDICTED WHITE BALLS (MAIN 1 - 58):")
    top_wb_str = ", ".join([f"#{b:02d} ({p*100:.1f}%)" for b, p in top_white[:10]])
    print(f"  Top 10: {top_wb_str}")

    print("\n[3] COMPUTED NEXT WINNING TICKETS (NO DUPLICATE NUMBERS):")
    print(
        f"  {'Rank':<6} {'Winning Ticket (Main + Extra)':<36} "
        f"{'Strategy':<12} {'Model':<8} {'Freq':<8} {'Bal':<8} {'Score':<8}"
    )
    print("  " + "-" * 88)

    for t in tickets:
        print(
            f"  #{t.rank:<5} {t.ticket_str:<36} "
            f"{t.strategy:<12} {t.model_score:6.3f}   {t.frequency_score:6.3f}   "
            f"{t.balance_score:6.3f}   {t.score:6.3f}"
        )

    print("=" * 80)
    print(" Verification: All generated tickets verified with 5 unique white balls.")
    print("=" * 80)


def export_tickets_to_csv(tickets: List[RankedTicket], output_path: str | Path):
    """Exports winning tickets to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank",
            "Num1",
            "Num2",
            "Num3",
            "Num4",
            "Num5",
            "Extra",
            "Ticket",
            "Strategy",
            "Score",
            "ModelScore",
            "FrequencyScore",
            "OverdueScore",
            "BalanceScore",
        ])
        for t in tickets:
            writer.writerow([
                t.rank,
                t.main_numbers[0],
                t.main_numbers[1],
                t.main_numbers[2],
                t.main_numbers[3],
                t.main_numbers[4],
                t.extra,
                t.ticket_str,
                t.strategy,
                f"{t.score:.4f}",
                f"{t.model_score:.4f}",
                f"{t.frequency_score:.4f}",
                f"{t.overdue_score:.4f}",
                f"{t.balance_score:.4f}",
            ])
    print(f"\nSaved {len(tickets)} winning tickets to: {out}")


# ---------------------------------------------------------------------------
# CLI & Main Entry Point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MLX model on past Millionaire for Life draws and compute next winning tickets without duplicates."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(find_default_history_csv()),
        help="Path to Millionaire for Life history CSV (default: auto-detected)",
    )
    parser.add_argument(
        "--counts-csv",
        type=str,
        default=str(find_default_counts_csv()) if find_default_counts_csv() else None,
        help="Optional path to number_counts.csv",
    )
    parser.add_argument(
        "--tickets",
        type=int,
        default=5,
        help="Number of winning tickets to compute (default: 5)",
    )
    parser.add_argument(
        "--ticket-type",
        type=str,
        default="mixed",
        choices=["model", "balanced", "hot", "overdue", "mixed"],
        help="Strategy for ticket generation (default: mixed)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of MLX training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Rolling history window size (default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV output file path to save computed winning tickets",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed MLX epoch loss logs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print(f"Loading Millionaire for Life history from: {args.csv}")
    main_history, extra_history, counts_dict, df = load_data(args.csv, args.counts_csv)
    print(f"Loaded {len(main_history)} historical draws.")

    if len(main_history) <= args.window:
        raise ValueError(
            f"History contains {len(main_history)} draws, which is not enough for window size {args.window}."
        )

    print(f"Building MLX training dataset (window={args.window})...")
    x_train, y_main, y_extra = make_dataset(main_history, extra_history, window=args.window)
    print(f"Training dataset: X={x_train.shape}, Y_main={y_main.shape}, Y_extra={y_extra.shape}")

    print(f"Training MLX Neural Network for {args.epochs} epochs (lr={args.lr})...")
    model = train_model(
        x_train,
        y_main,
        y_extra,
        epochs=args.epochs,
        lr=args.lr,
        verbose=args.verbose,
    )

    print("Computing probabilities and statistics for the next draw...")
    main_probs, extra_probs = compute_model_probabilities(
        model, main_history, extra_history, window=args.window
    )
    overdue_scores = compute_overdue_scores(main_history)

    # Rank extras and white balls
    top_extras = sorted(
        [(e + EXTRA_MIN, float(extra_probs[e])) for e in range(EXTRA_MAX)],
        key=lambda x: x[1],
        reverse=True,
    )
    top_white = sorted(
        [(b + MAIN_MIN, float(main_probs[b])) for b in range(MAIN_MAX)],
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"Generating {args.tickets} winning tickets (Strategy: {args.ticket_type})...")
    tickets = generate_winning_tickets(
        model_probs=main_probs,
        extra_probs=extra_probs,
        counts_dict=counts_dict,
        overdue_scores=overdue_scores,
        main_history=main_history,
        num_tickets=args.tickets,
        ticket_type=args.ticket_type,
        seed=args.seed,
    )

    print_winning_tickets_report(tickets, top_extras, top_white, len(main_history))

    if args.output:
        export_tickets_to_csv(tickets, args.output)


if __name__ == "__main__":
    main()
