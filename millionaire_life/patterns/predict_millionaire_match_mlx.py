#!/usr/bin/env python3
"""
predict_millionaire_match_mlx.py

MLX-powered predictor for the Millionaire for Life lottery.
Generates next-draw ticket candidates specifically optimized to match:
  1. The winning Millionaire Ball (Extra 1 - 5)
  2. At least one (and multiple) White Balls (Main 1 - 58)

Uses Apple MLX deep learning multi-task neural network trained on historical draws,
combining frequency distributions, overdue gap analysis, consecutive draw repeats,
and pair co-occurrence dynamics.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# Constants & Game Configuration
# ---------------------------------------------------------------------------
MAIN_MIN = 1
MAIN_MAX = 58
MAIN_PICK = 5
EXTRA_MIN = 1
EXTRA_MAX = 5
EXTRA_PICK = 1

DEFAULT_WINDOW = 50
DEFAULT_EPOCHS = 120
DEFAULT_LR = 0.008
DEFAULT_SEED = 42


def find_default_csv() -> Path:
    """Find the default history CSV file."""
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "millionaire_life_history.csv",
        Path(__file__).resolve().parent / "data" / "millionaire_life_history.csv",
        Path("millionaire_life/data/millionaire_life_history.csv"),
        Path("data/millionaire_life_history.csv"),
        Path("../data/millionaire_life_history.csv"),
        Path("../../data/millionaire_life_history.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------
def load_millionaire_history(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads Millionaire for Life history from CSV.
    Returns:
        main_history: (N, 5) ndarray of white balls in chronological order.
        extra_history: (N,) ndarray of millionaire balls in chronological order.
        df: Cleaned dataframe.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"History CSV file not found: {path}")

    df = pd.read_csv(path)
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
        # If no date, reverse assuming file is newest-first
        df = df.iloc[::-1].reset_index(drop=True)

    main_history = df[main_cols].values.astype(int)
    extra_history = df["Extra"].values.astype(int)

    return main_history, extra_history, df


def to_multihot_white(balls: np.ndarray | List[int]) -> np.ndarray:
    """Encodes white balls (1..58) to a multi-hot binary vector of length 58."""
    vec = np.zeros(MAIN_MAX, dtype=np.float32)
    for b in balls:
        if MAIN_MIN <= b <= MAIN_MAX:
            vec[b - MAIN_MIN] = 1.0
    return vec


def to_onehot_extra(extra: int) -> np.ndarray:
    """Encodes millionaire extra ball (1..5) to a one-hot vector of length 5."""
    vec = np.zeros(EXTRA_MAX, dtype=np.float32)
    if EXTRA_MIN <= extra <= EXTRA_MAX:
        vec[extra - EXTRA_MIN] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
def extract_draw_features(
    main_window: np.ndarray,
    extra_window: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Extracts deep numerical feature vector from a rolling history window:
      - Normalized recent draw sequences
      - Frequency vectors for white balls (58) and extra balls (5)
      - Overdue gap vectors
      - Consecutive draw repeat flags
      - Odd/Even & High/Low ratios
      - Co-occurrence context
    """
    w_len = len(main_window)
    main_dim = MAIN_MAX
    extra_dim = EXTRA_MAX

    # 1. Normalized flat recent history
    norm_main = (main_window.flatten().astype(np.float32) - MAIN_MIN) / (MAIN_MAX - MAIN_MIN)
    norm_extra = (extra_window.astype(np.float32) - EXTRA_MIN) / (EXTRA_MAX - EXTRA_MIN)

    # 2. White Ball Frequencies over window
    main_freq = np.zeros(main_dim, dtype=np.float32)
    for row in main_window:
        for b in row:
            if MAIN_MIN <= b <= MAIN_MAX:
                main_freq[b - MAIN_MIN] += 1.0
    main_freq /= max(1.0, float(w_len * MAIN_PICK))

    # 3. Millionaire Ball Frequencies over window
    extra_freq = np.zeros(extra_dim, dtype=np.float32)
    for e in extra_window:
        if EXTRA_MIN <= e <= EXTRA_MAX:
            extra_freq[e - EXTRA_MIN] += 1.0
    extra_freq /= max(1.0, float(w_len))

    # 4. White Ball Overdue Gaps (normalized 0..1, 1 = maximum overdue)
    main_gaps = np.ones(main_dim, dtype=np.float32)
    for b in range(MAIN_MIN, MAIN_MAX + 1):
        for dist, row in enumerate(reversed(main_window)):
            if b in row:
                main_gaps[b - MAIN_MIN] = float(dist) / float(window_size)
                break

    # 5. Millionaire Ball Overdue Gaps
    extra_gaps = np.ones(extra_dim, dtype=np.float32)
    for e in range(EXTRA_MIN, EXTRA_MAX + 1):
        for dist, val in enumerate(reversed(extra_window)):
            if e == val:
                extra_gaps[e - EXTRA_MIN] = float(dist) / float(window_size)
                break

    # 6. Last Draw Multi-Hot & One-Hot
    last_main_multihot = to_multihot_white(main_window[-1])
    last_extra_onehot = to_onehot_extra(extra_window[-1])

    # 7. Consecutive Repeat Flags (did ball appear in last 2 draws?)
    if w_len >= 2:
        main_consec_repeat = (
            to_multihot_white(main_window[-1]) * to_multihot_white(main_window[-2])
        )
        extra_consec_repeat = (
            to_onehot_extra(extra_window[-1]) * to_onehot_extra(extra_window[-2])
        )
    else:
        main_consec_repeat = np.zeros(main_dim, dtype=np.float32)
        extra_consec_repeat = np.zeros(extra_dim, dtype=np.float32)

    # 8. Statistical summary metrics (Odd ratio, High ratio, Sum mean)
    odd_count = sum(1 for row in main_window for b in row if b % 2 == 1)
    high_count = sum(1 for row in main_window for b in row if b > MAIN_MAX // 2)
    odd_ratio = np.array([odd_count / (w_len * MAIN_PICK)], dtype=np.float32)
    high_ratio = np.array([high_count / (w_len * MAIN_PICK)], dtype=np.float32)

    return np.concatenate([
        norm_main,
        norm_extra,
        main_freq,
        extra_freq,
        main_gaps,
        extra_gaps,
        last_main_multihot,
        last_extra_onehot,
        main_consec_repeat,
        extra_consec_repeat,
        odd_ratio,
        high_ratio,
    ])


def build_dataset(
    main_history: np.ndarray,
    extra_history: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates training features X and multi-target labels (Y_main, Y_extra)."""
    xs, ys_main, ys_extra = [], [], []
    total = len(main_history)

    for i in range(window, total):
        m_win = main_history[i - window : i]
        e_win = extra_history[i - window : i]

        feat = extract_draw_features(m_win, e_win, window)
        y_m = to_multihot_white(main_history[i])
        y_e = to_onehot_extra(extra_history[i])

        xs.append(feat)
        ys_main.append(y_m)
        ys_extra.append(y_e)

    return np.stack(xs), np.stack(ys_main), np.stack(ys_extra)


# ---------------------------------------------------------------------------
# MLX Multi-Task Neural Network Model
# ---------------------------------------------------------------------------
class MillionaireMatchModel(nn.Module):
    """
    MLX Deep Multi-Task Network predicting both:
      1. White Ball logits (58 outputs, multi-label probability)
      2. Millionaire Extra Ball logits (5 outputs, categorical distribution)
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


def train_mlx_model(
    x_train: np.ndarray,
    y_main: np.ndarray,
    y_extra: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    verbose: bool = False,
) -> MillionaireMatchModel:
    """Trains the MLX neural network using Adam optimizer."""
    input_dim = x_train.shape[1]
    model = MillionaireMatchModel(input_dim)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=lr)

    x_arr = mx.array(x_train)
    ym_arr = mx.array(y_main)
    ye_arr = mx.array(y_extra)

    def loss_fn(model_ref, x, ym, ye):
        logits_m, logits_e = model_ref(x)

        # Multi-label binary cross entropy with sigmoid for White Balls
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


# ---------------------------------------------------------------------------
# Prediction & Ticket Generation Targeting [Matched Extra + >=1 White Ball]
# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    extra_probs: np.ndarray  # Shape (5,)
    main_probs: np.ndarray  # Shape (58,)
    predicted_extra: int
    top_extras: List[Tuple[int, float]]
    top_white_balls: List[Tuple[int, float]]
    tickets: List[RankedTicket]


@dataclass
class RankedTicket:
    ticket_number: int
    main_numbers: List[int]
    extra: int
    match_score: float
    white_ball_match_prob: float
    extra_ball_match_prob: float
    joint_target_prob: float
    strategy: str


def predict_next_draw(
    model: MillionaireMatchModel,
    main_history: np.ndarray,
    extra_history: np.ndarray,
    window: int = DEFAULT_WINDOW,
    num_tickets: int = 5,
    seed: int = DEFAULT_SEED,
) -> PredictionResult:
    """
    Generates next draw predictions and ticket candidates designed to match:
      - The Millionaire Ball (Extra 1..5)
      - At least one White Ball (Main 1..58)
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Feature vector for the next unseen draw
    m_win = main_history[-window:]
    e_win = extra_history[-window:]
    feat = extract_draw_features(m_win, e_win, window)

    x_next = mx.array(feat[None, :])
    logits_m, logits_e = model(x_next)

    # Convert to probabilities
    main_probs = np.array(mx.sigmoid(logits_m))[0]
    extra_probs = np.array(mx.softmax(logits_e, axis=1))[0]

    # Normalize white ball probabilities
    main_prob_sum = np.sum(main_probs)
    if main_prob_sum > 0:
        main_norm_probs = main_probs / main_prob_sum
    else:
        main_norm_probs = np.ones(MAIN_MAX) / MAIN_MAX

    # Ranked Extra Balls
    extra_ranked = sorted(
        [(e + EXTRA_MIN, float(extra_probs[e])) for e in range(EXTRA_MAX)],
        key=lambda x: x[1],
        reverse=True,
    )
    predicted_extra = extra_ranked[0][0]

    # Ranked White Balls
    white_ranked = sorted(
        [(b + MAIN_MIN, float(main_probs[b])) for b in range(MAIN_MAX)],
        key=lambda x: x[1],
        reverse=True,
    )

    # Historical frequency and overdue mapping
    white_counts = np.zeros(MAIN_MAX, dtype=int)
    for row in main_history:
        for b in row:
            if MAIN_MIN <= b <= MAIN_MAX:
                white_counts[b - MAIN_MIN] += 1

    white_gaps = np.zeros(MAIN_MAX, dtype=int)
    for b in range(MAIN_MIN, MAIN_MAX + 1):
        for dist, row in enumerate(reversed(main_history)):
            if b in row:
                white_gaps[b - MAIN_MIN] = dist
                break

    # Pair co-occurrence matrix
    cooc = np.zeros((MAIN_MAX, MAIN_MAX), dtype=int)
    for row in main_history:
        for b1, b2 in itertools.combinations(row, 2):
            cooc[b1 - 1, b2 - 1] += 1
            cooc[b2 - 1, b1 - 1] += 1

    # -----------------------------------------------------------------------
    # Ticket Generation Strategies Specifically Designed to Hit Extra + >=1 WB
    # -----------------------------------------------------------------------
    generated_tickets: List[RankedTicket] = []
    seen_combos = set()

    top_wb_pool = [b for b, _ in white_ranked[:15]]
    top_extra_candidate = predicted_extra

    strategies = [
        "Model Consensus (Top MLX Probabilities)",
        "Max WB Coverage (Distributed Top Picks)",
        "Pattern Balanced (Odd/Even & High/Low)",
        "Hot & Overdue Hybrid (Momentum + Rebound)",
        "Pair Association (Top Co-occurrence Synergy)",
    ]

    for t_idx in range(num_tickets):
        strat = strategies[t_idx % len(strategies)]
        chosen_main: List[int] = []

        # Choose the Millionaire Extra Ball for this ticket
        # Strongest tickets use #1 predicted extra; secondary tickets can test top-2
        if t_idx < 3:
            chosen_extra = extra_ranked[0][0]
        else:
            # Distribute between top 2 extra predictions
            chosen_extra = extra_ranked[t_idx % 2][0]

        if strat.startswith("Model Consensus"):
            # Strategy 1: Top 5 absolute highest probability white balls
            chosen_main = sorted([b for b, _ in white_ranked[:5]])

        elif strat.startswith("Max WB Coverage"):
            # Strategy 2: Spread selections across the top 12 ranked balls to maximize
            # probability of catching at least 1-2 hits across tickets
            start_offset = (t_idx * 2) % (len(top_wb_pool) - 5)
            candidates = top_wb_pool[start_offset : start_offset + 5]
            if len(candidates) < 5:
                candidates = top_wb_pool[:5]
            chosen_main = sorted(candidates)

        elif strat.startswith("Pattern Balanced"):
            # Strategy 3: Select from top 15 white balls satisfying 2:3 or 3:2 odd/even
            pool = top_wb_pool.copy()
            odds = [b for b in pool if b % 2 == 1]
            evens = [b for b in pool if b % 2 == 0]
            if len(odds) >= 3 and len(evens) >= 2:
                chosen_main = sorted(odds[:3] + evens[:2])
            elif len(odds) >= 2 and len(evens) >= 3:
                chosen_main = sorted(odds[:2] + evens[:3])
            else:
                chosen_main = sorted(pool[:5])

        elif strat.startswith("Hot & Overdue"):
            # Strategy 4: 3 top hot predicted balls + 2 high overdue rebound balls
            hot_picks = [b for b, _ in white_ranked[:8]]
            overdue_ranked = sorted(
                range(MAIN_MIN, MAIN_MAX + 1),
                key=lambda b: (white_gaps[b - 1], main_probs[b - 1]),
                reverse=True,
            )
            top_overdue = [b for b in overdue_ranked if b not in hot_picks[:3]][:2]
            chosen_main = sorted(hot_picks[:3] + top_overdue)

        elif strat.startswith("Pair Association"):
            # Strategy 5: Anchor on the #1 predicted ball, add its best historical co-occurring partners
            anchor = white_ranked[0][0]
            partners = sorted(
                [b for b in range(MAIN_MIN, MAIN_MAX + 1) if b != anchor],
                key=lambda b: (cooc[anchor - 1, b - 1], main_probs[b - 1]),
                reverse=True,
            )
            chosen_main = sorted([anchor] + partners[:4])

        # Fallback if duplicate combination
        combo_key = (tuple(chosen_main), chosen_extra)
        if combo_key in seen_combos or len(chosen_main) != 5:
            # Sample weighted by main_probs
            avail = list(range(MAIN_MIN, MAIN_MAX + 1))
            weights = main_norm_probs.copy()
            chosen_main = sorted(
                np_rng.choice(avail, size=5, replace=False, p=weights).tolist()
            )

        seen_combos.add((tuple(chosen_main), chosen_extra))

        # Calculate exact probabilities for this ticket
        # P(at least 1 WB hit) = 1 - P(0 WB hits)
        # Assuming Bernoulli trials approximated by sum of inclusion probabilities
        wb_probs = [main_probs[b - 1] for b in chosen_main]
        prob_0_hits = math.prod(max(0.0, 1.0 - p) for p in wb_probs)
        prob_at_least_1_wb = float(1.0 - prob_0_hits)

        extra_prob = float(extra_probs[chosen_extra - 1])

        # Joint target probability: P(Match Extra AND Match >= 1 WB)
        joint_target_prob = extra_prob * prob_at_least_1_wb
        composite_score = joint_target_prob * (1.0 + np.mean(wb_probs))

        ticket = RankedTicket(
            ticket_number=t_idx + 1,
            main_numbers=chosen_main,
            extra=chosen_extra,
            match_score=float(composite_score),
            white_ball_match_prob=prob_at_least_1_wb,
            extra_ball_match_prob=extra_prob,
            joint_target_prob=joint_target_prob,
            strategy=strat,
        )
        generated_tickets.append(ticket)

    # Sort tickets by match_score descending
    generated_tickets.sort(key=lambda t: t.match_score, reverse=True)
    for idx, t in enumerate(generated_tickets):
        t.ticket_number = idx + 1

    return PredictionResult(
        extra_probs=extra_probs,
        main_probs=main_probs,
        predicted_extra=predicted_extra,
        top_extras=extra_ranked,
        top_white_balls=white_ranked[:10],
        tickets=generated_tickets,
    )


# ---------------------------------------------------------------------------
# Historical Walk-Forward Backtesting
# ---------------------------------------------------------------------------
def evaluate_historical_accuracy(
    main_history: np.ndarray,
    extra_history: np.ndarray,
    window: int = DEFAULT_WINDOW,
    eval_draws: int = 40,
    num_tickets: int = 5,
    epochs: int = 50,
) -> Dict[str, float]:
    """
    Evaluates MLX model accuracy across historical draws using walk-forward testing.
    Measures success rate of achieving:
      - Matched Millionaire Ball
      - Matched Millionaire Ball + At least 1 White Ball (1+MB Target)
      - Matched Millionaire Ball + At least 2 White Balls (2+MB)
    """
    total = len(main_history)
    start_idx = max(window + 10, total - eval_draws)
    num_eval = total - start_idx

    if num_eval <= 0:
        return {"eval_draws": 0}

    matched_extra_count = 0
    matched_target_count = 0  # Matched Extra + >= 1 WB
    matched_2plus_extra_count = 0  # Matched Extra + >= 2 WB
    matched_any_wb_count = 0

    print(f"\nRunning walk-forward historical simulation on {num_eval} past draws...")

    for i in range(start_idx, total):
        m_train = main_history[:i]
        e_train = extra_history[:i]

        actual_main = set(main_history[i])
        actual_extra = extra_history[i]

        x_tr, ym_tr, ye_tr = build_dataset(m_train, e_train, window)
        model = train_mlx_model(x_tr, ym_tr, ye_tr, epochs=epochs, lr=DEFAULT_LR, verbose=False)

        pred = predict_next_draw(
            model,
            m_train,
            e_train,
            window=window,
            num_tickets=num_tickets,
            seed=42 + i,
        )

        # Check ticket hits
        hit_extra = any(t.extra == actual_extra for t in pred.tickets)
        hit_target = any(
            t.extra == actual_extra and len(set(t.main_numbers) & actual_main) >= 1
            for t in pred.tickets
        )
        hit_2plus = any(
            t.extra == actual_extra and len(set(t.main_numbers) & actual_main) >= 2
            for t in pred.tickets
        )
        hit_any_wb = any(
            len(set(t.main_numbers) & actual_main) >= 1 for t in pred.tickets
        )

        if hit_extra:
            matched_extra_count += 1
        if hit_target:
            matched_target_count += 1
        if hit_2plus:
            matched_2plus_extra_count += 1
        if hit_any_wb:
            matched_any_wb_count += 1

    return {
        "eval_draws": num_eval,
        "matched_extra_rate": (matched_extra_count / num_eval) * 100.0,
        "matched_target_rate": (matched_target_count / num_eval) * 100.0,
        "matched_2plus_rate": (matched_2plus_extra_count / num_eval) * 100.0,
        "matched_any_wb_rate": (matched_any_wb_count / num_eval) * 100.0,
    }


# ---------------------------------------------------------------------------
# Formatting & Output
# ---------------------------------------------------------------------------
def print_prediction_report(res: PredictionResult, total_draws: int):
    """Prints a clear, comprehensive prediction and ticket candidate report."""
    print("=" * 80)
    print("  MILLIONAIRE FOR LIFE - MLX NEXT DRAW PREDICTOR")
    print("  Target Goal: Matched Millionaire Ball + At Least 1 White Ball")
    print("=" * 80)
    print(f"Historical Draws Analyzed: {total_draws}")

    # 1. Millionaire Ball Prediction
    print("\n[1] MILLIONAIRE BALL (EXTRA 1 - 5) PREDICTION & PROBABILITIES:")
    print("  ----------------------------------------------------------------------")
    print("  Rank  Millionaire Ball   Probability   Confidence Bar")
    print("  ----------------------------------------------------------------------")
    for r_idx, (extra_val, prob) in enumerate(res.top_extras, 1):
        bar = "█" * int(prob * 40)
        marker = " ◄ [PRIMARY SELECTION]" if r_idx == 1 else ""
        print(f"   #{r_idx:<3}       Ball {extra_val:<3}       {prob*100:>5.2f}%       {bar:<20}{marker}")

    # 2. Top White Ball Predictions
    print("\n[2] TOP 10 PREDICTED WHITE BALLS (MAIN 1 - 58):")
    wb_line1 = ", ".join(f"#{b:02d} ({p*100:.1f}%)" for b, p in res.top_white_balls[:5])
    wb_line2 = ", ".join(f"#{b:02d} ({p*100:.1f}%)" for b, p in res.top_white_balls[5:10])
    print(f"  Ranks 1-5:  {wb_line1}")
    print(f"  Ranks 6-10: {wb_line2}")

    # 3. Recommended Tickets Targeting Matched Extra + >=1 White Ball
    print("\n[3] GENERATED PREDICTION TICKETS (OPTIMIZED FOR EXTRA + >=1 WHITE BALL):")
    print("  " + "-" * 76)
    print("  Tkt  White Balls (5)        MB   Target Prob   P(>=1 WB)  P(MB)   Strategy")
    print("  " + "-" * 76)

    for t in res.tickets:
        wb_str = " ".join(f"{b:02d}" for b in t.main_numbers)
        print(
            f"  #{t.ticket_number:<2}  [{wb_str}]  [{t.extra}]"
            f"   {t.joint_target_prob*100:>5.2f}%"
            f"       {t.white_ball_match_prob*100:>5.1f}%"
            f"   {t.extra_ball_match_prob*100:>4.1f}%"
            f"  {t.strategy}"
        )
    print("  " + "-" * 76)


def export_tickets_to_csv(tickets: List[RankedTicket], output_path: str | Path):
    """Exports generated tickets to CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in tickets:
        rows.append({
            "Ticket": t.ticket_number,
            "Num1": t.main_numbers[0],
            "Num2": t.main_numbers[1],
            "Num3": t.main_numbers[2],
            "Num4": t.main_numbers[3],
            "Num5": t.main_numbers[4],
            "Millionaire_Ball": t.extra,
            "Target_Match_Prob": f"{t.joint_target_prob*100:.2f}%",
            "White_Ball_Match_Prob": f"{t.white_ball_match_prob*100:.2f}%",
            "Extra_Ball_Match_Prob": f"{t.extra_ball_match_prob*100:.2f}%",
            "Strategy": t.strategy,
        })

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSuccessfully exported {len(tickets)} prediction tickets to: {path}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Predict Millionaire for Life draws containing a matched Millionaire Ball and at least one White Ball using MLX."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(find_default_csv()),
        help="Path to Millionaire for Life history CSV (default: auto-detected)",
    )
    parser.add_argument(
        "--tickets",
        type=int,
        default=5,
        help="Number of prediction tickets to generate (default: 5)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Rolling history window size for feature extraction (default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run walk-forward simulation backtest on historical draws",
    )
    parser.add_argument(
        "--backtest-draws",
        type=int,
        default=30,
        help="Number of past draws to evaluate during backtesting (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save generated tickets as a CSV file",
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        mx.random.seed(args.seed)
    except AttributeError:
        pass

    # Load data
    print(f"Loading Millionaire for Life history from: {args.csv}")
    main_hist, extra_hist, df = load_millionaire_history(args.csv)
    print(f"Loaded {len(main_hist)} historical draws.")

    if len(main_hist) <= args.window:
        raise ValueError(
            f"Not enough historical draws ({len(main_hist)}) for window size {args.window}."
        )

    # Train model
    print(f"Building features with rolling window = {args.window} draws...")
    x_train, y_main, y_extra = build_dataset(main_hist, extra_hist, args.window)

    print(f"Training MLX Neural Network for {args.epochs} epochs...")
    model = train_mlx_model(x_train, y_main, y_extra, epochs=args.epochs, lr=args.lr, verbose=True)

    # Generate predictions
    res = predict_next_draw(
        model,
        main_hist,
        extra_hist,
        window=args.window,
        num_tickets=args.tickets,
        seed=args.seed,
    )

    # Display Report
    print_prediction_report(res, total_draws=len(main_hist))

    # Optional Backtesting
    if args.backtest:
        bt_results = evaluate_historical_accuracy(
            main_hist,
            extra_hist,
            window=args.window,
            eval_draws=args.backtest_draws,
            num_tickets=args.tickets,
            epochs=args.epochs // 2,
        )
        print("\n" + "=" * 80)
        print("  HISTORICAL BACKTEST RESULTS (WALK-FORWARD VALIDATION)")
        print("=" * 80)
        print(f"  Draws Evaluated:                       {bt_results['eval_draws']}")
        print(f"  Matched Millionaire Ball Rate:         {bt_results['matched_extra_rate']:.1f}%")
        print(f"  Matched Target (MB + >=1 White Ball):  {bt_results['matched_target_rate']:.1f}%")
        print(f"  Matched (MB + >=2 White Balls):        {bt_results['matched_2plus_rate']:.1f}%")
        print(f"  Any White Ball Match Rate:             {bt_results['matched_any_wb_rate']:.1f}%")
        print("=" * 80)

    # Export to CSV if specified
    if args.output:
        export_tickets_to_csv(res.tickets, args.output)


if __name__ == "__main__":
    main()
