#!/usr/bin/env python3
"""
millionaire_patterns_mlx.py

Discovers and analyzes numerical patterns in the Millionaire Life lottery balls (Main Balls 1-58
and Millionaire Ball/Extra 1-5) from historical draw records using Apple MLX tensor calculations.

Key Patterns Analyzed:
1. Main Ball and Millionaire Ball (Extra) frequencies, probabilities, and hot/cold rankings.
2. Back-to-back consecutive draw repeats (single, double, and triple ball overlaps).
3. Intra-draw consecutive number sequences (runs like 10-11, 23-24-25) and sequence frequencies.
4. Position-wise / Slot-wise consecutive repeats.
5. Draw arithmetic patterns: Sum distributions, spread (range), and sum bucket frequencies.
6. Odd / Even distribution ratios (e.g. 3:2, 2:3, 4:1) across historical draws.
7. High / Low distribution ratios (Low: 1-29, High: 30-58).
8. Decade / Tens distribution (1-9, 10-19, 20-29, 30-39, 40-49, 50-59).
9. Last digit (ending digit 0-9) patterns and repeating ending digits.
10. Pair co-occurrence matrix (58x58) computed via MLX matrix multiplication.
11. Overdue / Gap analysis (draws since last hit per number).
12. Pattern-based next ticket generator adhering to dominant historical patterns.

Usage:
    python millionaire_life/millionaire_patterns_mlx.py
    python millionaire_life/millionaire_patterns_mlx.py --generate-tickets 5
    python millionaire_life/millionaire_patterns_mlx.py --save-counts
    python millionaire_life/millionaire_patterns_mlx.py --output millionaire_patterns_summary.csv
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import mlx.core as mx
except ImportError as exc:
    raise SystemExit(
        "MLX is required. Install it with: pip install mlx"
    ) from exc

MAIN_MIN = 1
MAIN_MAX = 58
MAIN_COUNT = 5
EXTRA_MIN = 1
EXTRA_MAX = 5


@dataclass
class PatternReport:
    total_draws: int
    main_counts: np.ndarray
    extra_counts: np.ndarray
    main_gaps: np.ndarray
    extra_gaps: np.ndarray
    consecutive_draw_repeats_main: np.ndarray
    consecutive_draw_repeats_extra: np.ndarray
    pos_repeats: List[int]
    consecutive_runs_count: Dict[int, int]
    draws_with_consecutive_runs: int
    back_to_back_consecutive_runs: int
    odd_even_ratios: Dict[Tuple[int, int], int]
    high_low_ratios: Dict[Tuple[int, int], int]
    sum_mean: float
    sum_std: float
    sum_min: int
    sum_max: int
    sum_buckets: Dict[str, int]
    spread_mean: float
    decade_counts: Dict[str, int]
    last_digit_counts: Dict[int, int]
    top_pairs: List[Tuple[int, int, int]]
    bottom_pairs: List[Tuple[int, int, int]]
    cooccurrence_matrix: np.ndarray


def find_default_csv() -> Path:
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


def load_millionaire_history(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads Millionaire Life history and returns main balls, extra ball, and DataFrame.
    Data is arranged in chronological order (oldest to newest).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Millionaire Life history CSV not found: {path}")

    df = pd.read_csv(path)
    main_cols = ["Num1", "Num2", "Num3", "Num4", "Num5"]

    for col in main_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Extra" in df.columns:
        df["Extra"] = pd.to_numeric(df["Extra"], errors="coerce")
    else:
        df["Extra"] = np.nan

    df = df.dropna(subset=main_cols)

    # Sort chronologically (oldest to newest)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date", ascending=True).reset_index(drop=True)
    else:
        # Default CSV is newest first, so reverse if no Date
        df = df.iloc[::-1].reset_index(drop=True)

    main_data = df[main_cols].values.astype(int)
    extra_data = df["Extra"].fillna(1).values.astype(int)

    return main_data, extra_data, df


def analyze_patterns_mlx(main_data: np.ndarray, extra_data: np.ndarray) -> PatternReport:
    """
    Analyzes all numerical patterns in Millionaire Life draws using MLX tensors.
    """
    num_draws = main_data.shape[0]
    if num_draws == 0:
        raise ValueError("Cannot analyze empty dataset.")

    # Convert to MLX tensors
    mx_main = mx.array(main_data)  # Shape: (N, 5)
    mx_extra = mx.array(extra_data)  # Shape: (N,)

    # 1. Main Ball Hit Counts using Multi-Hot Encoding
    # Range 1..58
    main_range = mx.arange(MAIN_MIN, MAIN_MAX + 1)  # (58,)
    multi_hot = mx.zeros((num_draws, MAIN_MAX))  # (N, 58)
    for col in range(MAIN_COUNT):
        multi_hot += (mx_main[:, col, None] == main_range).astype(mx.float32)

    main_counts_mx = mx.sum(multi_hot, axis=0)  # (58,)
    main_counts = np.array(main_counts_mx, dtype=int)

    # 2. Millionaire Extra Ball Hit Counts
    extra_range = mx.arange(EXTRA_MIN, EXTRA_MAX + 1)  # (5,)
    extra_one_hot = (mx_extra[:, None] == extra_range).astype(mx.float32)  # (N, 5)
    extra_counts_mx = mx.sum(extra_one_hot, axis=0)
    extra_counts = np.array(extra_counts_mx, dtype=int)

    # 3. Consecutive Repeats Across Consecutive Draws (Draw t vs Draw t+1)
    present_main = multi_hot > 0  # (N, 58)
    consec_main_mx = mx.sum(present_main[1:] & present_main[:-1], axis=0)
    consec_repeats_main = np.array(consec_main_mx, dtype=int)

    present_extra = extra_one_hot > 0  # (N, 5)
    consec_extra_mx = mx.sum(present_extra[1:] & present_extra[:-1], axis=0)
    consec_repeats_extra = np.array(consec_extra_mx, dtype=int)

    # 4. Position-Wise Consecutive Repeats
    pos_repeats = []
    for col in range(MAIN_COUNT):
        col_matches = mx.sum(mx_main[1:, col] == mx_main[:-1, col]).item()
        pos_repeats.append(int(col_matches))

    # 5. Overdue / Gap Analysis (draws since last hit)
    main_gaps = np.zeros(MAIN_MAX, dtype=int)
    for num_idx in range(MAIN_MAX):
        hits = np.where(np.array(present_main[:, num_idx]))[0]
        if len(hits) > 0:
            main_gaps[num_idx] = (num_draws - 1) - hits[-1]
        else:
            main_gaps[num_idx] = num_draws

    extra_gaps = np.zeros(EXTRA_MAX, dtype=int)
    for ext_idx in range(EXTRA_MAX):
        hits = np.where(np.array(present_extra[:, ext_idx]))[0]
        if len(hits) > 0:
            extra_gaps[ext_idx] = (num_draws - 1) - hits[-1]
        else:
            extra_gaps[ext_idx] = num_draws

    # 6. Intra-Draw Consecutive Number Sequences (Runs e.g. 10-11, 23-24-25)
    sorted_draws = mx.sort(mx_main, axis=1)  # (N, 5)
    diffs = sorted_draws[:, 1:] - sorted_draws[:, :-1]  # (N, 4)
    is_consec = diffs == 1  # (N, 4)

    # Count runs of length 2, 3, 4, 5
    consec_runs_count = {2: 0, 3: 0, 4: 0, 5: 0}
    is_consec_np = np.array(is_consec)
    draws_with_runs_mask = np.zeros(num_draws, dtype=bool)

    for i in range(num_draws):
        row_consec = is_consec_np[i]  # shape (4,)
        if np.any(row_consec):
            draws_with_runs_mask[i] = True

        # Check run lengths
        # Look at consecutive trues in row_consec
        # 4 diffs: e.g. [True, True, False, False] -> 3 consecutive numbers (run of 3)
        # [True, False, True, False] -> two runs of 2
        run_len = 1
        for val in row_consec:
            if val:
                run_len += 1
            else:
                if run_len >= 2:
                    consec_runs_count[min(run_len, 5)] += 1
                run_len = 1
        if run_len >= 2:
            consec_runs_count[min(run_len, 5)] += 1

    draws_with_consecutive_runs = int(np.sum(draws_with_runs_mask))
    mx_draws_with_runs = mx.array(draws_with_runs_mask)
    back_to_back_runs = int(mx.sum(mx_draws_with_runs[1:] & mx_draws_with_runs[:-1]).item())

    # 7. Draw Arithmetic: Sum & Spread
    draw_sums_mx = mx.sum(mx_main, axis=1)  # (N,)
    draw_sums = np.array(draw_sums_mx, dtype=float)
    sum_mean = float(np.mean(draw_sums))
    sum_std = float(np.std(draw_sums))
    sum_min = int(np.min(draw_sums))
    sum_max = int(np.max(draw_sums))

    sum_buckets = {
        "< 100": int(np.sum(draw_sums < 100)),
        "100 - 130": int(np.sum((draw_sums >= 100) & (draw_sums <= 130))),
        "131 - 160": int(np.sum((draw_sums >= 131) & (draw_sums <= 160))),
        "161 - 190": int(np.sum((draw_sums >= 161) & (draw_sums <= 190))),
        "> 190": int(np.sum(draw_sums > 190)),
    }

    spreads_mx = mx.max(mx_main, axis=1) - mx.min(mx_main, axis=1)
    spread_mean = float(mx.mean(spreads_mx).item())

    # 8. Odd / Even Ratios
    odd_counts_mx = mx.sum(mx_main % 2 == 1, axis=1)  # (N,)
    odd_counts = np.array(odd_counts_mx, dtype=int)
    odd_even_ratios = {}
    for odd_k in range(MAIN_COUNT + 1):
        even_k = MAIN_COUNT - odd_k
        odd_even_ratios[(odd_k, even_k)] = int(np.sum(odd_counts == odd_k))

    # 9. High / Low Ratios (Low: 1-29, High: 30-58)
    high_counts_mx = mx.sum(mx_main >= 30, axis=1)  # (N,)
    high_counts = np.array(high_counts_mx, dtype=int)
    high_low_ratios = {}
    for high_k in range(MAIN_COUNT + 1):
        low_k = MAIN_COUNT - high_k
        high_low_ratios[(high_k, low_k)] = int(np.sum(high_counts == high_k))

    # 10. Decade / Tens Distribution
    decade_buckets = {
        "1 - 9": (1, 9),
        "10 - 19": (10, 19),
        "20 - 29": (20, 29),
        "30 - 39": (30, 39),
        "40 - 49": (40, 49),
        "50 - 58": (50, 58),
    }
    decade_counts = {}
    for name, (low, high) in decade_buckets.items():
        count = int(np.sum((main_data >= low) & (main_data <= high)))
        decade_counts[name] = count

    # 11. Last Digit (0-9) Distribution
    last_digits_flat = np.array(mx_main % 10).flatten()
    last_digit_counts = {}
    for d in range(10):
        last_digit_counts[d] = int(np.sum(last_digits_flat == d))

    # 12. Pair Co-occurrence Matrix (58x58) via MLX Matrix Multiplication: multi_hot.T @ multi_hot
    cooc_mx = mx.matmul(multi_hot.T, multi_hot)  # (58, 58)
    cooc_np = np.array(cooc_mx, dtype=int)
    # Zero out diagonal
    np.fill_diagonal(cooc_np, 0)

    pairs_list = []
    for i in range(MAIN_MAX):
        for j in range(i + 1, MAIN_MAX):
            pairs_list.append((i + 1, j + 1, int(cooc_np[i, j])))

    pairs_list.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs_list[:10]
    bottom_pairs = pairs_list[-10:]

    return PatternReport(
        total_draws=num_draws,
        main_counts=main_counts,
        extra_counts=extra_counts,
        main_gaps=main_gaps,
        extra_gaps=extra_gaps,
        consecutive_draw_repeats_main=consec_repeats_main,
        consecutive_draw_repeats_extra=consec_repeats_extra,
        pos_repeats=pos_repeats,
        consecutive_runs_count=consec_runs_count,
        draws_with_consecutive_runs=draws_with_consecutive_runs,
        back_to_back_consecutive_runs=back_to_back_runs,
        odd_even_ratios=odd_even_ratios,
        high_low_ratios=high_low_ratios,
        sum_mean=sum_mean,
        sum_std=sum_std,
        sum_min=sum_min,
        sum_max=sum_max,
        sum_buckets=sum_buckets,
        spread_mean=spread_mean,
        decade_counts=decade_counts,
        last_digit_counts=last_digit_counts,
        top_pairs=top_pairs,
        bottom_pairs=bottom_pairs,
        cooccurrence_matrix=cooc_np,
    )


def print_pattern_report(report: PatternReport):
    """
    Prints a rich, structured pattern analysis report to the console.
    """
    total = report.total_draws
    print("=" * 70)
    print(f"  MILLIONAIRE LIFE LOTTERY - MLX NUMERICAL PATTERN ANALYSIS")
    print(f"  Total Historical Draws Analyzed: {total}")
    print("=" * 70)

    # 1. Main Ball Frequency Overview
    print("\n[1] MAIN BALLS FREQUENCY & RANKINGS (1 - 58):")
    main_df = pd.DataFrame({
        "Ball": range(1, MAIN_MAX + 1),
        "Count": report.main_counts,
        "Gap": report.main_gaps,
        "Consecutive_Repeats": report.consecutive_draw_repeats_main,
    })
    sorted_hot = main_df.sort_values(by=["Count", "Consecutive_Repeats"], ascending=[False, False])
    sorted_overdue = main_df.sort_values(by="Gap", ascending=False)

    top5_hot = sorted_hot.head(5)
    top5_cold = sorted_hot.tail(5).iloc[::-1]
    top5_overdue = sorted_overdue.head(5)

    print("  Top 5 Hot Balls:     " + ", ".join(f"#{row.Ball} ({int(row.Count)} hits)" for _, row in top5_hot.iterrows()))
    print("  Top 5 Cold Balls:    " + ", ".join(f"#{row.Ball} ({int(row.Count)} hits)" for _, row in top5_cold.iterrows()))
    print("  Top 5 Overdue Balls: " + ", ".join(f"#{row.Ball} ({int(row.Gap)} draws ago)" for _, row in top5_overdue.iterrows()))

    # 2. Millionaire Extra Ball Analysis (Sorted by Consec Gap / Current Gap Ascending)
    print("\n[2] MILLIONAIRE BALL (EXTRA 1 - 5) PATTERN ANALYSIS:")
    extra_stats = []
    for b in range(1, EXTRA_MAX + 1):
        cnt = report.extra_counts[b - 1]
        pct = (cnt / total) * 100 if total > 0 else 0
        gap = report.extra_gaps[b - 1]
        consec = report.consecutive_draw_repeats_extra[b - 1]
        extra_stats.append({
            "ball": b,
            "cnt": cnt,
            "pct": pct,
            "gap": gap,
            "consec": consec,
        })
    # Sort by gap ascending, then consec repeats ascending, hits ascending, ball ascending
    extra_stats.sort(key=lambda x: (x["gap"], x["consec"], x["cnt"], x["ball"]))
    for s in extra_stats:
        print(f"  Millionaire Ball {s['ball']}: {s['cnt']:>3} hits ({s['pct']:>5.1f}%) | Current Gap: {s['gap']:>2} draws | Consec Repeats: {s['consec']:>2}")

    # 3. Intra-Draw Consecutive Numbers (Runs)
    print("\n[3] INTRA-DRAW CONSECUTIVE NUMBER RUNS (e.g. 10-11, 23-24-25):")
    runs_pct = (report.draws_with_consecutive_runs / total) * 100 if total > 0 else 0
    print(f"  Draws containing consecutive numbers: {report.draws_with_consecutive_runs} / {total} ({runs_pct:.1f}%)")
    print(f"  Back-to-back draws with runs:         {report.back_to_back_consecutive_runs}")
    print(f"  2-in-a-row sequences (Pairs):        {report.consecutive_runs_count[2]}")
    print(f"  3-in-a-row sequences (Triplets):     {report.consecutive_runs_count[3]}")
    print(f"  4-in-a-row sequences:                {report.consecutive_runs_count[4]}")

    # 4. Consecutive Draw Overlaps (Ball Repeats between Draw t and Draw t+1)
    print("\n[4] CONSECUTIVE DRAW OVERLAPS (Draw t vs Draw t+1):")
    total_consec_repeats = np.sum(report.consecutive_draw_repeats_main)
    print(f"  Total individual main ball back-to-back repeats: {total_consec_repeats}")
    print("  Position-wise repeats in consecutive draws:")
    for pos, count in enumerate(report.pos_repeats, 1):
        pct = (count / (total - 1)) * 100 if total > 1 else 0
        print(f"    Position {pos}: {count:>3} times ({pct:.2f}%)")

    # 5. Odd vs Even Distribution
    print("\n[5] ODD / EVEN PATTERN DISTRIBUTION:")
    for (odd_k, even_k), count in sorted(report.odd_even_ratios.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total) * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {odd_k} Odd / {even_k} Even : {count:>3} draws ({pct:>5.1f}%) {bar}")

    # 6. High vs Low Distribution (Low: 1-29, High: 30-58)
    print("\n[6] HIGH / LOW PATTERN DISTRIBUTION (Low: 1-29, High: 30-58):")
    for (high_k, low_k), count in sorted(report.high_low_ratios.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total) * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {high_k} High / {low_k} Low : {count:>3} draws ({pct:>5.1f}%) {bar}")

    # 7. Draw Sum and Spread Patterns
    print("\n[7] DRAW SUM & SPREAD ARITHMETIC PATTERNS:")
    print(f"  Sum Range: {report.sum_min} to {report.sum_max} | Average: {report.sum_mean:.1f} (StdDev: {report.sum_std:.1f})")
    print(f"  Average Number Spread (Max - Min): {report.spread_mean:.1f}")
    print("  Sum Distribution Buckets:")
    for bucket, count in report.sum_buckets.items():
        pct = (count / total) * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {bucket:>10}: {count:>3} draws ({pct:>5.1f}%) {bar}")

    # 8. Decade / Tens Distribution
    print("\n[8] DECADE / TENS DISTRIBUTION:")
    total_balls = total * MAIN_COUNT
    for decade, count in report.decade_counts.items():
        pct = (count / total_balls) * 100 if total_balls > 0 else 0
        print(f"  Decade {decade:>7}: {count:>4} balls ({pct:>5.1f}%)")

    # 9. Top Co-occurring Pairs
    print("\n[9] TOP 10 CO-OCCURRING BALL PAIRS (Drawn Together):")
    for rank, (b1, b2, cnt) in enumerate(report.top_pairs, 1):
        print(f"  {rank:>2}. Balls ({b1:>2}, {b2:>2}) - {cnt} times")

    print("=" * 70)


def generate_pattern_counts_dataframe(report: PatternReport) -> pd.DataFrame:
    """
    Builds a clean DataFrame of per-number metrics for saving to CSV.
    """
    total = report.total_draws
    df = pd.DataFrame({
        "Number": list(range(1, MAIN_MAX + 1)),
        "Count": report.main_counts,
        "Frequency_Pct": np.round((report.main_counts / total) * 100, 2) if total > 0 else 0,
        "Current_Gap": report.main_gaps,
        "Consecutive_Repeats": report.consecutive_draw_repeats_main,
    })

    # Sort by Count ascending then Number ascending (matches Idaho Cash standard format)
    return df.sort_values(by=["Count", "Number"], ascending=[True, True])


def generate_pattern_tickets(
    report: PatternReport,
    num_tickets: int = 5,
    seed: Optional[int] = None
) -> List[Tuple[List[int], int, float, str]]:
    """
    Generates candidate lottery tickets that match the most statistically dominant
    numerical patterns discovered in the historical Millionaire Life draws.
    """
    if seed is not None:
        np.random.seed(seed)

    # Determine dominant pattern characteristics from history
    # 1. Best Odd/Even ratio
    best_odd_even = max(report.odd_even_ratios.items(), key=lambda x: x[1])[0]  # (odd, even)
    # 2. Best High/Low ratio
    best_high_low = max(report.high_low_ratios.items(), key=lambda x: x[1])[0]  # (high, low)
    # 3. Sum bounds (within 1 std dev of mean)
    min_sum = int(report.sum_mean - report.sum_std)
    max_sum = int(report.sum_mean + report.sum_std)

    # 4. Weights for numbers based on Hot frequency + Overdue balance
    weights = report.main_counts.astype(float) + (report.main_gaps.astype(float) * 0.5)
    weights /= weights.sum()

    # 5. Best Millionaire Extra Ball (combining hotness and current gap)
    extra_weights = report.extra_counts.astype(float) + (report.extra_gaps.astype(float) * 0.4)
    extra_weights /= extra_weights.sum()

    tickets = []
    attempts = 0
    max_attempts = 10000

    while len(tickets) < num_tickets and attempts < max_attempts:
        attempts += 1
        # Sample 5 distinct numbers
        candidate = np.random.choice(range(1, MAIN_MAX + 1), size=MAIN_COUNT, replace=False, p=weights)
        candidate.sort()

        # Check Sum pattern
        c_sum = int(candidate.sum())
        if not (min_sum <= c_sum <= max_sum):
            continue

        # Check Odd/Even pattern
        odd_count = int(np.sum(candidate % 2 == 1))
        if odd_count != best_odd_even[0]:
            # Allow fallback to second best if needed
            if odd_count not in (2, 3):
                continue

        # Check High/Low pattern
        high_count = int(np.sum(candidate >= 30))
        if high_count not in (2, 3):
            continue

        # Sample Millionaire Ball
        extra_ball = int(np.random.choice(range(1, EXTRA_MAX + 1), p=extra_weights))

        # Check uniqueness
        cand_tuple = (tuple(candidate), extra_ball)
        if any(t[0] == list(candidate) and t[1] == extra_ball for t in tickets):
            continue

        # Compute score based on co-occurrence and individual ball weights
        score = float(np.sum([weights[n - 1] for n in candidate]))
        for i in range(MAIN_COUNT):
            for j in range(i + 1, MAIN_COUNT):
                n1, n2 = candidate[i], candidate[j]
                score += (report.cooccurrence_matrix[n1 - 1, n2 - 1] / max(report.total_draws, 1)) * 0.5

        pattern_desc = f"Sum={c_sum}, {odd_count}O/{MAIN_COUNT - odd_count}E, {high_count}H/{MAIN_COUNT - high_count}L"
        tickets.append((list(candidate), extra_ball, score, pattern_desc))

    # Sort tickets by pattern score descending
    tickets.sort(key=lambda x: x[2], reverse=True)
    return tickets


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and generate numerical patterns in Millionaire Life lottery balls using MLX"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(find_default_csv()),
        help="Path to millionaire_life_history.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save pattern report / summary CSV",
    )
    parser.add_argument(
        "--save-counts",
        action="store_true",
        help="Save number counts & stats to millionaire_life/data/number_counts.csv",
    )
    parser.add_argument(
        "--generate-tickets",
        type=int,
        default=5,
        help="Number of pattern-matching candidate tickets to generate (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for ticket generation",
    )

    args = parser.parse_args()

    # Load data
    main_data, extra_data, df = load_millionaire_history(args.csv)

    # Analyze patterns using MLX
    report = analyze_patterns_mlx(main_data, extra_data)

    # Print comprehensive pattern report
    print_pattern_report(report)

    # Generate pattern-matched candidate tickets
    if args.generate_tickets > 0:
        tickets = generate_pattern_tickets(report, num_tickets=args.generate_tickets, seed=args.seed)
        print("\n[10] PATTERN-OPTIMIZED CANDIDATE TICKETS FOR NEXT DRAW:")
        print("-" * 70)
        for rank, (nums, extra, score, desc) in enumerate(tickets, 1):
            nums_str = " ".join(f"{n:>2}" for n in nums)
            print(f"  Ticket #{rank}: [{nums_str}] + Millionaire Ball: {extra} | Score: {score:.4f} ({desc})")
        print("-" * 70)

    # Save number counts if requested or by default to data/number_counts.csv
    counts_df = generate_pattern_counts_dataframe(report)
    if args.save_counts:
        save_path = Path(args.csv).parent / "number_counts.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        counts_df.to_csv(save_path, index=False)
        print(f"\nSaved per-number pattern counts to: {save_path}")

    # Save summary report if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        counts_df.to_csv(out_path, index=False)
        print(f"Exported pattern data to: {out_path}")


if __name__ == "__main__":
    main()
