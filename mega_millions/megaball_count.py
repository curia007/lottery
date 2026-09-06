#!/usr/bin/env python3
"""
megaball_count.py

Analyzes Mega Millions historical draw data to count occurrences of each Mega Ball
and predict next Mega Ball candidates based on low hits, high hits, and balanced hits.

Usage:
    python megaball_count.py
    python megaball_count.py --predict
    python megaball_count.py --predict-mode high
    python megaball_count.py --predict-mode low
    python megaball_count.py --predict-mode balanced
    python megaball_count.py --predict-mode all --top 5
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

PredictMode = Literal["high", "low", "balanced", "all"]


@dataclass
class MegaBallPrediction:
    rank: int
    megaball: int
    mode: str
    score: float
    frequency: int
    percentage: float
    draws_since_last: int


def find_default_csv() -> Path:
    """Find the default Mega Millions history CSV file."""
    candidates = [
        Path(__file__).resolve().parent / "data" / "mega_millions_history.csv",
        Path(__file__).resolve().parents[1] / "mega_millions" / "data" / "mega_millions_history.csv",
        Path("mega_millions/data/mega_millions_history.csv"),
        Path("data/mega_millions_history.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def extract_ints(value) -> list[int]:
    """Extract all integers from a string or value."""
    return [int(x) for x in re.findall(r"\b\d{1,2}\b", str(value))]


def load_megaball_data(csv_path: str | Path) -> pd.Series:
    """
    Load Mega Millions CSV data and extract the MegaBall numbers.
    Supports various column naming schemes.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Mega Millions CSV file not found at: {path}")

    df = pd.read_csv(path)

    # Check for direct column match (case-insensitive)
    cols_lower = {c.lower().strip().replace(" ", "").replace("_", ""): c for c in df.columns}
    mega_col = (
        cols_lower.get("megaball")
        or cols_lower.get("mega")
        or cols_lower.get("meganumber")
        or cols_lower.get("megaballnumber")
        or cols_lower.get("mb")
        or cols_lower.get("ball6")
    )

    if mega_col:
        megaballs = pd.to_numeric(df[mega_col], errors="coerce").dropna().astype(int)
        return megaballs

    # Fallback: look for 6th number in multi-number columns
    winning_col = (
        cols_lower.get("winningnumbers")
        or cols_lower.get("numbers")
        or cols_lower.get("winningnumber")
        or cols_lower.get("result")
        or cols_lower.get("results")
    )
    if winning_col:
        mb_list = []
        for val in df[winning_col].dropna():
            nums = extract_ints(val)
            if len(nums) >= 6:
                mb_list.append(nums[-1])
        if mb_list:
            return pd.Series(mb_list, name="MegaBall", dtype=int)

    raise ValueError(f"Could not locate MegaBall column in {path}. Columns found: {list(df.columns)}")


def count_megaballs(
    megaballs: pd.Series,
    min_num: int = 1,
    max_num: int | None = None,
    sort_by: str = "count",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Calculate counts and percentages for each Mega Ball number.

    Args:
        megaballs: Series containing historical Mega Ball draws.
        min_num: Minimum Mega Ball number (default 1).
        max_num: Maximum Mega Ball number (defaults to max in data or 25).
        sort_by: 'number' or 'count' (default 'count').
        ascending: Sort order direction (default False for highest to lowest).

    Returns:
        DataFrame with columns: ['MegaBall', 'Count', 'Percentage']
    """
    if max_num is None:
        max_in_data = megaballs.max() if len(megaballs) > 0 else 25
        max_num = max(25, int(max_in_data))

    total_draws = len(megaballs)
    value_counts = megaballs.value_counts()

    rows = []
    for num in range(min_num, max_num + 1):
        cnt = int(value_counts.get(num, 0))
        pct = (cnt / total_draws * 100.0) if total_draws > 0 else 0.0
        rows.append({"MegaBall": num, "Count": cnt, "Percentage": round(pct, 2)})

    result_df = pd.DataFrame(rows)

    if sort_by.lower() == "count":
        result_df = result_df.sort_values(by=["Count", "MegaBall"], ascending=[ascending, True])
    else:
        result_df = result_df.sort_values(by=["MegaBall"], ascending=[ascending])

    result_df = result_df.reset_index(drop=True)
    return result_df


def compute_draws_since_last(megaballs: pd.Series, min_num: int = 1, max_num: int = 25) -> dict[int, int]:
    """
    Computes the number of draws since each Mega Ball number was last drawn.
    Assumes megaballs is in reverse chronological order (newest draw first at index 0).
    """
    arr = megaballs.to_numpy()
    total_draws = len(arr)
    gaps: dict[int, int] = {}

    for num in range(min_num, max_num + 1):
        indices = np.where(arr == num)[0]
        if len(indices) > 0:
            gaps[num] = int(indices[0])
        else:
            gaps[num] = total_draws
    return gaps


def predict_megaball(
    megaballs: pd.Series,
    mode: str = "all",
    window: int = 0,
    top_n: int = 5,
    min_num: int = 1,
    max_num: int | None = None,
) -> dict[str, list[MegaBallPrediction]]:
    """
    Predict next Mega Ball hit candidates based on low hits, high hits, and balanced hits.

    Modes:
      - 'high' (High Hits / Hot): Favors most frequently drawn Mega Balls.
      - 'low' (Low Hits / Cold / Overdue): Favors numbers with least hits and largest gaps since last hit.
      - 'balanced' (Balanced Hits): Blends overall frequency, recent window frequency, and overdue gap.
      - 'all': Returns predictions for high, low, and balanced hit strategies.

    Args:
        megaballs: Series containing historical Mega Ball draws (newest first).
        mode: 'high', 'low', 'balanced', or 'all'.
        window: Recent draw window to calculate localized frequency (0 = full history or auto 50).
        top_n: Number of predictions to return per mode.
        min_num: Minimum Mega Ball number (default 1).
        max_num: Maximum Mega Ball number (default 25 or data max).

    Returns:
        Dictionary mapping mode name to list of Ranked MegaBallPrediction objects.
    """
    if max_num is None:
        max_in_data = megaballs.max() if len(megaballs) > 0 else 25
        max_num = max(25, int(max_in_data))

    total_draws = len(megaballs)
    all_counts = megaballs.value_counts()
    gaps = compute_draws_since_last(megaballs, min_num, max_num)

    # Windowed frequency
    eff_window = window if window > 0 else min(50, total_draws)
    recent_mb = megaballs.iloc[:eff_window] if total_draws > 0 else megaballs
    recent_counts = recent_mb.value_counts()

    # Normalized metrics (0.0 to 1.0)
    max_all_cnt = max(all_counts.max() if len(all_counts) > 0 else 1, 1)
    min_all_cnt = min(all_counts.min() if len(all_counts) > 0 else 0, max_all_cnt)
    max_gap = max(max(gaps.values()) if gaps else 1, 1)

    all_numbers = list(range(min_num, max_num + 1))
    predictions: dict[str, list[MegaBallPrediction]] = {}

    modes_to_run = ["high", "low", "balanced"] if mode in ("all", "both") else [mode.lower()]

    for m in modes_to_run:
        scored_candidates = []
        for num in all_numbers:
            cnt = int(all_counts.get(num, 0))
            pct = (cnt / total_draws * 100.0) if total_draws > 0 else 0.0
            gap = gaps[num]
            rec_cnt = int(recent_counts.get(num, 0))

            # Standardized components (0 to 1)
            norm_freq = cnt / max_all_cnt
            norm_gap = gap / max_gap
            norm_rec_freq = rec_cnt / eff_window if eff_window > 0 else 0.0
            # Low hit component: 1.0 for fewest hits, down to 0.0 for most hits
            norm_low_freq = 1.0 - ((cnt - min_all_cnt) / max(1, (max_all_cnt - min_all_cnt)))

            if m in ("high", "high_hits", "hot"):
                # High hits strategy: prioritize high total frequency and recent momentum
                score = 0.65 * norm_freq + 0.35 * norm_rec_freq
                mode_key = "high_hits"
            elif m in ("low", "low_hits", "cold", "overdue"):
                # Low hits strategy: prioritize low hit frequency and overdue draws
                score = 0.50 * norm_low_freq + 0.50 * norm_gap
                mode_key = "low_hits"
            elif m in ("balanced", "balance", "balanced_hits"):
                # Balanced hits strategy: balanced blend of baseline frequency, recent trend, and gap
                score = 0.35 * norm_freq + 0.35 * norm_rec_freq + 0.30 * norm_gap
                mode_key = "balanced_hits"
            else:
                score = norm_freq
                mode_key = m

            scored_candidates.append({
                "megaball": num,
                "score": float(score),
                "frequency": cnt,
                "percentage": round(pct, 2),
                "draws_since_last": gap,
                "mode_key": mode_key,
            })

        # Sort descending by score, tiebreak by MegaBall number
        scored_candidates.sort(key=lambda x: (x["score"], -x["megaball"]), reverse=True)

        mode_key = scored_candidates[0]["mode_key"]
        ranked_list = []
        for rank, item in enumerate(scored_candidates[:top_n], start=1):
            ranked_list.append(
                MegaBallPrediction(
                    rank=rank,
                    megaball=item["megaball"],
                    mode=mode_key,
                    score=round(item["score"], 4),
                    frequency=item["frequency"],
                    percentage=item["percentage"],
                    draws_since_last=item["draws_since_last"],
                )
            )
        predictions[mode_key] = ranked_list

    return predictions


def print_counts_table(counts_df: pd.DataFrame, total_draws: int) -> None:
    """Print the Mega Ball counts in a clean, formatted table."""
    print("=" * 45)
    print("        MEGA MILLIONS - MEGABALL COUNTS      ")
    print("=" * 45)
    print(f"Total draws analyzed: {total_draws}")
    print("-" * 45)
    print(f"{'MegaBall':<12} {'Count':<10} {'Percentage':<12}")
    print("-" * 45)
    for _, row in counts_df.iterrows():
        mb = int(row["MegaBall"])
        cnt = int(row["Count"])
        pct = row["Percentage"]
        bar = "█" * int(pct // 0.5) if pct > 0 else ""
        print(f"Mega Ball {mb:<2} : {cnt:>5} draws ({pct:>5.2f}%)  {bar}")
    print("=" * 45)


def print_predictions_table(predictions: dict[str, list[MegaBallPrediction]]) -> None:
    """Print predicted Mega Ball hits for specified hit strategies."""
    headers = {
        "high_hits": "PREDICTED NEXT MEGABALL - HIGH HITS (HOT / FREQUENT)",
        "low_hits": "PREDICTED NEXT MEGABALL - LOW HITS (COLD / OVERDUE)",
        "balanced_hits": "PREDICTED NEXT MEGABALL - BALANCED HITS (WEIGHTED BLEND)",
    }

    for mode_key, pred_list in predictions.items():
        title = headers.get(mode_key, f"PREDICTED NEXT MEGABALL - {mode_key.upper()}")
        print()
        print("=" * 64)
        print(f"  {title}")
        print("=" * 64)
        print(f"{'Rank':<6}{'MegaBall':<12}{'Score':<10}{'Hits':<8}{'Percent':<10}{'Last Seen':<14}")
        print("-" * 64)
        for p in pred_list:
            seen_str = f"{p.draws_since_last} draw{'s' if p.draws_since_last != 1 else ''} ago"
            print(
                f"{p.rank:<6}"
                f"Ball {p.megaball:<6}"
                f"{p.score:<10.4f}"
                f"{p.frequency:<8}"
                f"{p.percentage:>5.2f}%    "
                f"{seen_str:<14}"
            )
        print("-" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce Mega Ball counts and predict the next Mega Ball hit based on low, high, and balanced hits."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to Mega Millions CSV history file. Defaults to data/mega_millions_history.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as CSV.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=["number", "count"],
        default="count",
        help="Sort results by 'count' (default) or 'number'.",
    )
    parser.add_argument(
        "--asc",
        action="store_true",
        help="Sort in ascending order (lowest to highest) instead of descending.",
    )
    parser.add_argument(
        "--desc",
        action="store_true",
        help="Sort in descending order (highest to lowest, default).",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=1,
        help="Minimum Mega Ball number (default: 1).",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum Mega Ball number (default: auto/25).",
    )

    # Prediction parameters
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run Mega Ball prediction across hit strategies.",
    )
    parser.add_argument(
        "--predict-mode",
        type=str,
        choices=["high", "low", "balanced", "all"],
        default="low",
        help="Prediction mode: 'high' (high hits), 'low' (low hits), 'balanced' (balanced hits), or 'all'.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top Mega Ball predictions to show (default: 5).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Recent draw window size used for trend calculations (default: 50).",
    )
    parser.add_argument(
        "--no-counts",
        action="store_true",
        help="Suppress the full historical counts table and display only predictions.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else find_default_csv()
    if not csv_path.exists():
        print(f"Error: Mega Millions history file not found at '{csv_path}'.")
        return

    megaballs = load_megaball_data(csv_path)

    # If --predict-mode is passed or --predict is set, run predictions
    should_predict = args.predict or (args.predict_mode is not None)
    pred_mode = args.predict_mode or "all"

    if not args.no_counts:
        ascending = args.asc and not args.desc
        counts_df = count_megaballs(
            megaballs=megaballs,
            min_num=args.min,
            max_num=args.max,
            sort_by=args.sort,
            ascending=ascending,
        )
        print_counts_table(counts_df, total_draws=len(megaballs))

        if args.output and not should_predict:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            counts_df.to_csv(out_path, index=False)
            print(f"\nSaved Mega Ball counts to: {out_path}")

    if should_predict:
        predictions = predict_megaball(
            megaballs=megaballs,
            mode=pred_mode,
            window=args.window,
            top_n=args.top,
            min_num=args.min,
            max_num=args.max,
        )
        print_predictions_table(predictions)

        if args.output:
            pred_rows = []
            for m_key, p_list in predictions.items():
                for p in p_list:
                    pred_rows.append({
                        "Mode": m_key,
                        "Rank": p.rank,
                        "MegaBall": p.megaball,
                        "Score": p.score,
                        "TotalHits": p.frequency,
                        "Percentage": p.percentage,
                        "DrawsSinceLast": p.draws_since_last,
                    })
            pred_df = pd.DataFrame(pred_rows)
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(out_path, index=False)
            print(f"\nSaved predictions to: {out_path}")


if __name__ == "__main__":
    main()
