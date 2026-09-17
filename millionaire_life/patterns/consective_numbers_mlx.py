#!/usr/bin/env python3
"""
consective_numbers_mlx.py (Millionaire Life)

Calculates the number of times consecutive patterns occur in Millionaire Life history.
Analyzes:
1. Exact combination repeats in back-to-back draws.
2. Individual ball repeats per position / slot.
3. Intra-draw consecutive numerical sequences (e.g., 10-11, 23-24-25).
4. Frequency of hits happening consecutively across draws (per number).
5. Per-number hit counts and consecutive repeats saved to CSV.

Uses MLX for tensor-based calculations.
"""

from pathlib import Path
import pandas as pd

try:
    import mlx.core as mx
except ImportError:
    print("Error: MLX not found. Please install with 'pip install mlx'")
    exit(1)


def calculate_consecutive_hits():
    # Define paths
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / 'data' / 'millionaire_life_history.csv',
        script_dir / 'data' / 'millionaire_life_history.csv',
        Path('millionaire_life/data/millionaire_life_history.csv'),
        Path('data/millionaire_life_history.csv'),
        Path('../data/millionaire_life_history.csv'),
        Path('../../data/millionaire_life_history.csv'),
    ]
    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = c
            break
    if csv_path is None:
        csv_path = candidates[0]

    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    # Load history data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=cols)

    # Sort chronologically (oldest to newest)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date', ascending=True)
    else:
        df = df.iloc[::-1]

    digits = df[cols].values.astype(int)
    data = mx.array(digits)
    num_draws = data.shape[0]

    print(f"Analyzing {num_draws} Millionaire Life draws...")
    print("=" * 50)

    # 1. Exact Combination Repeats (5/5 straight match in consecutive draws)
    comb_repeats = mx.all(data[1:] == data[:-1], axis=1)
    total_comb_repeats = mx.sum(comb_repeats).item()

    # 2. Position-wise Consecutive Hits
    pos_repeats = []
    for i in range(5):
        matches = (data[1:, i] == data[:-1, i])
        pos_repeats.append(mx.sum(matches).item())

    # 3. Numerical Consecutive Hits in same draw (e.g. 10-11, 23-24-25)
    sorted_data = mx.sort(data, axis=1)
    diffs = sorted_data[:, 1:] - sorted_data[:, :-1]
    has_consecutive = mx.any(diffs == 1, axis=1)
    total_num_consecutive = mx.sum(has_consecutive).item()

    # Back-to-back consecutive number draws
    double_consecutive = mx.sum(has_consecutive[1:] & has_consecutive[:-1]).item()

    # 4. Per-Number Hit Counts (1-58)
    all_digits_flat = data.flatten()
    hit_counts = []
    for n in range(1, 59):
        hit_counts.append(int(mx.sum(all_digits_flat == n).item()))

    # Consecutive repeats: number of times a number appears in draw i and draw i+1
    draw_counts = mx.zeros((num_draws, 58))
    digit_range = mx.array(list(range(1, 59)))
    for i in range(5):
        draw_counts += (data[:, i, None] == digit_range).astype(mx.float32)

    present = draw_counts > 0
    consecutive_hits = mx.sum(present[1:] & present[:-1], axis=0)
    consecutive_list = [int(c.item()) for c in consecutive_hits]

    # Repeat hits within draw (should be 0 for standard lotto)
    repeat_hits = mx.sum(mx.maximum(draw_counts - 1, 0), axis=0)
    repeat_hits_list = [int(r.item()) for r in repeat_hits]

    # Save to CSV
    counts_df = pd.DataFrame({
        'Number': range(1, 59),
        'Count': hit_counts,
        'Consecutive_Repeats': consecutive_list,
        'Repeat_Hits_In_Draw': repeat_hits_list
    })
    counts_df = counts_df.sort_values(by=['Count', 'Number'], ascending=[True, True])
    output_file = csv_path.parent / 'number_counts.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(output_file, index=False)

    # Display Results
    print(f"Total Combination Repeats (Straight): {int(total_comb_repeats)}")

    print("\nBall Repeats by Position (Consecutive draws):")
    for i, count in enumerate(pos_repeats):
        pct = (count / (num_draws - 1)) * 100 if num_draws > 1 else 0
        print(f"  Position {i+1}: {int(count):>4} times ({pct:.2f}%)")

    print("\nNumeric Consecutive Patterns:")
    print(f"  Draws with consecutive numbers (e.g. 10-11): {int(total_num_consecutive)}")
    print(f"  Back-to-back consecutive number draws:       {int(double_consecutive)}")

    print("\nPer-Number Statistics (Top 5 by Consecutive Repeats):")
    sorted_by_consecutive = counts_df.sort_values(by='Consecutive_Repeats', ascending=False)
    for _, row in sorted_by_consecutive.head(5).iterrows():
        print(f"  Number {int(row['Number']):>2}: Total Hits={int(row['Count']):>3}, "
              f"Consecutive Repeats={int(row['Consecutive_Repeats']):>2}")

    print(f"\nSuccessfully updated {output_file}")


if __name__ == "__main__":
    calculate_consecutive_hits()
