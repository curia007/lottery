#!/usr/bin/env python3
"""
consective_numbers_mlx.py (Pick 4)

Calculates the number of times consecutive patterns and intra-draw repeats occur in Idaho Pick 4 history.
Analyzes:
1. Exact combination repeats in back-to-back draws.
2. Individual digit repeats per position.
3. Intra-draw repeats (Doubles, Triples, Quadruples).
4. Frequency of hits happening consecutively (per number).
5. Per-number hit counts and repeats saved to CSV.

Uses MLX for tensor-based calculations.
"""

import pandas as pd
from pathlib import Path

try:
    import mlx.core as mx
except ImportError:
    print("Error: MLX not found. Please install with 'pip install mlx'")
    exit(1)

def calculate_consecutive_hits():
    # Define paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'data' / 'idaho_pick4_history.csv'
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    # Load the history data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Extract Num1, Num2, Num3, Num4
    # Reverse to process in chronological order
    digits = df[['Num1', 'Num2', 'Num3', 'Num4']].values[::-1]
    
    # Convert to MLX array
    data = mx.array(digits)
    num_draws = data.shape[0]
    
    print(f"Analyzing {num_draws} Idaho Pick 4 draws...")
    print("=" * 50)

    # 1. Exact Combination Repeats
    comb_repeats = mx.all(data[1:] == data[:-1], axis=1)
    total_comb_repeats = mx.sum(comb_repeats).item()

    # 2. Position-wise Consecutive Hits
    pos_repeats = []
    for i in range(4):
        matches = (data[1:, i] == data[:-1, i])
        pos_repeats.append(mx.sum(matches).item())

    # 3. Per-Number Hit Counts and Repeats (0-9)
    all_digits_flat = data.flatten()
    hit_counts = []
    for n in range(10):
        hit_counts.append(int(mx.sum(all_digits_flat == n).item()))

    # Intra-draw and Consecutive counts per number
    draw_counts = mx.zeros((num_draws, 10))
    digit_range = mx.array(list(range(10)))
    for i in range(4):
        draw_counts += (data[:, i, None] == digit_range).astype(mx.float32)
    
    present = draw_counts > 0
    consecutive_hits = mx.sum(present[1:] & present[:-1], axis=0)
    consecutive_list = [int(c.item()) for c in consecutive_hits]

    # Repeat hits in draw (extra hits beyond first)
    repeat_hits = mx.sum(mx.maximum(draw_counts - 1, 0), axis=0)
    repeat_hits_list = [int(r.item()) for r in repeat_hits]

    # Save Per-Number Stats to CSV
    counts_df = pd.DataFrame({
        'Number': range(10),
        'Count': hit_counts,
        'Consecutive_Repeats': consecutive_list,
        'Repeat_Hits_In_Draw': repeat_hits_list
    })
    counts_df = counts_df.sort_values(by=['Count', 'Number'], ascending=[True, True])
    output_file = script_dir / 'data' / 'number_counts.csv'
    counts_df.to_csv(output_file, index=False)

    # 4. Global Intra-draw Repeat Summary
    # In Pick 4:
    # Max count per draw tells us the highest repetition
    max_per_draw = mx.max(draw_counts, axis=1)
    # Also sum of counts > 1 tells us if there are multiple pairs
    # draw_counts > 1 gives 1 for each digit that repeats
    repeating_digits_count = mx.sum((draw_counts > 1).astype(mx.int32), axis=1)
    
    quads = int(mx.sum(max_per_draw == 4).item())
    triples = int(mx.sum(max_per_draw == 3).item())
    # Doubles: can be one double or two doubles
    any_repeats = int(mx.sum(max_per_draw > 1).item())
    two_doubles = int(mx.sum((max_per_draw == 2) & (repeating_digits_count == 2)).item())
    one_double = int(mx.sum((max_per_draw == 2) & (repeating_digits_count == 1)).item())

    # Display Results
    print(f"Total Combination Repeats (Straight): {int(total_comb_repeats)}")
    
    print("\nDigit Repeats by Position (Consecutive draws):")
    for i, count in enumerate(pos_repeats):
        pct = (count / (num_draws - 1)) * 100
        print(f"  Position {i+1}: {int(count):>4} times ({pct:.2f}%)")

    print("\nIntra-draw Repeat Patterns (Same number on one ticket):")
    print(f"  Draws with one Double:             {one_double}")
    print(f"  Draws with two Doubles:            {two_doubles}")
    print(f"  Draws with a Triple:               {triples}")
    print(f"  Draws with a Quadruple:            {quads}")
    print(f"  Total draws with repeated numbers: {any_repeats}")

    print("\nPer-Number Statistics (Top 5 by Consecutive Repeats):")
    sorted_by_consecutive = counts_df.sort_values(by='Consecutive_Repeats', ascending=False)
    for _, row in sorted_by_consecutive.head(5).iterrows():
        print(f"  Number {int(row['Number'])}: Total Hits={int(row['Count']):>4}, "
              f"Consecutive Repeats={int(row['Consecutive_Repeats']):>3}, "
              f"Repeat Hits In Draw={int(row['Repeat_Hits_In_Draw']):>3}")

    print(f"\nSuccessfully created {output_file}")

if __name__ == "__main__":
    calculate_consecutive_hits()
