 #!/usr/bin/env python3
"""
consective_numbers_mlx.py

Calculates the number of times consecutive patterns occur in Idaho Pick 3 history.
Analyzes:
1. Exact combination repeats in back-to-back draws.
2. Individual digit repeats per position.
3. Draws containing numerically consecutive numbers (e.g., 1-2-3).
4. Frequency of hits happening consecutively (per number).
5. Per-number hit counts and consecutive repeats saved to CSV.

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
    csv_path = script_dir / 'data' / 'idaho_pick3_history.csv'
    
    if not csv_path.exists():
        # Try relative to project root if script_dir doesn't work as expected
        csv_path = Path('idaho/pick3/data/idaho_pick3_history.csv')
        if not csv_path.exists():
            print(f"Error: {csv_path} not found.")
            return

    # Load the history data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Extract Num1, Num2, Num3
    # Reverse the dataframe to process in chronological order (oldest to newest)
    # The original CSV has newest at the top
    digits = df[['Num1', 'Num2', 'Num3']].values[::-1]
    
    # Convert to MLX array for fast tensor operations
    data = mx.array(digits)
    num_draws = data.shape[0]
    
    print(f"Analyzing {num_draws} Idaho Pick 3 draws...")
    print("=" * 50)

    # 1. Exact Combination Repeats (Straight Hit in consecutive draws)
    # Compare each draw with the one immediately preceding it
    # We use mx.all(..., axis=1) to check if all 3 digits match
    comb_repeats = mx.all(data[1:] == data[:-1], axis=1)
    total_comb_repeats = mx.sum(comb_repeats).item()

    # 2. Position-wise Consecutive Hits (Digit repeats in the same slot)
    pos_repeats = []
    for i in range(3):
        matches = (data[1:, i] == data[:-1, i])
        pos_repeats.append(mx.sum(matches).item())

    # 3. Numerical Consecutive Hits (e.g., 1-2-3, 5-6-7 in any order)
    # Sort digits in each draw to identify sequences
    sorted_data = mx.sort(data, axis=1)
    # Check if they form a sequence: x, x+1, x+2
    is_num_consecutive = (sorted_data[:, 1] == sorted_data[:, 0] + 1) & \
                         (sorted_data[:, 2] == sorted_data[:, 1] + 1)
    total_num_consecutive = mx.sum(is_num_consecutive).item()

    # 4. Boxed Combination Repeats (Same 3 digits in any order in consecutive draws)
    boxed_repeats = mx.all(sorted_data[1:] == sorted_data[:-1], axis=1)
    total_boxed_repeats = mx.sum(boxed_repeats).item()

    # 5. Consecutive "Consecutive" Draws
    # How many times a consecutive-number hit was followed by another consecutive-number hit
    double_consecutive = mx.sum(is_num_consecutive[1:] & is_num_consecutive[:-1]).item()

    # 6. Per-Number Hit Counts and Consecutive Repeats (0-9)
    # Total hits for each digit across all positions
    all_digits_flat = data.flatten()
    hit_counts = []
    for n in range(10):
        hit_counts.append(int(mx.sum(all_digits_flat == n).item()))

    # Consecutive repeats: number of times a digit appears in draw i and draw i+1
    draw_counts = mx.zeros((num_draws, 10))
    digit_range = mx.array(list(range(10)))
    for i in range(3):
        draw_counts += (data[:, i, None] == digit_range).astype(mx.float32)
    
    present = draw_counts > 0
    consecutive_hits = mx.sum(present[1:] & present[:-1], axis=0)
    consecutive_list = [int(c.item()) for c in consecutive_hits]

    # 7. Intra-draw Repeats (Same number on the same winning ticket)
    # count how many extra hits beyond the first one in a single draw
    repeat_hits = mx.sum(mx.maximum(draw_counts - 1, 0), axis=0)
    repeat_hits_list = [int(r.item()) for r in repeat_hits]

    # Save Per-Number Stats to CSV
    counts_df = pd.DataFrame({
        'Number': range(10),
        'Count': hit_counts,
        'Consecutive_Repeats': consecutive_list,
        'Repeat_Hits_In_Draw': repeat_hits_list
    })
    # Sort as done in Idaho Cash: Count ascending, then Number
    counts_df = counts_df.sort_values(by=['Count', 'Number'], ascending=[True, True])
    output_file = script_dir / 'data' / 'number_counts.csv'
    counts_df.to_csv(output_file, index=False)

    # Display Results
    print(f"Total Combination Repeats (Straight): {int(total_comb_repeats)}")
    print(f"Total Combination Repeats (Boxed):    {int(total_boxed_repeats)}")
    
    print("\nDigit Repeats by Position (Consecutive draws):")
    for i, count in enumerate(pos_repeats):
        pct = (count / (num_draws - 1)) * 100
        print(f"  Position {i+1}: {int(count):>4} times ({pct:.2f}%)")

    print("\nNumeric Consecutive Patterns:")
    print(f"  Draws with consecutive numbers (e.g. 1-2-3): {int(total_num_consecutive)}")
    print(f"  Back-to-back consecutive number draws:       {int(double_consecutive)}")

    # 8. Global Intra-draw Repeat Summary
    max_per_draw = mx.max(draw_counts, axis=1)
    total_doubles = int(mx.sum(max_per_draw == 2).item())
    total_triples = int(mx.sum(max_per_draw == 3).item())
    
    print("\nIntra-draw Repeat Patterns (Same number on one ticket):")
    print(f"  Draws with a Double (e.g. 5-5-7): {total_doubles}")
    print(f"  Draws with a Triple (e.g. 5-5-5): {total_triples}")
    print(f"  Total draws with repeated numbers: {total_doubles + total_triples}")

    print("\nPer-Number Statistics (Top 5 by Consecutive Repeats):")
    sorted_by_consecutive = counts_df.sort_values(by='Consecutive_Repeats', ascending=False)
    for _, row in sorted_by_consecutive.head(5).iterrows():
        print(f"  Number {int(row['Number'])}: Total Hits={int(row['Count']):>4}, "
              f"Consecutive Repeats={int(row['Consecutive_Repeats']):>3}, "
              f"Repeat Hits In Draw={int(row['Repeat_Hits_In_Draw']):>3}")

    print(f"\nSuccessfully updated {output_file}")

    # Specific common repeats
    if int(total_comb_repeats) > 0:
        print("\nNote: Exact combination repeats found! This indicates the same")
        print("number was drawn in two consecutive draws.")

if __name__ == "__main__":
    calculate_consecutive_hits()
