import pandas as pd
from pathlib import Path

try:
    import mlx.core as mx
except ImportError:
    print("Error: MLX not found. Please install with 'pip install mlx'")
    exit(1)

def generate_counts():
    # Define paths relative to the script location
    script_dir = Path(__file__).parent
    input_file = script_dir / 'data' / 'idaho_cash_history.csv'
    output_file = script_dir / 'data' / 'number_counts.csv'

    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    # Load the history data
    df = pd.read_csv(input_file)

    # Extract all numbers from Num1 to Num5
    number_columns = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5']
    
    # Ensure columns are numeric and drop rows with NaNs in these columns
    for col in number_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=number_columns)

    # Sort by Date if possible for correct consecutive calculation
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

    # Idaho Cash numbers are 1-45.
    # Convert to MLX array (shape N, 5)
    data = mx.array(df[number_columns].values.astype(int))
    num_draws = data.shape[0]

    # 1. Total hits for each number 1-45
    # Flatten and count occurrences
    all_numbers_flat = data.flatten()
    hit_counts = []
    for n in range(1, 46):
        hit_counts.append(int(mx.sum(all_numbers_flat == n).item()))

    # 2. Consecutive repeats for each number 1-45
    # Create multi-hot for each draw (N, 45)
    multi_hot = mx.zeros((num_draws, 45))
    numbers_range = mx.array(list(range(1, 46)))
    for i in range(5):
        # Broadcast comparison to get one-hot (N, 45)
        # data[:, i] has shape (N,)
        # data[:, i, None] has shape (N, 1)
        # comparison gives (N, 45)
        multi_hot += (data[:, i, None] == numbers_range).astype(mx.float32)
    
    # Check presence: in Idaho Cash, it's always 0 or 1 per draw
    present = multi_hot > 0
    # Sum overlaps between adjacent draws
    consecutive_hits = mx.sum(present[1:] & present[:-1], axis=0)
    consecutive_list = [int(c.item()) for c in consecutive_hits]

    # 3. Intra-draw Repeats (Same number on the same winning ticket)
    # In Idaho Cash (1-45), there are no repeats in a single draw.
    # We include this column for consistency with other games.
    repeat_hits = mx.sum(mx.maximum(multi_hot - 1, 0), axis=0)
    repeat_hits_list = [int(r.item()) for r in repeat_hits]

    # Create results dataframe
    counts = pd.DataFrame({
        'Number': range(1, 46),
        'Count': hit_counts,
        'Consecutive_Repeats': consecutive_list,
        'Repeat_Hits_In_Draw': repeat_hits_list
    })

    # Sort by Count in ascending order, then by Number as original
    counts = counts.sort_values(by=['Count', 'Number'], ascending=[True, True])

    # Save to CSV
    counts.to_csv(output_file, index=False)

    print(f"Successfully created {output_file}")
    print(f"Total unique numbers processed: {len(counts)}")
    print(f"Sample consecutive repeats: {counts.iloc[0]['Consecutive_Repeats']}")

if __name__ == "__main__":
    generate_counts()
