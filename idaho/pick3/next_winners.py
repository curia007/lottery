#!/usr/bin/env python3
"""
next_winners.py

Generates the next best possible winning tickets for Idaho Lottery games using MLX.
Supports Idaho Cash, Pick 3, and Pick 4.

Uses:
- Historical draw patterns (MLX Neural Network)
- Hot numbers (Frequency analysis)
- Overdue numbers (Gap analysis)
- Consecutive repeats (Historical trends)
- Intra-draw repeats (For Pick 3/4)
"""

import argparse
import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError:
    print("Error: MLX not found. Please install with 'pip install mlx'")
    exit(1)

# Game Configurations
GAMES = {
    'idaho_cash': {
        'history_path': 'idaho/idaho_cash/data/idaho_cash_history.csv',
        'counts_path': 'idaho/idaho_cash/data/number_counts.csv',
        'num_range': (1, 45),
        'pick_count': 5,
        'has_repeats': False,
        'cols': ['Num1', 'Num2', 'Num3', 'Num4', 'Num5']
    },
    'pick3': {
        'history_path': 'idaho/pick3/data/idaho_pick3_history.csv',
        'counts_path': 'idaho/pick3/data/number_counts.csv',
        'num_range': (0, 9),
        'pick_count': 3,
        'has_repeats': True,
        'cols': ['Num1', 'Num2', 'Num3']
    },
    'pick4': {
        'history_path': 'idaho/pick4/data/idaho_pick4_history.csv',
        'counts_path': 'idaho/pick4/data/number_counts.csv',
        'num_range': (0, 9),
        'pick_count': 4,
        'has_repeats': True,
        'cols': ['Num1', 'Num2', 'Num3', 'Num4']
    }
}

@dataclass
class RankedTicket:
    ticket: str
    score: float
    model_score: float
    stats_score: float

class LotteryModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, is_positional: bool = False, num_positions: int = 1):
        super().__init__()
        hidden_dim = 128
        self.is_positional = is_positional
        self.num_positions = num_positions
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * num_positions)
        )

    def __call__(self, x):
        return self.net(x)

def load_game_data(game_name: str):
    config = GAMES[game_name]
    script_dir = Path(__file__).parent
    history_path = script_dir / config['history_path']
    counts_path = script_dir / config['counts_path']
    
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    df = pd.read_csv(history_path)
    cols = config['cols']
    
    # Clean data
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=cols)
    
    # Sort chronologically
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    
    history_nums = df[cols].values.astype(int)
    
    stats_df = None
    if counts_path.exists():
        stats_df = pd.read_csv(counts_path)
    
    return history_nums, stats_df

def to_onehot(val: int, min_val: int, max_val: int):
    size = max_val - min_val + 1
    y = np.zeros(size, dtype=np.float32)
    y[val - min_val] = 1.0
    return y

def to_multihot(vals: List[int], min_val: int, max_val: int):
    size = max_val - min_val + 1
    y = np.zeros(size, dtype=np.float32)
    for v in vals:
        y[v - min_val] = 1.0
    return y

def make_features_and_labels(history: np.ndarray, config: dict, window: int):
    min_val, max_val = config['num_range']
    size = max_val - min_val + 1
    xs, ys = [], []
    
    for i in range(window, len(history)):
        recent = history[i - window : i]
        target = history[i]
        
        # Features: flattened recent draws + frequencies + last draw
        flattened = (recent.flatten().astype(np.float32) - min_val) / size
        
        freq = np.zeros(size, dtype=np.float32)
        for val in recent.flatten():
            freq[val - min_val] += 1
        freq /= recent.size
        
        last_draw = to_multihot(recent[-1], min_val, max_val)
        
        x = np.concatenate([flattened, freq, last_draw])
        
        if config['has_repeats']:
            # For Pick 3/4, target is position-wise
            y = np.concatenate([to_onehot(t, min_val, max_val) for t in target])
        else:
            y = to_multihot(target, min_val, max_val)
            
        xs.append(x)
        ys.append(y)
        
    return np.stack(xs), np.stack(ys)

def train_model(x_train, y_train, game_config, epochs=200):
    input_dim = x_train.shape[1]
    min_val, max_val = game_config['num_range']
    size = max_val - min_val + 1
    
    if game_config['has_repeats']:
        model = LotteryModel(input_dim, size, is_positional=True, num_positions=game_config['pick_count'])
    else:
        model = LotteryModel(input_dim, size)
        
    optimizer = optim.Adam(learning_rate=0.01)
    
    x_mx = mx.array(x_train)
    y_mx = mx.array(y_train)

    def loss_fn(model, x, y):
        logits = model(x)
        if game_config['has_repeats']:
            # Cross entropy for each position
            logits = logits.reshape((-1, game_config['pick_count'], size))
            targets = y.reshape((-1, game_config['pick_count'], size))
            # mx.argmax gives indices
            target_indices = mx.argmax(targets, axis=2)
            loss = 0
            for i in range(game_config['pick_count']):
                loss += mx.mean(nn.losses.cross_entropy(logits[:, i, :], target_indices[:, i]))
            return loss / game_config['pick_count']
        else:
            # Multi-label BCE
            return mx.mean(mx.maximum(logits, 0) - logits * y + mx.log1p(mx.exp(-mx.abs(logits))))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    
    for _ in range(epochs):
        loss, grads = loss_and_grad(model, x_mx, y_mx)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
    return model

def get_predictions(model, history, config, window):
    min_val, max_val = config['num_range']
    size = max_val - min_val + 1
    recent = history[-window:]
    
    flattened = (recent.flatten().astype(np.float32) - min_val) / size
    freq = np.zeros(size, dtype=np.float32)
    for val in recent.flatten():
        freq[val - min_val] += 1
    freq /= recent.size
    last_draw = to_multihot(recent[-1], min_val, max_val)
    
    x = np.concatenate([flattened, freq, last_draw])
    logits = model(mx.array(x[None, :]))
    
    if config['has_repeats']:
        logits = logits.reshape((config['pick_count'], size))
        probs = mx.softmax(logits, axis=1)
        return np.array(probs) # Shape (pick_count, size)
    else:
        probs = mx.sigmoid(logits)
        return np.array(probs).flatten() # Shape (size,)

def score_tickets(game_name, probs, stats_df, history, num_tickets):
    config = GAMES[game_name]
    min_val, max_val = config['num_range']
    size = max_val - min_val + 1
    
    # Prepare stats scores
    stats_score_map = {}
    if stats_df is not None:
        # Normalize counts and consecutive repeats
        max_count = stats_df['Count'].max() or 1
        max_consecutive = stats_df['Consecutive_Repeats'].max() or 1
        for _, row in stats_df.iterrows():
            num = int(row['Number'])
            s = (row['Count'] / max_count) * 0.5 + (row['Consecutive_Repeats'] / max_consecutive) * 0.5
            stats_score_map[num] = s
    else:
        stats_score_map = {n: 0.5 for n in range(min_val, max_val + 1)}

    tickets = []
    
    if game_name == 'idaho_cash':
        # Select top numbers to form combinations
        pool_scores = []
        for n in range(min_val, max_val + 1):
            m_s = probs[n - min_val]
            s_s = stats_score_map.get(n, 0)
            pool_scores.append((n, m_s * 0.7 + s_s * 0.3))
        
        pool_scores.sort(key=lambda x: x[1], reverse=True)
        pool = [n for n, _ in pool_scores[:15]]
        
        for combo in itertools.combinations(sorted(pool), 5):
            m_s = np.mean([probs[n - min_val] for n in combo])
            s_s = np.mean([stats_score_map.get(n, 0) for n in combo])
            # Balance score (simplified)
            odd_count = sum(1 for n in combo if n % 2 != 0)
            b_s = 1.0 - abs(odd_count - 2.5) / 2.5
            
            final_score = m_s * 0.5 + s_s * 0.3 + b_s * 0.2
            tickets.append(RankedTicket(
                ticket=" ".join(f"{n:02d}" for n in combo),
                score=final_score,
                model_score=m_s,
                stats_score=s_s
            ))
    # For Pick 3/4, we can generate based on position-wise probabilities.
    # We sample top combinations to keep it efficient.
    else:
        num_pos = config['pick_count']
        # Take top 4 most likely digits for each position
        top_indices = [np.argsort(probs[i])[-4:] for i in range(num_pos)]
        
        for combo in itertools.product(*top_indices):
            actual_combo = [idx + min_val for idx in combo]
            # Geometric mean of probabilities for the ticket score
            m_s = np.prod([probs[i, idx] for i, idx in enumerate(combo)]) ** (1/num_pos)
            s_s = np.mean([stats_score_map.get(n, 0) for n in actual_combo])
            
            # Combine model output with historical statistics
            final_score = m_s * 0.7 + s_s * 0.3
            tickets.append(RankedTicket(
                ticket="".join(str(n) for n in actual_combo),
                score=final_score,
                model_score=m_s,
                stats_score=s_s
            ))

    tickets.sort(key=lambda x: x.score, reverse=True)
    return tickets[:num_tickets]

def main():
    parser = argparse.ArgumentParser(description="Predict next winning lottery tickets using MLX.")
    parser.add_argument("--game", choices=list(GAMES.keys()), default="pick4", help="Lottery game to predict.")
    parser.add_argument("--tickets", type=int, default=5, help="Number of tickets to generate.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--window", type=int, default=20, help="Window size for features.")
    args = parser.parse_args()

    print(f"Loading data for {args.game}...")
    try:
        history, stats_df = load_game_data(args.game)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Training MLX model ({len(history)} draws)...")
    x, y = make_features_and_labels(history, GAMES[args.game], args.window)
    model = train_model(x, y, GAMES[args.game], epochs=args.epochs)
    
    print("Generating predictions...")
    probs = get_predictions(model, history, GAMES[args.game], args.window)
    ranked_tickets = score_tickets(args.game, probs, stats_df, history, args.tickets)
    
    print("\nNext Best Possible Winning Tickets:")
    print("=" * 40)
    for i, t in enumerate(ranked_tickets, 1):
        print(f"{i}. {t.ticket} (Score: {t.score:.4f})")
    print("=" * 40)
    print("Disclaimer: Lottery results are random. Play responsibly.")

if __name__ == "__main__":
    main()
