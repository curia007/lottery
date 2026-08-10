#!/usr/bin/env python3
"""
generate_next_idaho_cash_mlx.py

Generates Idaho Cash tickets using MLX and multiple strategies:
- Model: Based on neural network predictions.
- Balanced: Mix of odd/even, high/low, and historical sums.
- Hot Numbers: Based on frequency in number_counts.csv.
- Overdue Numbers: Based on time since last appearance.
- Mixed: A blend of all the above.

Uses MLX for the predictive model.
"""

import argparse
import csv
import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError:
    print("Error: MLX not found. Please install with 'pip install mlx'")
    exit(1)

# Idaho Cash Constants
MIN_NUM = 1
MAX_NUM = 45
NUM_COUNT = 5

@dataclass
class RankedTicket:
    ticket: str
    score: float
    ticket_type: str

class IdahoCashModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, MAX_NUM),
        )

    def __call__(self, x):
        return self.net(x)

def load_data():
    script_dir = Path(__file__).parent
    history_file = script_dir / 'data' / 'idaho_cash_history.csv'
    counts_file = script_dir / 'data' / 'number_counts.csv'

    if not history_file.exists():
        raise FileNotFoundError(f"History file not found: {history_file}")
    
    df = pd.read_csv(history_file)
    # Basic cleaning
    number_cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5']
    for col in number_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=number_cols)
    
    # Sort by Date if available
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    
    history_nums = df[number_cols].values.astype(int)
    
    counts_dict = {}
    if counts_file.exists():
        df_counts = pd.read_csv(counts_file)
        counts_dict = dict(zip(df_counts['Number'], df_counts['Count']))
    
    return history_nums, counts_dict

def to_multihot(row: Iterable[int]) -> np.ndarray:
    y = np.zeros(MAX_NUM, dtype=np.float32)
    for n in row:
        if MIN_NUM <= n <= MAX_NUM:
            y[int(n) - 1] = 1.0
    return y

def make_features_and_labels(numbers: np.ndarray, window: int):
    xs, ys = [], []
    for i in range(window, len(numbers)):
        recent = numbers[i - window : i]
        target = numbers[i]

        flattened = recent.flatten().astype(np.float32) / MAX_NUM
        
        freq = np.zeros(MAX_NUM, dtype=np.float32)
        for n in recent.flatten():
            freq[int(n) - 1] += 1.0
        freq /= recent.size

        last_draw = to_multihot(recent[-1])
        
        x = np.concatenate([flattened, freq, last_draw])
        y = to_multihot(target)
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.stack(ys)

def train_model(x_train, y_train, epochs=200, lr=0.01):
    model = IdahoCashModel(input_dim=x_train.shape[1])
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, x, y):
        logits = model(x)
        # Stable BCE with logits
        return mx.mean(mx.maximum(logits, 0) - logits * y + mx.log1p(mx.exp(-mx.abs(logits))))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    
    x_mx = mx.array(x_train)
    y_mx = mx.array(y_train)

    for _ in range(epochs):
        loss, grads = loss_and_grad(model, x_mx, y_mx)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    
    return model

def get_model_probs(model, numbers, window):
    recent = numbers[-window:]
    flattened = recent.flatten().astype(np.float32) / MAX_NUM
    freq = np.zeros(MAX_NUM, dtype=np.float32)
    for n in recent.flatten():
        freq[int(n) - 1] += 1.0
    freq /= recent.size
    last_draw = to_multihot(recent[-1])
    
    x = np.concatenate([flattened, freq, last_draw])
    logits = model(mx.array(x[None, :]))
    probs = mx.sigmoid(logits)
    return np.array(probs).flatten()

def get_overdue_scores(numbers):
    flat = numbers.flatten()
    gaps = {}
    for n in range(MIN_NUM, MAX_NUM + 1):
        indices = np.where(flat == n)[0]
        if len(indices) == 0:
            gaps[n] = len(flat)
        else:
            gaps[n] = len(flat) - 1 - indices[-1]
    
    max_gap = max(gaps.values()) if gaps else 1
    return {n: g / max_gap for n, g in gaps.items()}

def get_balance_score(ticket):
    nums = sorted(ticket)
    low_count = sum(1 for n in nums if n <= 22)
    odd_count = sum(1 for n in nums if n % 2 == 1)
    
    # Preferred 2/3 or 3/2 splits
    low_high_score = 1.0 - abs(low_count - 2.5) / 2.5
    odd_even_score = 1.0 - abs(odd_count - 2.5) / 2.5
    
    total = sum(nums)
    # Typical Idaho Cash sum center is around 115
    sum_score = max(0.0, 1.0 - abs(total - 115) / 100.0)
    
    return (low_high_score + odd_even_score + sum_score) / 3.0

def generate_tickets(type_name, model_probs, counts_dict, overdue_scores, history_nums, num_tickets=5):
    # Normalize counts
    all_counts = np.array([float(counts_dict.get(n, 0)) for n in range(MIN_NUM, MAX_NUM + 1)])
    max_c = max(all_counts) if any(all_counts) else 1
    freq_scores = {n: all_counts[n-1]/max_c for n in range(MIN_NUM, MAX_NUM + 1)}

    # Weighted scoring for number pool selection
    pool_scores = []
    for n in range(MIN_NUM, MAX_NUM + 1):
        m_s = model_probs[n-1]
        f_s = freq_scores[n]
        o_s = overdue_scores[n]
        
        if type_name == "hot":
            score = f_s * 0.8 + m_s * 0.2
        elif type_name == "overdue":
            score = o_s * 0.8 + m_s * 0.2
        elif type_name == "balanced":
            score = m_s * 0.4 + f_s * 0.3 + o_s * 0.3
        elif type_name == "mixed":
            score = m_s * 0.4 + f_s * 0.3 + o_s * 0.3
        else: # model
            score = m_s
        pool_scores.append((n, score))
    
    pool_scores.sort(key=lambda x: x[1], reverse=True)
    # Select top 15 numbers as a pool for combinations
    pool = [n for n, s in pool_scores[:15]]
    
    candidates = []
    # If pool is small, combinations might be few, but 15 choose 5 is 3003, which is plenty.
    for combo in itertools.combinations(sorted(pool), 5):
        m_score = np.mean([model_probs[n-1] for n in combo])
        f_score = np.mean([freq_scores[n] for n in combo])
        o_score = np.mean([overdue_scores[n] for n in combo])
        b_score = get_balance_score(combo)
        
        if type_name == "hot":
            final_score = f_score * 0.7 + m_score * 0.3
        elif type_name == "overdue":
            final_score = o_score * 0.7 + m_score * 0.3
        elif type_name == "balanced":
            final_score = b_score * 0.6 + m_score * 0.4
        elif type_name == "mixed":
            final_score = (m_score + f_score + o_score + b_score) / 4.0
        else: # model
            final_score = m_score
            
        candidates.append(RankedTicket(
            ticket=" ".join(f"{n:02d}" for n in combo),
            score=final_score,
            ticket_type=type_name
        ))
    
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:num_tickets]

def main():
    print("Loading data...")
    try:
        history_nums, counts_dict = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    window = 20
    print(f"Training MLX model (window size {window})...")
    x, y = make_features_and_labels(history_nums, window)
    model = train_model(x, y, epochs=200)
    
    model_probs = get_model_probs(model, history_nums, window)
    overdue_scores = get_overdue_scores(history_nums)
    
    types = ["model", "balanced", "hot", "mixed", "overdue"]
    
    print("\nGenerating tickets for all categories:")
    print("=" * 50)
    
    for t in types:
        tickets = generate_tickets(t, model_probs, counts_dict, overdue_scores, history_nums)
        print(f"\nType: {t.upper()}")
        print("-" * 20)
        for i, tick in enumerate(tickets, 1):
            print(f"{i}. {tick.ticket} (Score: {tick.score:.4f})")
    
    print("\n" + "=" * 50)
    print("Disclaimer: Lottery drawings are random. Use for entertainment only.")

if __name__ == "__main__":
    main()
