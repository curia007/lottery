from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError:
    print("Error: MLX not found. Please install with 'pip install mlx'")
    exit(1)

DEFAULT_CSV = "data/millionaire_life_history.csv"
DEFAULT_WINDOW = 20
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_SEED = 42
EXTRA_MIN = 1
EXTRA_MAX = 5

class ExtraModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 5):
        super().__init__()
        self.layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def load_extra_data(csv_path: str | Path):
    df = pd.read_csv(csv_path)
    # Ensure Extra column is numeric
    df['Extra'] = pd.to_numeric(df['Extra'], errors='coerce')
    df = df.dropna(subset=['Extra'])
    return df['Extra'].values.astype(int)[::-1] # Reverse to have chronological order (oldest to newest)

def to_onehot(val: int, size: int) -> np.ndarray:
    y = np.zeros(size, dtype=np.float32)
    y[val - EXTRA_MIN] = 1.0
    return y

def make_features_and_labels(extras: np.ndarray, window: int):
    xs = []
    ys = []
    e_dim = EXTRA_MAX - EXTRA_MIN + 1

    for i in range(window, len(extras)):
        recent = extras[i - window : i]
        target = extras[i]

        # 1. Recent values normalized
        norm_recent = recent.astype(np.float32) / EXTRA_MAX
        
        # 2. Frequencies
        freq = np.zeros(e_dim, dtype=np.float32)
        for e in recent:
            freq[e - EXTRA_MIN] += 1.0
        freq /= window

        # 3. Gaps
        gaps = np.ones(e_dim, dtype=np.float32)
        for e in range(EXTRA_MIN, EXTRA_MAX + 1):
            locations = np.where(recent == e)[0]
            if len(locations) > 0:
                gaps[e - EXTRA_MIN] = (window - 1 - locations[-1]) / window

        # 4. Last one hot
        last_onehot = to_onehot(recent[-1], e_dim)

        x = np.concatenate([norm_recent, freq, gaps, last_onehot])
        xs.append(x)
        ys.append(to_onehot(target, e_dim))

    return np.stack(xs), np.stack(ys)

def train_model(x_train, y_train, epochs, lr):
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    model = ExtraModel(input_dim, output_dim=output_dim)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, x, y):
        logits = model(x)
        return mx.mean(nn.losses.cross_entropy(logits, mx.argmax(y, axis=1)))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    x_train = mx.array(x_train)
    y_train = mx.array(y_train)

    for e in range(epochs):
        loss, grads = loss_and_grad(model, x_train, y_train)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if (e + 1) % 10 == 0:
            print(f"Epoch {e+1}: Loss {loss.item():.4f}")

    return model

def predict_next(model, extras, window):
    recent = extras[-window:]
    e_dim = EXTRA_MAX - EXTRA_MIN + 1

    norm_recent = recent.astype(np.float32) / EXTRA_MAX
    
    freq = np.zeros(e_dim, dtype=np.float32)
    for e in recent:
        freq[e - EXTRA_MIN] += 1.0
    freq /= window

    gaps = np.ones(e_dim, dtype=np.float32)
    for e in range(EXTRA_MIN, EXTRA_MAX + 1):
        locations = np.where(recent == e)[0]
        if len(locations) > 0:
            gaps[e - EXTRA_MIN] = (window - 1 - locations[-1]) / window

    last_onehot = to_onehot(recent[-1], e_dim)

    x = np.concatenate([norm_recent, freq, gaps, last_onehot])
    x = mx.array(x[None, :])
    
    logits = model(x)
    probs = mx.softmax(logits, axis=1)
    return np.array(probs)[0]

def main():
    parser = argparse.ArgumentParser(description="Predict the next Millionaire Life Extra number using MLX")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to history CSV")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Window size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    # mx.random.seed(args.seed) # Note: MLX might not have mx.random.seed in all versions, 
                               # but usually it's set via mx.array or similar if needed.
                               # Actually mx does have mx.random.seed in recent versions.
    try:
        mx.random.seed(args.seed)
    except AttributeError:
        pass

    print(f"Loading data from {args.csv}...")
    extras = load_extra_data(args.csv)
    print(f"Loaded {len(extras)} draws.")

    print("Preparing features...")
    x, y = make_features_and_labels(extras, args.window)
    
    print(f"Training model for {args.epochs} epochs...")
    model = train_model(x, y, args.epochs, args.lr)

    print("\nPredicting next Extra number...")
    probs = predict_next(model, extras, args.window)
    
    print("\nProbabilities:")
    for i, p in enumerate(probs):
        print(f"Number {i + EXTRA_MIN}: {p:.4f}")

    next_num = np.argmax(probs) + EXTRA_MIN
    print(f"\nPredicted Next Extra Number: {next_num}")

if __name__ == "__main__":
    main()
