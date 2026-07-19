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
    df['Extra'] = pd.to_numeric(df['Extra'], errors='coerce')
    df = df.dropna(subset=['Extra'])
    return df['Extra'].values.astype(int)[::-1] # Chronological order

def to_onehot(val: int, size: int) -> np.ndarray:
    y = np.zeros(size, dtype=np.float32)
    y[val - EXTRA_MIN] = 1.0
    return y

def get_features(recent: np.ndarray, window: int):
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
    return np.concatenate([norm_recent, freq, gaps, last_onehot])

def make_features_and_labels(extras: np.ndarray, window: int):
    xs = []
    ys = []
    e_dim = EXTRA_MAX - EXTRA_MIN + 1

    for i in range(window, len(extras)):
        recent = extras[i - window : i]
        target = extras[i]
        xs.append(get_features(recent, window))
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
    
    print(f"Final training loss: {loss.item():.4f}")
    return model

def predict_multi(model, extras, window, steps=5):
    current_extras = list(extras)
    predictions = []
    
    for _ in range(steps):
        recent = np.array(current_extras[-window:])
        x = get_features(recent, window)
        x = mx.array(x[None, :])
        
        logits = model(x)
        probs = mx.softmax(logits, axis=1)
        next_val = int(np.argmax(np.array(probs)[0]) + EXTRA_MIN)
        
        predictions.append(next_val)
        current_extras.append(next_val)
        
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    try:
        mx.random.seed(args.seed)
    except AttributeError:
        pass

    extras = load_extra_data(args.csv)
    x, y = make_features_and_labels(extras, args.window)
    model = train_model(x, y, args.epochs, args.lr)

    print("\nPredicting next 5 hits...")
    hits = predict_multi(model, extras, args.window, steps=5)
    
    # Run one single prediction to show probabilities for the first one
    recent = np.array(extras[-args.window:])
    x = get_features(recent, args.window)
    x = mx.array(x[None, :])
    logits = model(x)
    probs = mx.softmax(logits, axis=1)
    probs_np = np.array(probs)[0]
    
    print("\nProbabilities for the next draw:")
    for i, p in enumerate(probs_np):
        print(f"Number {i + EXTRA_MIN}: {p:.4f}")
    
    print("\nPredicted sequence of next 5 hits:")
    for i, val in enumerate(hits):
        print(f"Prediction {i+1}: {val}")

if __name__ == "__main__":
    main()
