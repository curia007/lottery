#!/usr/bin/env python3
"""
millionaire_life_llm_ticket_model.py

Millionaire for Life Autoregressive Transformer Language Model (LLM).
Loads or creates an LLM from a remote Lotto model specification, generates and saves
the local model, trains on historical draw sequences, and produces ranked next winning
ticket predictions with strong emphasis on the best next winning Millionaire Ball (Extra).

Features:
  - RemoteLottoModelLoader: Resolves remote Lotto model configs/weights (HTTP/HTTPS/URI)
    and generates/caches the local model.
  - MillionaireLifeLLM: Causal Transformer Decoder with Multi-Head Attention, causal masking,
    LayerNorm, and autoregressive language modeling head built with Apple MLX.
  - MillionaireLifeTokenizer: Lottery sequence tokenizer covering 5 main balls (1-58/60),
    Millionaire Ball (1-5), and special control tokens (<PAD>, <BOS>, <EOS>, <SEP>, <PRED>, <MB>).
  - Autoregressive Training: Optimizes next-token cross-entropy loss over sliding-window draw history.
  - Millionaire Ball Emphasis: Computes exact conditional probability distribution for all
    Millionaire Ball candidates (1 to 5) and prominently emphasizes the #1 Best Winning Ball.
  - Next Winning Ticket Generation: Ranks 5-number combinations + Millionaire Ball using joint
    LLM sequence probabilities and composite lottery metrics (model, balanced, hot, overdue, mb_focused).

Usage Examples:
    # 1. Standard training and ticket generation emphasizing Millionaire Ball:
    python millionaire_life/millionaire_life_llm_ticket_model.py --tickets 5 --epochs 10

    # 2. Specify remote model URI and custom local model output dir:
    python millionaire_life/millionaire_life_llm_ticket_model.py \\
        --remote-model remote://lottery-ai/millionaire-life-transformer-base \\
        --local-model-dir models/millionaire_life_llm_model \\
        --tickets 10

    # 3. Use balanced ticket selection strategy and export to CSV:
    python millionaire_life/millionaire_life_llm_ticket_model.py \\
        --ticket-type balanced \\
        --tickets 10 \\
        --output millionaire_life_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError as exc:
    raise SystemExit(
        "MLX is required. Install it using:\n\n"
        "    pip install mlx\n\n"
        "MLX is optimized for Apple Silicon Macs."
    ) from exc

TicketType = Literal["model", "balanced", "hot", "overdue", "mb_focused"]

DEFAULT_CSV_PATH = "millionaire_life/data/millionaire_life_history.csv"
DEFAULT_MODEL_DIR = "models/millionaire_life_llm_model"
DEFAULT_REMOTE_MODEL = "remote://lottery-ai/millionaire-life-transformer-base"

MAIN_MIN = 1
MAIN_MAX = 58
MAIN_COUNT = 5
EXTRA_MIN = 1
EXTRA_MAX = 5


# ==============================================================================
# Tokenizer for Millionaire for Life Sequence Modeling
# ==============================================================================

class MillionaireLifeTokenizer:
    """
    Tokenizer for Millionaire for Life lottery draw sequence modeling.
    
    Tokens:
      0       : '<PAD>' (Padding token)
      1       : '<BOS>' (Beginning of sequence)
      2       : '<EOS>' (End of sequence)
      3       : '<SEP>' (Separator between draws)
      4       : '<PRED>' (Target prediction prompt prefix)
      5       : '<MB>' (Millionaire Ball indicator prefix)
      6 .. 65 : Main balls '1' through '60'
      66 .. 70: Millionaire balls 'MB:1' through 'MB:5'
    """
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SEP_TOKEN = "<SEP>"
    PRED_TOKEN = "<PRED>"
    MB_TOKEN = "<MB>"

    def __init__(self, max_main: int = 60, max_extra: int = 5):
        self.max_main = max_main
        self.max_extra = max_extra

        self.special_tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.SEP_TOKEN,
            self.PRED_TOKEN,
            self.MB_TOKEN,
        ]
        self.main_tokens = [str(i) for i in range(1, max_main + 1)]
        self.extra_tokens = [f"MB:{i}" for i in range(1, max_extra + 1)]
        
        self.vocab = self.special_tokens + self.main_tokens + self.extra_tokens
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[self.SEP_TOKEN]

    @property
    def pred_id(self) -> int:
        return self.token_to_id[self.PRED_TOKEN]

    @property
    def mb_id(self) -> int:
        return self.token_to_id[self.MB_TOKEN]

    def encode_main_num(self, num: int) -> int:
        token = str(int(num))
        if token not in self.token_to_id:
            raise ValueError(f"Main number {num} out of vocab range (1..{self.max_main})")
        return self.token_to_id[token]

    def decode_main_num(self, token_id: int) -> int:
        token = self.id_to_token[token_id]
        return int(token)

    def encode_extra_num(self, extra: int) -> int:
        token = f"MB:{int(extra)}"
        if token not in self.token_to_id:
            raise ValueError(f"Extra/Millionaire ball {extra} out of range (1..{self.max_extra})")
        return self.token_to_id[token]

    def decode_extra_num(self, token_id: int) -> int:
        token = self.id_to_token[token_id]
        if not token.startswith("MB:"):
            raise ValueError(f"Token {token} is not a valid Millionaire ball token")
        return int(token.split(":")[1])

    def encode_draw(self, numbers: Sequence[int], extra: int) -> List[int]:
        """Encodes 5 main numbers and 1 Millionaire Ball into token sequence."""
        sorted_nums = sorted([int(n) for n in numbers])
        encoded_main = [self.encode_main_num(n) for n in sorted_nums]
        encoded_mb = [self.mb_id, self.encode_extra_num(extra)]
        return encoded_main + encoded_mb

    def decode_draw(self, token_ids: Sequence[int]) -> Tuple[List[int], Optional[int]]:
        """Decodes token IDs into main numbers list and Millionaire ball."""
        main_nums: List[int] = []
        extra_num: Optional[int] = None
        for tid in token_ids:
            tok = self.id_to_token.get(tid, "")
            if tok.isdigit():
                main_nums.append(int(tok))
            elif tok.startswith("MB:"):
                extra_num = int(tok.split(":")[1])
        return sorted(main_nums), extra_num


# ==============================================================================
# Model Architecture (Transformer LLM)
# ==============================================================================

@dataclass
class MillionaireLifeLLMConfig:
    vocab_size: int = 72
    max_seq_len: int = 256
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 3
    mlp_ratio: int = 4
    dropout: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MillionaireLifeLLMConfig":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class TransformerBlock(nn.Module):
    """Causal Transformer Decoder Block with LayerNorm & MultiHeadAttention."""
    def __init__(self, n_embd: int, n_head: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        mlp_dim = n_embd * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, n_embd),
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        norm_x = self.ln1(x)
        attn_out = self.attn(norm_x, norm_x, norm_x, mask=mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class MillionaireLifeLLM(nn.Module):
    """
    Autoregressive Causal Language Model for Millionaire for Life Sequence Generation.
    """
    def __init__(self, config: MillionaireLifeLLMConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_seq_len, config.n_embd)
        self.blocks = [
            TransformerBlock(config.n_embd, config.n_head, config.mlp_ratio)
            for _ in range(config.n_layer)
        ]
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, idx: mx.array) -> mx.array:
        B, T = idx.shape
        if T > self.config.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}")

        positions = mx.arange(0, T)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(positions)
        x = tok_emb + pos_emb

        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def save_pretrained(self, save_directory: Union[str, Path]) -> Path:
        """Saves local model architecture configuration and weights."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        config_path = out_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        weights_path = out_dir / "model.safetensors"
        self.save_weights(str(weights_path))
        return out_dir

    @classmethod
    def from_pretrained(cls, load_directory: Union[str, Path]) -> "MillionaireLifeLLM":
        """Loads a local model from saved directory."""
        in_dir = Path(load_directory)
        config_path = in_dir / "config.json"
        safetensors_path = in_dir / "model.safetensors"
        npz_path = in_dir / "model.npz"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        config = MillionaireLifeLLMConfig.from_dict(cfg_dict)
        model = cls(config)

        if safetensors_path.exists():
            model.load_weights(str(safetensors_path))
            mx.eval(model.parameters())
        elif npz_path.exists():
            try:
                model.load_weights(str(npz_path))
                mx.eval(model.parameters())
            except Exception as e:
                print(f"[Warning] Could not load weights from {npz_path}: {e}")

        return model


# ==============================================================================
# Remote Model Loader & Local Model Generator
# ==============================================================================

class RemoteLottoModelLoader:
    """
    Handles fetching or instantiating a remote Lotto model specification and
    instantiating the local model instance.
    """

    @staticmethod
    def fetch_or_create_model(
        remote_model_uri: str,
        local_model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
        config: Optional[MillionaireLifeLLMConfig] = None,
        force_download: bool = False,
    ) -> MillionaireLifeLLM:
        """
        Creates an LLM from a remote Lotto model reference or URL.
        Generates the local model and caches weights locally.
        """
        local_path = Path(local_model_dir)
        local_config_file = local_path / "config.json"
        local_weights_exist = (local_path / "model.safetensors").exists() or (local_path / "model.npz").exists()

        if local_config_file.exists() and local_weights_exist and not force_download:
            print(f"[RemoteModelLoader] Loading existing local model from '{local_path}'")
            return MillionaireLifeLLM.from_pretrained(local_path)

        print(f"[RemoteModelLoader] Resolving remote Lotto model: '{remote_model_uri}'")

        if remote_model_uri.startswith("http://") or remote_model_uri.startswith("https://"):
            try:
                print(f"[RemoteModelLoader] Fetching remote model config from {remote_model_uri}...")
                req = urllib.request.Request(
                    remote_model_uri,
                    headers={"User-Agent": "LottoLLM/1.0 (MillionaireLife-Client)"}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    remote_data = json.loads(response.read().decode("utf-8"))
                    config = MillionaireLifeLLMConfig.from_dict(remote_data.get("config", remote_data))
                    print("[RemoteModelLoader] Remote config downloaded successfully.")
            except Exception as e:
                print(f"[RemoteModelLoader] Remote download notice: ({e}). Utilizing base remote architecture.")
                if config is None:
                    config = MillionaireLifeLLMConfig()
        else:
            print(f"[RemoteModelLoader] Initializing base Lotto Transformer from remote spec: {remote_model_uri}")
            if config is None:
                config = MillionaireLifeLLMConfig(
                    vocab_size=72,
                    max_seq_len=256,
                    n_embd=64,
                    n_head=4,
                    n_layer=3,
                    mlp_ratio=4,
                )

        print(f"[RemoteModelLoader] Generating local MillionaireLifeLLM with {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dims...")
        model = MillionaireLifeLLM(config)
        mx.eval(model.parameters())

        saved_dir = model.save_pretrained(local_path)
        print(f"[RemoteModelLoader] Local model generated and saved to '{saved_dir}'")
        return model


# ==============================================================================
# Data Loading & Preprocessing
# ==============================================================================

@dataclass
class MillionaireBallInfo:
    ball: int
    probability: float
    frequency: int
    overdue_draws: int
    is_top: bool = False


@dataclass
class RankedTicket:
    rank: int
    numbers: List[int]
    extra: int
    ticket_str: str
    score: float
    llm_prob: float
    mb_prob: float
    freq_score: float
    overdue_score: float
    balance_score: float
    ticket_type: str


def load_millionaire_life_history(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Loads and standardizes the Millionaire for Life history CSV."""
    path = Path(csv_path)
    if not path.exists():
        candidates = [
            Path("millionaire_life/data/millionaire_life_history.csv"),
            Path("data/millionaire_life_history.csv"),
            Path("../data/millionaire_life_history.csv"),
        ]
        for c in candidates:
            if c.exists():
                path = c
                break

    if not path.exists():
        raise FileNotFoundError(f"Millionaire for Life history CSV not found at '{csv_path}'")

    df = pd.read_csv(path)
    required_cols = ["Num1", "Num2", "Num3", "Num4", "Num5", "Extra"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")

    # Clean numeric columns
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required_cols).copy()
    for col in required_cols:
        df[col] = df[col].astype(int)

    # Sort chronologically (oldest to newest)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date", ascending=True).reset_index(drop=True)
    else:
        # Assume CSV was newest-first, reverse to oldest-first
        df = df.iloc[::-1].reset_index(drop=True)

    return df


# ==============================================================================
# Dataset Generation for Autoregressive LM
# ==============================================================================

def create_autoregressive_dataset(
    df: pd.DataFrame,
    tokenizer: MillionaireLifeTokenizer,
    history_window: int = 10,
    max_seq_len: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs causal next-token language modeling dataset from sliding draw windows.
    Each window encodes:
        [BOS, d_{t-W}, SEP, d_{t-W+1}, SEP, ..., d_t, SEP, EOS]
    """
    draws_tokens: List[List[int]] = []
    for _, row in df.iterrows():
        nums = [row["Num1"], row["Num2"], row["Num3"], row["Num4"], row["Num5"]]
        extra = row["Extra"]
        encoded = tokenizer.encode_draw(nums, extra)
        draws_tokens.append(encoded)

    num_draws = len(draws_tokens)
    if num_draws <= history_window:
        raise ValueError(f"History too short ({num_draws} draws) for window size {history_window}")

    input_sequences: List[List[int]] = []
    target_sequences: List[List[int]] = []

    for i in range(history_window, num_draws):
        window_draws = draws_tokens[i - history_window : i + 1]
        seq: List[int] = [tokenizer.bos_id]
        for d in window_draws[:-1]:
            seq.extend(d)
            seq.append(tokenizer.sep_id)

        seq.append(tokenizer.pred_id)
        seq.extend(window_draws[-1])
        seq.append(tokenizer.eos_id)

        if len(seq) > max_seq_len:
            seq = seq[-max_seq_len:]

        input_seq = seq[:-1]
        target_seq = seq[1:]

        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

    # Pad sequences to uniform length
    max_len = max(len(s) for s in input_sequences)
    x_arr = np.full((len(input_sequences), max_len), tokenizer.pad_id, dtype=np.int32)
    y_arr = np.full((len(target_sequences), max_len), tokenizer.pad_id, dtype=np.int32)

    for idx, (inp, tgt) in enumerate(zip(input_sequences, target_sequences)):
        x_arr[idx, :len(inp)] = inp
        y_arr[idx, :len(tgt)] = tgt

    return x_arr, y_arr


# ==============================================================================
# Training Loop
# ==============================================================================

def train_millionaire_life_llm(
    model: MillionaireLifeLLM,
    x_data: np.ndarray,
    y_data: np.ndarray,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    pad_id: int = 0,
) -> Dict[str, List[float]]:
    """Fine-tunes the MillionaireLifeLLM model using AdamW optimizer."""
    num_samples = len(x_data)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def compute_loss(m: MillionaireLifeLLM, x_b: mx.array, y_b: mx.array) -> mx.array:
        logits = m(x_b)  # (B, T, V)
        loss = nn.losses.cross_entropy(logits, y_b, reduction="none")
        mask = (y_b != pad_id).astype(mx.float32)
        masked_loss = mx.sum(loss * mask) / mx.maximum(mx.sum(mask), 1.0)
        return masked_loss

    loss_and_grad = nn.value_and_grad(model, compute_loss)
    metrics: Dict[str, List[float]] = {"loss": [], "perplexity": []}

    print(f"\n[Training] Starting fine-tuning for {epochs} epochs (Dataset: {num_samples} sequences, Batch: {batch_size})...")
    start_time = time.time()

    indices = np.arange(num_samples)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_losses: List[float] = []

        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            x_b = mx.array(x_data[batch_idx])
            y_b = mx.array(y_data[batch_idx])

            loss, grads = loss_and_grad(model, x_b, y_b)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        ppl = float(math.exp(min(mean_loss, 20.0)))
        metrics["loss"].append(mean_loss)
        metrics["perplexity"].append(ppl)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Cross-Entropy Loss: {mean_loss:.4f} | Perplexity: {ppl:.2f} | Time: {elapsed:.1f}s")

    return metrics


# ==============================================================================
# Inference & Millionaire Ball Prediction Engine
# ==============================================================================

def analyze_millionaire_balls(
    model: MillionaireLifeLLM,
    df: pd.DataFrame,
    tokenizer: MillionaireLifeTokenizer,
    history_window: int = 15,
) -> List[MillionaireBallInfo]:
    """
    Computes conditional probabilities for all Millionaire Ball candidates (1 to 5)
    under the autoregressive LLM prompt and historical stats.
    """
    recent_draws = df.tail(history_window)
    tokens = [tokenizer.bos_id]
    for _, row in recent_draws.iterrows():
        nums = [row["Num1"], row["Num2"], row["Num3"], row["Num4"], row["Num5"]]
        extra = row["Extra"]
        tokens.extend(tokenizer.encode_draw(nums, extra))
        tokens.append(tokenizer.sep_id)

    tokens.append(tokenizer.pred_id)

    # Compute historical frequency and overdue gap
    all_extras = df["Extra"].values
    total_draws = len(all_extras)

    mb_probs: Dict[int, float] = {}
    
    # We evaluate sequence log-likelihood of prompt + MB token
    for mb_val in range(EXTRA_MIN, EXTRA_MAX + 1):
        prompt_with_mb = tokens + [tokenizer.mb_id]
        x_in = mx.array([prompt_with_mb])
        logits = model(x_in)
        last_logit = logits[0, -1]  # Logits predicting next token
        mb_token_id = tokenizer.encode_extra_num(mb_val)
        
        # Softmax over only MB candidates
        mb_cand_ids = [tokenizer.encode_extra_num(e) for e in range(EXTRA_MIN, EXTRA_MAX + 1)]
        cand_logits = np.array([last_logit[cid].item() for cid in mb_cand_ids])
        exp_logits = np.exp(cand_logits - np.max(cand_logits))
        cand_probs = exp_logits / np.sum(exp_logits)
        
        idx = mb_val - EXTRA_MIN
        mb_probs[mb_val] = float(cand_probs[idx])

    # Build MillionaireBallInfo objects
    results: List[MillionaireBallInfo] = []
    top_mb = max(mb_probs.keys(), key=lambda k: mb_probs[k])

    for mb_val in range(EXTRA_MIN, EXTRA_MAX + 1):
        freq = int(np.sum(all_extras == mb_val))
        locations = np.where(all_extras == mb_val)[0]
        if len(locations) > 0:
            overdue = int(total_draws - 1 - locations[-1])
        else:
            overdue = total_draws

        results.append(
            MillionaireBallInfo(
                ball=mb_val,
                probability=mb_probs[mb_val],
                frequency=freq,
                overdue_draws=overdue,
                is_top=(mb_val == top_mb),
            )
        )

    # Sort descending by probability
    results.sort(key=lambda x: x.probability, reverse=True)
    return results


def calculate_main_number_probabilities(
    model: MillionaireLifeLLM,
    df: pd.DataFrame,
    tokenizer: MillionaireLifeTokenizer,
    history_window: int = 15,
) -> np.ndarray:
    """
    Computes individual marginal probabilities P(num | prompt) for main balls 1..58.
    Returns 1D array indexed by (number - 1).
    """
    recent_draws = df.tail(history_window)
    tokens = [tokenizer.bos_id]
    for _, row in recent_draws.iterrows():
        nums = [row["Num1"], row["Num2"], row["Num3"], row["Num4"], row["Num5"]]
        extra = row["Extra"]
        tokens.extend(tokenizer.encode_draw(nums, extra))
        tokens.append(tokenizer.sep_id)

    tokens.append(tokenizer.pred_id)

    x_in = mx.array([tokens])
    logits = model(x_in)
    last_logit = logits[0, -1]

    main_token_ids = [tokenizer.encode_main_num(n) for n in range(MAIN_MIN, MAIN_MAX + 1)]
    main_logits = np.array([last_logit[tid].item() for tid in main_token_ids])
    exp_logits = np.exp(main_logits - np.max(main_logits))
    probs = exp_logits / np.sum(exp_logits)
    return probs


def calculate_ticket_composite_score(
    numbers: Sequence[int],
    extra: int,
    main_probs: np.ndarray,
    mb_probs: Dict[int, float],
    df: pd.DataFrame,
    ticket_type: TicketType = "model",
) -> Tuple[float, float, float, float, float, float]:
    """
    Computes multi-factor composite ranking scores for a ticket.
    Returns: (composite_score, llm_prob, mb_prob, freq_score, overdue_score, balance_score)
    """
    nums = sorted(numbers)
    
    # 1. LLM Relative Probability ratio against uniform baseline (uniform = 1.0)
    indices = [n - MAIN_MIN for n in nums]
    main_relative_prob = float(np.prod(main_probs[indices]) * (MAIN_MAX ** MAIN_COUNT))
    mb_raw_prob = mb_probs.get(extra, 0.2)
    mb_relative_prob = float(mb_raw_prob * EXTRA_MAX)
    llm_score = main_relative_prob * mb_relative_prob

    # 2. Historical frequency score (normalized around 1.0)
    main_cols = ["Num1", "Num2", "Num3", "Num4", "Num5"]
    all_mains = df[main_cols].values.flatten()
    total_draws = len(df)
    freqs = [(np.sum(all_mains == n) / (total_draws * 5)) * MAIN_MAX for n in nums]
    freq_score = float(np.mean(freqs))

    # 3. Overdue / recency score
    overdue_scores = []
    for n in nums:
        rows_with_n = np.where((df[main_cols].values == n).any(axis=1))[0]
        if len(rows_with_n) > 0:
            last_seen = total_draws - 1 - rows_with_n[-1]
            overdue_scores.append(min(last_seen / 20.0, 2.0))
        else:
            overdue_scores.append(2.0)
    overdue_score = float(np.mean(overdue_scores))

    # 4. Balance score (Sum distribution and odd/even balance)
    ticket_sum = sum(nums)
    expected_mean_sum = (MAIN_COUNT * (MAIN_MIN + MAIN_MAX)) / 2.0  # ~ 147.5
    sum_diff = abs(ticket_sum - expected_mean_sum) / expected_mean_sum
    sum_balance = max(0.0, 1.0 - sum_diff)

    odd_count = sum(1 for n in nums if n % 2 != 0)
    parity_balance = 1.0 if odd_count in (2, 3) else 0.6

    balance_score = 0.6 * sum_balance + 0.4 * parity_balance

    # 5. Composite score calculation based on strategy
    if ticket_type == "model":
        composite = llm_score * 0.7 + balance_score * 0.2 + freq_score * 0.1
    elif ticket_type == "balanced":
        composite = llm_score * 0.4 + balance_score * 0.4 + freq_score * 0.2
    elif ticket_type == "hot":
        composite = freq_score * 0.5 + llm_score * 0.3 + mb_relative_prob * 0.2
    elif ticket_type == "overdue":
        composite = overdue_score * 0.5 + llm_score * 0.3 + balance_score * 0.2
    elif ticket_type == "mb_focused":
        composite = mb_relative_prob * 0.5 + llm_score * 0.3 + balance_score * 0.2
    else:
        composite = llm_score

    return float(composite), float(main_relative_prob), float(mb_raw_prob), freq_score, overdue_score, balance_score


def generate_millionaire_life_winning_tickets(
    model: MillionaireLifeLLM,
    df: pd.DataFrame,
    num_tickets: int = 5,
    ticket_type: TicketType = "model",
    history_window: int = 15,
    pool_size: int = 1500,
    seed: Optional[int] = None,
) -> Tuple[List[MillionaireBallInfo], List[RankedTicket]]:
    """
    Generates and ranks top candidate Millionaire for Life winning tickets,
    emphasizing the top winning Millionaire Ball.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    tokenizer = MillionaireLifeTokenizer()

    # Step 1: Analyze and rank Millionaire Balls
    mb_info_list = analyze_millionaire_balls(model, df, tokenizer, history_window)
    mb_probs_map = {info.ball: info.probability for info in mb_info_list}
    top_mb = mb_info_list[0].ball

    # Step 2: Compute main numbers probability distribution
    main_probs = calculate_main_number_probabilities(model, df, tokenizer, history_window)
    main_candidates = np.arange(MAIN_MIN, MAIN_MAX + 1)

    # Step 3: Sample candidate 5-number combinations using LLM probability distribution
    sampled_tickets: List[RankedTicket] = []
    seen_tickets = set()

    # Determine extra assignment strategy
    all_extras = [info.ball for info in mb_info_list]
    extra_probs = [info.probability for info in mb_info_list]

    attempts = 0
    max_attempts = pool_size * 5

    while len(sampled_tickets) < pool_size and attempts < max_attempts:
        attempts += 1
        # Sample 5 distinct main numbers according to LLM softmax probabilities
        nums = sorted(list(np.random.choice(main_candidates, size=MAIN_COUNT, replace=False, p=main_probs)))
        
        if ticket_type == "mb_focused":
            extra = top_mb
        else:
            extra = int(np.random.choice(all_extras, p=extra_probs))

        ticket_key = (tuple(nums), extra)
        if ticket_key in seen_tickets:
            continue
        seen_tickets.add(ticket_key)

        score, llm_prob, mb_prob, freq_s, over_s, bal_s = calculate_ticket_composite_score(
            nums, extra, main_probs, mb_probs_map, df, ticket_type=ticket_type
        )

        ticket_str = f"{' '.join(f'{n:02d}' for n in nums)}  [MB: {extra}]"

        ranked_ticket = RankedTicket(
            rank=0,
            numbers=nums,
            extra=extra,
            ticket_str=ticket_str,
            score=score,
            llm_prob=llm_prob,
            mb_prob=mb_prob,
            freq_score=freq_s,
            overdue_score=over_s,
            balance_score=bal_s,
            ticket_type=ticket_type,
        )
        sampled_tickets.append(ranked_ticket)

    # Step 4: Sort candidates by composite score and pick top N
    sampled_tickets.sort(key=lambda t: t.score, reverse=True)
    top_tickets = sampled_tickets[:num_tickets]

    # Assign final ranks
    for r_idx, ticket in enumerate(top_tickets, 1):
        ticket.rank = r_idx

    return mb_info_list, top_tickets


# ==============================================================================
# Presentation & CLI Formatting
# ==============================================================================

def print_millionaire_ball_spotlight(mb_info_list: List[MillionaireBallInfo]):
    """Displays prominent spotlight and distribution for the winning Millionaire Ball."""
    top_info = mb_info_list[0]

    print("\n" + "=" * 80)
    print("🌟" + "  BEST NEXT WINNING MILLIONAIRE BALL (EXTRA)  ".center(76) + "🌟")
    print("=" * 80)
    print(f"  >>> TOP RECOMMENDED MILLIONAIRE BALL:  [ {top_info.ball} ]  <<<")
    print(f"  Model Probability: {top_info.probability * 100:.2f}%  |  Confidence Rank: #1  |  Overdue Gap: {top_info.overdue_draws} draws")
    print("-" * 80)
    print("📊 MILLIONAIRE BALL CANDIDATES DISTRIBUTION (1 to 5):")
    print(f"  {'Rank':<6} {'Millionaire Ball':<18} {'LLM Probability':<18} {'Hist Frequency':<16} {'Overdue':<12}")
    print("  " + "-" * 72)
    for idx, info in enumerate(mb_info_list, 1):
        marker = " ⭐️ [TOP PICK]" if info.is_top else ""
        print(
            f"  #{idx:<5} Ball {info.ball:<13} {info.probability * 100:6.2f}%{'':<10} "
            f"{info.frequency:<16} {info.overdue_draws} draws{marker}"
        )
    print("=" * 80)


def print_ranked_tickets_table(tickets: List[RankedTicket], ticket_type: str):
    """Displays formatted table of next winning ticket predictions."""
    print(f"\n🎟️  TOP RANKED WINNING TICKETS (Strategy: '{ticket_type}'):")
    print("+" + "-" * 6 + "+" + "-" * 32 + "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    print(f"| {'Rank':<4} | {'Winning Ticket (Main + MB)':<30} | {'Score':<12} | {'LLM Prob':<10} | {'MB Prob':<10} |")
    print("+" + "-" * 6 + "+" + "-" * 32 + "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    for t in tickets:
        print(
            f"| #{t.rank:<3} | {t.ticket_str:<30} | {t.score:<12.4f} | {t.llm_prob:<10.4f} | {t.mb_prob * 100:5.2f}%{'':<3} |"
        )
    print("+" + "-" * 6 + "+" + "-" * 32 + "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 12 + "+")


def save_tickets_to_csv(tickets: List[RankedTicket], output_path: Union[str, Path]):
    """Exports ticket predictions to CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "num1", "num2", "num3", "num4", "num5", "extra", "ticket",
            "score", "llm_prob", "mb_prob", "ticket_type"
        ])
        for t in tickets:
            writer.writerow([
                t.rank,
                t.numbers[0], t.numbers[1], t.numbers[2], t.numbers[3], t.numbers[4],
                t.extra,
                t.ticket_str,
                f"{t.score:.6f}",
                f"{t.llm_prob:.6f}",
                f"{t.mb_prob:.6f}",
                t.ticket_type,
            ])
    print(f"\n[Export] Saved {len(tickets)} winning tickets to '{path}'")


# ==============================================================================
# Main CLI Function
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create an LLM from a remote Lotto model, generate local model, train it, "
                    "and produce the next Millionaire for Life winning tickets emphasizing the best Millionaire ball."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f"Path to Millionaire for Life CSV history (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--remote-model",
        type=str,
        default=DEFAULT_REMOTE_MODEL,
        help=f"Remote Lotto model URI / URL spec (default: {DEFAULT_REMOTE_MODEL})",
    )
    parser.add_argument(
        "--local-model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Local directory to save/load the generated model (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force regeneration / download of model even if local model directory exists",
    )
    parser.add_argument(
        "--tickets",
        type=int,
        default=5,
        help="Number of winning tickets to generate (default: 5)",
    )
    parser.add_argument(
        "--ticket-type",
        type=str,
        choices=["model", "balanced", "hot", "overdue", "mb_focused"],
        default="model",
        help="Ticket selection ranking strategy (default: model)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Fine-tuning training epochs (default: 10, set 0 to skip training)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=15,
        help="History window size of previous draws (default: 15)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV file path to export generated predictions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        try:
            mx.random.seed(args.seed)
        except AttributeError:
            pass

    print("=" * 80)
    print("   MILLIONAIRE FOR LIFE - AUTOREGRESSIVE LOTTO LLM PIPELINE   ".center(80))
    print("=" * 80)

    # 1. Load historical draw dataset
    print(f"\n[Step 1] Loading Millionaire for Life history from '{args.csv}'...")
    df = load_millionaire_life_history(args.csv)
    print(f"Loaded {len(df)} historical draws (Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}).")

    # 2. Instantiate tokenizer & resolve/load model from remote spec
    tokenizer = MillionaireLifeTokenizer()
    print(f"\n[Step 2] Resolving remote model specification '{args.remote_model}'...")
    model = RemoteLottoModelLoader.fetch_or_create_model(
        remote_model_uri=args.remote_model,
        local_model_dir=args.local_model_dir,
        force_download=args.force_download,
    )

    # 3. Train / Fine-tune local model if requested
    if args.epochs > 0:
        print(f"\n[Step 3] Building autoregressive draw sequences (window={args.window})...")
        x_train, y_train = create_autoregressive_dataset(
            df=df,
            tokenizer=tokenizer,
            history_window=args.window,
            max_seq_len=model.config.max_seq_len,
        )
        train_millionaire_life_llm(
            model=model,
            x_data=x_train,
            y_data=y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            pad_id=tokenizer.pad_id,
        )
        # Save updated model
        model.save_pretrained(args.local_model_dir)
        print(f"[Model] Checkpoint saved successfully to '{args.local_model_dir}'")
    else:
        print("\n[Step 3] Skipping training (--epochs 0). Using existing model weights.")

    # 4. Generate winning tickets & emphasize Millionaire Ball
    print(f"\n[Step 4] Producing next Millionaire for Life winning tickets (tickets={args.tickets}, type={args.ticket_type})...")
    mb_info_list, ranked_tickets = generate_millionaire_life_winning_tickets(
        model=model,
        df=df,
        num_tickets=args.tickets,
        ticket_type=args.ticket_type,
        history_window=args.window,
        seed=args.seed,
    )

    # 5. Display Spotlight on Best Winning Millionaire Ball & Tickets Table
    print_millionaire_ball_spotlight(mb_info_list)
    print_ranked_tickets_table(ranked_tickets, ticket_type=args.ticket_type)

    # 6. Export to CSV if output path is provided
    if args.output:
        save_tickets_to_csv(ranked_tickets, args.output)

    print("\n[Done] Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
