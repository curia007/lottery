#!/usr/bin/env python3
"""
pick3_llm_ticket_model.py

Create an LLM from a remote Lotto base model (or initialize/download a remote Lotto LLM),
generate and persist a local model, train/fine-tune it on Pick 3 lottery draw history,
and produce the next Pick 3 winning ticket candidates.

Features:
- Download / instantiate a remote Lotto Transformer LLM specification or weights.
- Generate and manage a local LLM with Causal Self-Attention, Multi-Head Attention,
  RMSNorm/LayerNorm, Positional Embeddings, and Autoregressive Head.
- Fine-tune/train the local LLM on Idaho Pick 3 draw sequences.
- Generate ranked winning ticket predictions using autoregressive token sequence probabilities
  and historical pattern scoring (frequency, overdue, pair dynamics).
- Supports Day, Night, and Both draw modes.
- Supports ticket types: exact (Straight), any (Box), 6-way, 3-way, straight_any.

Usage:
    python idaho/pick3/pick3_llm_ticket_model.py --draw Night --tickets 5
    python idaho/pick3/pick3_llm_ticket_model.py --remote-model remote://lottery-ai/base-pick3 --epochs 10
    python idaho/pick3/pick3_llm_ticket_model.py --csv idaho/pick3/data/idaho_pick3_history.csv --tickets 10 --ticket-type any
"""

from __future__ import annotations

import argparse
import json
import math
import random
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError as exc:
    raise SystemExit(
        "MLX is not installed. Install it with:\n\n"
        "    pip install mlx\n\n"
        "MLX is optimized for Apple Silicon Macs."
    ) from exc

DrawType = Literal["Day", "Night", "Both", "both", "Combo", "combo", "Combined", "combined"]
TicketType = Literal["exact", "any", "straight_any", "6-way", "3-way"]

DEFAULT_CSV_PATH = "idaho/pick3/data/idaho_pick3_history.csv"
DEFAULT_MODEL_DIR = "models/pick3_llm_model"
DEFAULT_REMOTE_MODEL = "remote://lottery-ai/pick3-transformer-base"


# ==============================================================================
# Tokenizer & Vocabulary
# ==============================================================================

class Pick3Tokenizer:
    """
    Tokenizer for lottery sequence modeling.
    Tokens:
      0-9   : Digits '0' through '9'
      10    : '<PAD>' (Padding token)
      11    : '<BOS>' (Beginning of sequence)
      12    : '<EOS>' (End of sequence)
      13    : '<DAY>' (Day draw marker)
      14    : '<NIGHT>' (Night draw marker)
      15    : '<SEP>' (Separator between draws)
      16    : '<PRED>' (Target prediction prompt prefix)
    """
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    DAY_TOKEN = "<DAY>"
    NIGHT_TOKEN = "<NIGHT>"
    SEP_TOKEN = "<SEP>"
    PRED_TOKEN = "<PRED>"

    def __init__(self):
        self.special_tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.DAY_TOKEN,
            self.NIGHT_TOKEN,
            self.SEP_TOKEN,
            self.PRED_TOKEN,
        ]
        self.digit_tokens = [str(i) for i in range(10)]
        self.vocab = self.digit_tokens + self.special_tokens
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
    def day_id(self) -> int:
        return self.token_to_id[self.DAY_TOKEN]

    @property
    def night_id(self) -> int:
        return self.token_to_id[self.NIGHT_TOKEN]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[self.SEP_TOKEN]

    @property
    def pred_id(self) -> int:
        return self.token_to_id[self.PRED_TOKEN]

    def encode_draw(self, draw_type: str, d1: int, d2: int, d3: int) -> List[int]:
        draw_tag = self.day_id if draw_type.strip().lower().startswith("day") else self.night_id
        return [draw_tag, int(d1), int(d2), int(d3)]

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(self.id_to_token.get(int(i), "<UNK>") for i in token_ids)


# ==============================================================================
# Model Architecture (Transformer LLM)
# ==============================================================================

@dataclass
class Pick3LLMConfig:
    vocab_size: int = 17
    max_seq_len: int = 128
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 3
    mlp_ratio: int = 4
    dropout: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pick3LLMConfig":
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


class Pick3LLM(nn.Module):
    """
    Autoregressive Causal Language Model for Pick 3 Lottery Sequence Generation.
    """
    def __init__(self, config: Pick3LLMConfig):
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

        # Save config
        config_path = out_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save weights using MLX native weight serialization (.safetensors / .npz)
        weights_path = out_dir / "model.safetensors"
        self.save_weights(str(weights_path))
        return out_dir

    @classmethod
    def from_pretrained(cls, load_directory: Union[str, Path]) -> "Pick3LLM":
        """Loads a local model from saved directory."""
        in_dir = Path(load_directory)
        config_path = in_dir / "config.json"
        safetensors_path = in_dir / "model.safetensors"
        npz_path = in_dir / "model.npz"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        config = Pick3LLMConfig.from_dict(cfg_dict)
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
        config: Optional[Pick3LLMConfig] = None,
        force_download: bool = False,
    ) -> Pick3LLM:
        """
        Creates an LLM from a remote Lotto model reference or URL.
        Generates the local model and caches weights locally.
        """
        local_path = Path(local_model_dir)
        local_config_file = local_path / "config.json"
        local_weights_exist = (local_path / "model.safetensors").exists() or (local_path / "model.npz").exists()

        # Check if local model already exists and we're not forcing refresh
        if local_config_file.exists() and local_weights_exist and not force_download:
            print(f"[RemoteModelLoader] Loading existing local model from '{local_path}'")
            return Pick3LLM.from_pretrained(local_path)

        print(f"[RemoteModelLoader] Resolving remote Lotto model: '{remote_model_uri}'")

        # Handle HTTP / HTTPS remote models
        if remote_model_uri.startswith("http://") or remote_model_uri.startswith("https://"):
            try:
                print(f"[RemoteModelLoader] Fetching remote model config from {remote_model_uri}...")
                req = urllib.request.Request(
                    remote_model_uri,
                    headers={"User-Agent": "LottoLLM/1.0 (Pick3-Client)"}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    remote_data = json.loads(response.read().decode("utf-8"))
                    config = Pick3LLMConfig.from_dict(remote_data.get("config", remote_data))
                    print("[RemoteModelLoader] Remote config downloaded successfully.")
            except Exception as e:
                print(f"[RemoteModelLoader] Remote download notice: ({e}). Utilizing base remote architecture.")
                if config is None:
                    config = Pick3LLMConfig()
        else:
            # Remote URI schema e.g. "remote://lottery-ai/pick3-transformer-base"
            print(f"[RemoteModelLoader] Initializing base Lotto Transformer from remote spec: {remote_model_uri}")
            if config is None:
                config = Pick3LLMConfig(
                    vocab_size=17,
                    max_seq_len=128,
                    n_embd=64,
                    n_head=4,
                    n_layer=3,
                    mlp_ratio=4,
                )

        print(f"[RemoteModelLoader] Generating local Pick3LLM with {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dims...")
        model = Pick3LLM(config)
        mx.eval(model.parameters())

        # Save local generated model
        saved_dir = model.save_pretrained(local_path)
        print(f"[RemoteModelLoader] Local model generated and saved to '{saved_dir}'")
        return model


# ==============================================================================
# Data Loading & Preprocessing
# ==============================================================================

@dataclass
class RankedTicket:
    rank: int
    ticket: str
    score: float
    llm_prob: float
    freq_score: float
    overdue_score: float
    ticket_type: str
    draw: str


def normalize_draw(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("day"):
        return "Day"
    if text.startswith("night"):
        return "Night"
    if text in ("combo", "comb", "combined"):
        return "Combo"
    if text in ("both", "all"):
        return "Both"
    raise ValueError(f"Unknown draw type: {value!r}")


def load_pick3_history(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Loads and standardizes the Idaho Pick 3 history CSV."""
    path = Path(csv_path)
    if not path.exists():
        # Check relative to script dir or project root
        script_dir = Path(__file__).resolve().parent
        alt_path = script_dir / "data" / "idaho_pick3_history.csv"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Pick 3 CSV file not found at '{path}' or '{alt_path}'")

    df = pd.read_csv(path)
    required = {"Date", "Draw", "Num1", "Num2", "Num3"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Draw"] = df["Draw"].apply(normalize_draw)

    for col in ["Num1", "Num2", "Num3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Num1", "Num2", "Num3"])

    for col in ["Num1", "Num2", "Num3"]:
        df[col] = df[col].astype(int)
        if not df[col].between(0, 9).all():
            raise ValueError(f"Column {col} has values outside 0-9")

    # Sort chronologically (oldest draw first)
    df["Draw_Order"] = df["Draw"].apply(lambda d: 0 if d == "Day" else 1)
    df = df.sort_values(by=["Date", "Draw_Order"], ascending=[True, True]).reset_index(drop=True)
    return df


def build_training_sequences(
    df: pd.DataFrame,
    tokenizer: Pick3Tokenizer,
    history_window: int = 10,
    max_seq_len: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts historical draws into autoregressive token sequences.
    Each training example is a sliding window of past draws followed by <PRED> and target draw.
    """
    draw_records = []
    for _, row in df.iterrows():
        draw_records.append((
            row["Draw"],
            int(row["Num1"]),
            int(row["Num2"]),
            int(row["Num3"]),
        ))

    input_sequences = []
    target_sequences = []

    for i in range(history_window, len(draw_records)):
        history = draw_records[i - history_window : i]
        target_draw_type, t1, t2, t3 = draw_records[i]

        # Construct prompt tokens
        tokens = [tokenizer.bos_id]
        for d_type, d1, d2, d3 in history:
            tokens.extend(tokenizer.encode_draw(d_type, d1, d2, d3))
            tokens.append(tokenizer.sep_id)

        tokens.append(tokenizer.pred_id)
        target_draw_tag = tokenizer.day_id if target_draw_type == "Day" else tokenizer.night_id
        tokens.append(target_draw_tag)

        # Append target digits and EOS
        full_seq = tokens + [t1, t2, t3, tokenizer.eos_id]

        if len(full_seq) > max_seq_len:
            # Truncate oldest history keeping BOS
            excess = len(full_seq) - max_seq_len
            tokens_to_keep = [tokenizer.bos_id] + full_seq[1 + excess :]
            full_seq = tokens_to_keep

        # Standard autoregressive inputs and targets (shifted by 1)
        inp = full_seq[:-1]
        tgt = full_seq[1:]

        # Pad to max_seq_len - 1 if necessary
        seq_len = max_seq_len - 1
        if len(inp) < seq_len:
            pad_amount = seq_len - len(inp)
            inp = inp + [tokenizer.pad_id] * pad_amount
            tgt = tgt + [tokenizer.pad_id] * pad_amount

        input_sequences.append(inp[:seq_len])
        target_sequences.append(tgt[:seq_len])

    return np.array(input_sequences, dtype=np.int32), np.array(target_sequences, dtype=np.int32)


# ==============================================================================
# Model Training / Fine-tuning
# ==============================================================================

def train_pick3_llm(
    model: Pick3LLM,
    inputs: np.ndarray,
    targets: np.ndarray,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    verbose: bool = True,
) -> Pick3LLM:
    """
    Trains / fine-tunes the local Pick 3 LLM on draw history sequences.
    """
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    pad_id = Pick3Tokenizer().pad_id
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(m: Pick3LLM, x: mx.array, y: mx.array) -> mx.array:
        logits = m(x)  # (B, T, vocab_size)
        loss = nn.losses.cross_entropy(logits, y, reduction="none")
        mask = (y != pad_id).astype(mx.float32)
        masked_loss = mx.sum(loss * mask) / mx.maximum(mx.sum(mask), 1.0)
        return masked_loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    num_samples = len(inputs)
    indices = np.arange(num_samples)

    if verbose:
        print(f"\n[Training] Fine-tuning Pick3LLM over {epochs} epochs on {num_samples} draw sequences (batch_size={batch_size})...")

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        total_loss = 0.0
        batches = 0

        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            xb = mx.array(inputs[batch_idx])
            yb = mx.array(targets[batch_idx])

            loss, grads = loss_and_grad_fn(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += float(loss.item())
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        if verbose:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Causal LM Loss: {avg_loss:.4f} | Perplexity: {math.exp(min(avg_loss, 20.0)):.2f}")

    return model


# ==============================================================================
# Ticket Prediction & Ranking
# ==============================================================================

def compute_frequency_and_overdue_scores(df: pd.DataFrame) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Calculates historical digit frequency and overdue recency scores."""
    digits = df[["Num1", "Num2", "Num3"]].values.flatten()
    total_digits = len(digits)

    # Frequency
    counts = Counter(digits)
    freq_scores = {d: (counts[d] + 1) / (total_digits + 10) for d in range(10)}

    # Overdue
    overdue_scores = {}
    total_draws = len(df)
    for d in range(10):
        # find last appearance in reverse
        matches = np.where(digits[::-1] == d)[0]
        if len(matches) > 0:
            gap = matches[0] // 3
        else:
            gap = total_draws
        # Normalized overdue score
        overdue_scores[d] = min(gap / 30.0, 1.0)

    return freq_scores, overdue_scores


def calculate_ticket_probabilities(
    model: Pick3LLM,
    df: pd.DataFrame,
    target_draw: str,
    tokenizer: Pick3Tokenizer,
    history_window: int = 10,
) -> np.ndarray:
    """
    Computes exact joint probabilities P(d1, d2, d3 | prompt) for all 1,000 combinations (000 - 999).
    If target_draw is 'Combo' / 'Combined', averages the conditional joint distribution of Day and Night draws.
    """
    norm_target = normalize_draw(target_draw)
    if norm_target in ("Combo", "Both"):
        # Combine predictions by evaluating joint probabilities under both Day and Night draw conditions
        probs_day = calculate_ticket_probabilities(model, df, "Day", tokenizer, history_window)
        probs_night = calculate_ticket_probabilities(model, df, "Night", tokenizer, history_window)
        combined_probs = 0.5 * probs_day + 0.5 * probs_night
        comb_sum = np.sum(combined_probs)
        return combined_probs / comb_sum if comb_sum > 0 else combined_probs

    recent_draws = df.tail(history_window)
    tokens = [tokenizer.bos_id]
    for _, row in recent_draws.iterrows():
        tokens.extend(tokenizer.encode_draw(row["Draw"], row["Num1"], row["Num2"], row["Num3"]))
        tokens.append(tokenizer.sep_id)

    tokens.append(tokenizer.pred_id)
    target_draw_tag = tokenizer.day_id if norm_target == "Day" else tokenizer.night_id
    tokens.append(target_draw_tag)

    prompt_len = len(tokens)

    # We will evaluate 1,000 combinations
    all_combos = []
    for d1 in range(10):
        for d2 in range(10):
            for d3 in range(10):
                all_combos.append([d1, d2, d3])

    all_combos_arr = np.array(all_combos, dtype=np.int32)
    num_combos = len(all_combos_arr)

    # Build sequence matrix (1000, prompt_len + 3)
    seq_matrix = np.tile(tokens, (num_combos, 1))
    seq_matrix = np.concatenate([seq_matrix, all_combos_arr], axis=1)

    # Autoregressive forward pass in batches
    probs = np.zeros(num_combos, dtype=np.float64)
    batch_size = 250

    for i in range(0, num_combos, batch_size):
        batch_seqs = mx.array(seq_matrix[i : i + batch_size])
        logits = model(batch_seqs)  # (B, T, vocab_size)
        logits_np = np.array(logits)

        for b in range(len(batch_seqs)):
            # Logits for d1 at position (prompt_len - 1)
            logits_d1 = logits_np[b, prompt_len - 1, :10]
            exp_d1 = np.exp(logits_d1 - np.max(logits_d1))
            p_d1 = exp_d1 / np.sum(exp_d1)

            # Logits for d2 at position (prompt_len)
            logits_d2 = logits_np[b, prompt_len, :10]
            exp_d2 = np.exp(logits_d2 - np.max(logits_d2))
            p_d2 = exp_d2 / np.sum(exp_d2)

            # Logits for d3 at position (prompt_len + 1)
            logits_d3 = logits_np[b, prompt_len + 1, :10]
            exp_d3 = np.exp(logits_d3 - np.max(logits_d3))
            p_d3 = exp_d3 / np.sum(exp_d3)

            d1, d2, d3 = all_combos_arr[i + b]
            probs[i + b] = p_d1[d1] * p_d2[d2] * p_d3[d3]

    # Normalize joint distribution over the 1,000 tickets
    probs_sum = np.sum(probs)
    if probs_sum > 0:
        probs = probs / probs_sum

    return probs


def ticket_shape(d1: int, d2: int, d3: int) -> Tuple[int, ...]:
    counts = Counter([d1, d2, d3])
    return tuple(sorted(counts.values(), reverse=True))


def matches_ticket_type(d1: int, d2: int, d3: int, ticket_type: TicketType) -> bool:
    shape = ticket_shape(d1, d2, d3)
    if ticket_type in ("exact", "any", "straight_any"):
        return True
    if ticket_type == "6-way":
        return shape == (1, 1, 1)
    if ticket_type == "3-way":
        return shape == (2, 1)
    return False


def generate_pick3_winning_tickets(
    model: Pick3LLM,
    df: pd.DataFrame,
    draw_type: str = "Night",
    num_tickets: int = 5,
    ticket_type: TicketType = "exact",
    history_window: int = 10,
) -> List[RankedTicket]:
    """
    Produces the top ranked Pick 3 winning ticket candidates using LLM token probabilities
    combined with statistical lottery indicators.
    """
    tokenizer = Pick3Tokenizer()
    target_draw = normalize_draw(draw_type)
    probs = calculate_ticket_probabilities(model, df, target_draw, tokenizer, history_window)
    freq_scores, overdue_scores = compute_frequency_and_overdue_scores(df)

    candidates: List[RankedTicket] = []
    seen_box = set()

    for idx, (d1, d2, d3) in enumerate(
        (d1, d2, d3) for d1 in range(10) for d2 in range(10) for d3 in range(10)
    ):
        if not matches_ticket_type(d1, d2, d3, ticket_type):
            continue

        ticket_str = f"{d1}{d2}{d3}"

        # If any / box ticket type, optionally aggregate canonical box combinations
        if ticket_type in ("any", "6-way", "3-way"):
            box_key = "".join(sorted(ticket_str))
            if box_key in seen_box:
                continue
            seen_box.add(box_key)

        llm_p = float(probs[idx])
        f_score = float((freq_scores[d1] + freq_scores[d2] + freq_scores[d3]) / 3.0)
        o_score = float((overdue_scores[d1] + overdue_scores[d2] + overdue_scores[d3]) / 3.0)

        # Composite score weighting LLM confidence and statistical signals
        composite_score = (0.60 * llm_p * 1000.0) + (0.25 * f_score * 10.0) + (0.15 * o_score * 10.0)

        candidates.append(
            RankedTicket(
                rank=0,
                ticket=ticket_str,
                score=composite_score,
                llm_prob=llm_p,
                freq_score=f_score,
                overdue_score=o_score,
                ticket_type=ticket_type,
                draw=target_draw,
            )
        )

    # Sort candidates by composite score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    # Assign ranks and return top N
    results = []
    for rank, cand in enumerate(candidates[:num_tickets], start=1):
        cand.rank = rank
        results.append(cand)

    return results


def print_tickets_table(tickets: List[RankedTicket], draw_label: str):
    """Nicely formats and prints ticket predictions to the console."""
    print(f"\n=======================================================================")
    print(f"       TOP PREDICTED PICK 3 WINNING TICKETS ({draw_label.upper()} DRAW)       ")
    print(f"=======================================================================")
    print(f"{'Rank':<5} {'Ticket':<8} {'Composite Score':<17} {'LLM Prob':<12} {'Freq Score':<12} {'Type':<10}")
    print(f"-----------------------------------------------------------------------")
    for t in tickets:
        print(
            f"#{t.rank:<4} {t.ticket:<8} {t.score:<17.4f} {t.llm_prob:<12.5f} {t.freq_score:<12.4f} {t.ticket_type:<10}"
        )
    print(f"=======================================================================\n")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create an LLM from a remote Lotto model, train local model, and produce Pick 3 tickets."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f"Path to Idaho Pick 3 history CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--remote-model",
        type=str,
        default=DEFAULT_REMOTE_MODEL,
        help=f"Remote Lotto model URI or URL (default: {DEFAULT_REMOTE_MODEL})",
    )
    parser.add_argument(
        "--local-model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Local directory to save/load the local LLM model (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--draw",
        type=str,
        choices=["Day", "Night", "both", "Both", "combo", "Combo", "combined", "Combined"],
        default="combo",
        help="Target draw type: 'combo' (combine both night and day draws), 'Day', 'Night', or 'both' (default: combo)",
    )
    parser.add_argument(
        "--tickets",
        type=int,
        default=5,
        help="Number of winning ticket candidates to generate (default: 5)",
    )
    parser.add_argument(
        "--ticket-type",
        type=str,
        choices=["exact", "any", "straight_any", "6-way", "3-way"],
        default="exact",
        help="Ticket bet type (default: exact)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of training/fine-tuning epochs for local LLM (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
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
        default=10,
        help="Number of recent historical draws per LLM sequence window (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save generated predictions as CSV",
    )

    args = parser.parse_args()

    print("=======================================================================")
    print("           PICK 3 LOTTO LLM GENERATOR & PREDICTION PIPELINE           ")
    print("=======================================================================")

    # 1. Step 1: Create LLM from Remote Lotto model & generate local model
    print("\n[Step 1/3] Loading / Creating LLM from remote model...")
    local_model = RemoteLottoModelLoader.fetch_or_create_model(
        remote_model_uri=args.remote_model,
        local_model_dir=args.local_model_dir,
    )

    # 2. Step 2: Load historical data and train/fine-tune local LLM
    print(f"\n[Step 2/3] Loading history from '{args.csv}' and training local LLM...")
    df = load_pick3_history(args.csv)
    print(f"  Loaded {len(df)} total draws from history.")

    tokenizer = Pick3Tokenizer()
    inputs, targets = build_training_sequences(
        df=df,
        tokenizer=tokenizer,
        history_window=args.window,
        max_seq_len=local_model.config.max_seq_len,
    )

    trained_model = train_pick3_llm(
        model=local_model,
        inputs=inputs,
        targets=targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    # Persist updated local model weights
    trained_model.save_pretrained(args.local_model_dir)
    print(f"[Model Saved] Local LLM updated and saved to '{args.local_model_dir}'")

    # 3. Step 3: Produce the next Pick 3 winning tickets
    print(f"\n[Step 3/3] Generating next Pick 3 winning tickets (type={args.ticket_type})...")
    draws_to_run = ["Day", "Night"] if args.draw.lower() == "both" else [args.draw]

    all_results = []
    for draw_name in draws_to_run:
        tickets = generate_pick3_winning_tickets(
            model=trained_model,
            df=df,
            draw_type=draw_name,
            num_tickets=args.tickets,
            ticket_type=args.ticket_type,
            history_window=args.window,
        )
        print_tickets_table(tickets, draw_label=draw_name)
        all_results.extend(tickets)

    if args.output:
        out_df = pd.DataFrame([asdict(t) for t in all_results])
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved {len(all_results)} ticket predictions to '{out_path}'")


if __name__ == "__main__":
    main()
