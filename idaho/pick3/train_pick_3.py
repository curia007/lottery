#!/usr/bin/env python3
"""
train_pick_3.py (root wrapper)

Sequence training pipeline for the Idaho Pick 3 Transformer using Apple MLX.
Delegates to `idaho/pick3/train_pick_3.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from idaho.pick3.train_pick_3 import (
    DEFAULT_CSV_PATH,
    DEFAULT_JSONL_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_REMOTE_MODEL,
    Pick3DrawRecord,
    TrainingHistory,
    build_training_and_validation_datasets,
    evaluate_model_metrics,
    load_draw_records,
    main,
    parse_draw_date,
    run_training_pipeline,
    train_pick_3_model,
)

__all__ = [
    "DEFAULT_CSV_PATH",
    "DEFAULT_JSONL_PATH",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_REMOTE_MODEL",
    "Pick3DrawRecord",
    "TrainingHistory",
    "build_training_and_validation_datasets",
    "evaluate_model_metrics",
    "load_draw_records",
    "main",
    "parse_draw_date",
    "run_training_pipeline",
    "train_pick_3_model",
]

if __name__ == "__main__":
    main()
