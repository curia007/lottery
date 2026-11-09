#!/usr/bin/env python3
"""
CLI entry point for the Idaho Pick 3 MLX training pipeline.

Usage:
    python train_pick_3.py
    python train_pick_3.py --epochs 10 --batch-size 32 --lr 1e-3
    python train_pick_3.py --jsonl idaho/pick3/data/idaho_pick3_last_picks.jsonl --predict --tickets 5
"""

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
    run_training_pipeline,
    train_pick_3_model,
)

if __name__ == "__main__":
    main()
