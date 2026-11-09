#!/usr/bin/env python3
"""
CLI entry point to evaluate Pick 3 MLX models and compute loss and accuracy
across batches on JSONL draw data.

Usage:
    python evaluate_pick3_mlx.py
    python evaluate_pick3_mlx.py --jsonl idaho/pick3/data/idaho_pick3_last_picks.jsonl --batch-size 64
"""

from idaho.pick3.evaluate_pick3_mlx import (
    DEFAULT_JSONL_PATH,
    DEFAULT_MODEL_DIR,
    Pick3EvalTokenizer,
    compute_masked_metrics,
    create_evaluation_batch,
    evaluate_model_on_jsonl,
    evaluate_step,
    load_draw_jsonl,
    main,
)

if __name__ == "__main__":
    main()
