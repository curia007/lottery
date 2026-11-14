import json
import tempfile
import unittest
from pathlib import Path

from idaho.pick3.evaluate_pick3_mlx import (
    Pick3EvalTokenizer,
    compute_masked_metrics,
    create_evaluation_batch,
    evaluate_model_on_jsonl,
    evaluate_step,
    load_draw_jsonl,
    resolve_default_jsonl,
)

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    mx = None
    nn = None


class TestEvaluatePick3MLX(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Pick3EvalTokenizer()
        self.sample_records = [
            {"date": "2026-08-01", "draw": "Day", "num1": 1, "num2": 2, "num3": 3},
            {"date": "2026-08-01", "draw": "Night", "num1": 4, "num2": 5, "num3": 6},
            {"date": "2026-08-02", "draw": "Day", "num1": 7, "num2": 8, "num3": 9},
            {"date": "2026-08-02", "draw": "Night", "num1": 0, "num2": 1, "num3": 2},
            {"date": "2026-08-03", "draw": "Day", "num1": 3, "num2": 4, "num3": 5},
            {"date": "2026-08-03", "draw": "Night", "num1": 6, "num2": 7, "num3": 8},
        ]

    def test_load_draw_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_p = Path(tmp_dir) / "test_picks.jsonl"
            # Write unsorted lines to verify chronological sorting
            unsorted = [
                {"date": "2026-08-02", "draw": "Night", "num1": 0, "num2": 1, "num3": 2},
                {"date": "2026-08-01", "draw": "Day", "num1": 1, "num2": 2, "num3": 3},
                {"date": "2026-08-01", "draw": "Night", "num1": 4, "num2": 5, "num3": 6},
            ]
            with file_p.open("w", encoding="utf-8") as f:
                for r in unsorted:
                    f.write(json.dumps(r) + "\n")

            loaded = load_draw_jsonl(file_p)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0]["date"], "2026-08-01")
            self.assertEqual(loaded[0]["draw"], "Day")
            self.assertEqual(loaded[1]["date"], "2026-08-01")
            self.assertEqual(loaded[1]["draw"], "Night")
            self.assertEqual(loaded[2]["date"], "2026-08-02")
            self.assertEqual(loaded[2]["draw"], "Night")

    def test_create_evaluation_batch_masking(self):
        x, y, mask = create_evaluation_batch(
            records=self.sample_records,
            window_size=2,
            batch_size=2,
            max_len=32,
            tokenizer=self.tokenizer,
            token_aware=False,
        )

        if mx is not None:
            self.assertEqual(x.shape, (2, 32))
            self.assertEqual(y.shape, (2, 32))
            self.assertEqual(mask.shape, (2, 32))

            # Convert to list for inspection
            x_list = x.tolist()
            y_list = y.tolist()
            mask_list = mask.tolist()
        else:
            x_list = x
            y_list = y
            mask_list = mask

        for b_idx in range(len(x_list)):
            row_x = x_list[b_idx]
            row_y = y_list[b_idx]
            row_mask = mask_list[b_idx]

            # Verify prompt positions have mask == 0.0
            self.assertEqual(row_mask[0], 0.0)

            # Verify target digits are masked with 1.0
            active_indices = [idx for idx, m in enumerate(row_mask) if m == 1.0]
            self.assertEqual(len(active_indices), 4)  # 3 digits + EOS

            # Verify padding tokens have mask == 0.0
            pad_indices = [idx for idx, t in enumerate(row_x) if t == self.tokenizer.pad_id]
            for p_idx in pad_indices:
                self.assertEqual(row_mask[p_idx], 0.0)

    @unittest.skipIf(mx is None, "MLX is not installed in current environment")
    def test_compute_masked_metrics_accuracy(self):
        # Construct synthetic logits, y, and mask
        # Batch=2, Seq=4, Vocab=17
        vocab_size = self.tokenizer.vocab_size
        y = mx.array([[1, 2, 3, 12], [4, 5, 6, 12]], dtype=mx.int32)
        mask = mx.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], dtype=mx.float32)

        # Construct logits that predict y perfectly
        logits_list = [[[0.0] * vocab_size for _ in range(4)] for _ in range(2)]
        logits_list[0][0][1] = 10.0
        logits_list[0][1][2] = 10.0
        logits_list[0][2][3] = 10.0
        logits_list[0][3][12] = 10.0

        logits_list[1][0][4] = 10.0
        logits_list[1][1][5] = 10.0
        logits_list[1][2][6] = 10.0
        logits_list[1][3][12] = 10.0

        logits = mx.array(logits_list, dtype=mx.float32)

        loss, token_acc, exact_acc = compute_masked_metrics(logits, y, mask)
        mx.eval(loss, token_acc, exact_acc)

        self.assertAlmostEqual(float(token_acc.item()), 1.0, places=4)
        self.assertAlmostEqual(float(exact_acc.item()), 1.0, places=4)
        self.assertLess(float(loss.item()), 0.01)

    @unittest.skipIf(mx is None, "MLX is not installed in current environment")
    def test_evaluate_model_pipeline(self):
        class MockModel(nn.Module):
            def __init__(self, vocab_size=17):
                super().__init__()
                self.vocab_size = vocab_size

            def __call__(self, x):
                B, T = x.shape
                # Predict token 0 everywhere
                out = mx.zeros((B, T, self.vocab_size), dtype=mx.float32)
                return out

        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_file = Path(tmp_dir) / "picks.jsonl"
            with jsonl_file.open("w", encoding="utf-8") as f:
                for r in self.sample_records * 3:
                    f.write(json.dumps(r) + "\n")

            model = MockModel(vocab_size=self.tokenizer.vocab_size)
            metrics = evaluate_model_on_jsonl(
                model=model,
                jsonl_path=jsonl_file,
                window_size=2,
                batch_size=4,
                max_len=24,
            )

            self.assertIn("mean_cross_entropy", metrics)
            self.assertIn("mean_token_accuracy", metrics)
            self.assertIn("exact_draw_match_rate", metrics)
            self.assertGreater(metrics["total_samples"], 0)
            self.assertGreater(metrics["num_batches"], 0)


    def test_resolve_default_jsonl(self):
        resolved = resolve_default_jsonl()
        self.assertEqual(resolved.name, "idaho_pick3_last_picks.jsonl")
        self.assertEqual(resolved.parent.name, "data")

    def test_tokens_per_draw(self):
        """Each draw encodes to 5 tokens: <DRAW_TAG> + d1 + d2 + d3 + <SEP>"""
        self.assertEqual(Pick3EvalTokenizer.tokens_per_draw(), 5)

    def test_max_draws_for_token_budget(self):
        """Token-aware window calculation for various sequence lengths."""
        # max_seq_len=128: overhead=3, reserve=4, available=121, draws=121//5=24
        self.assertEqual(Pick3EvalTokenizer.max_draws_for_token_budget(128), 24)
        # max_seq_len=64: overhead=3, reserve=4, available=57, draws=57//5=11
        self.assertEqual(Pick3EvalTokenizer.max_draws_for_token_budget(64), 11)
        # max_seq_len=20: overhead=3, reserve=4, available=13, draws=13//5=2
        self.assertEqual(Pick3EvalTokenizer.max_draws_for_token_budget(20), 2)
        # Very small budget: should return 0
        self.assertEqual(Pick3EvalTokenizer.max_draws_for_token_budget(5), 0)

    def test_build_token_aware_window(self):
        """Token-aware window builds prompt within token budget."""
        tok = Pick3EvalTokenizer()
        # With 128 max tokens, should fit all 6 sample records (6 * 5 = 30 tokens)
        prompt, prompt_len = tok.build_token_aware_window(
            records=self.sample_records,
            target_draw={"draw": "Night"},
            max_seq_len=128,
        )
        # Prompt = BOS(1) + 6 draws * 5 tokens + PRED(1) + target_tag(1) = 33 tokens
        self.assertEqual(prompt_len, 33)
        self.assertEqual(prompt[0], tok.bos_id)
        self.assertEqual(prompt[-2], tok.pred_id)

    def test_token_aware_batch_construction(self):
        """Batch construction with token_aware=True uses optimal window size."""
        # Use enough records for token-aware window (need > window + batch)
        extended_records = self.sample_records * 5  # 30 records
        x, y, mask = create_evaluation_batch(
            records=extended_records,
            window_size=2,
            batch_size=4,
            max_len=128,
            tokenizer=self.tokenizer,
            token_aware=True,
        )
        # Should work without error and produce batches
        if mx is not None:
            self.assertEqual(x.shape[1], 128)
            self.assertGreater(x.shape[0], 0)

    def test_token_aware_disabled_falls_back_to_window_size(self):
        """When token_aware=False, uses the specified window_size."""
        x, y, mask = create_evaluation_batch(
            records=self.sample_records,
            window_size=2,
            batch_size=2,
            max_len=32,
            tokenizer=self.tokenizer,
            token_aware=False,
        )
        # Should work with fixed window_size=2
        if mx is not None:
            self.assertEqual(x.shape[0], 2)  # 2 batches


if __name__ == "__main__":
    unittest.main()
