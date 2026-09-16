import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from millionaire_life.millionaire_life_llm_ticket_model import (
    MillionaireLifeLLM,
    MillionaireLifeLLMConfig,
    MillionaireLifeTokenizer,
    RemoteLottoModelLoader,
    analyze_millionaire_balls,
    calculate_main_number_probabilities,
    calculate_ticket_composite_score,
    create_autoregressive_dataset,
    generate_millionaire_life_winning_tickets,
    load_millionaire_life_history,
    train_millionaire_life_llm,
)


class TestMillionaireLifeLLM(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MillionaireLifeTokenizer(max_main=58, max_extra=5)
        # 30 synthetic draws
        rows = []
        for i in range(30):
            rows.append({
                "Date": f"2026-08-{i+1:02d}",
                "Num1": (i * 2) % 50 + 1,
                "Num2": (i * 2 + 3) % 50 + 2,
                "Num3": (i * 2 + 7) % 50 + 3,
                "Num4": (i * 2 + 11) % 50 + 4,
                "Num5": (i * 2 + 15) % 50 + 5,
                "Extra": (i % 5) + 1,
            })
        self.sample_df = pd.DataFrame(rows)

    def test_tokenizer(self):
        self.assertEqual(self.tokenizer.vocab_size, 6 + 58 + 5)
        encoded = self.tokenizer.encode_draw([5, 12, 23, 34, 45], extra=3)
        self.assertEqual(len(encoded), 7)  # 5 main + MB tag + extra token
        
        main_nums, extra = self.tokenizer.decode_draw(encoded)
        self.assertEqual(main_nums, [5, 12, 23, 34, 45])
        self.assertEqual(extra, 3)

    def test_remote_model_loader_and_local_save(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_millionaire_model"
            model = RemoteLottoModelLoader.fetch_or_create_model(
                remote_model_uri="remote://lottery-ai/base-millionaire-test",
                local_model_dir=tmp_path,
                config=MillionaireLifeLLMConfig(n_embd=32, n_head=2, n_layer=2, max_seq_len=64),
            )
            self.assertIsInstance(model, MillionaireLifeLLM)
            self.assertTrue((tmp_path / "config.json").exists())
            self.assertTrue((tmp_path / "model.safetensors").exists())

            # Test loading from pretrained
            loaded_model = MillionaireLifeLLM.from_pretrained(tmp_path)
            self.assertEqual(loaded_model.config.n_embd, 32)
            self.assertEqual(loaded_model.config.n_layer, 2)

    def test_training_and_millionaire_ball_analysis(self):
        config = MillionaireLifeLLMConfig(vocab_size=self.tokenizer.vocab_size, max_seq_len=128, n_embd=32, n_head=2, n_layer=2)
        model = MillionaireLifeLLM(config)

        x_train, y_train = create_autoregressive_dataset(
            df=self.sample_df,
            tokenizer=self.tokenizer,
            history_window=5,
            max_seq_len=128,
        )
        self.assertGreater(len(x_train), 0)
        self.assertEqual(x_train.shape, y_train.shape)

        metrics = train_millionaire_life_llm(
            model=model,
            x_data=x_train,
            y_data=y_train,
            epochs=2,
            batch_size=8,
            lr=1e-3,
            pad_id=self.tokenizer.pad_id,
        )
        self.assertEqual(len(metrics["loss"]), 2)
        self.assertTrue(metrics["loss"][-1] > 0)

        # Test Millionaire Ball conditional analysis
        mb_info = analyze_millionaire_balls(model, self.sample_df, self.tokenizer, history_window=5)
        self.assertEqual(len(mb_info), 5)
        self.assertTrue(any(info.is_top for info in mb_info))
        prob_sum = sum(info.probability for info in mb_info)
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

        # Test Main Number probabilities
        main_probs = calculate_main_number_probabilities(model, self.sample_df, self.tokenizer, history_window=5)
        self.assertEqual(len(main_probs), 58)
        self.assertAlmostEqual(float(np.sum(main_probs)), 1.0, places=4)

    def test_ticket_generation_strategies(self):
        config = MillionaireLifeLLMConfig(vocab_size=self.tokenizer.vocab_size, max_seq_len=128, n_embd=32, n_head=2, n_layer=2)
        model = MillionaireLifeLLM(config)

        for strat in ["model", "balanced", "hot", "overdue", "mb_focused"]:
            mb_info, tickets = generate_millionaire_life_winning_tickets(
                model=model,
                df=self.sample_df,
                num_tickets=4,
                ticket_type=strat,
                history_window=5,
                pool_size=100,
                seed=42,
            )
            self.assertEqual(len(tickets), 4)
            self.assertEqual(len(mb_info), 5)
            for t in tickets:
                self.assertEqual(len(t.numbers), 5)
                self.assertTrue(1 <= t.extra <= 5)
                self.assertGreater(t.score, 0.0)
                if strat == "mb_focused":
                    self.assertEqual(t.extra, mb_info[0].ball)

    def test_cli_execution(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_csv = Path(tmp_dir) / "millionaire_preds.csv"
            model_dir = Path(tmp_dir) / "test_model_cli"

            cmd = [
                sys.executable,
                "millionaire_life/millionaire_life_llm_ticket_model.py",
                "--csv", "millionaire_life/data/millionaire_life_history.csv",
                "--local-model-dir", str(model_dir),
                "--tickets", "5",
                "--ticket-type", "mb_focused",
                "--epochs", "1",
                "--batch-size", "32",
                "--output", str(output_csv),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Millionaire CLI failed: {res.stderr}")
            self.assertTrue(output_csv.exists())
            df_out = pd.read_csv(output_csv)
            self.assertEqual(len(df_out), 5)
            self.assertIn("rank", df_out.columns)
            self.assertIn("extra", df_out.columns)


if __name__ == "__main__":
    unittest.main()
