import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from idaho.pick3.pick3_llm_ticket_model import (
    Pick3LLM,
    Pick3LLMConfig,
    Pick3Tokenizer,
    RemoteLottoModelLoader,
    build_training_sequences,
    calculate_ticket_probabilities,
    compute_frequency_and_overdue_scores,
    generate_pick3_winning_tickets,
    load_pick3_history,
    matches_ticket_type,
    normalize_draw,
    train_pick3_llm,
)


class TestPick3LLM(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Pick3Tokenizer()
        self.sample_df = pd.DataFrame({
            "Date": ["2026-08-01", "2026-08-01", "2026-08-02", "2026-08-02", "2026-08-03"],
            "Draw": ["Day", "Night", "Day", "Night", "Day"],
            "Num1": [1, 5, 2, 8, 9],
            "Num2": [2, 6, 3, 4, 1],
            "Num3": [3, 7, 4, 0, 5],
        })

    def test_tokenizer(self):
        self.assertEqual(self.tokenizer.vocab_size, 17)
        draw_tokens = self.tokenizer.encode_draw("Day", 1, 2, 3)
        self.assertEqual(draw_tokens, [self.tokenizer.day_id, 1, 2, 3])
        night_tokens = self.tokenizer.encode_draw("Night", 9, 0, 5)
        self.assertEqual(night_tokens, [self.tokenizer.night_id, 9, 0, 5])

    def test_normalize_draw(self):
        self.assertEqual(normalize_draw("day"), "Day")
        self.assertEqual(normalize_draw("NIGHT"), "Night")
        with self.assertRaises(ValueError):
            normalize_draw("InvalidDraw")

    def test_ticket_types(self):
        self.assertTrue(matches_ticket_type(1, 2, 3, "6-way"))
        self.assertFalse(matches_ticket_type(1, 1, 3, "6-way"))
        self.assertTrue(matches_ticket_type(1, 1, 3, "3-way"))
        self.assertFalse(matches_ticket_type(1, 2, 3, "3-way"))
        self.assertTrue(matches_ticket_type(1, 1, 1, "exact"))

    def test_remote_model_loader_and_local_save(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_model"
            model = RemoteLottoModelLoader.fetch_or_create_model(
                remote_model_uri="remote://lottery-ai/base-pick3-test",
                local_model_dir=tmp_path,
                config=Pick3LLMConfig(n_embd=32, n_head=2, n_layer=2, max_seq_len=64),
            )
            self.assertIsInstance(model, Pick3LLM)
            self.assertTrue((tmp_path / "config.json").exists())
            self.assertTrue((tmp_path / "model.safetensors").exists())

            # Load back
            loaded_model = Pick3LLM.from_pretrained(tmp_path)
            self.assertEqual(loaded_model.config.n_embd, 32)
            self.assertEqual(loaded_model.config.n_layer, 2)

    def test_training_and_generation(self):
        # Create small dataset for rapid test
        rows = []
        for i in range(25):
            rows.append({
                "Date": f"2026-01-{i+1:02d}",
                "Draw": "Day" if i % 2 == 0 else "Night",
                "Num1": (i * 2) % 10,
                "Num2": (i * 3) % 10,
                "Num3": (i * 5) % 10,
            })
        df = pd.DataFrame(rows)

        config = Pick3LLMConfig(vocab_size=17, max_seq_len=64, n_embd=32, n_head=2, n_layer=2)
        model = Pick3LLM(config)

        inputs, targets = build_training_sequences(
            df=df,
            tokenizer=self.tokenizer,
            history_window=4,
            max_seq_len=64,
        )
        self.assertGreater(len(inputs), 0)
        self.assertEqual(inputs.shape, targets.shape)

        trained_model = train_pick3_llm(
            model=model,
            inputs=inputs,
            targets=targets,
            epochs=2,
            batch_size=8,
            lr=1e-3,
            verbose=False,
        )

        # Generate tickets
        tickets = generate_pick3_winning_tickets(
            model=trained_model,
            df=df,
            draw_type="Night",
            num_tickets=5,
            ticket_type="exact",
            history_window=4,
        )
        self.assertEqual(len(tickets), 5)
        for t in tickets:
            self.assertEqual(len(t.ticket), 3)
            self.assertGreater(t.score, 0.0)
            self.assertEqual(t.draw, "Night")

        # Test any / 6-way / 3-way
        tickets_6way = generate_pick3_winning_tickets(
            model=trained_model,
            df=df,
            draw_type="Day",
            num_tickets=3,
            ticket_type="6-way",
            history_window=4,
        )
        self.assertLessEqual(len(tickets_6way), 3)
        for t in tickets_6way:
            self.assertEqual(len(set(t.ticket)), 3)

    def test_cli_execution(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_csv = Path(tmp_dir) / "preds.csv"
            model_dir = Path(tmp_dir) / "test_model_cli"

            cmd = [
                sys.executable,
                "idaho/pick3/pick3_llm_ticket_model.py",
                "--csv", "idaho/pick3/data/idaho_pick3_history.csv",
                "--local-model-dir", str(model_dir),
                "--draw", "both",
                "--tickets", "3",
                "--epochs", "1",
                "--batch-size", "64",
                "--output", str(output_csv),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI failed: {res.stderr}")
            self.assertTrue(output_csv.exists())
            df_out = pd.read_csv(output_csv)
            self.assertEqual(len(df_out), 6)  # 3 Day + 3 Night


if __name__ == "__main__":
    unittest.main()
