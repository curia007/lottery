import json
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

from idaho.pick3.pick3_llm_ticket_model import Pick3LLM, Pick3LLMConfig, Pick3Tokenizer
from idaho.pick3.train_pick_3 import (
    Pick3DrawRecord,
    TrainingHistory,
    build_training_and_validation_datasets,
    evaluate_model_metrics,
    load_draw_records,
    parse_draw_date,
    run_training_pipeline,
    train_pick_3_model,
)

try:
    import mlx.core as mx
except ImportError:
    mx = None


class TestTrainPick3(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Pick3Tokenizer()
        self.sample_draws = [
            {"date": "2026-01-01", "draw": "Day", "num1": 1, "num2": 2, "num3": 3},
            {"date": "2026-01-01", "draw": "Night", "num1": 4, "num2": 5, "num3": 6},
            {"date": "2026-01-02", "draw": "Day", "num1": 7, "num2": 8, "num3": 9},
            {"date": "2026-01-02", "draw": "Night", "num1": 0, "num2": 1, "num3": 2},
            {"date": "2026-01-03", "draw": "Day", "num1": 3, "num2": 4, "num3": 5},
            {"date": "2026-01-03", "draw": "Night", "num1": 6, "num2": 7, "num3": 8},
            {"date": "2026-01-04", "draw": "Day", "num1": 9, "num2": 0, "num3": 1},
            {"date": "2026-01-04", "draw": "Night", "num1": 2, "num2": 3, "num3": 4},
            {"date": "2026-01-05", "draw": "Day", "num1": 5, "num2": 6, "num3": 7},
            {"date": "2026-01-05", "draw": "Night", "num1": 8, "num2": 9, "num3": 0},
            {"date": "2026-01-06", "draw": "Day", "num1": 1, "num2": 3, "num3": 5},
            {"date": "2026-01-06", "draw": "Night", "num1": 2, "num2": 4, "num3": 6},
        ]

    def test_parse_draw_date(self):
        dt = parse_draw_date("2026-11-07")
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 11)
        self.assertEqual(dt.day, 7)

    def test_load_draw_records_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "test_picks.jsonl"
            # Write unsorted to test sorting
            reversed_draws = list(reversed(self.sample_draws))
            with jsonl_path.open("w", encoding="utf-8") as f:
                for d in reversed_draws:
                    f.write(json.dumps(d) + "\n")

            records = load_draw_records(jsonl_path)
            self.assertEqual(len(records), len(self.sample_draws))
            # Verify chronological ordering
            self.assertEqual(records[0].date, "2026-01-01")
            self.assertEqual(records[0].draw_normalized, "Day")
            self.assertEqual(records[1].date, "2026-01-01")
            self.assertEqual(records[1].draw_normalized, "Night")
            self.assertEqual(records[-1].date, "2026-01-06")
            self.assertEqual(records[-1].draw_normalized, "Night")

    def test_load_draw_records_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_picks.csv"
            with csv_path.open("w", encoding="utf-8") as f:
                f.write("Date,Draw,Num1,Num2,Num3\n")
                for d in self.sample_draws:
                    f.write(f"{d['date']},{d['draw']},{d['num1']},{d['num2']},{d['num3']}\n")

            records = load_draw_records(csv_path)
            self.assertEqual(len(records), len(self.sample_draws))
            self.assertEqual(records[0].digits, (1, 2, 3))
            self.assertEqual(records[-1].digits, (2, 4, 6))

    def test_build_training_and_validation_datasets(self):
        records = [
            Pick3DrawRecord(
                date=d["date"],
                draw=d["draw"],
                num1=d["num1"],
                num2=d["num2"],
                num3=d["num3"],
            )
            for d in self.sample_draws
        ]

        train_data, val_data = build_training_and_validation_datasets(
            records=records,
            tokenizer=self.tokenizer,
            window_size=3,
            max_seq_len=32,
            val_split=0.2,
            target_loss_only=True,
        )

        train_x, train_y, train_mask = train_data
        self.assertGreater(len(train_x), 0)
        self.assertEqual(len(train_x), len(train_y))
        self.assertEqual(len(train_x), len(train_mask))
        self.assertIsNotNone(val_data)
        val_x, val_y, val_mask = val_data
        self.assertGreater(len(val_x), 0)

        if mx is not None:
            self.assertEqual(train_x.shape[1], 31)  # max_seq_len - 1
            self.assertEqual(train_y.shape[1], 31)
            self.assertEqual(train_mask.shape[1], 31)

    @unittest.skipIf(mx is None, "MLX is not installed")
    def test_train_and_evaluate_model(self):
        # Generate 30 synthetic draw records
        records = []
        for i in range(30):
            records.append(
                Pick3DrawRecord(
                    date=f"2026-01-{(i // 2) + 1:02d}",
                    draw="Day" if i % 2 == 0 else "Night",
                    num1=(i * 3) % 10,
                    num2=(i * 7) % 10,
                    num3=(i * 2) % 10,
                )
            )

        train_data, val_data = build_training_and_validation_datasets(
            records=records,
            tokenizer=self.tokenizer,
            window_size=4,
            max_seq_len=48,
            val_split=0.2,
        )

        config = Pick3LLMConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=48,
            n_embd=32,
            n_head=2,
            n_layer=2,
        )
        model = Pick3LLM(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "test_checkpoint"
            trained_model, history = train_pick_3_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                epochs=2,
                batch_size=8,
                lr=1e-3,
                model_dir=model_dir,
                verbose=False,
            )

            self.assertEqual(len(history.epochs), 2)
            self.assertEqual(len(history.train_loss), 2)
            self.assertEqual(len(history.val_loss), 2)
            self.assertTrue((model_dir / "config.json").exists())
            self.assertTrue((model_dir / "model.safetensors").exists())

            # Evaluate metrics function
            val_x, val_y, val_mask = val_data
            metrics = evaluate_model_metrics(trained_model, val_x, val_y, val_mask, batch_size=4)
            self.assertIn("loss", metrics)
            self.assertIn("token_accuracy", metrics)
            self.assertIn("exact_match_rate", metrics)
            self.assertGreater(metrics["loss"], 0.0)

    @unittest.skipIf(mx is None, "MLX is not installed")
    def test_run_training_pipeline_and_prediction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "picks.jsonl"
            model_dir = Path(tmp_dir) / "pick3_model"
            tickets_csv = Path(tmp_dir) / "tickets.csv"
            history_json = Path(tmp_dir) / "history.json"

            with jsonl_path.open("w", encoding="utf-8") as f:
                for i in range(25):
                    rec = {
                        "date": f"2026-02-{(i // 2) + 1:02d}",
                        "draw": "Day" if i % 2 == 0 else "Night",
                        "num1": (i * 2) % 10,
                        "num2": (i * 4) % 10,
                        "num3": (i * 6) % 10,
                    }
                    f.write(json.dumps(rec) + "\n")

            trained_model, history = run_training_pipeline(
                data_path=jsonl_path,
                model_dir=model_dir,
                epochs=2,
                batch_size=8,
                window_size=4,
                max_seq_len=48,
                val_split=0.2,
                n_embd=32,
                n_head=2,
                n_layer=2,
                predict=True,
                num_tickets=3,
                draw_type="Night",
                output_tickets_csv=tickets_csv,
                history_json=history_json,
                verbose=False,
            )

            self.assertTrue(model_dir.exists())
            self.assertTrue(tickets_csv.exists())
            self.assertTrue(history_json.exists())

    def test_cli_execution(self):
        repo_root = Path(__file__).resolve().parent.parent
        script_path = repo_root / "train_pick_3.py"
        if not script_path.exists():
            script_path = repo_root / "idaho" / "pick3" / "train_pick_3.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "test_history.jsonl"
            model_dir = Path(tmp_dir) / "cli_model"
            history_json = Path(tmp_dir) / "cli_history.json"

            with jsonl_path.open("w", encoding="utf-8") as f:
                for i in range(20):
                    rec = {
                        "date": f"2026-03-{(i // 2) + 1:02d}",
                        "draw": "Day" if i % 2 == 0 else "Night",
                        "num1": (i + 1) % 10,
                        "num2": (i + 3) % 10,
                        "num3": (i + 5) % 10,
                    }
                    f.write(json.dumps(rec) + "\n")

            cmd = [
                sys.executable,
                str(script_path),
                "--jsonl", str(jsonl_path),
                "--model-dir", str(model_dir),
                "--epochs", "1",
                "--batch-size", "8",
                "--window-size", "3",
                "--max-seq-len", "32",
                "--n-embd", "32",
                "--n-head", "2",
                "--n-layer", "2",
                "--history-json", str(history_json),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
            self.assertEqual(res.returncode, 0, f"CLI execution failed:\n{res.stderr}")
            self.assertTrue((model_dir / "config.json").exists())
            self.assertTrue(history_json.exists())


if __name__ == "__main__":
    unittest.main()
