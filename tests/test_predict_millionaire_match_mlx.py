import io
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from millionaire_life.patterns.predict_millionaire_match_mlx import (
    load_millionaire_history,
    build_dataset,
    train_mlx_model,
    predict_next_draw,
    evaluate_historical_accuracy,
    print_prediction_report,
    export_tickets_to_csv,
    MAIN_MIN,
    MAIN_MAX,
    EXTRA_MIN,
    EXTRA_MAX,
)


class TestPredictMillionaireMatchMLX(unittest.TestCase):
    def setUp(self):
        # Create synthetic dataset with 30 historical draws
        rows = []
        for i in range(30):
            rows.append({
                "Date": f"2026-08-{i+1:02d}",
                "Num1": 1 + (i % 5),
                "Num2": 10 + (i % 8),
                "Num3": 20 + (i % 10),
                "Num4": 35 + (i % 10),
                "Num5": 48 + (i % 10),
                "Extra": (i % 5) + 1,
            })
        self.sample_df = pd.DataFrame(rows)

    def test_pipeline_with_synthetic_data(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            self.sample_df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            main_hist, extra_hist, df = load_millionaire_history(csv_path)
            self.assertEqual(len(main_hist), 30)
            self.assertEqual(len(extra_hist), 30)
            self.assertEqual(main_hist.shape[1], 5)

            # Test feature extraction & dataset building
            window = 10
            x_train, y_main, y_extra = build_dataset(main_hist, extra_hist, window=window)
            self.assertEqual(len(x_train), 30 - window)
            self.assertEqual(y_main.shape[1], MAIN_MAX)
            self.assertEqual(y_extra.shape[1], EXTRA_MAX)

            # Test MLX Model training
            model = train_mlx_model(x_train, y_main, y_extra, epochs=15, lr=0.01, verbose=False)
            self.assertIsNotNone(model)

            # Test prediction generation
            res = predict_next_draw(model, main_hist, extra_hist, window=window, num_tickets=5, seed=42)
            self.assertEqual(len(res.tickets), 5)
            self.assertEqual(len(res.top_extras), EXTRA_MAX)
            self.assertTrue(1 <= res.predicted_extra <= EXTRA_MAX)
            self.assertEqual(len(res.top_white_balls), 10)

            for t in res.tickets:
                self.assertEqual(len(t.main_numbers), 5)
                self.assertTrue(all(MAIN_MIN <= n <= MAIN_MAX for n in t.main_numbers))
                self.assertTrue(EXTRA_MIN <= t.extra <= EXTRA_MAX)
                self.assertGreater(t.match_score, 0)
                self.assertGreater(t.white_ball_match_prob, 0)
                self.assertGreater(t.extra_ball_match_prob, 0)
                self.assertGreater(t.joint_target_prob, 0)

            # Test report printing
            captured = io.StringIO()
            sys.stdout = captured
            try:
                print_prediction_report(res, total_draws=30)
            finally:
                sys.stdout = sys.__stdout__
            report_str = captured.getvalue()
            self.assertIn("MILLIONAIRE FOR LIFE - MLX NEXT DRAW PREDICTOR", report_str)
            self.assertIn("Target Goal: Matched Millionaire Ball + At Least 1 White Ball", report_str)

            # Test CSV export
            with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as out_f:
                out_path = out_f.name
            try:
                export_tickets_to_csv(res.tickets, out_path)
                out_df = pd.read_csv(out_path)
                self.assertEqual(len(out_df), 5)
                self.assertIn("Millionaire_Ball", out_df.columns)
                self.assertIn("Target_Match_Prob", out_df.columns)
            finally:
                Path(out_path).unlink(missing_ok=True)

            # Test historical evaluation
            bt = evaluate_historical_accuracy(
                main_hist, extra_hist, window=10, eval_draws=5, num_tickets=5, epochs=5
            )
            self.assertIn("matched_target_rate", bt)
            self.assertIn("eval_draws", bt)
            self.assertEqual(bt["eval_draws"], 5)

        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_real_dataset_prediction(self):
        csv_path = Path("millionaire_life/data/millionaire_life_history.csv")
        if csv_path.exists():
            main_hist, extra_hist, df = load_millionaire_history(csv_path)
            self.assertGreater(len(main_hist), 100)
            x_train, y_main, y_extra = build_dataset(main_hist, extra_hist, window=20)
            model = train_mlx_model(x_train, y_main, y_extra, epochs=10, lr=0.01)
            res = predict_next_draw(model, main_hist, extra_hist, window=20, num_tickets=3, seed=123)
            self.assertEqual(len(res.tickets), 3)
            for t in res.tickets:
                self.assertEqual(len(t.main_numbers), 5)
                self.assertTrue(1 <= t.extra <= 5)


if __name__ == "__main__":
    unittest.main()
