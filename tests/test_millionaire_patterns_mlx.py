import tempfile
import unittest
from pathlib import Path
import pandas as pd

from millionaire_life.patterns.millionaire_patterns_mlx import (
    load_millionaire_history,
    analyze_patterns_mlx,
    generate_pattern_tickets,
    generate_pattern_counts_dataframe,
    print_pattern_report,
    MAIN_MAX,
    EXTRA_MAX,
)


class TestMillionairePatternsMLX(unittest.TestCase):
    def setUp(self):
        # Create synthetic draws for testing
        rows = []
        for i in range(25):
            rows.append({
                "Date": f"2026-07-{i+1:02d}",
                "Num1": 2 + (i % 5),
                "Num2": 3 + (i % 5),  # Consecutive with Num1 for run tests
                "Num3": 15 + (i % 10),
                "Num4": 30 + (i % 10),
                "Num5": 45 + (i % 10),
                "Extra": (i % 5) + 1,
            })
        self.sample_df = pd.DataFrame(rows)

    def test_load_and_analyze_patterns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            self.sample_df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            main_data, extra_data, df = load_millionaire_history(csv_path)
            self.assertEqual(len(main_data), 25)
            self.assertEqual(len(extra_data), 25)

            report = analyze_patterns_mlx(main_data, extra_data)
            self.assertEqual(report.total_draws, 25)
            self.assertEqual(len(report.main_counts), MAIN_MAX)
            self.assertEqual(len(report.extra_counts), EXTRA_MAX)

            # Test consecutive runs detection
            self.assertGreater(report.draws_with_consecutive_runs, 0)
            self.assertGreater(report.consecutive_runs_count[2], 0)

            # Test Odd/Even and High/Low
            self.assertGreater(len(report.odd_even_ratios), 0)
            self.assertGreater(len(report.high_low_ratios), 0)

            # Test Co-occurrence matrix
            self.assertEqual(report.cooccurrence_matrix.shape, (MAIN_MAX, MAIN_MAX))
            self.assertGreater(len(report.top_pairs), 0)

            # Test dataframe generation
            counts_df = generate_pattern_counts_dataframe(report)
            self.assertEqual(len(counts_df), MAIN_MAX)
            self.assertIn("Number", counts_df.columns)
            self.assertIn("Count", counts_df.columns)
            self.assertIn("Consecutive_Repeats", counts_df.columns)

            # Test ticket generation
            tickets = generate_pattern_tickets(report, num_tickets=3, seed=42)
            self.assertEqual(len(tickets), 3)
            for nums, extra, score, desc in tickets:
                self.assertEqual(len(nums), 5)
                self.assertTrue(1 <= extra <= EXTRA_MAX)
                self.assertGreater(score, 0)

            # Test print report output
            import io
            import sys
            captured_out = io.StringIO()
            sys.stdout = captured_out
            try:
                print_pattern_report(report)
            finally:
                sys.stdout = sys.__stdout__
            output_str = captured_out.getvalue()
            self.assertIn("[2] MILLIONAIRE BALL (EXTRA 1 - 5) PATTERN ANALYSIS:", output_str)
        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_real_dataset(self):
        real_csv = Path("millionaire_life/data/millionaire_life_history.csv")
        if real_csv.exists():
            main_data, extra_data, df = load_millionaire_history(real_csv)
            report = analyze_patterns_mlx(main_data, extra_data)
            self.assertGreater(report.total_draws, 100)
            self.assertEqual(len(report.main_counts), MAIN_MAX)
            self.assertGreater(report.sum_mean, 50)


if __name__ == "__main__":
    unittest.main()
