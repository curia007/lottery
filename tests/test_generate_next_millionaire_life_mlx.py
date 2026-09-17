import io
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from millionaire_life.patterns.generate_next_millionaire_life_mlx import (
    load_data,
    make_dataset,
    train_model,
    compute_model_probabilities,
    compute_overdue_scores,
    generate_winning_tickets,
    print_winning_tickets_report,
    export_tickets_to_csv,
    MAIN_MIN,
    MAIN_MAX,
    MAIN_PICK,
    EXTRA_MIN,
    EXTRA_MAX,
    RankedTicket,
)


class TestGenerateNextMillionaireLifeMLX(unittest.TestCase):
    def setUp(self):
        # Create synthetic dataset with 25 draws
        rows = []
        for i in range(25):
            # Ensure 5 distinct numbers in range 1..58
            nums = sorted(list({(1 + (i + j * 7) % 58) for j in range(5)}))
            while len(nums) < 5:
                next_val = ((nums[-1] + 1) % 58) + 1
                if next_val not in nums:
                    nums.append(next_val)
                nums.sort()

            rows.append({
                "Date": f"2026-08-{i+1:02d}",
                "Num1": nums[0],
                "Num2": nums[1],
                "Num3": nums[2],
                "Num4": nums[3],
                "Num5": nums[4],
                "Extra": (i % 5) + 1,
            })
        self.sample_df = pd.DataFrame(rows)

    def test_pipeline_and_no_duplicates(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            self.sample_df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            main_hist, extra_hist, counts_dict, df = load_data(csv_path)
            self.assertEqual(len(main_hist), 25)
            self.assertEqual(len(extra_hist), 25)
            self.assertEqual(main_hist.shape[1], 5)

            # Test dataset generation
            window = 10
            x_train, y_main, y_extra = make_dataset(main_hist, extra_hist, window=window)
            self.assertEqual(len(x_train), 25 - window)
            self.assertEqual(y_main.shape[1], MAIN_MAX)
            self.assertEqual(y_extra.shape[1], EXTRA_MAX)

            # Test MLX Model training
            model = train_model(x_train, y_main, y_extra, epochs=10, lr=0.01, verbose=False)
            self.assertIsNotNone(model)

            # Probabilities & Statistics
            main_probs, extra_probs = compute_model_probabilities(
                model, main_hist, extra_hist, window=window
            )
            self.assertEqual(len(main_probs), MAIN_MAX)
            self.assertEqual(len(extra_probs), EXTRA_MAX)

            overdue_scores = compute_overdue_scores(main_hist)
            self.assertEqual(len(overdue_scores), MAIN_MAX)

            # Test ticket generation for all strategies
            for strat in ["model", "balanced", "hot", "overdue", "mixed"]:
                tickets = generate_winning_tickets(
                    model_probs=main_probs,
                    extra_probs=extra_probs,
                    counts_dict=counts_dict,
                    overdue_scores=overdue_scores,
                    main_history=main_hist,
                    num_tickets=5,
                    ticket_type=strat,
                    seed=42,
                )
                self.assertEqual(len(tickets), 5)

                seen_ticket_combos = set()
                for t in tickets:
                    # STRICT DUPLICATE NUMBER CHECKS:
                    # 1. Ticket must contain exactly 5 white ball numbers
                    self.assertEqual(len(t.main_numbers), MAIN_PICK)
                    # 2. No duplicate numbers within the ticket
                    self.assertEqual(len(set(t.main_numbers)), MAIN_PICK, f"Duplicate found in ticket: {t.main_numbers}")
                    # 3. All numbers in valid range
                    self.assertTrue(all(MAIN_MIN <= n <= MAIN_MAX for n in t.main_numbers))
                    # 4. Extra ball in valid range
                    self.assertTrue(EXTRA_MIN <= t.extra <= EXTRA_MAX)

                    combo_tuple = tuple(sorted(t.main_numbers))
                    self.assertNotIn(combo_tuple, seen_ticket_combos, "Duplicate ticket generated in ticket batch!")
                    seen_ticket_combos.add(combo_tuple)

            # Test Report Printing
            captured = io.StringIO()
            sys.stdout = captured
            try:
                top_extras = [(e + 1, float(extra_probs[e])) for e in range(5)]
                top_white = [(b + 1, float(main_probs[b])) for b in range(10)]
                print_winning_tickets_report(tickets, top_extras, top_white, 25)
            finally:
                sys.stdout = sys.__stdout__
            report_str = captured.getvalue()
            self.assertIn("MILLIONAIRE FOR LIFE - MLX PREDICTED WINNING TICKETS", report_str)
            self.assertIn("NO DUPLICATE NUMBERS", report_str)
            self.assertIn("Verification: All generated tickets verified with 5 unique white balls.", report_str)

            # Test CSV Export
            with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as out_f:
                out_path = out_f.name
            try:
                export_tickets_to_csv(tickets, out_path)
                out_df = pd.read_csv(out_path)
                self.assertEqual(len(out_df), 5)
                self.assertIn("Num1", out_df.columns)
                self.assertIn("Extra", out_df.columns)
                self.assertIn("Ticket", out_df.columns)
                # Verify CSV contents have no duplicates per row
                for _, row in out_df.iterrows():
                    nums = [row["Num1"], row["Num2"], row["Num3"], row["Num4"], row["Num5"]]
                    self.assertEqual(len(set(nums)), 5)
            finally:
                Path(out_path).unlink(missing_ok=True)

        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_duplicate_rejection_in_dataclass(self):
        # Verify RankedTicket rejects duplicates upon instantiation
        with self.assertRaises(ValueError):
            RankedTicket(
                rank=1,
                ticket_str="01 01 03 04 05 [MB: 1]",
                main_numbers=[1, 1, 3, 4, 5],  # Duplicate '1'
                extra=1,
                score=1.0,
                model_score=1.0,
                frequency_score=1.0,
                overdue_score=1.0,
                balance_score=1.0,
                strategy="Model",
            )

    def test_real_history_execution(self):
        csv_path = Path("millionaire_life/data/millionaire_life_history.csv")
        if csv_path.exists():
            main_hist, extra_hist, counts_dict, df = load_data(csv_path)
            x_train, y_main, y_extra = make_dataset(main_hist, extra_hist, window=20)
            model = train_model(x_train, y_main, y_extra, epochs=5, lr=0.01)
            main_probs, extra_probs = compute_model_probabilities(model, main_hist, extra_hist, window=20)
            overdue_scores = compute_overdue_scores(main_hist)
            tickets = generate_winning_tickets(
                model_probs=main_probs,
                extra_probs=extra_probs,
                counts_dict=counts_dict,
                overdue_scores=overdue_scores,
                main_history=main_hist,
                num_tickets=10,
                ticket_type="mixed",
                seed=999,
            )
            self.assertEqual(len(tickets), 10)
            for t in tickets:
                self.assertEqual(len(t.main_numbers), 5)
                self.assertEqual(len(set(t.main_numbers)), 5)
                self.assertTrue(1 <= t.extra <= 5)


if __name__ == "__main__":
    unittest.main()
