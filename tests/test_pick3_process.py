import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from idaho.pick3.hugging_face.pick3_process import (
    _load_prediction_generator,
    compile_and_export_report,
    parse_drawing_data,
    resolve_model_source,
)


class TestPick3Process(unittest.TestCase):
    def setUp(self):
        self.drawings = [[i % 10, (i + 1) % 10, (i + 2) % 10] for i in range(10)]

    def test_parse_ten_labeled_text_draws(self):
        text = "\n".join(
            f"Drawing {index + 1}: {drawing[0]}-{drawing[1]}-{drawing[2]}"
            for index, drawing in enumerate(self.drawings)
        )

        self.assertEqual(parse_drawing_data(text), self.drawings)

    def test_parse_dated_csv_in_chronological_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "draws.csv"
            rows = [
                "Date,Draw,Num1,Num2,Num3",
                *(
                    f"2026-01-{day:02d},{'Day' if day % 2 else 'Night'},"
                    f"{drawing[0]},{drawing[1]},{drawing[2]}"
                    for day, drawing in reversed(list(enumerate(self.drawings, start=1)))
                ),
            ]
            csv_path.write_text("\n".join(rows), encoding="utf-8")

            parsed = parse_drawing_data(csv_path)
            self.assertEqual(parsed, self.drawings)
            self.assertEqual(parsed.draw_types, ["Day" if day % 2 else "Night" for day in range(1, 11)])

    def test_local_checkpoint_precedes_default_hub_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "local_model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").touch()

            source_type, source = resolve_model_source(
                model_override="curia007/idaho-pick3-sequence-transformer",
                model_dirs=[model_dir],
            )

        self.assertEqual(source_type, "local")
        self.assertEqual(source, model_dir)

    def test_missing_local_checkpoint_does_not_select_default_hub_model(self):
        source_type, source = resolve_model_source(model_override="", model_dirs=[])

        self.assertEqual(source_type, "none")
        self.assertIsNone(source)

    def test_missing_local_checkpoint_without_override_fails_instead_of_mocking(self):
        with patch("idaho.pick3.hugging_face.pick3_process.DEFAULT_MODEL_DIRS", ()):
            with patch.dict("os.environ", {"PICK3_HF_MODEL": ""}):
                with self.assertRaisesRegex(FileNotFoundError, "PICK3_HF_MODEL"):
                    _load_prediction_generator()

    def test_report_uses_injected_predictor_and_temporary_export_dir(self):
        def predictor(history, temperature, top_k, top_p):
            self.assertEqual(history, self.drawings)
            self.assertEqual((temperature, top_k, top_p), (0.8, 5, 0.9))
            return np.zeros((3, 10)), [1, 2, 3]

        with tempfile.TemporaryDirectory() as tmp_dir:
            report, probabilities, export_path = compile_and_export_report(
                self.drawings,
                temperature=0.8,
                top_k=5,
                top_p=0.9,
                prediction_generator=predictor,
                export_dir=tmp_dir,
            )

            self.assertIn("[1-2-3]", report)
            self.assertEqual(len(probabilities), 10)
            self.assertTrue(Path(export_path).is_file())
            self.assertEqual(Path(export_path).parent, Path(tmp_dir))


if __name__ == "__main__":
    unittest.main()