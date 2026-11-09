import json
import tempfile
import unittest
from pathlib import Path

from idaho.pick3.scrape_idaho_pick3_jsonl import (
    Pick3Draw,
    parse_draws,
    resolve_default_output,
    write_jsonl,
)


class TestScrapeIdahoPick3Jsonl(unittest.TestCase):
    def test_parse_draws_from_html(self):
        sample_html = """
        <html>
            <head><title>History</title></head>
            <body>
                <script>var x = 123;</script>
                <div>
                    2026-11-06 Night 4 2 8
                    2026-11-06 Day 6 9 0
                    2026-11-05 Night 7 2 5
                    2026-11-05 Day 9 1 0
                </div>
            </body>
        </html>
        """
        draws = parse_draws(sample_html)
        self.assertEqual(len(draws), 4)

        self.assertEqual(
            draws[0],
            Pick3Draw(date="2026-11-06", draw="Night", num1=4, num2=2, num3=8),
        )
        self.assertEqual(
            draws[1],
            Pick3Draw(date="2026-11-06", draw="Day", num1=6, num2=9, num3=0),
        )
        self.assertEqual(
            draws[2],
            Pick3Draw(date="2026-11-05", draw="Night", num1=7, num2=2, num3=5),
        )
        self.assertEqual(
            draws[3],
            Pick3Draw(date="2026-11-05", draw="Day", num1=9, num2=1, num3=0),
        )

    def test_parse_draws_deduplication_and_sorting(self):
        sample_html = """
        2026-10-01 Day 1 2 3
        2026-10-02 Night 9 8 7
        2026-10-02 Day 4 5 6
        2026-10-01 Day 1 2 3
        """
        draws = parse_draws(sample_html)
        self.assertEqual(len(draws), 3)

        self.assertEqual(draws[0].date, "2026-10-02")
        self.assertEqual(draws[0].draw, "Night")
        self.assertEqual((draws[0].num1, draws[0].num2, draws[0].num3), (9, 8, 7))

        self.assertEqual(draws[1].date, "2026-10-02")
        self.assertEqual(draws[1].draw, "Day")
        self.assertEqual((draws[1].num1, draws[1].num2, draws[1].num3), (4, 5, 6))

        self.assertEqual(draws[2].date, "2026-10-01")
        self.assertEqual(draws[2].draw, "Day")
        self.assertEqual((draws[2].num1, draws[2].num2, draws[2].num3), (1, 2, 3))

    def test_write_jsonl(self):
        draws = [
            Pick3Draw(date="2026-11-06", draw="Night", num1=4, num2=2, num3=8),
            Pick3Draw(date="2026-11-06", draw="Day", num1=6, num2=9, num3=0),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "sub" / "output.jsonl"
            write_jsonl(draws, out_file)

            self.assertTrue(out_file.is_file())
            lines = out_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

            data0 = json.loads(lines[0])
            self.assertEqual(data0, {"date": "2026-11-06", "draw": "Night", "num1": 4, "num2": 2, "num3": 8})

            data1 = json.loads(lines[1])
            self.assertEqual(data1, {"date": "2026-11-06", "draw": "Day", "num1": 6, "num2": 9, "num3": 0})

    def test_resolve_default_output(self):
        output = resolve_default_output()
        self.assertEqual(output.name, "idaho_pick3_last_picks.jsonl")
        self.assertEqual(output.parent.name, "data")


if __name__ == "__main__":
    unittest.main()
