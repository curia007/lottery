#!/usr/bin/env python3
"""
Scrape Idaho Lottery Pick 3 drawing history and save it to JSONL format.

Source:
    https://www.idaholottery.com/drawgame/history/pick-3

Output format:
    Each line in the JSONL file is a JSON object with:
    {"date": "YYYY-MM-DD", "draw": "Day|Night", "num1": int, "num2": int, "num3": int}

Usage:
    python scrape_idaho_pick3_jsonl.py
    python scrape_idaho_pick3_jsonl.py --output idaho/pick3/data/idaho_pick3_last_picks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List

URL = "https://www.idaholottery.com/drawgame/history/pick-3"
DEFAULT_OUTPUT = "idaho/pick3/data/idaho_pick3_last_picks.jsonl"


@dataclass(frozen=True)
class Pick3Draw:
    date: str
    draw: str
    num1: int
    num2: int
    num3: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "draw": self.draw,
            "num1": self.num1,
            "num2": self.num2,
            "num3": self.num3,
        }


DRAW_LINE_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<draw>Day|Night)\s+"
    r"(?P<num1>\d)\s+(?P<num2>\d)\s+(?P<num3>\d)",
    re.IGNORECASE,
)


def fetch_html(url: str = URL, timeout: int = 30) -> str:
    """Download the Idaho Lottery Pick 3 history page."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        import requests  # type: ignore

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except ImportError:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")


def parse_draws(html: str) -> List[Pick3Draw]:
    """
    Parse draw rows from the page HTML into structured Pick3Draw objects.

    The site renders text like:
        2026-05-15 Night 7 7 5
        2026-05-15 Day 6 9 5

    This parser strips noise (scripts, styles, tags) and extracts each matching draw line.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
    except ImportError:
        # Fallback without BeautifulSoup
        cleaned = re.sub(
            r"<(script|style|noscript)[^>]*>.*?</\1>",
            "",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"<[^>]+>", "\n", cleaned)

    draws: List[Pick3Draw] = []
    seen: set[tuple[str, str, int, int, int]] = set()

    for match in DRAW_LINE_RE.finditer(text):
        row = Pick3Draw(
            date=match.group("date"),
            draw=match.group("draw").capitalize(),
            num1=int(match.group("num1")),
            num2=int(match.group("num2")),
            num3=int(match.group("num3")),
        )
        key = (row.date, row.draw, row.num1, row.num2, row.num3)
        if key not in seen:
            seen.add(key)
            draws.append(row)

    # Sort newest first, Night before Day when dates are equal.
    draws.sort(
        key=lambda r: (
            datetime.strptime(r.date, "%Y-%m-%d"),
            1 if r.draw.lower() == "night" else 0,
        ),
        reverse=True,
    )

    return draws


def write_jsonl(draws: Iterable[Pick3Draw], output_path: str | Path) -> None:
    """Write draw rows to a JSONL file (one JSON object per line)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in draws:
            f.write(json.dumps(row.to_dict()) + "\n")


def resolve_default_output() -> Path:
    """Find appropriate data directory based on execution context."""
    # When executed from within idaho/pick3 directory
    if Path("data").is_dir() and Path("scrape_idaho_pick3.py").exists():
        return Path("data") / "idaho_pick3_last_picks.jsonl"
    # When executed from repository root
    if Path("idaho/pick3/data").is_dir():
        return Path("idaho/pick3/data") / "idaho_pick3_last_picks.jsonl"
    script_dir = Path(__file__).resolve().parent
    return script_dir / "data" / "idaho_pick3_last_picks.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Idaho Lottery Pick 3 history and output as JSONL."
    )
    parser.add_argument(
        "--url",
        default=URL,
        help=f"Page URL to scrape. Default: {URL}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"JSONL output file. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else resolve_default_output()

    html = fetch_html(args.url)
    draws = parse_draws(html)

    if not draws:
        raise RuntimeError(
            "No Pick 3 draw rows were found. The page structure may have changed."
        )

    write_jsonl(draws, output_path)
    print(f"Saved {len(draws):,} Pick 3 draw rows to {output_path}")


if __name__ == "__main__":
    main()
