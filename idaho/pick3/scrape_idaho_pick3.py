#!/usr/bin/env python3
"""
Scrape Idaho Lottery Pick 3 drawing history and save it to CSV.

Source:
    https://www.idaholottery.com/drawgame/history/pick-3

Output columns:
    Date, Draw, Num1, Num2, Num3

Usage:
    python scrape_idaho_pick3.py
    python scrape_idaho_pick3.py --output idaho_pick3_history.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup

URL = "https://www.idaholottery.com/drawgame/history/pick-3"
DEFAULT_OUTPUT = "data/idaho_pick3_history.csv"


@dataclass(frozen=True)
class Pick3Draw:
    date: str
    draw: str
    num1: int
    num2: int
    num3: int


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

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_draws(html: str) -> List[Pick3Draw]:
    """
    Parse draw rows from the page.

    The site renders text like:
        2026-05-15 Night 7 7 5
        2026-05-15 Day 6 9 5

    This parser reads all visible text, then extracts each matching draw line.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style noise before extracting visible text.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)

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


def write_csv(draws: Iterable[Pick3Draw], output_path: str | Path) -> None:
    """Write draw rows to CSV."""
    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Draw", "Num1", "Num2", "Num3"])
        for row in draws:
            writer.writerow([row.date, row.draw, row.num1, row.num2, row.num3])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Idaho Lottery Pick 3 history to CSV."
    )
    parser.add_argument(
        "--url",
        default=URL,
        help=f"Page URL to scrape. Default: {URL}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"CSV output file. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    html = fetch_html(args.url)
    draws = parse_draws(html)

    if not draws:
        raise RuntimeError(
            "No Pick 3 draw rows were found. The page structure may have changed."
        )

    write_csv(draws, args.output)
    print(f"Saved {len(draws):,} Pick 3 draw rows to {args.output}")


if __name__ == "__main__":
    main()
