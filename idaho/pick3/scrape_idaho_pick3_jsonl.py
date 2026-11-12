#!/usr/bin/env python3
"""
scrape_idaho_pick3_jsonl.py

Scrapes historical Idaho Lottery Pick 3 draws from the official website
and outputs them as a JSONL file sorted chronologically (newest first).
"""

from __future__ import annotations

import json
import re
import sys
import argparse
from collections import OrderedDict
from pathlib import Path
from typing import List, NamedTuple
from urllib.request import urlopen, Request

# Ensure project root is on sys.path for package imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --- Constants ---
URL = "https://www.idaholottery.com/drawgame/history/pick-3"
DEFAULT_OUTPUT = _SCRIPT_DIR / "data" / "idaho_pick3_last_picks.jsonl"


class Pick3Draw(NamedTuple):
    date: str
    draw: str
    num1: int
    num2: int
    num3: int


def resolve_default_output() -> Path:
    """Resolve the default output path, ensuring the parent directory exists."""
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT


def fetch_html(url: str = URL) -> str:
    """Fetch HTML content from the Idaho Lottery Pick 3 history page."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")


def parse_draws(html: str) -> List[Pick3Draw]:
    """
    Parse Pick 3 draw records from HTML content.
    
    Returns deduplicated draws sorted chronologically (newest first).
    """
    # Match lines like: 2026-11-06 Night 4 2 8
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})\s+(Day|Night)\s+(\d)\s+(\d)\s+(\d)", re.IGNORECASE)
    
    draws = []
    seen = set()
    for match in pattern.finditer(html):
        date, draw_type, d1, d2, d3 = match.groups()
        key = (date, draw_type.capitalize())
        if key not in seen:
            seen.add(key)
            draws.append(Pick3Draw(
                date=date,
                draw=draw_type.capitalize(),
                num1=int(d1),
                num2=int(d2),
                num3=int(d3),
            ))
    
    # Sort by date descending (newest first), then Night before Day within same date
    # Night=0, Day=1 so Night comes first when sorting ascending within a date
    # We want newest dates first, so reverse the date order
    draws.sort(key=lambda d: (d.date, 0 if d.draw == "Night" else 1))
    # Now sort by date descending while preserving Night-before-Day within each date
    from itertools import groupby
    grouped = []
    for date, group in groupby(draws, key=lambda d: d.date):
        grouped.append(list(group))
    grouped.sort(key=lambda g: g[0].date, reverse=True)
    draws = [d for group in grouped for d in group]
    return draws


def write_jsonl(draws: List[Pick3Draw], output_path: Path) -> None:
    """Write draw records to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for draw in draws:
            record = OrderedDict([
                ("date", draw.date),
                ("draw", draw.draw),
                ("num1", draw.num1),
                ("num2", draw.num2),
                ("num3", draw.num3),
            ])
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Idaho Lottery Pick 3 historical draws and save as JSONL"
    )
    parser.add_argument(
        "--url", type=str, default=URL, help="URL to scrape (default: official Idaho Lottery)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSONL file path"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else resolve_default_output()
    
    print(f"Fetching Pick 3 history from {args.url}...")
    html = fetch_html(args.url)
    
    print("Parsing draws...")
    draws = parse_draws(html)
    print(f"Found {len(draws)} unique draws.")
    
    print(f"Writing JSONL to {output_path}...")
    write_jsonl(draws, output_path)
    print(f"Done. Saved {len(draws)} draws to {output_path}")


if __name__ == "__main__":
    main()
