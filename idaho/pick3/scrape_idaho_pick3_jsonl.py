#!/usr/bin/env python3
"""
scrape_idaho_pick3_jsonl.py

Scrapes historical Idaho Lottery Pick 3 draws from the official website
and outputs them as a JSONL file sorted chronologically (newest first).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

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


def fetch_html(url: str = URL, timeout: int = 30) -> str:
    """Download the Idaho Lottery Pick 3 history page with urllib fallback."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    if requests is not None:
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception:
            pass

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_draws(html: str) -> List[Pick3Draw]:
    """
    Parse Pick 3 draw records from HTML content.
    
    Returns deduplicated draws sorted chronologically (newest first).
    """
    text = html
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
        except Exception:
            text = html
    else:
        # Strip script, style, and HTML tags using regex fallback
        clean_text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", clean_text)

    # Match lines like: 2026-11-06 Night 4 2 8
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})\s+(Day|Night)\s+(\d)\s+(\d)\s+(\d)", re.IGNORECASE)
    
    draws = []
    seen = set()
    for match in pattern.finditer(text):
        date, draw_type, d1, d2, d3 = match.groups()
        draw_type_cap = draw_type.capitalize()
        key = (date, draw_type_cap, int(d1), int(d2), int(d3))
        if key not in seen:
            seen.add(key)
            draws.append(Pick3Draw(
                date=date,
                draw=draw_type_cap,
                num1=int(d1),
                num2=int(d2),
                num3=int(d3),
            ))
    
    # Sort newest first, Night before Day when dates are equal.
    draws.sort(
        key=lambda r: (
            datetime.strptime(r.date, "%Y-%m-%d"),
            1 if r.draw.lower() == "night" else 0,
        ),
        reverse=True,
    )
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
