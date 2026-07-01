#!/usr/bin/env python3
"""
scrape_mega_millions.py

Scrapes Mega Millions draw history from:
https://www.idaholottery.com/drawgame/history/mega-millions

Outputs a CSV file with:

    Date,Num1,Num2,Num3,Num4,Num5,MegaBall

Install:
    pip install requests beautifulsoup4

Run:
    python scrape_mega_millions.py

Optional:
    python scrape_mega_millions.py --output mega_millions_history.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup


URL = "https://www.idaholottery.com/drawgame/history/mega-millions"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "mega_millions_history.csv"
MAIN_MIN = 1
MAIN_MAX = 70
MEGA_MIN = 1
MEGA_MAX = 24


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.0 Safari/605.1.15"
        )
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def parse_date(text: str) -> str | None:
    text = " ".join(str(text).split())

    patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4}\b",
    ]

    formats = [
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y-%m-%d",
        "%b %d, %Y",
        "%B %d, %Y",
        "%b. %d, %Y",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        raw_date = match.group(0).replace("Sept.", "Sep.").replace(".", "")

        for fmt in formats:
            clean_fmt = fmt.replace(".", "")
            try:
                return datetime.strptime(raw_date, clean_fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass

    return None


def remove_dates(text: str) -> str:
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", " ", text)
    text = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", " ", text)
    text = re.sub(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4}\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return text


def extract_numbers(text: str) -> list[int]:
    return [int(token) for token in re.findall(r"\b\d{1,2}\b", str(text))]


def valid_main_numbers(nums: Iterable[int]) -> list[int]:
    return [int(n) for n in nums if MAIN_MIN <= int(n) <= MAIN_MAX]


def valid_mega_numbers(nums: Iterable[int]) -> list[int]:
    return [int(n) for n in nums if MEGA_MIN <= int(n) <= MEGA_MAX]


def make_row(date: str, nums: list[int]) -> dict[str, int | str] | None:
    """
    Mega Millions uses 5 main numbers from 1-70 and 1 Mega Ball from 1-24.
    """
    if len(nums) < 6:
        return None

    main = valid_main_numbers(nums[:5])
    mega_candidates = valid_mega_numbers(nums[5:])

    if len(main) != 5 or not mega_candidates:
        return None

    mega_ball = mega_candidates[0]

    return {
        "Date": date,
        "Num1": main[0],
        "Num2": main[1],
        "Num3": main[2],
        "Num4": main[3],
        "Num5": main[4],
        "MegaBall": mega_ball,
    }


def parse_row(text: str) -> dict[str, int | str] | None:
    date = parse_date(text)
    if not date:
        return None

    cleaned = remove_dates(text).replace("MB:", " ")
    nums = extract_numbers(cleaned)

    return make_row(date, nums)


def parse_table_rows(soup: BeautifulSoup) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [
                " ".join(cell.get_text(" ", strip=True).split())
                for cell in tr.find_all(["td", "th"])
            ]

            if not cells:
                continue

            row = parse_row(" ".join(cells))
            if row:
                rows.append(row)

    return rows


def parse_page_text_fallback(soup: BeautifulSoup) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = [line.strip() for line in soup.get_text("\n").splitlines()]
    lines = [line for line in lines if line]

    for line in lines:
        row = parse_row(line)
        if row:
            rows.append(row)

    return rows


def deduplicate_rows(rows: Iterable[dict[str, int | str]]) -> list[dict[str, int | str]]:
    seen = set()
    clean_rows: list[dict[str, int | str]] = []

    for row in rows:
        key = (
            row["Date"],
            row["Num1"],
            row["Num2"],
            row["Num3"],
            row["Num4"],
            row["Num5"],
            row["MegaBall"],
        )

        if key in seen:
            continue

        seen.add(key)
        clean_rows.append(row)

    clean_rows.sort(key=lambda r: str(r["Date"]), reverse=True)
    return clean_rows


def write_csv(rows: list[dict[str, int | str]], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Date", "Num1", "Num2", "Num3", "Num4", "Num5", "MegaBall"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scrape_mega_millions(url: str = URL) -> list[dict[str, int | str]]:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    rows = parse_table_rows(soup)

    if not rows:
        rows = parse_page_text_fallback(soup)

    rows = deduplicate_rows(rows)

    if not rows:
        raise RuntimeError(
            "No Mega Millions draw rows were found. "
            "The page may have changed or may require JavaScript rendering."
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Mega Millions lottery history to CSV.")
    parser.add_argument("--url", default=URL, help="Mega Millions history URL")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"CSV output path. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    rows = scrape_mega_millions(args.url)
    write_csv(rows, args.output)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
