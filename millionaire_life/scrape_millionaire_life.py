#!/usr/bin/env python3
"""
scrape_millionaire_life.py

Scrapes Millionaire for Life draw history from:
https://www.idaholottery.com/drawgame/history/millionaire-life

Outputs a CSV file with:

    Date,Num1,Num2,Num3,Num4,Num5,Extra,WinningNumbers

Notes:
    - The scraper is intentionally flexible because Idaho Lottery pages can
      change layout or labels.
    - If the game returns fewer than 6 numbers, unused fields are left blank.
    - WinningNumbers preserves the parsed numbers as a space-delimited string.

Install:
    pip install requests beautifulsoup4

Run:
    python scrape_millionaire_life.py

Optional:
    python scrape_millionaire_life.py --output millionaire_life_history.csv
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


URL = "https://www.idaholottery.com/drawgame/history/millionaire-life"
DEFAULT_OUTPUT = "data/millionaire_life_history.csv"


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
    """
    Extract numeric tokens likely to be lottery numbers.

    This keeps 1-2 digit numbers and ignores obvious large values such as years.
    """
    numbers = []

    for token in re.findall(r"\b\d{1,2}\b", str(text)):
        value = int(token)
        if 0 <= value <= 99:
            numbers.append(value)

    return numbers


def make_row(date: str, nums: list[int]) -> dict[str, int | str | None] | None:
    if not nums:
        return None

    nums = nums[:6]

    padded: list[int | None] = nums + [None] * (6 - len(nums))
    winning_numbers = " ".join(str(n) for n in nums)

    return {
        "Date": date,
        "Num1": padded[0],
        "Num2": padded[1],
        "Num3": padded[2],
        "Num4": padded[3],
        "Num5": padded[4],
        "Extra": padded[5],
        "WinningNumbers": winning_numbers,
    }


def parse_table_rows(soup: BeautifulSoup) -> list[dict[str, int | str | None]]:
    rows: list[dict[str, int | str | None]] = []

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [
                " ".join(cell.get_text(" ", strip=True).split())
                for cell in tr.find_all(["td", "th"])
            ]

            if not cells:
                continue

            row_text = " ".join(cells)
            date = parse_date(row_text)

            if not date:
                continue

            non_date_cells = [cell for cell in cells if not parse_date(cell)]
            candidate_text = " ".join(non_date_cells)

            nums = extract_numbers(candidate_text)

            if not nums:
                nums = extract_numbers(remove_dates(row_text))

            row = make_row(date, nums)

            if row:
                rows.append(row)

    return rows


def parse_page_text_fallback(soup: BeautifulSoup) -> list[dict[str, int | str | None]]:
    rows: list[dict[str, int | str | None]] = []

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = [line.strip() for line in soup.get_text("\n").splitlines()]
    lines = [line for line in lines if line]

    for idx, line in enumerate(lines):
        date = parse_date(line)

        if not date:
            continue

        nearby = " ".join(lines[idx : idx + 10])
        nearby = remove_dates(nearby)
        nums = extract_numbers(nearby)

        row = make_row(date, nums)

        if row:
            rows.append(row)

    return rows


def deduplicate_rows(rows: Iterable[dict[str, int | str | None]]) -> list[dict[str, int | str | None]]:
    seen = set()
    clean_rows = []

    for row in rows:
        key = (
            row["Date"],
            row["Num1"],
            row["Num2"],
            row["Num3"],
            row["Num4"],
            row["Num5"],
            row["Extra"],
            row["WinningNumbers"],
        )

        if key in seen:
            continue

        seen.add(key)
        clean_rows.append(row)

    clean_rows.sort(key=lambda r: str(r["Date"]), reverse=True)
    return clean_rows


def write_csv(rows: list[dict[str, int | str | None]], output_file: str | Path) -> None:
    output_path = Path(output_file)
    fieldnames = ["Date", "Num1", "Num2", "Num3", "Num4", "Num5", "Extra", "WinningNumbers"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scrape_millionaire_life(url: str = URL) -> list[dict[str, int | str | None]]:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    rows = parse_table_rows(soup)

    if not rows:
        rows = parse_page_text_fallback(soup)

    rows = deduplicate_rows(rows)

    if not rows:
        raise RuntimeError(
            "No Millionaire for Life draw rows were found. "
            "The page may have changed or may require JavaScript rendering."
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Millionaire for Life lottery history to CSV."
    )
    parser.add_argument("--url", default=URL, help="Millionaire for Life history URL")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file. Default: {DEFAULT_OUTPUT}",
    )

    args = parser.parse_args()

    rows = scrape_millionaire_life(args.url)
    write_csv(rows, args.output)

    print(f"Scraped {len(rows)} Millionaire for Life draw rows.")
    print(f"Saved CSV: {args.output}")


if __name__ == "__main__":
    main()
