#!/usr/bin/env python3
"""
scrape_lotto_america.py

Scrapes Lotto America draw history from:
https://www.idaholottery.com/drawgame/history/lotto-america

Outputs a CSV file with:

    Date,Num1,Num2,Num3,Num4,Num5,StarBall

Install:
    pip install requests beautifulsoup4

Run:
    python scrape_lotto_america.py

Optional:
    python scrape_lotto_america.py --output lotto_america_history.csv
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


URL = "https://www.idaholottery.com/drawgame/history/lotto-america"
DEFAULT_OUTPUT = "data/lotto_america_history.csv"


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
    return [int(n) for n in nums if 1 <= int(n) <= 52]


def valid_star_numbers(nums: Iterable[int]) -> list[int]:
    return [int(n) for n in nums if 1 <= int(n) <= 10]


def make_row(date: str, nums: list[int]) -> dict[str, int | str] | None:
    """
    Lotto America uses 5 main numbers from 1-52 and 1 Star Ball from 1-10.

    Most pages list the five main numbers first, followed by Star Ball.
    """
    if len(nums) < 6:
        return None

    main = valid_main_numbers(nums[:5])
    star_candidates = valid_star_numbers(nums[5:])

    if len(main) != 5 or not star_candidates:
        return None

    star = star_candidates[0]

    return {
        "Date": date,
        "Num1": main[0],
        "Num2": main[1],
        "Num3": main[2],
        "Num4": main[3],
        "Num5": main[4],
        "StarBall": star,
    }


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

            row_text = " ".join(cells)
            date = parse_date(row_text)

            if not date:
                continue

            non_date_cells = [cell for cell in cells if not parse_date(cell)]
            candidate_text = " ".join(non_date_cells)

            nums = extract_numbers(candidate_text)

            if len(nums) < 6:
                nums = extract_numbers(remove_dates(row_text))

            row = make_row(date, nums)

            if row:
                rows.append(row)

    return rows


def parse_page_text_fallback(soup: BeautifulSoup) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []

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


def deduplicate_rows(rows: Iterable[dict[str, int | str]]) -> list[dict[str, int | str]]:
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
            row["StarBall"],
        )

        if key in seen:
            continue

        seen.add(key)
        clean_rows.append(row)

    clean_rows.sort(key=lambda r: str(r["Date"]), reverse=True)
    return clean_rows


def write_csv(rows: list[dict[str, int | str]], output_file: str | Path) -> None:
    output_path = Path(output_file)
    fieldnames = ["Date", "Num1", "Num2", "Num3", "Num4", "Num5", "StarBall"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scrape_lotto_america(url: str = URL) -> list[dict[str, int | str]]:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    rows = parse_table_rows(soup)

    if not rows:
        rows = parse_page_text_fallback(soup)

    rows = deduplicate_rows(rows)

    if not rows:
        raise RuntimeError(
            "No Lotto America draw rows were found. "
            "The page may have changed or may require JavaScript rendering."
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Lotto America lottery history to CSV.")
    parser.add_argument("--url", default=URL, help="Lotto America history URL")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file. Default: {DEFAULT_OUTPUT}",
    )

    args = parser.parse_args()

    rows = scrape_lotto_america(args.url)
    write_csv(rows, args.output)

    print(f"Scraped {len(rows)} Lotto America draw rows.")
    print(f"Saved CSV: {args.output}")


if __name__ == "__main__":
    main()
