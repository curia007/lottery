#!/usr/bin/env python3
"""
scrape_idaho_pick4.py

Scrapes Idaho Pick 4 draw history from:
https://www.idaholottery.com/drawgame/history/pick-4

Outputs:
    Date,Draw,Num1,Num2,Num3,Num4

Install:
    pip install requests beautifulsoup4

Run:
    python scrape_idaho_pick4.py
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


URL = "https://www.idaholottery.com/drawgame/history/pick-4"
DEFAULT_OUTPUT = "data/idaho_pick4_history.csv"


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 AppleWebKit/605.1.15 Safari/605.1.15"
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
            try:
                return datetime.strptime(raw_date, fmt.replace(".", "")).strftime("%Y-%m-%d")
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


def detect_draw(text: str) -> str:
    lower = str(text).lower()
    if "night" in lower or "evening" in lower:
        return "Night"
    if "day" in lower or "midday" in lower:
        return "Day"
    return ""


def extract_digits(text: str) -> list[int]:
    text = str(text)

    separated = [int(x) for x in re.findall(r"\b\d\b", text)]
    if len(separated) >= 4:
        return separated[:4]

    compact_values = re.findall(r"\b\d{4}\b", text)
    if compact_values:
        return [int(ch) for ch in compact_values[0]]

    return separated


def make_row(date: str, draw: str, digits: list[int]) -> dict[str, int | str] | None:
    if len(digits) < 4:
        return None

    digits = digits[:4]

    if any(d < 0 or d > 9 for d in digits):
        return None

    return {
        "Date": date,
        "Draw": draw,
        "Num1": digits[0],
        "Num2": digits[1],
        "Num3": digits[2],
        "Num4": digits[3],
    }


def parse_table_rows(soup: BeautifulSoup) -> list[dict[str, int | str]]:
    rows = []

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

            candidate_text = " ".join(cell for cell in cells if not parse_date(cell))
            candidate_text = remove_dates(candidate_text)
            draw = detect_draw(row_text)
            digits = extract_digits(candidate_text)

            row = make_row(date, draw, digits)

            if row:
                rows.append(row)

    return rows


def parse_text_fallback(soup: BeautifulSoup) -> list[dict[str, int | str]]:
    rows = []

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = [line.strip() for line in soup.get_text("\n").splitlines()]
    lines = [line for line in lines if line]

    for index, line in enumerate(lines):
        date = parse_date(line)

        if not date:
            continue

        nearby = " ".join(lines[index:index + 10])
        draw = detect_draw(nearby)
        digits = extract_digits(remove_dates(nearby))

        row = make_row(date, draw, digits)

        if row:
            rows.append(row)

    return rows


def deduplicate_rows(rows: Iterable[dict[str, int | str]]) -> list[dict[str, int | str]]:
    seen = set()
    clean = []

    for row in rows:
        key = (
            row["Date"],
            row["Draw"],
            row["Num1"],
            row["Num2"],
            row["Num3"],
            row["Num4"],
        )

        if key in seen:
            continue

        seen.add(key)
        clean.append(row)

    clean.sort(key=lambda r: (str(r["Date"]), str(r["Draw"])), reverse=True)
    return clean


def write_csv(rows: list[dict[str, int | str]], output_file: str | Path) -> None:
    fieldnames = ["Date", "Draw", "Num1", "Num2", "Num3", "Num4"]

    with Path(output_file).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scrape_idaho_pick4(url: str = URL) -> list[dict[str, int | str]]:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    rows = parse_table_rows(soup)

    if not rows:
        rows = parse_text_fallback(soup)

    rows = deduplicate_rows(rows)

    if not rows:
        raise RuntimeError(
            "No Idaho Pick 4 draw rows were found. "
            "The page may have changed or may require JavaScript rendering."
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Idaho Pick 4 history to CSV.")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = scrape_idaho_pick4(args.url)
    write_csv(rows, args.output)

    print(f"Scraped {len(rows)} Idaho Pick 4 draw rows.")
    print(f"Saved CSV: {args.output}")


if __name__ == "__main__":
    main()
