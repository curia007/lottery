#!/usr/bin/env python3
"""
scrape_idaho_pick3_jsonl.py (root wrapper)

Scrapes historical Idaho Lottery Pick 3 draws and outputs to JSONL.
Delegates to `idaho/pick3/scrape_idaho_pick3_jsonl.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from idaho.pick3.scrape_idaho_pick3_jsonl import (
    DEFAULT_OUTPUT,
    URL,
    Pick3Draw,
    fetch_html,
    main,
    parse_draws,
    resolve_default_output,
    write_jsonl,
)

__all__ = [
    "DEFAULT_OUTPUT",
    "URL",
    "Pick3Draw",
    "fetch_html",
    "main",
    "parse_draws",
    "resolve_default_output",
    "write_jsonl",
]

if __name__ == "__main__":
    main()
