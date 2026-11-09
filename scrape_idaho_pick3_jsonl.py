#!/usr/bin/env python3
"""
CLI entry point to scrape Idaho Lottery Pick 3 drawing history into JSONL format.

Source:
    https://www.idaholottery.com/drawgame/history/pick-3

Output:
    idaho/pick3/data/idaho_pick3_last_picks.jsonl
"""

from idaho.pick3.scrape_idaho_pick3_jsonl import (
    DEFAULT_OUTPUT,
    URL,
    Pick3Draw,
    fetch_html,
    main,
    parse_draws,
    write_jsonl,
)

if __name__ == "__main__":
    main()
