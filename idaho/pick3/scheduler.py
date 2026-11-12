#!/usr/bin/env python3
"""
scheduler.py

Automated task scheduler to run lottery scrapers for Pick 3, Idaho Cash, and Lotto America.
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Process
from pathlib import Path

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
except ImportError:
    BlockingScheduler = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_pick3() -> None:
    scraper = PROJECT_ROOT / "idaho" / "pick3" / "scrape_idaho_pick3.py"
    if scraper.exists():
        os.system(f"{sys.executable} {scraper}")
    else:
        os.system(f"{sys.executable} scrape_idaho_pick3.py")


def run_cash() -> None:
    scraper = PROJECT_ROOT / "idaho" / "idaho_cash" / "scrape_idaho_cash.py"
    if scraper.exists():
        os.system(f"{sys.executable} {scraper}")
    else:
        os.system(f"{sys.executable} scrape_idaho_cash.py")


def run_lotto() -> None:
    scraper = PROJECT_ROOT / "lotto_america" / "scrape_lotto_america.py"
    if scraper.exists():
        os.system(f"{sys.executable} {scraper}")
    else:
        os.system(f"{sys.executable} scrape_lotto_america.py")


def run_all() -> None:
    p1 = Process(target=run_pick3)
    p2 = Process(target=run_cash)
    p3 = Process(target=run_lotto)

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()


def start_scheduler() -> None:
    if BlockingScheduler is None:
        print("Error: 'apscheduler' is not installed. Install it with: pip install apscheduler")
        return

    scheduler = BlockingScheduler()
    # Run all scripts every day at 7 PM
    scheduler.add_job(run_all, "cron", hour=19, minute=0)

    print("Scheduler running... (Press Ctrl+C to stop)")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--run-now", "-r", "run"):
        print("Running all scrapers once now...")
        run_all()
    else:
        start_scheduler()
