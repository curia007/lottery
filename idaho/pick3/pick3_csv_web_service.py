"""
pick3_csv_web_service.py

A small FastAPI web service that returns the contents of the CSV file
created by scrape_idaho_pick3.py.

Expected CSV format:
Date,Draw,Num1,Num2,Num3

Install:
    pip install fastapi uvicorn

Run:
    uvicorn pick3_csv_web_service:app --reload

Example URLs:
    http://127.0.0.1:8000/
    http://127.0.0.1:8000/pick3
    http://127.0.0.1:8000/pick3/download
"""

from pathlib import Path
import csv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI(title="Idaho Pick 3 CSV Web Service")

# Change this if your scraper writes to a different CSV filename.
CSV_FILE = Path("idaho_pick3_history.csv")


@app.get("/")
def home():
    return {
        "message": "Idaho Pick 3 CSV Web Service is running",
        "json_endpoint": "/pick3",
        "download_endpoint": "/pick3/download",
    }


@app.get("/pick3")
def get_pick3_csv_contents():
    """Return CSV contents as JSON."""

    if not CSV_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=f"CSV file not found: {CSV_FILE}. Run scrape_idaho_pick3.py first.",
        )

    try:
        with CSV_FILE.open("r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        return {
            "file": str(CSV_FILE),
            "record_count": len(rows),
            "data": rows,
        }

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading CSV file: {ex}",
        )


@app.get("/pick3/download")
def download_pick3_csv():
    """Download the raw CSV file."""

    if not CSV_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=f"CSV file not found: {CSV_FILE}. Run scrape_idaho_pick3.py first.",
        )

    return FileResponse(
        path=CSV_FILE,
        filename=CSV_FILE.name,
        media_type="text/csv",
    )
