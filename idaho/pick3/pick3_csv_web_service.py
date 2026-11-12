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
import sys

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
except ImportError:
    FastAPI = None
    HTTPException = None
    FileResponse = None

if FastAPI is not None:
    app = FastAPI(title="Idaho Pick 3 CSV Web Service")
else:
    app = None

# Change this if your scraper writes to a different CSV filename.
CSV_FILE = Path("idaho_pick3_history.csv")


def resolve_csv_file() -> Path:
    """Finds the CSV file across common relative paths."""
    candidates = [
        CSV_FILE,
        Path("data/idaho_pick3_history.csv"),
        Path("idaho/pick3/data/idaho_pick3_history.csv"),
        Path(__file__).resolve().parent / "data" / "idaho_pick3_history.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return CSV_FILE


if app is not None:
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
        csv_path = resolve_csv_file()
        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"CSV file not found: {csv_path}. Run scrape_idaho_pick3.py first.",
            )

        try:
            with csv_path.open("r", newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)

            return {
                "file": str(csv_path),
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
        csv_path = resolve_csv_file()
        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"CSV file not found: {csv_path}. Run scrape_idaho_pick3.py first.",
            )

        return FileResponse(
            path=csv_path,
            filename=csv_path.name,
            media_type="text/csv",
        )


if __name__ == "__main__":
    if FastAPI is None:
        print("Error: 'fastapi' or 'uvicorn' is not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
