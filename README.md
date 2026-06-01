# Idaho Lottery MLX Prediction Suite

## Overview

This project provides utilities for:

- Scraping Idaho Lottery historical results
- Exporting CSV files
- Training MLX machine-learning models
- Generating ranked ticket candidates
- Serving lottery data through REST APIs

Supported Games:

- Idaho Pick 3
- Idaho Cash

## Files

- scrape_idaho_pick3.py
- scrape_idaho_cash.py
- pick3_mlx_ticket_model.py
- idaho_cash_mlx_ticket_model.py
- pick3_csv_web_service.py
- README.md

## Quick Start

Install dependencies:

```bash
pip install pandas numpy requests beautifulsoup4 fastapi uvicorn mlx
```

Scrape data:

```bash
python scrape_idaho_pick3.py
python scrape_idaho_cash.py
```

Generate Pick 3 tickets:

```bash
python pick3_mlx_ticket_model.py --csv idaho_pick3_history.csv --draw Night --tickets 5
```

Generate Idaho Cash tickets:

```bash
python idaho_cash_mlx_ticket_model.py --csv idaho_cash_history.csv --tickets 5
```

## Disclaimer

Lottery drawings are random. These models analyze historical patterns and rank candidate tickets but cannot predict future lottery results with certainty.
