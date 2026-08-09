# Idaho Lottery MLX Prediction Suite

## Overview

This project provides a complete lottery analytics, scraping, prediction, and REST service framework for Idaho Lottery games using Python and Apple's MLX machine learning framework.

Features include:

* Historical draw scraping
* CSV export generation
* MLX machine learning models
* Ranked ticket generation
* Hot number analysis
* Overdue number analysis
* Exact, Any Order, 3-way, and 6-way ticket support
* REST API services
* Automated scheduling support

Supported Games:

* Idaho Pick 3
* Idaho Pick 4
* Idaho Cash
* Lotto America
* Mega Millions
* Millionaire for Life

---

# Project Structure

```text
.
├── idaho/
│   ├── pick3/
│   │   ├── scrape_idaho_pick3.py
│   │   ├── pick3_mlx_ticket_model.py
│   │   ├── pick3_csv_web_service.py
│   │   └── data/
│   │       └── idaho_pick3_history.csv
│   ├── pick4/
│   │   ├── scrape_idaho_pick4.py
│   │   ├── pick4_mlx_number_select_model.py
│   │   └── data/
│   │       └── idaho_pick4_history.csv
│   └── idaho_cash/
│       ├── scrape_idaho_cash.py
│       ├── idaho_cash_mlx_ticket_model.py
│       └── data/
│           └── idaho_cash_history.csv
├── lotto_america/
│   ├── scrape_lotto_america.py
│   ├── lotto_america_mlx_ticket_model.py
│   └── data/
│       └── lotto_america_history.csv
├── mega_millions/
│   ├── scrape_mega_millions.py
│   ├── mega_millions_mlx_ticket_model.py
│   └── data/
│       └── mega_millions_history.csv
├── millionaire_life/
│   ├── scrape_millionaire_life.py
│   ├── millionaire_life_mlx_ticket_model.py
│   └── data/
│       └── millionaire_life_history.csv
├── main.py
├── LICENSE
└── README.md
```

---

# Requirements

## Python

Python 3.11+

```bash
python --version
```

## Install Dependencies

```bash
pip install pandas numpy requests beautifulsoup4 fastapi uvicorn
```

### MLX (Apple Silicon)

```bash
pip install mlx
```

---

# Idaho Pick 3

## Scrape History

```bash
python idaho/pick3/scrape_idaho_pick3.py
```

Output:

```text
idaho/pick3/data/idaho_pick3_history.csv
```

## Generate Night Draw Predictions

```bash
python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw Night \
    --tickets 10
```

## Generate Any-Order Predictions

```bash
python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw both \
    --ticket-type any \
    --tickets 10
```

## Generate 6-Way Predictions

```bash
python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw both \
    --ticket-type 6-way \
    --tickets 10
```

## Generate 3-Way Predictions

```bash
python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw both \
    --ticket-type 3-way \
    --tickets 10
```

## Generate Day Draw Predictions

```bash
python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw Day \
    --tickets 10
```

## Combined Draw Training

Train on both Day and Night history combined for the next draw:

```bash
python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw combo \
    --tickets 10
```

### Advanced Options

* `--exclude-recent N`: Exclude tickets that appeared in the last N draws.
* `--weights model,freq,overdue,pair`: Set custom weights for scoring (e.g., `0.5,0.2,0.2,0.1`).
* `--seed N`: Set a random seed for reproducible training.
* `--output FILE`: Save ranked results to a CSV file.

### Ticket Types

```text
exact
any
6-way
3-way
straight_any
```

`6-way` is the any-order play for three distinct digits. `3-way` is the any-order play for one pair and one single digit. `straight_any` provides exact-order ranking but labeled for Straight/Any play.

---

# Idaho Pick 4

## Scrape History

```bash
python idaho/pick4/scrape_idaho_pick4.py
```

Output:

```text
idaho/pick4/data/idaho_pick4_history.csv
```

CSV Format:

```csv
Date,Draw,Num1,Num2,Num3,Num4
2026-01-01,Night,1,2,3,4
```

## Generate Exact Order Predictions

```bash
python idaho/pick4/pick4_mlx_number_select_model.py \
    --csv idaho/pick4/data/idaho_pick4_history.csv \
    --draw Night \
    --number-select exact \
    --ticket-type model \
    --tickets 10
```

## Generate Any Order Predictions

```bash
python idaho/pick4/pick4_mlx_number_select_model.py \
    --csv idaho/pick4/data/idaho_pick4_history.csv \
    --draw Night \
    --number-select any \
    --ticket-type balanced \
    --tickets 10
```

## Generate Day Draw Predictions

```bash
python idaho/pick4/pick4_mlx_number_select_model.py \
    --csv idaho/pick4/data/idaho_pick4_history.csv \
    --draw Day \
    --number-select exact \
    --tickets 10
```

## Generate Both Day and Night Predictions

```bash
python idaho/pick4/pick4_mlx_number_select_model.py \
    --csv idaho/pick4/data/idaho_pick4_history.csv \
    --draw both \
    --number-select exact \
    --tickets 10
```

### Ticket Types

```text
model
balanced
hot
overdue
hot_overdue
```

### Number Selection Types

```text
exact
any
boxed
```

## Export Predictions

```bash
python idaho/pick4/pick4_mlx_number_select_model.py \
    --csv idaho/pick4/data/idaho_pick4_history.csv \
    --draw Night \
    --number-select exact \
    --tickets 20 \
    --output idaho/pick4/data/pick4_predictions.csv
```

---

# Idaho Cash

## Scrape History

```bash
python idaho/idaho_cash/scrape_idaho_cash.py
```

Output:

```text
idaho/idaho_cash/data/idaho_cash_history.csv
```

## Generate Balanced Tickets

```bash
python idaho/idaho_cash/idaho_cash_mlx_ticket_model.py \
    --csv idaho/idaho_cash/data/idaho_cash_history.csv \
    --ticket-type balanced \
    --tickets 10
```

## Hot Numbers

```bash
python idaho/idaho_cash/idaho_cash_mlx_ticket_model.py \
    --csv idaho/idaho_cash/data/idaho_cash_history.csv \
    --ticket-type hot \
    --tickets 10
```

## Overdue Numbers

```bash
python idaho/idaho_cash/idaho_cash_mlx_ticket_model.py \
    --csv idaho/idaho_cash/data/idaho_cash_history.csv \
    --ticket-type overdue \
    --tickets 10
```

---

# Lotto America

## Scrape History

```bash
python lotto_america/scrape_lotto_america.py
```

Output:

```text
lotto_america/data/lotto_america_history.csv
```

## Generate Balanced Tickets

```bash
python lotto_america/lotto_america_mlx_ticket_model.py \
    --csv lotto_america/data/lotto_america_history.csv \
    --ticket-type balanced \
    --star-mode balanced \
    --tickets 10
```

## Hot + Overdue

```bash
python lotto_america/lotto_america_mlx_ticket_model.py \
    --csv lotto_america/data/lotto_america_history.csv \
    --ticket-type hot_overdue \
    --star-mode overdue \
    --tickets 10
```

## Star Ball Predictions

The Lotto America model also prints Star Ball predictions in four modes:
hot, balanced, mix, and low hit.

```bash
python lotto_america/lotto_america_mlx_ticket_model.py \
    --csv lotto_america/data/lotto_america_history.csv \
    --star-hit-window 50 \
    --star-top 5
```

## Star Ball Modes

```bash
python lotto_america/lotto_america_mlx_ticket_model.py \
    --star-mode default

python lotto_america/lotto_america_mlx_ticket_model.py \
    --star-mode balanced

python lotto_america/lotto_america_mlx_ticket_model.py \
    --star-mode overdue

python lotto_america/lotto_america_mlx_ticket_model.py \
    --star-mode less_hits
```

## Star Ball Hits Window

Use `--star-hit-window` to count how many times each Star Ball was drawn from the most recent N draws. Use `0` to include the full history.

```bash
python lotto_america/lotto_america_mlx_ticket_model.py \
    --csv lotto_america/data/lotto_america_history.csv \
    --star-hit-window 50 \
    --star-mode less_hits
```

Use `--star-top` to control how many Star Ball predictions are shown per mode.

---

# Mega Millions

## Scrape History

```bash
python mega_millions/scrape_mega_millions.py
```

Output:

```text
mega_millions/data/mega_millions_history.csv
```

CSV Format:

```csv
Date,Num1,Num2,Num3,Num4,Num5,MegaBall
```

## Generate Balanced Tickets

```bash
python mega_millions/mega_millions_mlx_ticket_model.py \
    --csv mega_millions/data/mega_millions_history.csv \
    --ticket-type balanced \
    --tickets 10
```

## Hot Numbers

```bash
python mega_millions/mega_millions_mlx_ticket_model.py \
    --csv mega_millions/data/mega_millions_history.csv \
    --ticket-type hot \
    --tickets 10
```

## Overdue Numbers

```bash
python mega_millions/mega_millions_mlx_ticket_model.py \
    --csv mega_millions/data/mega_millions_history.csv \
    --ticket-type overdue \
    --tickets 10
```

## Export Predictions

```bash
python mega_millions/mega_millions_mlx_ticket_model.py \
    --csv mega_millions/data/mega_millions_history.csv \
    --ticket-type balanced \
    --tickets 10 \
    --output mega_millions/data/mega_millions_predictions.csv
```

---

# Millionaire for Life

## Scrape History

```bash
python millionaire_life/scrape_millionaire_life.py
```

Output:

```text
millionaire_life/data/millionaire_life_history.csv
```

CSV Format:

```csv
Date,Num1,Num2,Num3,Num4,Num5,Extra,WinningNumbers
```

## Generate Balanced Tickets

```bash
python millionaire_life/millionaire_life_mlx_ticket_model.py \
    --csv millionaire_life/data/millionaire_life_history.csv \
    --ticket-type balanced \
    --tickets 10
```

## Hot Numbers

```bash
python millionaire_life/millionaire_life_mlx_ticket_model.py \
    --csv millionaire_life/data/millionaire_life_history.csv \
    --ticket-type hot \
    --tickets 10
```

## Overdue Numbers

```bash
python millionaire_life/millionaire_life_mlx_ticket_model.py \
    --csv millionaire_life/data/millionaire_life_history.csv \
    --ticket-type overdue \
    --tickets 10
```

## Include Extra Number

```bash
python millionaire_life/millionaire_life_mlx_ticket_model.py \
    --csv millionaire_life/data/millionaire_life_history.csv \
    --include-extra \
    --tickets 10
```

## Export Predictions

```bash
python millionaire_life/millionaire_life_mlx_ticket_model.py \
    --csv millionaire_life/data/millionaire_life_history.csv \
    --tickets 20 \
    --output millionaire_life/data/millionaire_life_predictions.csv
```

---

# REST API

## Pick 3 CSV Service

Start server:

```bash
uvicorn idaho.pick3.pick3_csv_web_service:app --reload
```

Endpoints:

```text
GET /pick3
GET /pick3/download
```

Example:

```text
http://127.0.0.1:8000/pick3
http://127.0.0.1:8000/pick3/download
```

---

# Automated Daily Workflow

```bash
python idaho/pick3/scrape_idaho_pick3.py
python idaho/pick4/scrape_idaho_pick4.py
python idaho/idaho_cash/scrape_idaho_cash.py
python lotto_america/scrape_lotto_america.py
python millionaire_life/scrape_millionaire_life.py

python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw combo \
    --tickets 10

python idaho/pick4/pick4_mlx_number_select_model.py \
    --csv idaho/pick4/data/idaho_pick4_history.csv \
    --draw Night \
    --number-select exact \
    --tickets 10

python idaho/idaho_cash/idaho_cash_mlx_ticket_model.py \
    --csv idaho/idaho_cash/data/idaho_cash_history.csv \
    --ticket-type balanced \
    --tickets 10

python lotto_america/lotto_america_mlx_ticket_model.py \
    --csv lotto_america/data/lotto_america_history.csv \
    --ticket-type balanced \
    --star-mode balanced \
    --tickets 10

python millionaire_life/millionaire_life_mlx_ticket_model.py \
    --csv millionaire_life/data/millionaire_life_history.csv \
    --ticket-type balanced \
    --tickets 10
```

---

# Recommended Strategies

| Game                 | Conservative | Balanced      | Aggressive  |
| -------------------- | ------------ | ------------- | ----------- |
| Pick 3               | 6-Way        | 3-Way         | Exact       |
| Pick 4               | Any Order    | Exact         | Model       |
| Idaho Cash           | Balanced     | Hot + Overdue | Model       |
| Lotto America        | Balanced     | Hot + Overdue | Model       |
| Millionaire for Life | Balanced     | Hot + Overdue | Model       |

---

# Disclaimer

Lottery drawings are designed to be random.

The MLX models in this project:

* Analyze historical data
* Identify statistical patterns
* Rank candidate tickets

They do not predict future lottery results with certainty.

Use these tools for research, experimentation, and entertainment purposes only.
