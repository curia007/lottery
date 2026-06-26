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
* Millionaire for Life

---

# Project Structure

```text
.
├── scrape_idaho_pick3.py
├── scrape_idaho_pick4.py
├── scrape_idaho_cash.py
├── scrape_lotto_america.py
├── scrape_millionaire_life.py
│
├── pick3_mlx_ticket_model.py
├── pick4_mlx_number_select_model.py
├── idaho_cash_mlx_ticket_model.py
├── lotto_america_mlx_ticket_model.py
├── millionaire_life_mlx_ticket_model.py
│
├── pick3_csv_web_service.py
│
├── idaho_pick3_history.csv
├── idaho_pick4_history.csv
├── idaho_cash_history.csv
├── lotto_america_history.csv
├── millionaire_life_history.csv
│
├── README.md
└── requirements.txt
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
python scrape_idaho_pick3.py
```

Output:

```text
idaho_pick3_history.csv
```

## Generate Night Draw Predictions

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw Night \
    --tickets 10
```

## Generate Any-Order Predictions

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw both \
    --ticket-type any \
    --tickets 10
```

## Generate 6-Way Predictions

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw both \
    --ticket-type 6-way \
    --tickets 10
```

## Generate 3-Way Predictions

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw both \
    --ticket-type 3-way \
    --tickets 10
```

## Generate Day Draw Predictions

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw Day \
    --tickets 10
```

### Ticket Types

```text
exact
any
6-way
3-way
straight_any
```

`6-way` is the any-order play for three distinct digits. `3-way` is the any-order play for one pair and one single digit.

---

# Idaho Pick 4

## Scrape History

```bash
python scrape_idaho_pick4.py
```

Output:

```text
idaho_pick4_history.csv
```

CSV Format:

```csv
Date,Draw,Num1,Num2,Num3,Num4
2026-01-01,Night,1,2,3,4
```

## Generate Exact Order Predictions

```bash
python pick4_mlx_number_select_model.py \
    --csv idaho_pick4_history.csv \
    --draw Night \
    --number-select exact \
    --ticket-type model \
    --tickets 10
```

## Generate Any Order Predictions

```bash
python pick4_mlx_number_select_model.py \
    --csv idaho_pick4_history.csv \
    --draw Night \
    --number-select any \
    --ticket-type balanced \
    --tickets 10
```

## Generate Day Draw Predictions

```bash
python pick4_mlx_number_select_model.py \
    --csv idaho_pick4_history.csv \
    --draw Day \
    --number-select exact \
    --tickets 10
```

## Generate Both Day and Night Predictions

```bash
python pick4_mlx_number_select_model.py \
    --csv idaho_pick4_history.csv \
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
python pick4_mlx_number_select_model.py \
    --draw Night \
    --number-select exact \
    --tickets 20 \
    --output pick4_predictions.csv
```

---

# Idaho Cash

## Scrape History

```bash
python scrape_idaho_cash.py
```

Output:

```text
idaho_cash_history.csv
```

## Generate Balanced Tickets

```bash
python idaho_cash_mlx_ticket_model.py \
    --csv data/idaho_cash_history.csv \
    --ticket-type balanced \
    --tickets 10
```

## Hot Numbers

```bash
python idaho_cash_mlx_ticket_model.py \
    --ticket-type hot \
    --tickets 10
```

## Overdue Numbers

```bash
python idaho_cash_mlx_ticket_model.py \
    --ticket-type overdue \
    --tickets 10
```

---

# Lotto America

## Scrape History

```bash
python scrape_lotto_america.py
```

Output:

```text
lotto_america_history.csv
```

## Generate Balanced Tickets

```bash
python lotto_america_mlx_ticket_model.py \
    --csv data/lotto_america_history.csv \
    --ticket-type balanced \
    --tickets 10
```

## Hot + Overdue

```bash
python lotto_america_mlx_ticket_model.py \
    --ticket-type hot_overdue \
    --tickets 10
```

---

# Millionaire for Life

## Scrape History

```bash
python scrape_millionaire_life.py
```

Output:

```text
millionaire_life_history.csv
```

CSV Format:

```csv
Date,Num1,Num2,Num3,Num4,Num5,Extra,WinningNumbers
```

## Generate Balanced Tickets

```bash
python millionaire_life_mlx_ticket_model.py \
    --csv data/millionaire_life_history.csv \
    --ticket-type balanced \
    --tickets 10
```

## Hot Numbers

```bash
python millionaire_life_mlx_ticket_model.py \
    --ticket-type hot \
    --tickets 10
```

## Overdue Numbers

```bash
python millionaire_life_mlx_ticket_model.py \
    --ticket-type overdue \
    --tickets 10
```

## Include Extra Number

```bash
python millionaire_life_mlx_ticket_model.py \
    --include-extra \
    --tickets 10
```

## Export Predictions

```bash
python millionaire_life_mlx_ticket_model.py \
    --tickets 20 \
    --output data/millionaire_life_predictions.csv
```

---

# REST API

## Pick 3 CSV Service

Start server:

```bash
uvicorn pick3_csv_web_service:app --reload
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
python scrape_idaho_pick3.py
python scrape_idaho_pick4.py
python scrape_idaho_cash.py
python scrape_lotto_america.py
python scrape_millionaire_life.py

python pick3_mlx_ticket_model.py \
    --csv data/idaho_pick3_history.csv \
    --draw Night \
    --tickets 10

python pick4_mlx_number_select_model.py \
    --draw Night \
    --number-select exact \
    --tickets 10

python idaho_cash_mlx_ticket_model.py \
    --ticket-type balanced \
    --tickets 10

python lotto_america_mlx_ticket_model.py \
    --ticket-type balanced \
    --tickets 10

python millionaire_life_mlx_ticket_model.py \
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
