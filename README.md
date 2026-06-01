# Idaho Lottery MLX Prediction Suite

## Overview

This project provides a complete lottery analytics and prediction framework for Idaho Lottery games using Python and Apple's MLX framework.

Features include:

* Historical draw scraping
* CSV export generation
* MLX machine learning models
* Ranked ticket generation
* Hot number analysis
* Overdue number analysis
* REST API services
* Automated scheduling support

Supported Games:

* Idaho Pick 3
* Idaho Cash
* Lotto America
* Millionaire for Life

---

# Project Structure

```text
.
├── scrape_idaho_pick3.py
├── scrape_idaho_cash.py
├── scrape_lotto_america.py
├── scrape_millionaire_life.py
│
├── pick3_mlx_ticket_model.py
├── idaho_cash_mlx_ticket_model.py
├── lotto_america_mlx_ticket_model.py
├── millionaire_life_mlx_ticket_model.py
│
├── pick3_csv_web_service.py
│
├── idaho_pick3_history.csv
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
    --tickets 5
```

## Generate Day Draw Predictions

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw Day \
    --tickets 5
```

## Exact Order

```bash
python pick3_mlx_ticket_model.py \
    --draw Night \
    --ticket-type exact \
    --tickets 10
```

## Any Order

```bash
python pick3_mlx_ticket_model.py \
    --draw Night \
    --ticket-type any \
    --tickets 10
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
    --csv idaho_cash_history.csv \
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

## Hot + Overdue

```bash
python idaho_cash_mlx_ticket_model.py \
    --ticket-type hot_overdue \
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

CSV format:

```csv
Date,Num1,Num2,Num3,Num4,Num5,StarBall
```

## Generate Balanced Tickets

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type balanced \
    --tickets 10
```

## Hot Numbers

```bash
python lotto_america_mlx_ticket_model.py \
    --ticket-type hot \
    --tickets 10
```

## Overdue Numbers

```bash
python lotto_america_mlx_ticket_model.py \
    --ticket-type overdue \
    --tickets 10
```

## Hot + Overdue

```bash
python lotto_america_mlx_ticket_model.py \
    --ticket-type hot_overdue \
    --tickets 10
```

## Export Predictions

```bash
python lotto_america_mlx_ticket_model.py \
    --tickets 20 \
    --output lotto_america_predictions.csv
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

CSV format:

```csv
Date,Num1,Num2,Num3,Num4,Num5,Extra,WinningNumbers
```

## Generate Balanced Tickets

```bash
python millionaire_life_mlx_ticket_model.py \
    --csv millionaire_life_history.csv \
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

## Hot + Overdue

```bash
python millionaire_life_mlx_ticket_model.py \
    --ticket-type hot_overdue \
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
    --output millionaire_life_predictions.csv
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

Examples:

```text
http://127.0.0.1:8000/pick3
http://127.0.0.1:8000/pick3/download
```

---

# Automated Daily Workflow

```bash
python scrape_idaho_pick3.py
python scrape_idaho_cash.py
python scrape_lotto_america.py
python scrape_millionaire_life.py

python pick3_mlx_ticket_model.py \
    --draw Night \
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
| Pick 3               | Any Order    | Exact         | Exact Top 5 |
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
