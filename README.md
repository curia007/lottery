# Idaho Lottery MLX Prediction Suite

## Overview

This project provides a complete lottery analysis platform for Idaho Lottery games.

Features include:

* Historical lottery data scraping
* CSV export generation
* MLX machine learning models
* Ranked ticket generation
* Statistical analysis
* REST API services
* Automated scheduling support

Supported Games:

* Idaho Pick 3
* Idaho Cash
* Lotto America

---

# Project Structure

```text
.
├── scrape_idaho_pick3.py
├── scrape_idaho_cash.py
├── scrape_lotto_america.py
│
├── pick3_mlx_ticket_model.py
├── idaho_cash_mlx_ticket_model.py
├── lotto_america_mlx_ticket_model.py
│
├── pick3_csv_web_service.py
│
├── idaho_pick3_history.csv
├── idaho_cash_history.csv
├── lotto_america_history.csv
│
├── README.md
└── requirements.txt
```

---

# Requirements

## Python

Python 3.11+

Verify version:

```bash
python --version
```

---

## Install Dependencies

### General Dependencies

```bash
pip install pandas numpy requests beautifulsoup4 fastapi uvicorn
```

### MLX

Apple Silicon Macs:

```bash
pip install mlx
```

Optional:

```bash
pip install certifi
```

---

# Scraping Historical Data

## Idaho Pick 3

```bash
python scrape_idaho_pick3.py
```

Output:

```text
idaho_pick3_history.csv
```

CSV format:

```csv
Date,Draw,Num1,Num2,Num3
2026-05-01,Day,4,8,3
2026-05-01,Night,7,1,5
```

---

## Idaho Cash

```bash
python scrape_idaho_cash.py
```

Output:

```text
idaho_cash_history.csv
```

CSV format:

```csv
Date,Num1,Num2,Num3,Num4,Num5
2026-05-01,4,13,24,33,43
```

---

## Lotto America

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
2026-05-01,3,8,19,25,47,6
```

---

# Pick 3 MLX Model

Train an MLX neural network using historical Pick 3 results.

## Night Draw

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw Night \
    --tickets 5
```

## Day Draw

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw Day \
    --tickets 5
```

## Both Draws

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw both \
    --tickets 5
```

## Exact Order

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw Night \
    --ticket-type exact \
    --tickets 10
```

## Any Order

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw Night \
    --ticket-type any \
    --tickets 10
```

---

# Idaho Cash MLX Model

Train an MLX neural network using historical Idaho Cash drawings.

## Balanced Tickets

```bash
python idaho_cash_mlx_ticket_model.py \
    --csv idaho_cash_history.csv \
    --ticket-type balanced \
    --tickets 10
```

## Hot Numbers

```bash
python idaho_cash_mlx_ticket_model.py \
    --csv idaho_cash_history.csv \
    --ticket-type hot \
    --tickets 10
```

## Overdue Numbers

```bash
python idaho_cash_mlx_ticket_model.py \
    --csv idaho_cash_history.csv \
    --ticket-type overdue \
    --tickets 10
```

## Hot + Overdue

```bash
python idaho_cash_mlx_ticket_model.py \
    --csv idaho_cash_history.csv \
    --ticket-type hot_overdue \
    --tickets 10
```

---

# Lotto America MLX Model

Train an MLX neural network using historical Lotto America drawings.

The model ranks:

* Main Numbers (1-52)
* Star Ball (1-10)

## Balanced Strategy

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type balanced \
    --tickets 10
```

## Hot Numbers

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type hot \
    --tickets 10
```

## Overdue Numbers

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type overdue \
    --tickets 10
```

## Hot + Overdue

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type hot_overdue \
    --tickets 10
```

## Export Predictions

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --tickets 20 \
    --output lotto_america_predictions.csv
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

# Automation

Daily update workflow:

```bash
python scrape_idaho_pick3.py
python scrape_idaho_cash.py
python scrape_lotto_america.py

python pick3_mlx_ticket_model.py \
    --draw Night \
    --tickets 10

python idaho_cash_mlx_ticket_model.py \
    --ticket-type balanced \
    --tickets 10

python lotto_america_mlx_ticket_model.py \
    --ticket-type balanced \
    --tickets 10
```

---

# Recommended Strategies

## Pick 3

Conservative:

```text
Any Order
```

Aggressive:

```text
Exact Order
```

---

## Idaho Cash

Conservative:

```text
Balanced
```

Aggressive:

```text
Hot + Overdue
```

---

## Lotto America

Conservative:

```text
Balanced
```

Aggressive:

```text
Hot + Overdue
```

---

# Disclaimer

Lottery drawings are designed to be random.

The MLX models in this project:

* Analyze historical data
* Rank candidate tickets
* Identify statistical trends

They cannot predict future lottery results with certainty.

Use these tools for research, experimentation, and entertainment purposes only.

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --tickets 20 \
    --output lotto_america_predictions.csv
# Idaho Lottery MLX Prediction Suite

## Overview

This project provides a complete lottery analysis platform for Idaho Lottery games.

Features include:

* Historical lottery data scraping
* CSV export generation
* MLX machine learning models
* Ranked ticket generation
* Statistical analysis
* REST API services
* Automated scheduling support

Supported Games:

* Idaho Pick 3
* Idaho Cash
* Lotto America

---

# Project Structure

```text
.
├── scrape_idaho_pick3.py
├── scrape_idaho_cash.py
├── scrape_lotto_america.py
│
├── pick3_mlx_ticket_model.py
├── idaho_cash_mlx_ticket_model.py
├── lotto_america_mlx_ticket_model.py
│
├── pick3_csv_web_service.py
│
├── idaho_pick3_history.csv
├── idaho_cash_history.csv
├── lotto_america_history.csv
│
├── README.md
└── requirements.txt
```

---

# Requirements

## Python

Python 3.11+

Verify version:

```bash
python --version
```

---

## Install Dependencies

### General Dependencies

```bash
pip install pandas numpy requests beautifulsoup4 fastapi uvicorn
```

### MLX

Apple Silicon Macs:

```bash
pip install mlx
```

Optional:

```bash
pip install certifi
```

---

# Scraping Historical Data

## Idaho Pick 3

```bash
python scrape_idaho_pick3.py
```

Output:

```text
idaho_pick3_history.csv
```

CSV format:

```csv
Date,Draw,Num1,Num2,Num3
2026-05-01,Day,4,8,3
2026-05-01,Night,7,1,5
```

---

## Idaho Cash

```bash
python scrape_idaho_cash.py
```

Output:

```text
idaho_cash_history.csv
```

CSV format:

```csv
Date,Num1,Num2,Num3,Num4,Num5
2026-05-01,4,13,24,33,43
```

---

## Lotto America

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
2026-05-01,3,8,19,25,47,6
```

---

# Pick 3 MLX Model

Train an MLX neural network using historical Pick 3 results.

## Night Draw

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw Night \
    --tickets 5
```

## Day Draw

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw Day \
    --tickets 5
```

## Both Draws

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw both \
    --tickets 5
```

## Exact Order

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw Night \
    --ticket-type exact \
    --tickets 10
```

## Any Order

```bash
python pick3_mlx_ticket_model.py \
    --csv idaho_pick3_history.csv \
    --draw Night \
    --ticket-type any \
    --tickets 10
```

---

# Idaho Cash MLX Model

Train an MLX neural network using historical Idaho Cash drawings.

## Balanced Tickets

```bash
python idaho_cash_mlx_ticket_model.py \
    --csv idaho_cash_history.csv \
    --ticket-type balanced \
    --tickets 10
```

## Hot Numbers

```bash
python idaho_cash_mlx_ticket_model.py \
    --csv idaho_cash_history.csv \
    --ticket-type hot \
    --tickets 10
```

## Overdue Numbers

```bash
python idaho_cash_mlx_ticket_model.py \
    --csv idaho_cash_history.csv \
    --ticket-type overdue \
    --tickets 10
```

## Hot + Overdue

```bash
python idaho_cash_mlx_ticket_model.py \
    --csv idaho_cash_history.csv \
    --ticket-type hot_overdue \
    --tickets 10
```

---

# Lotto America MLX Model

Train an MLX neural network using historical Lotto America drawings.

The model ranks:

* Main Numbers (1-52)
* Star Ball (1-10)

## Balanced Strategy

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type balanced \
    --tickets 10
```

## Hot Numbers

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type hot \
    --tickets 10
```

## Overdue Numbers

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type overdue \
    --tickets 10
```

## Hot + Overdue

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type hot_overdue \
    --tickets 10
```

## Export Predictions

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --tickets 20 \
    --output lotto_america_predictions.csv
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

# Automation

Daily update workflow:

```bash
python scrape_idaho_pick3.py
python scrape_idaho_cash.py
python scrape_lotto_america.py

python pick3_mlx_ticket_model.py \
    --draw Night \
    --tickets 10

python idaho_cash_mlx_ticket_model.py \
    --ticket-type balanced \
    --tickets 10

python lotto_america_mlx_ticket_model.py \
    --ticket-type balanced \
    --tickets 10
```

---

# Recommended Strategies

## Pick 3

Conservative:

```text
Any Order
```

Aggressive:

```text
Exact Order
```

---

## Idaho Cash

Conservative:

```text
Balanced
```

Aggressive:

```text
Hot + Overdue
```

---

## Lotto America

Conservative:

```text
Balanced
```

Aggressive:

```text
Hot + Overdue
```

---

# Disclaimer

Lottery drawings are designed to be random.

The MLX models in this project:

* Analyze historical data
* Rank candidate tickets
* Identify statistical trends

They cannot predict future lottery results with certainty.

Use these tools for research, experimentation, and entertainment purposes only.
