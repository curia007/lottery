# Idaho Lottery MLX & Transformer Prediction Suite

## Overview

This project provides a comprehensive lottery analytics, web scraping, machine learning prediction, Transformer LLM fine-tuning, and REST service framework for Idaho Lottery and multi-state lottery games using Python and Apple's **MLX** machine learning framework.

Key capabilities include:

* **Automated Web Scraping:** Real-time historical draw scraping and CSV dataset generation across all games.
* **Apple Silicon MLX Deep Learning:** High-performance neural network architectures for probability distribution estimation and ticket scoring.
* **Autoregressive Transformer LLMs:** Fine-tune causal Transformer models on sequential lottery draw histories with local `.safetensors` model persistence and remote model loading.
* **Unified Multi-Game Winner Predictor:** Quick-run unified CLI tool (`next_winners.py`) for multi-game ticket ranking.
* **Specialized Ball Analyzers:** Deep statistical and neural models for Millionaire Ball (Extra 1–5), Mega Ball (1–24), and Star Ball (1–10).
* **Pattern & Tensor Dynamics:** Intra-draw consecutive runs, back-to-back draw repeats, odd/even and high/low distributions, and pair co-occurrence matrices.
* **Strict Duplicate Protection:** Guaranteed unique white ball generation and deduplicated ticket candidate pools.
* **REST API & Daily Scheduling:** FastAPI web endpoints and APScheduler automation.
* **Automated Unit Testing:** Full `unittest` test suite covering LLMs, pattern generators, and match predictors.

### Supported Games

* **Idaho Pick 3** (Digits 0–9 across 3 positions, Day/Night/Combo draws)
* **Idaho Pick 4** (Digits 0–9 across 4 positions, Exact/Any/Boxed number selection)
* **Idaho Cash** (5 of 45 white balls)
* **Lotto America** (5 of 52 white balls + 1 Star Ball 1–10)
* **Mega Millions** (5 of 70 white balls + 1 Mega Ball 1–24)
* **Millionaire for Life** (5 of 58 white balls + 1 Millionaire Ball 1–5)

---

# Project Structure

```text
.
├── idaho/
│   ├── pick3/
│   │   ├── scrape_idaho_pick3.py            # Historical draw scraper
│   │   ├── pick3_mlx_ticket_model.py        # MLX ticket model (Exact, 6-way, 3-way, Combo)
│   │   ├── pick3_llm_ticket_model.py        # Autoregressive Transformer LLM & fine-tuning
│   │   ├── pick3_mlx_model.py               # MLX neural network probability model
│   │   ├── pick3_mlx_model_with_top.py      # MLX model with top candidate filtering
│   │   ├── consective_numbers_mlx.py        # Consecutive patterns & number counts
│   │   ├── pick3_csv_web_service.py         # FastAPI REST service
│   │   ├── scheduler.py                     # Daily scraping job scheduler
│   │   ├── models/                          # Saved local Pick 3 LLM weights
│   │   └── data/
│   │       ├── idaho_pick3_history.csv
│   │       └── number_counts.csv
│   ├── pick4/
│   │   ├── scrape_idaho_pick4.py            # Historical draw scraper
│   │   ├── pick4_mlx_number_select_model.py # MLX model with Exact/Any/Boxed selection
│   │   ├── pick4_mlx_ticket_model.py        # MLX ticket scoring model
│   │   ├── consective_numbers_mlx.py        # Consecutive patterns & frequency analysis
│   │   └── data/
│   │       ├── idaho_pick4_history.csv
│   │       └── number_counts.csv
│   └── idaho_cash/
│       ├── scrape_idaho_cash.py             # Historical draw scraper
│       ├── idaho_cash_mlx_ticket_model.py   # MLX ticket ranking model
│       ├── generate_next_idaho_cash_mlx.py  # Next draw generator (Model, Balanced, Hot, Mixed)
│       ├── generate_number_counts.py        # Statistical hit and consecutive repeats counter
│       └── data/
│           ├── idaho_cash_history.csv
│           └── number_counts.csv
├── lotto_america/
│   ├── scrape_lotto_america.py              # Historical draw scraper
│   ├── lotto_america_mlx_ticket_model.py    # MLX ticket model & Star Ball predictor
│   └── data/
│       └── lotto_america_history.csv
├── mega_millions/
│   ├── scrape_mega_millions.py              # Historical draw scraper
│   ├── mega_millions_mlx_ticket_model.py    # MLX ticket ranking model
│   ├── megaball_count.py                    # Mega Ball frequency analysis & predictor
│   └── data/
│       └── mega_millions_history.csv
├── millionaire_life/
│   ├── scrape_millionaire_life.py           # Historical draw scraper
│   ├── millionaire_life_mlx_ticket_model.py # MLX ticket model with Extra number support
│   ├── millionaire_life_llm_ticket_model.py # Transformer LLM with Millionaire Ball spotlight
│   ├── predict_extra_mlx.py                 # MLX neural predictor for Millionaire Ball
│   ├── predict_5_extra.py                   # Multi-step Extra ball sequence predictor
│   ├── patterns/
│   │   ├── generate_next_millionaire_life_mlx.py # Next winner generator (No duplicate white balls)
│   │   ├── predict_millionaire_match_mlx.py      # Match-optimized predictor (Extra + White Ball)
│   │   ├── millionaire_patterns_mlx.py          # Tensor pattern analysis & pair co-occurrence
│   │   ├── generate_patterns_mlx.py             # Pattern generation utilities
│   │   └── consective_numbers_mlx.py            # Consecutive hits analysis
│   ├── notebooks/
│   │   └── FindingTheNextWinner.ipynb       # Interactive analysis and visualization notebook
│   ├── models/                              # Saved local Millionaire Life LLM weights
│   └── data/
│       ├── millionaire_life_history.csv
│       └── number_counts.csv
├── models/                                  # Pretrained base & fine-tuned LLM checkpoints
│   ├── pick3_llm_model/
│   │   ├── config.json
│   │   └── model.safetensors
│   └── millionaire_life_llm_model/
│       ├── config.json
│       └── model.safetensors
├── tests/                                   # Automated unit and integration test suite
│   ├── test_pick3_llm.py
│   ├── test_millionaire_life_llm.py
│   ├── test_generate_next_millionaire_life_mlx.py
│   ├── test_millionaire_patterns_mlx.py
│   └── test_predict_millionaire_match_mlx.py
├── next_winners.py                          # Unified multi-game winner generator CLI
├── main.py
├── LICENSE
└── README.md
```

---

# Requirements & Installation

## Python Version

Python 3.11+ is required.

```bash
python --version
```

## Install Dependencies

```bash
pip install pandas numpy requests beautifulsoup4 fastapi uvicorn safetensors apscheduler
```

### Apple Silicon MLX

MLX requires macOS on Apple Silicon (M1/M2/M3/M4):

```bash
pip install mlx
```

---

# Unified Multi-Game Winner Predictor

Use `next_winners.py` at the project root as a quick unified interface to train and rank the next best possible winning tickets across games using MLX neural networks and statistical gap analysis:

```bash
# Idaho Cash (default: 5 tickets)
python next_winners.py --game idaho_cash --tickets 10

# Pick 3
python next_winners.py --game pick3 --tickets 5 --epochs 200

# Pick 4
python next_winners.py --game pick4 --tickets 10 --window 20
```

---

# Idaho Pick 3

## 1. Scrape History

Scrape historical draws into `idaho/pick3/data/idaho_pick3_history.csv`:

```bash
python idaho/pick3/scrape_idaho_pick3.py
```

## 2. MLX Ticket Model

Train an MLX neural network and rank tickets using composite scoring:

```bash
# Night draw predictions (Exact order)
python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw Night \
    --tickets 10

# Day draw predictions
python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw Day \
    --tickets 10

# Any-Order / Boxed predictions
python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw both \
    --ticket-type any \
    --tickets 10

# 6-Way predictions (3 distinct digits)
python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw both \
    --ticket-type 6-way \
    --tickets 10

# 3-Way predictions (1 pair + 1 single digit)
python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw both \
    --ticket-type 3-way \
    --tickets 10

# Combined Day and Night training
python idaho/pick3/pick3_mlx_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw combo \
    --tickets 10
```

### Advanced Options:
* `--exclude-recent N`: Exclude tickets drawn in the last `N` drawings.
* `--weights model,freq,overdue,pair`: Customize score weighting (e.g. `0.5,0.2,0.2,0.1`).
* `--seed N`: Set random seed for reproducible runs.
* `--output FILE`: Save ranked results to a CSV file.

## 3. Autoregressive Transformer LLM Fine-Tuning

Instantiate a Transformer Language Model, fine-tune autoregressively on Pick 3 draw sequences, and generate ranked winning tickets:

```bash
# Combine Day and Night draws to predict next winning tickets (default: --draw combo)
python idaho/pick3/pick3_llm_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw combo \
    --tickets 5 \
    --epochs 8

# Fine-tune local model and generate Night winning tickets
python idaho/pick3/pick3_llm_ticket_model.py \
    --csv idaho/pick3/data/idaho_pick3_history.csv \
    --draw Night \
    --tickets 5 \
    --epochs 8

# Specify custom local model directory or remote model specification
python idaho/pick3/pick3_llm_ticket_model.py \
    --remote-model remote://lottery-ai/pick3-transformer-base \
    --local-model-dir models/pick3_llm_model \
    --draw combo \
    --ticket-type any \
    --tickets 10 \
    --output idaho/pick3/data/pick3_llm_predictions.csv
```

## 4. Consecutive Numbers & Statistical Analysis

Analyze consecutive pattern repeats, slot-wise repeats, and generate `idaho/pick3/data/number_counts.csv`:

```bash
python idaho/pick3/consective_numbers_mlx.py
```

---

# Idaho Pick 4

## 1. Scrape History

```bash
python idaho/pick4/scrape_idaho_pick4.py
```

Output: `idaho/pick4/data/idaho_pick4_history.csv`

## 2. MLX Number Select Model

Generate Pick 4 predictions with granular control over draw types, ticket strategies, and number selection modes:

```bash
# Exact order predictions (Night draw, model strategy)
python idaho/pick4/pick4_mlx_number_select_model.py \
    --csv idaho/pick4/data/idaho_pick4_history.csv \
    --draw Night \
    --number-select exact \
    --ticket-type model \
    --tickets 10

# Any order / balanced tickets
python idaho/pick4/pick4_mlx_number_select_model.py \
    --csv idaho/pick4/data/idaho_pick4_history.csv \
    --draw Night \
    --number-select any \
    --ticket-type balanced \
    --tickets 10

# Boxed number selection across Day and Night draws
python idaho/pick4/pick4_mlx_number_select_model.py \
    --csv idaho/pick4/data/idaho_pick4_history.csv \
    --draw both \
    --number-select boxed \
    --tickets 10

# Export predictions to CSV
python idaho/pick4/pick4_mlx_number_select_model.py \
    --csv idaho/pick4/data/idaho_pick4_history.csv \
    --draw Night \
    --number-select exact \
    --tickets 20 \
    --output idaho/pick4/data/pick4_predictions.csv
```

### Supported Strategies:
* `--ticket-type`: `model`, `balanced`, `hot`, `overdue`, `hot_overdue`
* `--number-select`: `exact`, `any`, `boxed`
* `--draw`: `Night`, `Day`, `both`, `combo`

## 3. Consecutive Numbers Analysis

```bash
python idaho/pick4/consective_numbers_mlx.py
```

---

# Idaho Cash

## 1. Scrape History

```bash
python idaho/idaho_cash/scrape_idaho_cash.py
```

Output: `idaho/idaho_cash/data/idaho_cash_history.csv`

## 2. Generate Next Winning Tickets

Generate tickets using MLX neural network probability modeling combined with balanced, hot, and overdue gap analysis:

```bash
# Generate tickets with balanced strategy
python idaho/idaho_cash/idaho_cash_mlx_ticket_model.py \
    --csv idaho/idaho_cash/data/idaho_cash_history.csv \
    --ticket-type balanced \
    --tickets 10

# Hot numbers strategy
python idaho/idaho_cash/idaho_cash_mlx_ticket_model.py \
    --ticket-type hot \
    --tickets 10

# Overdue numbers strategy
python idaho/idaho_cash/idaho_cash_mlx_ticket_model.py \
    --ticket-type overdue \
    --tickets 10
```

Alternative generator (`generate_next_idaho_cash_mlx.py`):

```bash
# Model, balanced, hot, overdue, or mixed strategies
python idaho/idaho_cash/generate_next_idaho_cash_mlx.py \
    --ticket-type mixed \
    --tickets 10
```

## 3. Generate Number Counts

Compute historical frequency, consecutive draw overlaps, and intra-draw statistics:

```bash
python idaho/idaho_cash/generate_number_counts.py
```

---

# Lotto America

## 1. Scrape History

```bash
python lotto_america/scrape_lotto_america.py
```

Output: `lotto_america/data/lotto_america_history.csv`

## 2. Generate Winning Tickets & Star Ball Predictions

Predict 5 white balls (1–52) and the bonus Star Ball (1–10):

```bash
# Balanced ticket strategy with balanced Star Ball selection
python lotto_america/lotto_america_mlx_ticket_model.py \
    --csv lotto_america/data/lotto_america_history.csv \
    --ticket-type balanced \
    --star-mode balanced \
    --tickets 10

# Hot + Overdue ticket strategy with overdue Star Ball
python lotto_america/lotto_america_mlx_ticket_model.py \
    --ticket-type hot_overdue \
    --star-mode overdue \
    --tickets 10
```

### Star Ball Prediction Modes:
* `--star-mode`: `default`, `balanced`, `overdue`, `less_hits`
* `--star-hit-window N`: Number of recent draws to evaluate for Star Ball hits (`0` for full history).
* `--star-top N`: Number of top Star Ball candidates to display.

```bash
python lotto_america/lotto_america_mlx_ticket_model.py \
    --star-hit-window 50 \
    --star-mode less_hits \
    --star-top 5
```

---

# Mega Millions

## 1. Scrape History

```bash
python mega_millions/scrape_mega_millions.py
```

Output: `mega_millions/data/mega_millions_history.csv`

## 2. MLX Ticket Model

```bash
# Balanced strategy
python mega_millions/mega_millions_mlx_ticket_model.py \
    --csv mega_millions/data/mega_millions_history.csv \
    --ticket-type balanced \
    --tickets 10

# Export ranked predictions to CSV
python mega_millions/mega_millions_mlx_ticket_model.py \
    --ticket-type model \
    --tickets 20 \
    --output mega_millions/data/mega_millions_predictions.csv
```

## 3. Mega Ball Frequency & Predictor

Analyze Mega Ball (1–24) frequencies, overdue draws, hit percentages, and predict upcoming candidates:

```bash
# Display full Mega Ball frequency table
python mega_millions/megaball_count.py

# Predict Mega Ball candidates across all modes
python mega_millions/megaball_count.py --predict-mode all --top 5

# Predict based on high, low, or balanced hits
python mega_millions/megaball_count.py --predict-mode balanced
python mega_millions/megaball_count.py --predict-mode low
```

---

# Millionaire for Life

## 1. Scrape History

```bash
python millionaire_life/scrape_millionaire_life.py
```

Output: `millionaire_life/data/millionaire_life_history.csv`

## 2. Transformer LLM Model (Millionaire Ball Spotlight)

Fine-tune an autoregressive Transformer language model on historical draws (5 white balls 1–58 and 1 Millionaire Ball 1–5) and produce tickets with a spotlight on the **Best Next Winning Millionaire Ball**:

```bash
# Fine-tune local LLM, spotlight top Millionaire Ball, and generate tickets
python millionaire_life/millionaire_life_llm_ticket_model.py \
    --csv millionaire_life/data/millionaire_life_history.csv \
    --tickets 5 \
    --epochs 10

# Prioritize top winning Millionaire Ball across generated tickets (mb_focused)
python millionaire_life/millionaire_life_llm_ticket_model.py \
    --ticket-type mb_focused \
    --tickets 10 \
    --epochs 10

# Load from remote specification / save local checkpoint and export to CSV
python millionaire_life/millionaire_life_llm_ticket_model.py \
    --remote-model remote://lottery-ai/millionaire-life-transformer-base \
    --local-model-dir models/millionaire_life_llm_model \
    --ticket-type balanced \
    --tickets 10 \
    --output millionaire_life/data/millionaire_life_predictions.csv
```

## 3. Strict Duplicate Protection Next Winner Generator

Generate next winning tickets using a multi-task MLX neural network with strict validation ensuring **no duplicate numbers** exist within any ticket and across candidate pools:

```bash
# Mixed strategy (Model + Balanced + Hot + Overdue)
python millionaire_life/patterns/generate_next_millionaire_life_mlx.py \
    --ticket-type mixed \
    --tickets 10

# Export deduplicated tickets to CSV
python millionaire_life/patterns/generate_next_millionaire_life_mlx.py \
    --ticket-type model \
    --tickets 20 \
    --output millionaire_life/data/millionaire_winning_tickets.csv
```

## 4. Match-Optimized Predictor (Extra + White Ball Match)

Train a deep learning model specifically optimized to maximize the joint probability of matching:
1. The winning Millionaire Ball (Extra 1–5), and
2. At least one (and multiple) White Balls (1–58).

Includes historical walk-forward backtesting evaluation:

```bash
# Predict next draw candidates optimized for target match
python millionaire_life/patterns/predict_millionaire_match_mlx.py \
    --tickets 10 \
    --epochs 120

# Run historical backtesting simulation
python millionaire_life/patterns/predict_millionaire_match_mlx.py \
    --backtest \
    --backtest-draws 10
```

## 5. Tensor Pattern Analysis & Co-Occurrence Matrix

Perform deep mathematical and tensor-based pattern discovery on historical draws:

* Main Ball & Extra Ball frequency and probability distributions
* Back-to-back consecutive draw repeats
* Intra-draw consecutive runs (e.g., 12-13, 34-35-36)
* Odd/Even and High/Low (1–29 vs 30–58) balance ratios
* Tens/decade and ending-digit distribution patterns
* **58x58 Pair Co-occurrence Matrix** computed via MLX matrix operations

```bash
# Generate full pattern report
python millionaire_life/patterns/millionaire_patterns_mlx.py

# Generate pattern-based tickets and save counts
python millionaire_life/patterns/millionaire_patterns_mlx.py \
    --generate-tickets 5 \
    --save-counts
```

## 6. Extra Number Neural Predictors

```bash
# Predict the single next Extra number
python millionaire_life/predict_extra_mlx.py --epochs 100

# Predict the multi-step sequence of the next 5 Extra hits
python millionaire_life/predict_5_extra.py --epochs 100
```

## 7. Interactive Jupyter Notebook

Explore statistical distributions and model predictions interactively:

```bash
jupyter notebook millionaire_life/notebooks/FindingTheNextWinner.ipynb
```

---

# REST API Service

A FastAPI microservice is available for serving Idaho Pick 3 historical data and CSV downloads.

### Start the Server:

```bash
uvicorn idaho.pick3.pick3_csv_web_service:app --reload --port 8000
```

### Endpoints:

* `GET /pick3` — Returns JSON summary of latest Pick 3 history.
* `GET /pick3/download` — Direct download of `idaho_pick3_history.csv`.

```text
http://127.0.0.1:8000/pick3
http://127.0.0.1:8000/pick3/download
```

---

# Automated Daily Workflow & Scheduling

### Automated Daily Runner Script (`scheduler.py`):

```bash
python idaho/pick3/scheduler.py
```

### Manual Full-Pipeline Execution:

```bash
# 1. Scrape all games
python idaho/pick3/scrape_idaho_pick3.py
python idaho/pick4/scrape_idaho_pick4.py
python idaho/idaho_cash/scrape_idaho_cash.py
python lotto_america/scrape_lotto_america.py
python mega_millions/scrape_mega_millions.py
python millionaire_life/scrape_millionaire_life.py

# 2. Run statistical counters
python idaho/pick3/consective_numbers_mlx.py
python idaho/pick4/consective_numbers_mlx.py
python idaho/idaho_cash/generate_number_counts.py

# 3. Generate daily prediction batches
python next_winners.py --game idaho_cash --tickets 10
python next_winners.py --game pick3 --tickets 10
python next_winners.py --game pick4 --tickets 10
python lotto_america/lotto_america_mlx_ticket_model.py --ticket-type balanced --tickets 10
python mega_millions/mega_millions_mlx_ticket_model.py --ticket-type balanced --tickets 10
python millionaire_life/patterns/generate_next_millionaire_life_mlx.py --ticket-type mixed --tickets 10
```

---

# Running the Test Suite

The project includes unit tests for the Transformer LLMs, pattern discovery algorithms, match predictors, and duplicate validation.

Run all tests:

```bash
python3 -m unittest discover tests
```

---

# Recommended Strategies

| Game | Conservative | Balanced | Aggressive | Target Match |
| :--- | :--- | :--- | :--- | :--- |
| **Pick 3** | 6-Way Boxed | 3-Way Boxed | Exact Order (Model) | LLM Transformer Combo |
| **Pick 4** | Any Order / Boxed | Exact Order (Balanced) | Model Strategy | Hot + Overdue |
| **Idaho Cash** | Balanced Pool | Hot + Overdue | Model Strategy | Mixed Blend |
| **Lotto America** | Balanced + Low Hits Star | Hot + Overdue | Model + Overdue Star | Balanced Star Ball |
| **Mega Millions** | Balanced + High MegaBall | Hot + Overdue | Model Strategy | Balanced MegaBall |
| **Millionaire for Life** | Balanced + Top MB | Hot + Overdue | `mb_focused` LLM | Match-Optimized Predictor |

---

# Disclaimer

Lottery drawings are random processes designed to ensure independence between drawings.

The machine learning models, Transformer LLMs, and statistical tools in this project:
* Analyze historical patterns and frequency distributions
* Evaluate overdue gap dynamics and digit co-occurrences
* Rank candidate tickets based on empirical and neural probabilities

They do not guarantee future lottery results. Use these tools for research, educational, and entertainment purposes only. Play responsibly.
