# Repository guidance

## Project overview

This is a Python lottery analytics repository with historical draw scrapers,
statistical analysis, Apple MLX neural models, Transformer ticket generators,
and a small FastAPI service. Predictions are experimental rankings, not
guarantees of future results.

## Layout

- `idaho/pick3/`, `idaho/pick4/`, and `idaho/idaho_cash/`: game-specific scrapers,
  models, and historical CSV data.
- `lotto_america/`, `mega_millions/`, and `millionaire_life/`: multi-state game
  implementations and datasets.
- `millionaire_life/patterns/`: pattern analysis, ticket generation, and
  match prediction.
- `next_winners.py`: unified CLI for Idaho Cash, Pick 3, and Pick 4.
- `tests/`: standard-library `unittest` tests with synthetic draw fixtures.
- `models/` and game-local `models/`: saved model configuration and weights.
- `README.md`: installation instructions and game-specific CLI examples.

## Environment and commands

Use Python 3.11 or newer. An existing `.venv/` is available; prefer its Python
when it has the required dependencies. MLX code requires macOS on Apple Silicon.
There is no root dependency manifest or configured lint/format tool; follow the
installation instructions in `README.md` rather than assuming one exists.

Run commands from the repository root unless a script requires another working
directory. Common commands:

```sh
# Full test suite
python3 -m unittest discover -s tests

# Focused test file
python3 -m unittest discover -s tests -p 'test_pick3_llm.py'

# Unified prediction example (trains a model)
python3 next_winners.py --game pick3 --tickets 5 --epochs 2

# Pick 3 API
python3 -m uvicorn idaho.pick3.pick3_csv_web_service:app --reload --port 8000
```

The API currently reads `idaho_pick3_history.csv` relative to its working
directory. Check `CSV_FILE` and the scraper's output location when working on
the service; do not assume the documented game data directory is used.

## Implementation conventions

- Follow the surrounding Python style: four-space indentation, snake_case
  functions and variables, PascalCase classes, and uppercase constants. Preserve
  existing script names and public CLI arguments, including historical spellings.
- Keep game rules and CSV schemas specific to each game. Pick 3/4 digits are
  positional and may repeat; five-white-ball games require unique main numbers
  and separate bonus-ball validation. Check the implementation's constants before
  changing number ranges or ticket constraints.
- Preserve chronological ordering and Day/Night handling when preparing draw
  histories. Backtests must train only on draws preceding the evaluated draw.
- Preserve seed controls, ticket deduplication, model save/load compatibility,
  and CSV output contracts when modifying model or generation code.
- Check how each script resolves paths. Prefer explicit `--csv` and output paths
  for local experiments, and script-relative defaults for new bundled-data paths.
- Update `README.md` when changing supported commands or observable behavior.

## Validation and repository hygiene

- Run the relevant `unittest` files for behavioral changes, then the full suite
  when shared model or data-loading behavior changes. If MLX or other dependencies
  prevent validation, report that limitation explicitly.
- Use small synthetic datasets, short training runs, fixed seeds, and temporary
  directories for tests. Avoid network calls and writing to bundled checkpoints
  or historical datasets during verification.
- Scrapers, schedulers, and training CLIs can fetch data, run for a long time, or
  overwrite outputs. Inspect their side effects before using them as checks.
- Keep unrelated working-tree changes intact. Do not include `.DS_Store`,
  `__pycache__`, virtual environments, generated predictions, or model weights in
  code changes unless the task specifically concerns those artifacts.
- Treat historical CSVs, notebooks, and saved `.safetensors` files as project
  assets; avoid regenerating or rewriting them for unrelated changes.
