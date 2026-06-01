# Lotto America MLX Model

Train an MLX model using historical Lotto America drawings.

Generate ranked ticket candidates based on:

- ML probability
- Hot numbers
- Overdue numbers
- Pair frequency
- Number balance
- Star Ball probability

## Generate 5 Tickets

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --tickets 5
```

## Balanced Strategy

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type balanced \
    --tickets 10
```

## Hot Numbers Strategy

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type hot \
    --tickets 10
```

## Overdue Strategy

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type overdue \
    --tickets 10
```

## Hot + Overdue Strategy

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --ticket-type hot_overdue \
    --tickets 10
```

## Export Results

```bash
python lotto_america_mlx_ticket_model.py \
    --csv lotto_america_history.csv \
    --tickets 20 \
    --output lotto_america_predictions.csv
```