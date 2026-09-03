import argparse

def main():
    parser = argparse.ArgumentParser(description="Pick 3 MLX Model")
    parser.add_argument("--csv", required=False, help="CSV file")
    parser.add_argument("--draw", choices=["Day", "Night"], default="Night")
    parser.add_argument("--top", type=int, default=1,
                        help="Number of ranked ticket predictions to return")
    args = parser.parse_args()

    print(f"Loading: {args.csv}")
    print(f"Draw Type: {args.draw}")
    print(f"Returning Top {args.top} ticket(s)")

    sample_predictions = [
        ("483", 0.92),
        ("482", 0.90),
        ("583", 0.88),
        ("584", 0.87),
        ("493", 0.85),
    ]

    print("\nPredictions:")
    for ticket, score in sample_predictions[:args.top]:
        print(f"{ticket}  score={score:.4f}")

if __name__ == "__main__":
    main()
