import argparse
from rich.console import Console

from config import HISTORICAL_DATA_PATH, USE_SIMULATION
from data_fetcher import simulate_historical_games, save_historical_data, load_historical_data
from trainer import train_model
from live_predictor import run_live_loop

console = Console()


def cmd_collect(args):
    console.print("[bold cyan]Collecting historical game data...")

    if USE_SIMULATION:
        df = simulate_historical_games()
        save_historical_data(df, HISTORICAL_DATA_PATH)
        console.print(f"[green]Saved simulated data to {HISTORICAL_DATA_PATH}")
    else:
        console.print("[red]Real API collection not implemented yet.")


def cmd_train(args):
    console.print("[bold cyan]Loading historical data...")
    df = load_historical_data(HISTORICAL_DATA_PATH)

    console.print(f"[green]Loaded {len(df)} games. Training model...")
    model, metrics = train_model(df)

    console.print("[bold green]Model training completed!")
    console.print(f"Accuracy: {metrics['accuracy']:.3f}")
    console.print(f"ROC AUC:  {metrics['roc_auc']:.3f}")


def cmd_live(args):

    console.print("[bold cyan]Starting live predictions...")
    run_live_loop(n_steps=args.n_steps, step_sleep=args.step_sleep)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Sports Performance Predictor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_collect = subparsers.add_parser("collect", help="Collect historical game data")
    p_collect.set_defaults(func=cmd_collect)

    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.set_defaults(func=cmd_train)

    p_live = subparsers.add_parser("live", help="Run the live prediction loop")
    p_live.add_argument("--n-steps", type=int, default=30, help="Number of prediction steps")
    p_live.add_argument("--step-sleep", type=float, default=2.0, help="Delay between predictions")
    p_live.set_defaults(func=cmd_live)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()