from typing import Dict, Any

from rich.console import Console
from rich.table import Table

from config import USE_SIMULATION
from data_fetcher import simulate_live_game_stream, fetch_live_game_state_from_api
from featurizer import featurize_live_frame
from trainer import load_model

console = Console()


def _print_prediction_frame(t: int, features: Dict[str, Any], prob_a_win: float):
    table = Table(title=f"Live Game Prediction - Time Step {t}")

    table.add_column("Feature", justify="left")
    table.add_column("Value", justify="right")

    for k, v in features.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.3f}")
        else:
            table.add_row(k, str(v))

    table.add_row("Predicted P(Team A Win)", f"{prob_a_win:.3f}")

    console.print(table)


def run_live_loop(n_steps: int = 30, step_sleep: float = 2.0):
    model = load_model()

    console.rule("[bold green]Starting Real-Time Sports Performance Predictor")

    if USE_SIMULATION:
        stream = simulate_live_game_stream(n_steps=n_steps, step_sleep=step_sleep)

        for frame in stream:
            t = frame["time_step"]
            features = {k: v for k, v in frame.items() if k != "time_step"}

            X_live = featurize_live_frame(features)

            prob_a_win = float(model.predict_proba(X_live)[0, 1])

            _print_prediction_frame(t, features, prob_a_win)
    else:
        import time

        t = 0
        while True:
            frame = fetch_live_game_state_from_api()
            features = frame

            X_live = featurize_live_frame(features)
            prob_a_win = float(model.predict_proba(X_live)[0, 1])

            _print_prediction_frame(t, features, prob_a_win)

            t += 1
            time.sleep(step_sleep)