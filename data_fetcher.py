import os
import time
from typing import List, Dict, Any, Generator

import numpy as np
import pandas as pd
import requests

from config import (
    USE_SIMULATION,
    HISTORICAL_DATA_PATH,
    SPORTS_API_KEY,
    SPORTS_API_BASE_URL,
    N_SIMULATED_GAMES,
    RANDOM_SEED,
)


def simulate_historical_games(n_games: int = N_SIMULATED_GAMES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    team_a_offense = rng.normal(loc=0.0, scale=1.0, size=n_games)
    team_a_defense = rng.normal(loc=0.0, scale=1.0, size=n_games)
    team_b_offense = rng.normal(loc=0.0, scale=1.0, size=n_games)
    team_b_defense = rng.normal(loc=0.0, scale=1.0, size=n_games)

    team_a_fatigue = rng.uniform(0, 1, size=n_games)
    team_b_fatigue = rng.uniform(0, 1, size=n_games)

    team_a_strength = 1.2 * team_a_offense + 0.8 * team_a_defense - 0.5 * team_a_fatigue
    team_b_strength = 1.2 * team_b_offense + 0.8 * team_b_defense - 0.5 * team_b_fatigue

    margin = team_a_strength - team_b_strength
    prob_a_win = 1 / (1 + np.exp(-margin / 2.0))
    a_wins = rng.binomial(1, prob_a_win)

    df = pd.DataFrame(
        {
            "team_a_offense": team_a_offense,
            "team_a_defense": team_a_defense,
            "team_b_offense": team_b_offense,
            "team_b_defense": team_b_defense,
            "team_a_fatigue": team_a_fatigue,
            "team_b_fatigue": team_b_fatigue,
            "team_a_win": a_wins,
            "prob_a_win_true": prob_a_win,
        }
    )

    return df


def save_historical_data(df: pd.DataFrame, path: str = HISTORICAL_DATA_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_historical_data(path: str = HISTORICAL_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No historical data found at {path}. Run `python main.py collect` first."
        )
    return pd.read_csv(path)


def _headers() -> Dict[str, str]:
    if not SPORTS_API_KEY:
        return {}
    return {
        "Authorization": f"Bearer {SPORTS_API_KEY}",
        "Accept": "application/json",
    }


def fetch_historical_games_from_api(season: str = "2024") -> pd.DataFrame:
    raise NotImplementedError("Implement this if you add a real sports API.")


def simulate_live_game_stream(
    n_steps: int = 50, step_sleep: float = 2.0, seed: int = RANDOM_SEED
) -> Generator[Dict[str, Any], None, None]:
    rng = np.random.default_rng(seed)

    a_off = rng.normal(0.5, 0.3)
    a_def = rng.normal(0.3, 0.3)
    b_off = rng.normal(0.4, 0.3)
    b_def = rng.normal(0.4, 0.3)

    a_fatigue = 0.2
    b_fatigue = 0.2

    for t in range(n_steps):
        a_fatigue = min(1.0, a_fatigue + rng.uniform(0.0, 0.03))
        b_fatigue = min(1.0, b_fatigue + rng.uniform(0.0, 0.03))

        frame = {
            "time_step": t,
            "team_a_offense": a_off + rng.normal(0, 0.1),
            "team_a_defense": a_def + rng.normal(0, 0.1),
            "team_b_offense": b_off + rng.normal(0, 0.1),
            "team_b_defense": b_def + rng.normal(0, 0.1),
            "team_a_fatigue": a_fatigue,
            "team_b_fatigue": b_fatigue,
        }

        yield frame
        time.sleep(step_sleep)


def fetch_live_game_state_from_api() -> Dict[str, Any]:
    raise NotImplementedError("Implement this when using a real API.")
