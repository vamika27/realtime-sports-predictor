import numpy as np
import pandas as pd
import os

INPUT_PATH = "data/sports_betting_predictive_analysis.csv"
OUTPUT_PATH = "data/historical_games.csv"

RANDOM_SEED = 42


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Make sure the Kaggle CSV is in the data/ folder."
        )

    df = pd.read_csv(INPUT_PATH)

    df = df.dropna(subset=["Home_Team_Odds", "Away_Team_Odds", "Actual_Winner"])

    df = df[df["Actual_Winner"] != "Draw"].copy()

    home_odds = df["Home_Team_Odds"].astype(float)
    away_odds = df["Away_Team_Odds"].astype(float)

    draw_odds = df["Draw_Odds"].fillna(1e9).astype(float)

    implied_home = 1.0 / home_odds
    implied_away = 1.0 / away_odds
    implied_draw = 1.0 / draw_odds

    total = implied_home + implied_away + implied_draw
    prob_home = implied_home / total

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df)

    team_a_offense = implied_home
    team_a_defense = implied_home
    team_b_offense = implied_away
    team_b_defense = implied_away
    team_a_fatigue = rng.uniform(0, 1, size=n)
    team_b_fatigue = rng.uniform(0, 1, size=n)

    team_a_win = (df["Actual_Winner"] == df["Home_Team"]).astype(int)

    out = pd.DataFrame(
        {
            "team_a_offense": team_a_offense,
            "team_a_defense": team_a_defense,
            "team_b_offense": team_b_offense,
            "team_b_defense": team_b_defense,
            "team_a_fatigue": team_a_fatigue,
            "team_b_fatigue": team_b_fatigue,
            "team_a_win": team_a_win,
            "prob_a_win_true": prob_home,
        }
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote processed dataset with {len(out)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
