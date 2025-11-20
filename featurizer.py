from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split

from config import TEST_SIZE, RANDOM_SEED

FEATURE_COLUMNS = [
    "team_a_offense",
    "team_a_defense",
    "team_b_offense",
    "team_b_defense",
    "team_a_fatigue",
    "team_b_fatigue",
]

TARGET_COLUMN = "team_a_win"


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def featurize_live_frame(frame: Dict[str, Any]) -> pd.DataFrame:
    data = {col: [frame[col]] for col in FEATURE_COLUMNS}
    return pd.DataFrame(data)