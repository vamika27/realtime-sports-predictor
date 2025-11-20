import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from config import MODELS_DIR, MODEL_PATH
from featurizer import prepare_training_data


def train_model(df):

    X_train, X_test, y_train, y_test = prepare_training_data(df)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics = {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model, metrics


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python main.py train` after we finish wiring up main.py."
        )

    model = joblib.load(MODEL_PATH)
    return model
