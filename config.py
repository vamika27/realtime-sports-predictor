import os
from dotenv import load_dotenv

load_dotenv()

USE_SIMULATION = True

DATA_DIR = "data"
MODELS_DIR = "models"
HISTORICAL_DATA_PATH = os.path.join(DATA_DIR, "historical_games.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "game_outcome_model.pkl")

SPORTS_API_KEY = os.getenv("SPORTS_API_KEY", "")
SPORTS_API_BASE_URL = os.getenv("SPORTS_API_BASE_URL", "")

N_SIMULATED_GAMES = 2000
RANDOM_SEED = 42
TEST_SIZE = 0.2