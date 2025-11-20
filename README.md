# Real-Time Sports Performance Predictor

This project builds a **real-time predictive model** using **Python, machine learning, and live-style data streams** to estimate the probability that a team will win a game.  
It uses a real historical dataset of sports betting odds to train the model and then runs a simulated live feed to generate **real-time win probability updates**.

---

## Tech Stack

- Python
- Pandas, NumPy
- scikit-learn (Logistic Regression model)
- Requests (for future real API integration)
- `python-dotenv` for environment variables
- `rich` for a nice CLI dashboard

---

## Project Structure

```text
realtime-sports-predictor/
  data/
    sports_betting_predictive_analysis.csv   # raw Kaggle data
    historical_games.csv                     # processed training data
  models/
    game_outcome_model.pkl                   # trained model
  config.py
  data_fetcher.py
  featurizer.py
  live_predictor.py
  main.py
  prepare_kaggle_data.py                     # converts Kaggle data to model-ready data
  trainer.py
  requirements.txt
  .env.example
  README.md
