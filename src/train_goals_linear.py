"""Fit the linear goal-difference model on every completed season and export it as JSON.

Same 29 inputs as the neural network, so the shared feature builder (nn_features.build_pairs)
feeds both. The JSON is committed, like models/goals_nn_v1.npz.

Run from ~/epl_webapp/epl-predictor:   python src/train_goals_linear.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from goals_linear import LinearGoalsModel
from nn_infer import GoalsPredictor
from train_final import LIVE_SEASON

FEATURES_FILE = Path("data/processed/features_2005.csv")
MODEL_FILE = Path("models/goals_linear_v1.json")

_FORM = [f"{s}_last5" for s in
         ("points", "goals_for", "goals_against", "shots_for", "shots_against", "sot_for", "sot_against")]
_EXTRA = (["rest_days"] + [f"{s}_venue5" for s in ("points", "goals_for", "goals_against")]
          + ["goal_diff_ewm20", "goal_diff_ewm40"])
FEATURE_COLUMNS = (
    ["elo_home", "elo_away", "elo_diff"]
    + [f"{side}_{f}" for side in ("home", "away") for f in _FORM]
    + [f"{side}_{f}" for side in ("home", "away") for f in _EXTRA]
)


def main() -> None:
    assert FEATURE_COLUMNS == GoalsPredictor().columns, \
        "feature list differs from the neural network's; the shared builder would feed the wrong columns"

    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    missing = [c for c in FEATURE_COLUMNS + ["season", "home_goals", "away_goals", "result"] if c not in df.columns]
    assert not missing, f"{FEATURES_FILE} is missing columns: {missing}"

    train = df[df["season"] < LIVE_SEASON]          # the live season is never used for fitting
    goal_diff = (train["home_goals"] - train["away_goals"]).to_numpy()
    X = train[FEATURE_COLUMNS].to_numpy()
    model = LinearGoalsModel(FEATURE_COLUMNS).fit(X, goal_diff)

    model.save(MODEL_FILE)
    again = LinearGoalsModel.load(MODEL_FILE)
    probs = model.predict_probs(X)
    assert np.array_equal(again.predict_probs(X), probs), "save and load changed the predictions"

    print(f"Trained on {len(train)} matches, seasons {int(train['season'].min())}..{int(train['season'].max())}")
    print(f"Saved {MODEL_FILE} ({MODEL_FILE.stat().st_size} bytes). sigma {model.sigma:.4f}, intercept {model.intercept:+.4f}")
    print("\nSanity check, in-sample so only a rough guide (average predicted vs actual):")
    actual = train["result"].map({"H": 0, "D": 1, "A": 2}).value_counts(normalize=True).sort_index().to_numpy()
    print(f"  predicted H/D/A {probs.mean(axis=0).round(3)}")
    print(f"  actual    H/D/A {actual.round(3)}")


if __name__ == "__main__":
    main()
