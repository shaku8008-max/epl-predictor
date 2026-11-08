"""Fit the linear goal-difference model (plus home/away goal heads) and export it as JSON.

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
    X = train[FEATURE_COLUMNS].to_numpy()
    home_goals = train["home_goals"].to_numpy()
    away_goals = train["away_goals"].to_numpy()

    old = LinearGoalsModel.load(MODEL_FILE) if MODEL_FILE.exists() else None
    model = LinearGoalsModel(FEATURE_COLUMNS).fit(X, home_goals - away_goals, home_goals, away_goals)

    probs = model.predict_probs(X)
    if old is not None:
        gap = np.abs(old.predict_probs(X) - probs).max()
        print(f"Probabilities vs the currently saved model: largest difference {gap:.2e}")
        assert gap < 1e-9, "the probabilities changed; the tested model must not change. Stop and check."

    goals = model.predict_goals(X)
    link = np.abs((goals[:, 0] - goals[:, 1]) - model.predict_mu(X)).max()
    print(f"Expected home goals minus away goals vs the expected margin: largest difference {link:.2e}")
    assert link < 1e-9, "home minus away does not equal the margin"

    model.save(MODEL_FILE)
    again = LinearGoalsModel.load(MODEL_FILE)
    assert np.array_equal(again.predict_probs(X), probs), "save and load changed the probabilities"
    assert np.array_equal(again.predict_goals(X), goals), "save and load changed the goals"

    print(f"\nTrained on {len(train)} matches, seasons {int(train['season'].min())}..{int(train['season'].max())}")
    print(f"Saved {MODEL_FILE} ({MODEL_FILE.stat().st_size} bytes). sigma {model.sigma:.4f}")
    print("\nExpected goals on the training rows (in-sample, so only a rough guide):")
    print(f"  home: predicted mean {goals[:, 0].mean():.3f} vs actual {home_goals.mean():.3f} | "
          f"range {goals[:, 0].min():.2f} to {goals[:, 0].max():.2f}")
    print(f"  away: predicted mean {goals[:, 1].mean():.3f} vs actual {away_goals.mean():.3f} | "
          f"range {goals[:, 1].min():.2f} to {goals[:, 1].max():.2f}")
    print(f"  rows with a predicted side below 0.1 goals: {int((goals < 0.1).any(axis=1).sum())} of {len(goals)}")


if __name__ == "__main__":
    main()
