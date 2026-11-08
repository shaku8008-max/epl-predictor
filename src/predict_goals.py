"""Predict every pairing with the linear goal-difference model; write goals_predictions.csv.

Mirrors predict_nn.py and shares its feature builder (nn_features.build_pairs), so the
neural network and this model see identical inputs.

Columns: the three probabilities, expected_diff (expected home goals minus away goals) and
xg_home / xg_away (expected goals per side). The per-side numbers are display only: home
minus away equals expected_diff exactly, and the probabilities use only expected_diff.

Run from ~/epl_webapp/epl-predictor:   python src/predict_goals.py
"""
import contextlib
import io
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

import download_data
from goals_linear import LinearGoalsModel
from nn_features import MATCHES_FILE, build_pairs
from predict_nn import NEAR_DAYS, ROUND_DAYS, load_fixtures
from train_final import LIVE_SEASON

OUT_DIR = Path("data/predictions")
OUT_FILE = OUT_DIR / "goals_predictions.csv"
NN_FILE = OUT_DIR / "nn_predictions.csv"
MODEL_FILE = Path("models/goals_linear_v1.json")
MODEL_NAME = "goals_linear"
MODEL_VERSION = "v1"


def main() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        download_data.main(2005, MATCHES_FILE)           # football-data results, quietly
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["date"])
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)

    model = LinearGoalsModel.load(MODEL_FILE)
    if not model.has_goal_heads:
        raise SystemExit(f"{MODEL_FILE} has no goal heads: run python src/train_goals_linear.py and commit it")
    this_season = matches[matches["season"] == LIVE_SEASON]
    teams = sorted(set(this_season["home"]) | set(this_season["away"]))
    if len(teams) != 20:
        raise SystemExit(f"expected 20 teams this season, found {len(teams)}: {teams}")

    # Fixtures, minus any pairing football-data already shows as played this season.
    fixtures = load_fixtures(teams)
    played = this_season[["home", "away"]].assign(played=True)
    fixtures = fixtures.merge(played, on=["home", "away"], how="left")
    fixtures = fixtures[fixtures["played"].isna()].drop(columns="played")

    pairs = pd.DataFrame(list(permutations(teams, 2)), columns=["home", "away"])
    pairs = pairs.merge(fixtures, on=["home", "away"], how="left")
    pairs["fixture_date"] = pd.to_datetime(pairs["fixture_date"])

    last_result = matches["date"].max()
    near = pairs["fixture_date"].where(pairs["fixture_date"] <= last_result + pd.Timedelta(days=NEAR_DAYS))
    inputs = build_pairs(
        matches,
        pd.DataFrame({"home": pairs["home"], "away": pairs["away"], "date": near}),
        model.columns,
        teams,
    )

    X = inputs[model.columns].to_numpy(dtype=float)
    probs = model.predict_probs(X)
    assert probs.shape == (len(pairs), 3) and np.isfinite(probs).all(), "bad probabilities"
    assert np.allclose(probs.sum(axis=1), 1), "probabilities do not sum to 1"
    expected_diff = model.predict_mu(X)
    goals = model.predict_goals(X)
    assert goals.shape == (len(pairs), 2) and np.isfinite(goals).all(), "bad expected goals"
    assert np.abs((goals[:, 0] - goals[:, 1]) - expected_diff).max() < 1e-9, "home minus away != margin"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out = pairs[["home", "away"]].copy()
    out[["p_home", "p_draw", "p_away"]] = probs.round(4)
    out[["xg_home", "xg_away"]] = goals.round(3)
    out["expected_diff"] = expected_diff.round(3)
    out["fixture_date"] = pairs["fixture_date"].dt.strftime("%Y-%m-%d")
    out["model"] = MODEL_NAME
    out["model_version"] = MODEL_VERSION
    out["predicted_at"] = now
    assert len(out) == 380, f"expected 380 pairings, got {len(out)}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    upcoming = out.dropna(subset=["fixture_date"]).sort_values("fixture_date")
    print(f"Results through {last_result.date()} | {len(teams)} teams, {len(out)} pairings, "
          f"{len(upcoming)} upcoming fixtures | blank model inputs: {int(inputs.isna().sum().sum())}")
    print(f"Expected goals across all pairings: home {goals[:, 0].min():.2f} to {goals[:, 0].max():.2f}, "
          f"away {goals[:, 1].min():.2f} to {goals[:, 1].max():.2f} | "
          f"pairings with a side below 0.1: {int((goals < 0.1).any(axis=1).sum())}")
    print(f"Saved {OUT_FILE}")

    if len(upcoming):
        first = pd.Timestamp(upcoming["fixture_date"].min())
        cutoff = (first + pd.Timedelta(days=ROUND_DAYS)).strftime("%Y-%m-%d")
        print("\nNext round:")
        print(upcoming[upcoming["fixture_date"] <= cutoff][
            ["fixture_date", "home", "away", "p_home", "p_draw", "p_away", "xg_home", "xg_away", "expected_diff"]
        ].to_string(index=False))

    # Feed sanity check: another results-only model should be broadly similar, not identical.
    if NN_FILE.exists():
        cols = ["p_home", "p_draw", "p_away", "xg_home", "xg_away"]
        nn = pd.read_csv(NN_FILE)[["home", "away"] + cols]
        both = out.merge(nn, on=["home", "away"], suffixes=("", "_nn"), validate="one_to_one")
        print("\nAgreement with the neural network's predictions (all 380 pairings):")
        for c in cols:
            gap = (both[c] - both[f"{c}_nn"]).abs().mean()
            print(f"  {c}: average gap {gap:.3f} | goals model mean {both[c].mean():.3f}, "
                  f"neural network mean {both[f'{c}_nn'].mean():.3f}")


if __name__ == "__main__":
    main()
