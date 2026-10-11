import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate import calibration, class_index, league_average_probs, load_features, market_probs, score
from models import BoostingModel, LogisticModel, PoissonModel, feature_columns

SELECTED_FILE = Path("data/processed/selected_features.json")
STACK_SEASONS = list(range(2018, 2024))   # stacker learns from these walk-forward predictions
TEST_SEASONS = [2024, 2025]


def base_models(df: pd.DataFrame, with_odds: bool) -> dict:
    """Name -> (function that builds the model, feature columns). Settings come from stage 4."""
    cols = feature_columns(df, with_odds)
    elo_cols = json.loads(SELECTED_FILE.read_text())["no_odds"]
    if with_odds:
        return {
            "poisson": (lambda: PoissonModel(alpha=0.1), cols),
            "logistic": (lambda: LogisticModel(C=0.01), cols),
            "boosting": (lambda: BoostingModel(max_depth=2, max_iter=100), cols),
            "elo_only": (lambda: LogisticModel(C=0.1), elo_cols),
        }
    return {
        "poisson": (lambda: PoissonModel(alpha=1.0), cols),
        "logistic": (lambda: LogisticModel(C=0.003), cols),
        "boosting": (lambda: BoostingModel(max_depth=2, max_iter=100), cols),
        "elo_only": (lambda: LogisticModel(C=0.1), elo_cols),
    }


def walk_forward(df: pd.DataFrame, build, cols: list[str], seasons: list[int]) -> np.ndarray:
    """For each season, fit on all earlier seasons and predict that season."""
    out = np.full((len(df), 3), np.nan)
    for season in seasons:
        past = (df["season"] < season).to_numpy()
        now = (df["season"] == season).to_numpy()
        model = build().fit(df[past], cols)
        out[now] = model.predict_proba(df[now])
    return out


def run_stack(df: pd.DataFrame, with_odds: bool) -> dict[str, np.ndarray]:
    """Return test-season probabilities for each base model, their average and the stacker."""
    seasons = STACK_SEASONS + TEST_SEASONS
    preds = {
        name: walk_forward(df, build, cols, seasons)
        for name, (build, cols) in base_models(df, with_odds).items()
    }

    stack_rows = df["season"].isin(STACK_SEASONS).to_numpy()
    test_rows = df["season"].isin(TEST_SEASONS).to_numpy()

    X = np.hstack([np.log(np.clip(p, 1e-6, 1)) for p in preds.values()])
    y = class_index(df["result"])
    stacker = LogisticRegression(C=1.0, max_iter=2000).fit(X[stack_rows], y[stack_rows])

    results = {name: p[test_rows] for name, p in preds.items()}
    results["average"] = np.mean([p[test_rows] for p in preds.values()], axis=0)
    results["stacker"] = stacker.predict_proba(X[test_rows])
    return results


def main() -> None:
    df = load_features()
    test = df[df["season"].isin(TEST_SEASONS)]
    before_test = df[df["season"] < min(TEST_SEASONS)]

    finals = {
        "league average": league_average_probs(before_test, len(test)),
        "bookmaker": market_probs(test),
    }
    rows = [{"model": name, "odds": "-", **score(test["result"], probs)} for name, probs in finals.items()]

    for with_odds in (False, True):
        label = "yes" if with_odds else "no"
        results = run_stack(df, with_odds)
        for name, probs in results.items():
            rows.append({"model": name, "odds": label, **score(test["result"], probs)})
        finals[f"stacker ({label} odds)"] = results["stacker"]

    table = pd.DataFrame(rows).drop(columns="matches").round(4)
    print("Test seasons 2024-25 and 2025-26:")
    print(table.to_string(index=False))

    print("\nLog loss by test season:")
    by_season = {}
    for name, probs in finals.items():
        by_season[name] = {
            season: score(test.loc[test["season"] == season, "result"], probs[(test["season"] == season).to_numpy()])["log_loss"]
            for season in TEST_SEASONS
        }
    print(pd.DataFrame(by_season).T.round(4).to_string())

    print("\nStacker (no odds) calibration on test (home win):")
    print(calibration(test["result"], finals["stacker (no odds)"]))


if __name__ == "__main__":
    main()