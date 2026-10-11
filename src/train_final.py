from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate import class_index, market_probs, score
from models import BoostingModel, LogisticModel, PoissonModel, feature_columns
from stack import base_models, walk_forward

LEAN_FEATURES = Path("data/processed/features_2005_ewm.csv")   # built with xG
FALLBACK_FEATURES = Path("data/processed/features_2005.csv")   # built without xG
MODEL_DIR = Path("models")

LEAN_FIRST_SEASON = 2014                  # xG starts here
TUNE_SEASONS = list(range(2018, 2024))    # settings are chosen on these
STACK_SEASONS = list(range(2018, 2026))   # stacker learns from these walk-forward predictions
LIVE_SEASON = 2026                        # season in progress, never used for fitting

LEAN_CANDIDATES = {
    "poisson": [(f"alpha={a}", lambda a=a: PoissonModel(alpha=a)) for a in (1.0, 0.1, 0.01)],
    "logistic": [(f"C={c}", lambda c=c: LogisticModel(C=c)) for c in (0.003, 0.03, 0.3)],
    "boosting": [("depth=2, iter=100", lambda: BoostingModel(max_depth=2, max_iter=100))],
}


def stack_inputs(probs_list: list[np.ndarray]) -> np.ndarray:
    """Log-probabilities of every base model side by side, in a fixed order."""
    return np.hstack([np.log(np.clip(p, 1e-6, 1)) for p in probs_list])


def train(label: str, df: pd.DataFrame, specs: dict) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """specs: model name -> (list of (setting label, builder), feature columns)."""
    y = class_index(df["result"])
    stack_rows = df["season"].isin(STACK_SEASONS).to_numpy()
    tune_rows = df["season"].isin(TUNE_SEASONS).to_numpy()

    print(f"\n{label}: seasons {df['season'].min()} to {df['season'].max()}")
    chosen, walk = {}, []
    for name, (candidates, cols) in specs.items():
        best = None
        for setting, build in candidates:
            probs = walk_forward(df, build, cols, STACK_SEASONS)
            loss = score(df.loc[tune_rows, "result"], probs[tune_rows])["log_loss"]
            if best is None or loss < best[0]:
                best = (loss, setting, build, probs)
        loss, setting, build, probs = best
        print(f"  {name:9s} {setting:18s} {len(cols):3d} features, log loss {loss:.4f}")
        chosen[name] = (setting, build, cols)
        walk.append(probs[stack_rows])

    # Stacker, fitted on honest walk-forward predictions.
    stacker = LogisticRegression(C=1.0, max_iter=2000).fit(stack_inputs(walk), y[stack_rows])

    # Base models, fitted on every finished season.
    history = df[df["season"] < LIVE_SEASON]
    fitted = {name: build().fit(history, cols) for name, (_, build, cols) in chosen.items()}
    print(f"  final fit on {len(history)} matches, {history['date'].min().date()} to {history['date'].max().date()}")

    live = df[df["season"] == LIVE_SEASON]
    live_probs = stacker.predict_proba(stack_inputs([m.predict_proba(live) for m in fitted.values()]))

    bundle = {
        "name": label,
        "version": "v1",
        "trained_on": date.today().isoformat(),
        "trained_through_season": LIVE_SEASON - 1,
        "settings": {name: setting for name, (setting, _, _) in chosen.items()},
        "base_models": fitted,
        "stacker": stacker,
    }
    return bundle, live, live_probs


def main() -> None:
    # Lean: xG and Elo only, from 2014.
    lean_df = pd.read_csv(LEAN_FEATURES, parse_dates=["date"])
    lean_df = lean_df[lean_df["season"] >= LEAN_FIRST_SEASON].reset_index(drop=True)
    all_cols = feature_columns(lean_df)
    lean_cols = [c for c in all_cols if "xg" in c] + [c for c in all_cols if c.startswith("elo")]
    lean_specs = {name: (candidates, lean_cols) for name, candidates in LEAN_CANDIDATES.items()}
    lean, live, lean_probs = train("lean", lean_df, lean_specs)

    # Fallback: every no-xG feature, from 2005, stage 5 settings.
    fb_df = pd.read_csv(FALLBACK_FEATURES, parse_dates=["date"])
    fb_specs = {
        name: ([("stage 5 setting", build)], cols)
        for name, (build, cols) in base_models(fb_df, with_odds=False).items()
    }
    fallback, fb_live, fb_probs = train("fallback", fb_df, fb_specs)

    rows = [
        {"model": "bookmaker", **score(live["result"], market_probs(live))},
        {"model": "lean", **score(live["result"], lean_probs)},
        {"model": "fallback", **score(fb_live["result"], fb_probs)},
    ]
    print(f"\nSeason {LIVE_SEASON} so far (too few matches to judge):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    recent = live[["date", "home", "away", "result"]].tail(8).copy()
    recent[["p_home", "p_draw", "p_away"]] = lean_probs[-8:].round(3)
    print("\nMost recent lean predictions:")
    print(recent.to_string(index=False))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for bundle in (lean, fallback):
        path = MODEL_DIR / f"{bundle['name']}_v1.joblib"
        joblib.dump(bundle, path)
        print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()