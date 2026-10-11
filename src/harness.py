import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import CLASSES, market_probs, score
from stack import STACK_SEASONS, base_models, walk_forward

DEFAULT_FILE = Path("data/processed/features.csv")


def league_average_walk_forward(df: pd.DataFrame, seasons: list[int]) -> np.ndarray:
    """For each season, predict the H, D, A rates of all earlier seasons."""
    out = np.full((len(df), 3), np.nan)
    for season in seasons:
        past = df[df["season"] < season]
        now = (df["season"] == season).to_numpy()
        out[now] = past["result"].value_counts(normalize=True).reindex(CLASSES).to_numpy()
    return out


def run(path: Path) -> None:
    df = pd.read_csv(path, parse_dates=["date"])
    scored_rows = df["season"].isin(STACK_SEASONS).to_numpy()
    scored = df[scored_rows]

    preds = {
        "league average": league_average_walk_forward(df, STACK_SEASONS),
        "bookmaker": market_probs(df),
    }
    base = {
        name: walk_forward(df, build, cols, STACK_SEASONS)
        for name, (build, cols) in base_models(df, with_odds=False).items()
    }
    preds.update(base)
    preds["average"] = np.mean(list(base.values()), axis=0)

    overall, by_season = [], {}
    for name, probs in preds.items():
        overall.append({"model": name, **score(scored["result"], probs[scored_rows])})
        by_season[name] = {
            season: score(
                df.loc[df["season"] == season, "result"],
                probs[(df["season"] == season).to_numpy()],
            )["log_loss"]
            for season in STACK_SEASONS
        }

    print(f"File: {path}")
    print(f"Seasons in file: {df['season'].min()} to {df['season'].max()}, {len(df.columns)} columns")
    print(f"\nWalk-forward, seasons {STACK_SEASONS[0]} to {STACK_SEASONS[-1]}:")
    print(pd.DataFrame(overall).round(4).to_string(index=False))
    print("\nLog loss by season:")
    print(pd.DataFrame(by_season).T.round(4).to_string())


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE)