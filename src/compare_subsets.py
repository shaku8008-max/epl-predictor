from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import market_probs, score
from models import BoostingModel, LogisticModel, PoissonModel, feature_columns
from stack import STACK_SEASONS, walk_forward

FEATURES_FILE = Path("data/processed/features_2005_ewm.csv")
FIRST_SEASON = 2014     # xG starts here, so every subset trains from the same point

# A few settings per model; the best one is reported for each feature set.
CANDIDATES = {
    "poisson": [lambda a=a: PoissonModel(alpha=a) for a in (1.0, 0.1, 0.01)],
    "logistic": [lambda c=c: LogisticModel(C=c) for c in (0.003, 0.03, 0.3)],
    "boosting": [lambda: BoostingModel(max_depth=2, max_iter=100)],
}


def main() -> None:
    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    df = df[df["season"] >= FIRST_SEASON].reset_index(drop=True)
    rows = df["season"].isin(STACK_SEASONS).to_numpy()
    results = df.loc[rows, "result"]

    all_cols = feature_columns(df)
    xg_cols = [c for c in all_cols if "xg" in c]
    elo_cols = [c for c in all_cols if c.startswith("elo")]
    subsets = {
        "xg only": xg_cols,
        "xg + elo": xg_cols + elo_cols,
        "everything": all_cols,
    }

    table = []
    for label, cols in subsets.items():
        line = {"features": label, "count": len(cols)}
        best_probs = []
        for name, builders in CANDIDATES.items():
            tried = [walk_forward(df, build, cols, STACK_SEASONS)[rows] for build in builders]
            losses = [score(results, p)["log_loss"] for p in tried]
            best = int(np.argmin(losses))
            line[name] = losses[best]
            best_probs.append(tried[best])
        average = score(results, np.mean(best_probs, axis=0))
        line["average"] = average["log_loss"]
        line["home_acc"] = average["home_acc"]
        table.append(line)

    print(f"Walk-forward {STACK_SEASONS[0]} to {STACK_SEASONS[-1]}, trained from {FIRST_SEASON}:")
    print(pd.DataFrame(table).round(4).to_string(index=False))
    print(f"\nBookmaker on the same matches: {score(results, market_probs(df)[rows])['log_loss']:.4f}")


if __name__ == "__main__":
    main()