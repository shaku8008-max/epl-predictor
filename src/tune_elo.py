import itertools
from pathlib import Path

import pandas as pd

from evaluate import score
from features import add_elo
from models import LogisticModel
from stack import STACK_SEASONS, walk_forward

IN_FILE = Path("data/processed/matches_2005.csv")
ELO_COLS = ["elo_home", "elo_away", "elo_diff"]

GRID = {
    "k": (10, 15, 20, 25, 30),
    "home_adv": (40, 60, 80),
    "carry": (1.0, 0.9, 0.8, 0.7),
    "margin_power": (0.0, 0.5, 1.0),
    "new_team": (1350, 1400, 1450),
}


def elo_log_loss(matches: pd.DataFrame, **params) -> float:
    """Walk-forward log loss of a logistic model that sees only the Elo columns."""
    df, _ = add_elo(matches.copy(), **params)
    probs = walk_forward(df, lambda: LogisticModel(C=0.1), ELO_COLS, STACK_SEASONS)
    rows = df["season"].isin(STACK_SEASONS).to_numpy()
    return score(df.loc[rows, "result"], probs[rows])["log_loss"]


def main() -> None:
    matches = pd.read_csv(IN_FILE, parse_dates=["date"])
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)

    print(f"Current settings: {elo_log_loss(matches):.4f}")

    combos = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    rows = []
    for i, params in enumerate(combos, start=1):
        rows.append({**params, "log_loss": elo_log_loss(matches, **params)})
        if i % 60 == 0:
            print(f"  {i}/{len(combos)} done")

    table = pd.DataFrame(rows).sort_values("log_loss").round(4)
    print("\nBest 10 settings:")
    print(table.head(10).to_string(index=False))
    print("\nWorst 3 settings:")
    print(table.tail(3).to_string(index=False))

    print("\nAverage log loss for each value of each setting:")
    for name in GRID:
        print(table.groupby(name)["log_loss"].mean().round(4).to_string())
        print()


if __name__ == "__main__":
    main()