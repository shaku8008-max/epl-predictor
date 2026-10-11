from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_FILE = Path("data/processed/features.csv")

CLASSES = ["H", "D", "A"]          # column order for every probability array

SPLITS = {
    "train": range(2015, 2023),    # 2015-16 to 2022-23
    "valid": [2023],               # 2023-24
    "test": [2024, 2025],          # 2024-25 and 2025-26
    "live": [2026],                # season in progress, never used for fitting or scoring
}


def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_FILE, parse_dates=["date"])


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split by season and confirm the parts do not overlap in time."""
    parts = {name: df[df["season"].isin(seasons)].copy() for name, seasons in SPLITS.items()}
    order = ["train", "valid", "test", "live"]
    for earlier, later in zip(order, order[1:]):
        assert parts[earlier]["date"].max() < parts[later]["date"].min(), f"{earlier} overlaps {later}"
    return parts


def class_index(results: pd.Series) -> np.ndarray:
    """Map H, D, A to 0, 1, 2."""
    return results.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


def score(results: pd.Series, probs: np.ndarray) -> dict:
    """Score an (n, 3) array of H, D, A probabilities against the real results."""
    y = class_index(results)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-15, 1)
    p_true = probs[np.arange(len(y)), y]
    home_won = y == 0
    return {
        "matches": len(y),
        "log_loss": -np.log(p_true).mean(),
        "accuracy": (probs.argmax(axis=1) == y).mean(),
        "home_acc": ((probs[:, 0] > 0.5) == home_won).mean(),
    }


def calibration(results: pd.Series, probs: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Compare predicted home-win probability with how often the home team won."""
    table = pd.DataFrame({
        "predicted": np.asarray(probs)[:, 0],
        "actual": class_index(results) == 0,
    })
    table["bin"] = pd.cut(table["predicted"], np.linspace(0, 1, bins + 1), include_lowest=True)
    return (
        table.groupby("bin", observed=True)
        .agg(matches=("actual", "size"), predicted=("predicted", "mean"), actual=("actual", "mean"))
        .round(3)
    )


def league_average_probs(train: pd.DataFrame, n: int) -> np.ndarray:
    """Baseline 1: the training seasons' H, D, A rates for every match."""
    rates = train["result"].value_counts(normalize=True).reindex(CLASSES).to_numpy()
    return np.tile(rates, (n, 1))


def market_probs(df: pd.DataFrame) -> np.ndarray:
    """Baseline 2: bookmaker probabilities."""
    return df[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy()


def main() -> None:
    parts = split(load_features())

    print("Split sizes:")
    for name, part in parts.items():
        print(f"  {name:5s} {len(part):5d} matches, {part['date'].min().date()} to {part['date'].max().date()}")

    rows = []
    for name in ("valid", "test"):
        part = parts[name]
        baselines = {
            "league average": league_average_probs(parts["train"], len(part)),
            "bookmaker": market_probs(part),
        }
        for model, probs in baselines.items():
            rows.append({"model": model, "split": name, **score(part["result"], probs)})

    print("\nBaselines:")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\nBookmaker calibration on test (home win):")
    print(calibration(parts["test"]["result"], market_probs(parts["test"])))


if __name__ == "__main__":
    main()