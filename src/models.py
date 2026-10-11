import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import class_index, league_average_probs, load_features, market_probs, score, split

ID_COLS = ["match_id", "date", "season", "home", "away"]
TARGET_COLS = ["home_goals", "away_goals", "result", "home_win"]
MARKET_COLS = ["mkt_home", "mkt_draw", "mkt_away"]


def feature_columns(df: pd.DataFrame, with_odds: bool = False) -> list[str]:
    """Every pre-match column, with or without the bookmaker probabilities."""
    exclude = ID_COLS + TARGET_COLS + ([] if with_odds else MARKET_COLS)
    return [c for c in df.columns if c not in exclude]


def scaled(estimator):
    """Fill blanks with the training median, standardise, then fit."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def outcome_probs(lam_home: np.ndarray, lam_away: np.ndarray, max_goals: int = 10) -> np.ndarray:
    """Turn expected goals into H, D, A probabilities by summing over scorelines."""
    goals = np.arange(max_goals + 1)
    p_home = poisson.pmf(goals[None, :], lam_home[:, None])
    p_away = poisson.pmf(goals[None, :], lam_away[:, None])
    grid = p_home[:, :, None] * p_away[:, None, :]      # [match, home goals, away goals]
    home = np.tril(grid, k=-1).sum(axis=(1, 2))
    draw = np.trace(grid, axis1=1, axis2=2)
    away = np.triu(grid, k=1).sum(axis=(1, 2))
    probs = np.column_stack([home, draw, away])
    return probs / probs.sum(axis=1, keepdims=True)


class PoissonModel:
    name = "poisson"
    grid = [{"alpha": a} for a in (0.01, 0.1, 1.0, 3.0, 10.0)]

    def __init__(self, alpha: float = 0.1):
        self.home = scaled(PoissonRegressor(alpha=alpha, max_iter=300))
        self.away = scaled(PoissonRegressor(alpha=alpha, max_iter=300))

    def fit(self, df: pd.DataFrame, cols: list[str]):
        self.cols = cols
        self.home.fit(df[cols], df["home_goals"])
        self.away.fit(df[cols], df["away_goals"])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return outcome_probs(self.home.predict(df[self.cols]), self.away.predict(df[self.cols]))


class LogisticModel:
    name = "logistic"
    grid = [{"C": c} for c in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)]

    def __init__(self, C: float = 0.03):
        self.pipe = scaled(LogisticRegression(C=C, max_iter=2000))

    def fit(self, df: pd.DataFrame, cols: list[str]):
        self.cols = cols
        self.pipe.fit(df[cols], class_index(df["result"]))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(df[self.cols])


class BoostingModel:
    name = "boosting"
    grid = [{"max_depth": d, "max_iter": n} for d in (2, 3) for n in (50, 100, 200, 400)]

    def __init__(self, max_depth: int = 2, max_iter: int = 100):
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_depth=max_depth,
            max_iter=max_iter,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        )

    def fit(self, df: pd.DataFrame, cols: list[str]):
        self.cols = cols
        self.model.fit(df[cols], class_index(df["result"]))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.cols])


MODELS = [PoissonModel, LogisticModel, BoostingModel]


def tune(model_class, train: pd.DataFrame, valid: pd.DataFrame, cols: list[str]) -> dict:
    """Fit every setting in the model's grid and keep the best on validation log loss."""
    best = None
    for params in model_class.grid:
        model = model_class(**params).fit(train, cols)
        result = score(valid["result"], model.predict_proba(valid))
        if best is None or result["log_loss"] < best["log_loss"]:
            best = {**result, "params": params}
    return best


def main() -> None:
    parts = split(load_features())
    train, valid = parts["train"], parts["valid"]

    rows = []
    baselines = {
        "league average": league_average_probs(train, len(valid)),
        "bookmaker": market_probs(valid),
    }
    for name, probs in baselines.items():
        rows.append({"model": name, "odds": "-", **score(valid["result"], probs), "params": ""})

    for with_odds in (False, True):
        cols = feature_columns(train, with_odds)
        print(f"{'with' if with_odds else 'without'} odds: {len(cols)} features") 
        for model_class in MODELS:
            best = tune(model_class, train, valid, cols)
            params = best.pop("params")
            rows.append({
                "model": model_class.name,
                "odds": "yes" if with_odds else "no",
                **best,
                "params": str(params),
            })

    table = pd.DataFrame(rows).drop(columns="matches").round(4)
    print("\nValidation season 2023-24:")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()