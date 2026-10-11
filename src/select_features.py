import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluate import class_index, load_features, split
from models import MODELS, BoostingModel, feature_columns, tune

OUT_FILE = Path("data/processed/selected_features.json")
FOLD_SEASONS = [2018, 2019, 2020, 2021, 2022]   # each is scored using only earlier seasons
MIN_FEATURES = 3


def season_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-in-time folds: fit on seasons before S, score on season S."""
    seasons = train["season"].to_numpy()
    return [(np.flatnonzero(seasons < s), np.flatnonzero(seasons == s)) for s in FOLD_SEASONS]


def run_rfecv(train: pd.DataFrame, cols: list[str]) -> RFECV:
    """Drop the weakest feature one at a time and score each feature count."""
    filled = SimpleImputer(strategy="median").fit_transform(train[cols])
    X = StandardScaler().fit_transform(filled)
    y = class_index(train["result"])
    selector = RFECV(
        LogisticRegression(C=0.03, max_iter=2000),
        step=1,
        cv=season_folds(train),
        scoring="neg_log_loss",
        min_features_to_select=MIN_FEATURES,
    )
    return selector.fit(X, y)


def boosting_importance(train: pd.DataFrame, cols: list[str]) -> pd.Series:
    """How much worse the log loss gets when each feature is shuffled."""
    last = max(FOLD_SEASONS)
    fit_part = train[train["season"] < last]
    held_out = train[train["season"] == last]
    model = BoostingModel().fit(fit_part, cols)
    result = permutation_importance(
        model.model,
        held_out[cols],
        class_index(held_out["result"]),
        scoring="neg_log_loss",
        n_repeats=20,
        random_state=0,
    )
    return pd.Series(result.importances_mean, index=cols).sort_values(ascending=False)


def main() -> None:
    parts = split(load_features())
    train, valid = parts["train"], parts["valid"]
    cols = feature_columns(train, with_odds=False)

    selector = run_rfecv(train, cols)
    selected = [c for c, keep in zip(cols, selector.support_) if keep]

    curve = pd.Series(
        -selector.cv_results_["mean_test_score"],
        index=pd.RangeIndex(MIN_FEATURES, len(cols) + 1, name="n_features"),
        name="cv_log_loss",
    )
    print("Cross-validated log loss by number of features:")
    print(curve.round(4).to_string())

    print(f"\nRFECV picked {len(selected)} of {len(cols)} features:")
    for c in selected:
        print(f"  {c}")

    print("\nBoosting permutation importance, top 12:")
    print(boosting_importance(train, cols).head(12).round(4).to_string())

    rows = []
    for label, subset in (("all", cols), ("selected", selected)):
        for model_class in MODELS:
            best = tune(model_class, train, valid, subset)
            params = best.pop("params")
            rows.append({
                "model": model_class.name,
                "features": f"{label} ({len(subset)})",
                **best,
                "params": str(params),
            })
    table = pd.DataFrame(rows).drop(columns="matches").round(4)
    print("\nValidation season 2023-24, no odds:")
    print(table.to_string(index=False))

    OUT_FILE.write_text(json.dumps({"no_odds": selected}, indent=2))
    print(f"\nSaved selected features to {OUT_FILE}")


if __name__ == "__main__":
    main()