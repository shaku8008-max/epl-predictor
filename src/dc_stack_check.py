import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dixon_coles import DixonColes
from evaluate import class_index, load_features, score
from stack import base_models, walk_forward

SEASONS = list(range(2018, 2024))   # tuning seasons only. Never add 2024 or 2025 here.
FIT, CHECK = [2018, 2019, 2020], [2021, 2022, 2023]

df = load_features()
hist = pd.read_csv("data/processed/matches_2005.csv", parse_dates=["date"])

preds = {
    name: walk_forward(df, build, cols, SEASONS)
    for name, (build, cols) in base_models(df, with_odds=False).items()
}
dc = walk_forward(df, lambda: DixonColes(reg=3.0, xi=0.5, history=hist), [], SEASONS)

y = class_index(df["result"])
fit_rows = df["season"].isin(FIT).to_numpy()
chk_rows = df["season"].isin(CHECK).to_numpy()
truth = df.loc[chk_rows, "result"]


def stacked(models: dict) -> np.ndarray:
    X = np.hstack([np.log(np.clip(p, 1e-6, 1)) for p in models.values()])
    stacker = LogisticRegression(C=1.0, max_iter=2000).fit(X[fit_rows], y[fit_rows])
    return stacker.predict_proba(X[chk_rows])


print("Scored on 2021-2023 (stacker learned from 2018-2020):")
for name, p in {**preds, "dixon_coles": dc}.items():
    s = score(truth, p[chk_rows])
    print(f"  {name:12s} alone     log_loss {s['log_loss']:.4f}  accuracy {s['accuracy']:.4f}")

for label, models in (("stack without DC", preds), ("stack with DC", {**preds, "dixon_coles": dc})):
    s = score(truth, stacked(models))
    print(f"  {label:17s}  log_loss {s['log_loss']:.4f}  accuracy {s['accuracy']:.4f}")