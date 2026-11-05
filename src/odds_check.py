import numpy as np
from sklearn.linear_model import LogisticRegression

from evaluate import class_index, load_features, market_probs, score
from stack import base_models, walk_forward

SEASONS = list(range(2018, 2024))   # tuning seasons only. Never add 2024 or 2025 here.
FIT, CHECK = [2018, 2019, 2020], [2021, 2022, 2023]

df = load_features()
y = class_index(df["result"])
fit_rows = df["season"].isin(FIT).to_numpy()
chk_rows = df["season"].isin(CHECK).to_numpy()
truth = df.loc[chk_rows, "result"]

mkt = market_probs(df)
print("matches with missing odds in 2018-2023:", int(np.isnan(mkt[df["season"].isin(SEASONS).to_numpy()]).any(axis=1).sum()))

preds = {}
for with_odds in (False, True):
    preds[with_odds] = {
        name: walk_forward(df, build, cols, SEASONS)
        for name, (build, cols) in base_models(df, with_odds).items()
    }


def stacked(models: dict) -> np.ndarray:
    X = np.hstack([np.log(np.clip(p, 1e-6, 1)) for p in models.values()])
    stacker = LogisticRegression(C=1.0, max_iter=2000).fit(X[fit_rows], y[fit_rows])
    return stacker.predict_proba(X[chk_rows])


def show(label, probs):
    s = score(truth, probs)
    print(f"  {label:26s} log_loss {s['log_loss']:.4f}  accuracy {s['accuracy']:.4f}")


print("\nScored on 2021-2023 (stacker learned from 2018-2020):")
show("bookmaker alone", mkt[chk_rows])
for name, p in preds[True].items():
    if name != "elo_only":   # elo_only gets Elo columns only, with or without odds
        show(f"{name} (with odds)", p[chk_rows])
show("stack WITHOUT odds", stacked(preds[False]))
show("stack WITH odds", stacked(preds[True]))