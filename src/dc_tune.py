import itertools

import pandas as pd

from dixon_coles import DixonColes
from evaluate import load_features, score
from stack import walk_forward

SEASONS = list(range(2018, 2024))   # tuning seasons only. Never add 2024 or 2025 here.

df = load_features()
hist = pd.read_csv("data/processed/matches_2005.csv", parse_dates=["date"])
rows = df["season"].isin(SEASONS).to_numpy()

results = []
for xi, reg in itertools.product([0.0, 0.15, 0.3, 0.5, 0.8], [2.0, 5.0, 10.0, 20.0]):
    probs = walk_forward(df, lambda: DixonColes(reg=reg, xi=xi, history=hist), [], SEASONS)
    s = score(df.loc[rows, "result"], probs[rows])
    results.append({"xi": xi, "reg": reg, "log_loss": s["log_loss"], "accuracy": s["accuracy"]})
    print(f"xi={xi:<4} reg={reg:<5} log_loss {s['log_loss']:.4f}  accuracy {s['accuracy']:.4f}", flush=True)

table = pd.DataFrame(results).sort_values("log_loss").round(4)
print("\nBest first (old Poisson on the same seasons: 0.9624):")
print(table.to_string(index=False))