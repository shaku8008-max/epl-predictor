import pandas as pd

from dixon_coles import DixonColes
from evaluate import load_features, score
from stack import base_models, walk_forward

SEASONS = list(range(2018, 2024))   # tuning seasons only. Never add 2024 or 2025 here.

df = load_features()
hist = pd.read_csv("data/processed/matches_2005.csv", parse_dates=["date"])

dc = walk_forward(df, lambda: DixonColes(history=hist), [], SEASONS)
build, cols = base_models(df, with_odds=False)["poisson"]
old = walk_forward(df, build, cols, SEASONS)

rows = df["season"].isin(SEASONS).to_numpy()
print("Seasons 2018-2023, walk-forward:")
for name, probs in (("dixon_coles", dc), ("old poisson", old)):
    s = score(df.loc[rows, "result"], probs[rows])
    print(f"  {name:12s} log_loss {s['log_loss']:.4f}  accuracy {s['accuracy']:.4f}")

model = DixonColes(history=hist).fit(df[df["season"] < 2024])
print(f"\nFitted before 2024: home edge {model.edge:.3f}, base rate {model.mu:.3f}")
table = model.ratings()
current = set(df.loc[df["season"] == 2023, "home"])
table = table[table.index.isin(current)]
print("\nTop 5 and bottom 5 of the 2023-24 teams:")
print(pd.concat([table.head(5), table.tail(5)]).to_string())