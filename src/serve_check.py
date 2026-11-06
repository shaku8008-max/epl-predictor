import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate import class_index, score
from models import feature_columns
from stack import base_models, walk_forward
from train_final import (FALLBACK_FEATURES, LEAN_CANDIDATES, LEAN_FEATURES,
                         LEAN_FIRST_SEASON, TUNE_SEASONS, stack_inputs)

ALL_SEASONS = list(range(2018, 2026))
STACK_FIT = list(range(2018, 2024))   # stacker learns from these only, never 2024-25
TEST = [2024, 2025]
KEYS = ["date", "home", "away"]
NN_FILE = "../epl_nn/data/processed/network_test_preds.csv"
OLD_FILE = "../epl_nn/data/processed/old_models_test_preds.csv"


def honest_stack(df: pd.DataFrame, specs: dict) -> pd.DataFrame:
    """Pick settings on the tuning seasons, fit the stacker on 2018-2023, return test predictions."""
    y = class_index(df["result"])
    tune = df["season"].isin(TUNE_SEASONS).to_numpy()
    fit = df["season"].isin(STACK_FIT).to_numpy()
    test = df["season"].isin(TEST).to_numpy()

    walk = []
    for name, (candidates, cols) in specs.items():
        best = None
        for setting, build in candidates:
            probs = walk_forward(df, build, cols, ALL_SEASONS)
            loss = score(df.loc[tune, "result"], probs[tune])["log_loss"]
            if best is None or loss < best[0]:
                best = (loss, setting, probs)
        print(f"  {name:9s} {best[1]}", flush=True)
        walk.append(best[2])

    X = stack_inputs(walk)
    stacker = LogisticRegression(C=1.0, max_iter=2000).fit(X[fit], y[fit])
    out = df.loc[test, KEYS + ["result"]].reset_index(drop=True)
    out[["p_home", "p_draw", "p_away"]] = stacker.predict_proba(X[test])
    return out


def tag(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Keep the match keys and rename the three probability columns."""
    out = frame[KEYS].copy()
    out[[f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]] = frame[["p_home", "p_draw", "p_away"]].to_numpy()
    return out


# Lean: xG and Elo only, from 2014 (same set-up as train_final.py).
print("Lean:")
lean_df = pd.read_csv(LEAN_FEATURES, parse_dates=["date"])
lean_df = lean_df[lean_df["season"] >= LEAN_FIRST_SEASON].reset_index(drop=True)
all_cols = feature_columns(lean_df)
lean_cols = [c for c in all_cols if "xg" in c] + [c for c in all_cols if c.startswith("elo")]
lean_specs = {name: (cands, lean_cols) for name, cands in LEAN_CANDIDATES.items()}
lean = honest_stack(lean_df, lean_specs)

# Fallback: every no-xG feature, from 2005.
print("Fallback:")
fb_df = pd.read_csv(FALLBACK_FEATURES, parse_dates=["date"])
fb_specs = {
    name: ([("stage 5 setting", build)], cols)
    for name, (build, cols) in base_models(fb_df, with_odds=False).items()
}
fallback = honest_stack(fb_df, fb_specs)
print(f"test rows: lean {len(lean)}, fallback {len(fallback)}")

# Join on the match, never on row position.
net = pd.read_csv(NN_FILE, parse_dates=["date"])
old = pd.read_csv(OLD_FILE, parse_dates=["date"])[KEYS + ["result", "mkt_home", "mkt_draw", "mkt_away"]]

lean_t = tag(lean, "lean")
lean_t["result"] = lean["result"]
df = lean_t.merge(tag(fallback, "fb"), on=KEYS, validate="one_to_one")
df = df.merge(tag(net, "nn"), on=KEYS, validate="one_to_one")
df = df.merge(old.rename(columns={"result": "result_old"}), on=KEYS, validate="one_to_one")
assert len(df) == 760, f"expected 760 matched rows, got {len(df)} (team names may differ between files)"
assert (df["result"] == df["result_old"]).all(), "results disagree, the join is wrong"

y = class_index(df["result"])
models = {
    "lean (fresh xG)": df[["lean_home", "lean_draw", "lean_away"]].to_numpy(),
    "fallback (no xG)": df[["fb_home", "fb_draw", "fb_away"]].to_numpy(),
    "neural network": df[["nn_home", "nn_draw", "nn_away"]].to_numpy(),
    "bookmaker": df[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy(),
}


def per_match_loss(p):
    p = np.clip(p, 1e-15, 1)
    return -np.log(p[np.arange(len(y)), y])


losses = {name: per_match_loss(p) for name, p in models.items()}
print("\nTest seasons 2024-25 and 2025-26, 760 matches:")
for name, p in models.items():
    print(f"  {name:17s} log_loss {losses[name].mean():.4f}  accuracy {(p.argmax(axis=1) == y).mean():.4f}")

rng = np.random.default_rng(0)
idx = rng.integers(0, len(y), size=(10_000, len(y)))


def compare(a, b):
    """Positive gain = model a has lower log loss than model b."""
    diff = losses[b] - losses[a]
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    print(f"  {a} vs {b}: gain {diff.mean():+.4f}  95% interval [{low:+.4f}, {high:+.4f}]")


print("\nPaired bootstrap (positive gain = first model is better):")
compare("fallback (no xG)", "lean (fresh xG)")
compare("neural network", "lean (fresh xG)")
compare("neural network", "fallback (no xG)")