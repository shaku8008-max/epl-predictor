"""Season-phase diagnostic (dev only, no training, no hold-out).
Rule: notes/phase_decision_rule.md (written before running).
Run from epl-predictor:  python src/phase_check.py
"""
import numpy as np
import pandas as pd

PROC = "../epl_nn/data/processed/"
N_BOOT = 10_000
SEED = 0
BINS = [0, 127, 253, 380]
LABELS = ["early", "mid", "late"]
CLASS_INDEX = {"H": 0, "D": 1, "A": 2}


def parse_dates(s):
    s = s.astype(str)
    if s.str.contains("/").any():
        return pd.to_datetime(s, dayfirst=True)
    return pd.to_datetime(s)


def match_loss(probs, result):
    """Per-match log loss: -log(prob given to what happened)."""
    idx = result.map(CLASS_INDEX).to_numpy()
    p = probs[np.arange(len(idx)), idx]
    return -np.log(np.clip(p, 1e-12, 1.0))


# ---- 1. load and merge (exactly 4,560 rows) ----
a = pd.read_csv(PROC + "dev_logistic_preds.csv")
b = pd.read_csv(PROC + "goals_check_preds.csv")
print("results values:", sorted(a["result"].unique()))
assert set(a["result"].unique()) <= set(CLASS_INDEX), "unexpected result codes"

keep_b = ["date", "home", "away",
          "logistic_home", "logistic_draw", "logistic_away",
          "linear_home", "linear_draw", "linear_away",
          "poisson_home", "poisson_draw", "poisson_away"]
df = a.merge(b[keep_b], on=["date", "home", "away"],
             how="inner", validate="one_to_one")
assert len(df) == 4560, f"expected 4560 rows, got {len(df)}"

# ---- 2. phases by date rank within each season ----
df["date"] = parse_dates(df["date"])
df = df.sort_values(["season", "date"], kind="stable").reset_index(drop=True)
df["rank"] = df.groupby("season").cumcount() + 1
counts = df.groupby("season").size()
assert (counts == 380).all(), f"season sizes not all 380:\n{counts}"
df["phase"] = pd.cut(df["rank"], bins=BINS, labels=LABELS)
print("\nmatches per phase:\n", df["phase"].value_counts()[LABELS].to_string())

# sanity: date ranges of the phases (2020 COVID season shown separately)
chk = df.groupby(["season", "phase"], observed=True)["date"].agg(["min", "max"])
print("\n2020 season phase date ranges:\n", chk.loc[2020].to_string())

# ---- 3. per-match losses ----
res = df["result"]
loss = {
    "model":   match_loss(df[["p_home", "p_draw", "p_away"]].to_numpy(), res),
    "market":  match_loss(df[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy(), res),
    "logi2":   match_loss(df[["logistic_home", "logistic_draw", "logistic_away"]].to_numpy(), res),
    "linear":  match_loss(df[["linear_home", "linear_draw", "linear_away"]].to_numpy(), res),
    "poisson": match_loss(df[["poisson_home", "poisson_draw", "poisson_away"]].to_numpy(), res),
}
print("\noverall mean log loss (sanity check):")
for k, v in loss.items():
    print(f"  {k:8s} {v.mean():.4f}")

gap = loss["model"] - loss["market"]            # positive = bookmaker ahead
goals_gain = loss["logi2"] - loss["poisson"]    # positive = poisson better
lin_gain = loss["logi2"] - loss["linear"]


def boot_means(x, rng):
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    return x[idx].mean(axis=1)


def ci(v):
    return np.percentile(v, [2.5, 97.5])


# ---- 4. per-phase numbers and PRIMARY TEST ----
rng = np.random.default_rng(SEED)
phase = df["phase"].to_numpy()
boots_gap = {}
print("\n=== Context: per-phase numbers (95% bootstrap intervals) ===")
for ph in LABELS:
    m = phase == ph
    bg = boot_means(gap[m], rng)
    boots_gap[ph] = bg
    lo, hi = ci(bg)
    print(f"{ph:6s} n={m.sum()}  gap {gap[m].mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]")

print("\n=== Context: goals-model gain over logistic, per phase ===")
for ph in LABELS:
    m = phase == ph
    for name, g in (("poisson", goals_gain), ("linear", lin_gain)):
        bg = boot_means(g[m], rng)
        lo, hi = ci(bg)
        print(f"{ph:6s} {name:8s} gain {g[m].mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]")

diff = boots_gap["early"] - boots_gap["late"]
lo, hi = ci(diff)
obs = gap[phase == "early"].mean() - gap[phase == "late"].mean()
print("\n=== PRIMARY TEST: (early gap) - (late gap) ===")
print(f"observed difference {obs:+.4f}   95% interval [{lo:+.4f}, {hi:+.4f}]")
if lo > 0:
    print("RESULT: phase matters (bookmaker's lead is larger early).")
elif hi < 0:
    print("RESULT: phase matters (bookmaker's lead is larger late).")
else:
    print("RESULT: no detectable phase effect (interval includes 0).")
