import json
import sys
from math import factorial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import ELO_NEW_TEAM, add_elo
from train_final import stack_inputs


def group_of(col: str) -> str:
    """The table row a model column belongs to: both teams' versions of a stat share one."""
    if col.startswith("elo"):
        return "elo"
    if col.startswith(("home_", "away_")):
        return col[5:]
    return "other"


def home_win_prob(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Same path predict.py uses: base models, then the stacker. Column 0 is a home win."""
    models = bundle["base_models"]
    inputs = stack_inputs([m.predict_proba(frame) for m in models.values()])
    return bundle["stacker"].predict_proba(inputs)[:, 0]


def explain(bundle: dict, pairs: pd.DataFrame) -> pd.DataFrame:
    """Exact Shapley contribution of each stat row to the home win chance, in points.

    Baseline is every input at the league average (so Elo difference is 0).
    Each stat row is switched from the baseline to the pair's real values, in every
    possible order, and the average effect is kept. Contributions sum to
    (model home win chance - baseline home win chance).
    """
    cols = next(iter(bundle["base_models"].values())).cols
    groups: dict[str, list[int]] = {}
    for i, col in enumerate(cols):
        groups.setdefault(group_of(col), []).append(i)
    names = list(groups)
    k = len(names)
    n = len(pairs)

    actual = pairs[cols].to_numpy(dtype=float)
    base = actual.mean(axis=0)

    # use_actual[mask, column]: True when that column takes the pair's real value.
    masks = np.arange(2 ** k)
    use_actual = np.zeros((len(masks), len(cols)), dtype=bool)
    for j, name in enumerate(names):
        for mask in masks:
            if mask >> j & 1:
                use_actual[mask, groups[name]] = True

    stacked = np.where(use_actual[:, None, :], actual[None, :, :], base[None, None, :])
    frame = pd.DataFrame(stacked.reshape(-1, len(cols)), columns=cols)
    v = home_win_prob(bundle, frame).reshape(len(masks), n)   # v[mask, pair]

    contrib = np.zeros((n, k))
    for j in range(k):
        for mask in masks:
            if mask >> j & 1:
                continue
            size = bin(mask).count("1")
            weight = factorial(size) * factorial(k - size - 1) / factorial(k)
            contrib[:, j] += weight * (v[mask | (1 << j)] - v[mask])

    out = pairs[["home", "away"]].reset_index(drop=True).copy()
    out["start"] = v[0]
    for j, name in enumerate(names):
        out[name] = contrib[:, j] * 100          # percentage points
    out["end"] = v[-1]
    return out


def write_explain(bundle: dict, pairs: pd.DataFrame, updated_at: str, out_file: Path) -> int:
    """Explain every pairing and save the result for upload. Returns the row count."""
    result = explain(bundle, pairs)
    stat_cols = [c for c in result.columns if c not in ("home", "away", "start", "end")]
    rows = [
        {
            "home": row["home"],
            "away": row["away"],
            "start_pct": round(row["start"] * 100, 4),
            "end_pct": round(row["end"] * 100, 4),
            "steps": {c: round(row[c], 4) for c in stat_cols},
            "model": bundle["name"],
            "model_version": bundle["version"],
            "updated_at": updated_at,
        }
        for row in result.to_dict("records")
    ]
    out_file.write_text(json.dumps(rows))
    return len(rows)


def main() -> None:
    import predict  # imported here because predict.py imports this file

    bundle = joblib.load(predict.MODEL_FILE)
    cols = next(iter(bundle["base_models"].values())).cols

    current = predict.understat_table(predict.refresh_data())
    matches, _ = predict.load_history(current)
    teams = sorted(set(current["home"]) | set(current["away"]))
    _, elo = add_elo(matches.copy())
    ratings = {team: elo.get(team, ELO_NEW_TEAM) for team in teams}
    state = predict.team_state(matches, teams)
    pairs = predict.build_pairs(state, ratings, teams, cols)

    if len(sys.argv) == 3:
        home, away = sys.argv[1], sys.argv[2]
    else:
        nxt = current[~current["played"]].sort_values("date").iloc[0]
        home, away = nxt["home"], nxt["away"]

    result = explain(bundle, pairs)

    # Check 1: the last step must equal what predict.py would publish.
    direct = home_win_prob(bundle, pairs)
    print(f"Max gap to predict.py home win chance: {np.abs(result['end'].to_numpy() - direct).max():.2e}")

    row = result[(result["home"] == home) & (result["away"] == away)]
    if row.empty:
        print(f"No pairing {home} v {away}. Teams: {', '.join(teams)}")
        return
    row = row.iloc[0]

    stat_cols = [c for c in result.columns if c not in ("home", "away", "start", "end")]
    steps = row[stat_cols].sort_values(key=abs, ascending=False)

    print(f"\n{home} v {away}: home win chance")
    print(f"  {'Average match':22s} {row['start'] * 100:6.1f}%")
    for name, points in steps.items():
        print(f"  {name:22s} {points:+6.1f}")
    print(f"  {'Model':22s} {row['end'] * 100:6.1f}%")
    print(f"\nSteps add to {steps.sum():+.2f}, start to end is {(row['end'] - row['start']) * 100:+.2f}")


if __name__ == "__main__":
    main()