import json
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

from nn_infer import GoalsPredictor

# Which group each model input belongs to. Home and away versions of a stat share a group.
GROUP_STATS = {
    "recent_form": {"points_last5", "goals_for_last5", "goals_against_last5"},
    "shots": {"shots_for_last5", "shots_against_last5", "sot_for_last5", "sot_against_last5"},
    "venue_form": {"points_venue5", "goals_for_venue5", "goals_against_venue5"},
    "goal_difference": {"goal_diff_ewm20", "goal_diff_ewm40"},
    "rest_days": {"rest_days"},
}


def group_of(col: str) -> str:
    if col.startswith("elo"):
        return "elo"
    stat = col[5:] if col.startswith(("home_", "away_")) else col
    for group, stats in GROUP_STATS.items():
        if stat in stats:
            return group
    raise KeyError(f"input column {col!r} has no explain group: add it to GROUP_STATS")


def home_win_prob(model: GoalsPredictor, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_probs(frame)[:, 0]


def explain(model: GoalsPredictor, pairs: pd.DataFrame) -> pd.DataFrame:
    """Exact Shapley contribution of each group to the home win chance, in percentage points.

    Baseline: every input at the average over all pairings (so Elo difference is about 0).
    Contributions add up to (model home win chance - baseline home win chance).
    """
    cols = model.columns
    groups: dict[str, list[int]] = {}
    for i, col in enumerate(cols):
        groups.setdefault(group_of(col), []).append(i)
    names = list(groups)
    k, n = len(names), len(pairs)

    actual = pairs[cols].to_numpy(dtype=float)
    base = np.nanmean(actual, axis=0)

    masks = np.arange(2 ** k)
    use_actual = np.zeros((len(masks), len(cols)), dtype=bool)
    for j, name in enumerate(names):
        for mask in masks:
            if mask >> j & 1:
                use_actual[mask, groups[name]] = True

    stacked = np.where(use_actual[:, None, :], actual[None, :, :], base[None, None, :])
    frame = pd.DataFrame(stacked.reshape(-1, len(cols)), columns=cols)
    v = home_win_prob(model, frame).reshape(len(masks), n)      # v[mask, pair]

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
        out[name] = contrib[:, j] * 100
    out["end"] = v[-1]
    return out


def write_explain(model: GoalsPredictor, pairs: pd.DataFrame, updated_at: str,
                  name: str, version: str, out_file: Path) -> pd.DataFrame:
    """Explain every pairing, save for upload, and check the maths. Returns the explained frame."""
    result = explain(model, pairs)
    group_cols = [c for c in result.columns if c not in ("home", "away", "start", "end")]

    direct = home_win_prob(model, pairs[model.columns])
    gap_end = np.abs(result["end"].to_numpy() - direct).max()
    gap_sum = np.abs(result[group_cols].sum(axis=1) / 100 - (result["end"] - result["start"])).max()
    print(f"Explain check: end vs model home win chance {gap_end:.1e}, steps vs start-to-end {gap_sum:.1e}")
    assert gap_end < 1e-9 and gap_sum < 1e-9, "explain does not add up"

    rows = [
        {
            "home": r["home"],
            "away": r["away"],
            "start_pct": round(r["start"] * 100, 4),
            "end_pct": round(r["end"] * 100, 4),
            "steps": {c: round(r[c], 4) for c in group_cols},
            "model": name,
            "model_version": version,
            "updated_at": updated_at,
        }
        for r in result.to_dict("records")
    ]
    out_file.write_text(json.dumps(rows))
    return result


def describe(result: pd.DataFrame, home: str, away: str) -> None:
    row = result[(result["home"] == home) & (result["away"] == away)].iloc[0]
    group_cols = [c for c in result.columns if c not in ("home", "away", "start", "end")]
    steps = row[group_cols].sort_values(key=abs, ascending=False)
    print(f"\n{home} v {away}: home win chance")
    print(f"  {'Average match':16s} {row['start'] * 100:6.1f}%")
    for name, points in steps.items():
        print(f"  {name:16s} {points:+6.1f}")
    print(f"  {'Network':16s} {row['end'] * 100:6.1f}%")