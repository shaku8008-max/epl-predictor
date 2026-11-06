"""Season outlook: play out the remaining fixtures many times using the neural
network's expected goals, then count how often each team finishes where."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nn_features import MATCHES_FILE
from train_final import LIVE_SEASON

PRED_FILE = Path("data/predictions/nn_predictions.csv")
OUT_FILE = Path("data/predictions/nn_season_outlook.csv")
N_SIMS = 10_000
SEED = 2026
GAMES = 38
MODEL_NAME = "goals_nn"
MODEL_VERSION = "v1"


def current_table(matches: pd.DataFrame, teams: list[str]):
    """Played, points, goals for and goals against so far this season."""
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    played, points, gf, ga = (np.zeros(n) for _ in range(4))
    done = matches[matches["season"] == LIVE_SEASON].dropna(subset=["home_goals", "away_goals"])
    for r in done.itertuples():
        h, a = idx[r.home], idx[r.away]
        played[h] += 1
        played[a] += 1
        gf[h] += r.home_goals
        ga[h] += r.away_goals
        gf[a] += r.away_goals
        ga[a] += r.home_goals
        if r.home_goals > r.away_goals:
            points[h] += 3
        elif r.home_goals < r.away_goals:
            points[a] += 3
        else:
            points[h] += 1
            points[a] += 1
    return played, points, gf, ga


def main() -> None:
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["date"])
    preds = pd.read_csv(PRED_FILE)
    teams = sorted(set(preds["home"]) | set(preds["away"]))
    n = len(teams)
    if n != 20:
        raise SystemExit(f"expected 20 teams, found {n}")
    idx = {t: i for i, t in enumerate(teams)}

    played, points, gf, ga = current_table(matches, teams)
    fixtures = preds.dropna(subset=["fixture_date"]).reset_index(drop=True)
    n_fix = len(fixtures)

    # One-hot matrices: H[f, t] = 1 if team t is the home side in fixture f.
    H = np.zeros((n_fix, n))
    A = np.zeros((n_fix, n))
    H[np.arange(n_fix), fixtures["home"].map(idx).to_numpy()] = 1
    A[np.arange(n_fix), fixtures["away"].map(idx).to_numpy()] = 1

    remaining = H.sum(axis=0) + A.sum(axis=0)
    short = [(teams[i], int(played[i]), int(remaining[i])) for i in range(n)
             if played[i] + remaining[i] != GAMES]
    print(f"{int(played.sum() / 2)} matches played, {n_fix} fixtures left to simulate.")
    if short:
        print("WARNING: these teams do not add up to 38 games (team, played, remaining):")
        print(short)

    rng = np.random.default_rng(SEED)
    hg = rng.poisson(fixtures["xg_home"].to_numpy(), size=(N_SIMS, n_fix)).astype(float)
    ag = rng.poisson(fixtures["xg_away"].to_numpy(), size=(N_SIMS, n_fix)).astype(float)

    home_pts = 3 * (hg > ag) + (hg == ag)
    away_pts = 3 * (ag > hg) + (hg == ag)
    pts = points + home_pts @ H + away_pts @ A
    gf_s = gf + hg @ H + ag @ A
    ga_s = ga + ag @ H + hg @ A
    gd = gf_s - ga_s

    # Rank by points, then goal difference, then goals scored; a tiny random
    # number settles any remaining tie (the real league uses head-to-head).
    key = pts * 1e6 + gd * 1e3 + gf_s + rng.random(pts.shape) * 0.5
    positions = (-key).argsort(axis=1).argsort(axis=1) + 1          # [sim, team], 1 = champions
    pos_probs = np.stack([(positions == p).mean(axis=0) for p in range(1, n + 1)], axis=1)

    out = pd.DataFrame({
        "team": teams,
        "played": played.astype(int),
        "points": points.astype(int),
        "goal_diff": (gf - ga).astype(int),
        "expected_points": pts.mean(axis=0).round(1),
        "expected_position": positions.mean(axis=0).round(2),
        "p_title": pos_probs[:, 0].round(4),
        "p_top4": pos_probs[:, :4].sum(axis=1).round(4),
        "p_relegation": pos_probs[:, n - 3:].sum(axis=1).round(4),
        "position_probs": [json.dumps(np.round(row, 4).tolist()) for row in pos_probs],
        "simulations": N_SIMS,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }).sort_values("expected_points", ascending=False).reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    show = out.drop(columns=["simulations", "model", "model_version", "position_probs"]).copy()
    for col in ("p_title", "p_top4", "p_relegation"):
        show[col] = (show[col] * 100).round(1)
    print(f"\nSimulated {N_SIMS:,} seasons.\n")
    print(show.to_string(index=False))

    print("\nSanity checks (the simulation should agree with the model):")
    print(f"  home wins: simulated {(hg > ag).mean():.3f} vs model average p_home {fixtures['p_home'].mean():.3f}")
    print(f"  draws:     simulated {(hg == ag).mean():.3f} vs model average p_draw {fixtures['p_draw'].mean():.3f}")
    print(f"  away wins: simulated {(ag > hg).mean():.3f} vs model average p_away {fixtures['p_away'].mean():.3f}")
    print(f"  title probabilities add to {out['p_title'].sum():.3f} (should be 1.000)")
    print(f"  top-4 probabilities add to {out['p_top4'].sum():.3f} (should be 4.000)")
    print(f"  relegation probabilities add to {out['p_relegation'].sum():.3f} (should be 3.000)")
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()