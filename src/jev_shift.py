"""Jev shift test: real FPL news vs no news for one fixture."""
import os
import sys
from pathlib import Path

import pandas as pd
import requests

import fpl_news
from jev_test import LADDER, MODEL, URL, describe, label, load_env

PREDICTIONS_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "predictions" / "predictions.csv"
)

INSTRUCTIONS = (
    "The statistical model's probabilities already reflect team strength and "
    "recent form, but know nothing about injuries or suspensions. Given the "
    "team news for both teams, how should the home team's win probability be "
    "adjusted? Key players missing from the home side lower it. Key players "
    "missing from the away side raise it. If the news is minor or balanced "
    "between the teams, choose no_change."
)


def load_model_probs(home, away):
    df = pd.read_csv(PREDICTIONS_CSV)
    mask = (df["home"].str.lower() == home.lower()) & (
        df["away"].str.lower() == away.lower()
    )
    rows = df[mask]
    if rows.empty:
        names = ", ".join(sorted(df["home"].unique()))
        sys.exit(f"No row for {home} v {away} in predictions.csv. Team names: {names}")
    row = rows.iloc[0]
    fixture = "none" if pd.isna(row["fixture_date"]) else row["fixture_date"]
    print(
        f"Model probabilities from predictions.csv: model={row['model']} "
        f"{row['model_version']}, predicted_at={row['predicted_at']}, "
        f"fixture_date={fixture}"
    )
    return (
        float(row["p_home"]) * 100,
        float(row["p_draw"]) * 100,
        float(row["p_away"]) * 100,
    )


def build_state(home, away, probs, home_news, away_news):
    h, d, a = probs
    return (
        f"Premier League match: {home} (home) vs {away} (away).\n"
        f"Statistical model probabilities: home win {h:.0f}%, draw {d:.0f}%, "
        f"away win {a:.0f}%.\n"
        f"Team news for {home}: {home_news}\n"
        f"Team news for {away}: {away_news}"
    )


def ask_jev(api_key, state):
    body = {
        "model": MODEL,
        "state": state,
        "questions": {
            "shift": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {label(n): describe(n) for n in LADDER},
            }
        },
    }
    resp = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"Jev error {resp.status_code}: {resp.text}")
        resp.raise_for_status()
    answer = resp.json()["answers"]["shift"]
    probs = answer["probabilities"]
    expected = sum(probs.get(label(n), 0.0) * n for n in LADDER)
    return answer["choice"], answer["confidence"], expected, probs


def adjust(probs, shift):
    h, d, a = probs
    new_h = min(max(h + shift, 1.0), 98.0)
    remaining = 100.0 - new_h
    new_d = remaining * d / (d + a)
    new_a = remaining - new_d
    return new_h, new_d, new_a


def run_scenario(title, api_key, home, away, probs, home_news, away_news):
    print(f"\n########## {title} ##########")
    print(f"{home}: {home_news}")
    print(f"{away}: {away_news}")
    state = build_state(home, away, probs, home_news, away_news)
    choice, confidence, expected, jev_probs = ask_jev(api_key, state)
    top3 = sorted(jev_probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    new_h, new_d, new_a = adjust(probs, expected)
    print(f"\nJev's pick: {choice} (confidence {confidence})")
    print("Top 3 options: " + ", ".join(f"{k}={v}" for k, v in top3))
    print(f"Probability-weighted shift: {expected:+.1f} points")
    print(
        f"Model:    home {probs[0]:.1f}%  draw {probs[1]:.1f}%  away {probs[2]:.1f}%"
    )
    print(f"Adjusted: home {new_h:.1f}%  draw {new_d:.1f}%  away {new_a:.1f}%")
    return expected


def main():
    load_env()
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        sys.exit("TYPESAFE_API_KEY not found in epl-predictor/.env")

    home_name = sys.argv[1] if len(sys.argv) > 1 else "Arsenal"
    away_name = sys.argv[2] if len(sys.argv) > 2 else "Chelsea"
    if len(sys.argv) > 5:
        probs = (float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
    else:
        probs = load_model_probs(home_name, away_name)
    if abs(sum(probs) - 100) > 1:
        sys.exit("The three probabilities should add up to about 100.")

    data = fpl_news.get_data()
    teams = []
    for name in (home_name, away_name):
        team = fpl_news.find_team(data, name)
        if team is None:
            sys.exit(f"No FPL team matches '{name}'.")
        teams.append(team)
    home_info = fpl_news.team_news(data, teams[0])
    away_info = fpl_news.team_news(data, teams[1])

    real = run_scenario(
        "REAL NEWS",
        api_key,
        home_info["team"],
        away_info["team"],
        probs,
        fpl_news.news_text(home_info),
        fpl_news.news_text(away_info),
    )

    empty_home = {"team": home_info["team"], "players": []}
    empty_away = {"team": away_info["team"], "players": []}
    none = run_scenario(
        "NO NEWS (control)",
        api_key,
        home_info["team"],
        away_info["team"],
        probs,
        fpl_news.news_text(empty_home),
        fpl_news.news_text(empty_away),
    )

    print("\n########## VERDICT ##########")
    if abs(none) <= 1.0:
        print(f"Control OK: no news gives a shift of {none:+.1f} (close to zero).")
    else:
        print(f"WARNING: no news still gives {none:+.1f}. The ladder wording needs work.")
    print(f"News effect on the shift: {real - none:+.1f} points.")


if __name__ == "__main__":
    main()