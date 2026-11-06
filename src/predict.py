import contextlib
import io
import json
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import joblib
import pandas as pd

import download_data
import download_xg
from fpl_fixtures import load_fixtures
from explain import write_explain
from features import ELO_NEW_TEAM, add_elo, add_rolling, team_rows
from train_final import LIVE_SEASON, stack_inputs

MATCHES_FILE = Path("data/processed/matches_2005.csv")
XG_FILE = Path("data/processed/xg.csv")
MODEL_FILE = Path("models/lean_v1.joblib")
OUT_DIR = Path("data/predictions")
XG_MAX_LAG_DAYS = 3     # xG this many days behind the latest results (or more) counts as stale

# Shown under the prediction on the website.
DISPLAY_STATS = [
    "points_last5", "goals_for_last5", "goals_against_last5",
    "xg_for_last5", "xg_against_last5", "xg_for_last10", "xg_against_last10",
    "xg_diff_ewm20", "xg_diff_ewm40",
]


def refresh_data() -> None:
    """Re-download football-data results and Understat's xG copy, quietly."""
    with contextlib.redirect_stdout(io.StringIO()):
        download_data.main(2005, MATCHES_FILE)
        download_xg.main()


def load_history() -> pd.DataFrame:
    """All played matches from football-data, with xG where Understat's copy has it."""
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["date"])
    matches = matches.merge(pd.read_csv(XG_FILE), on=["season", "home", "away"], how="left")
    return matches.sort_values("date", kind="stable").reset_index(drop=True)


def team_state(matches: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Each team's form going into its next match, using the training feature code."""
    future = matches["date"].max() + pd.Timedelta(days=1)
    placeholder = pd.DataFrame({
        "season": LIVE_SEASON,
        "date": future,
        "home": teams[0::2],
        "away": teams[1::2],
    })
    both = pd.concat([matches, placeholder], ignore_index=True)
    both["match_id"] = both.index
    rows = add_rolling(team_rows(both))
    return rows[rows["match_id"] >= len(matches)].set_index("team")


def build_pairs(state: pd.DataFrame, ratings: dict, teams: list[str], cols: list[str]) -> pd.DataFrame:
    """One row per ordered pair of teams, with the model's feature columns."""
    records = []
    for home, away in permutations(teams, 2):
        record = {
            "home": home,
            "away": away,
            "elo_home": ratings[home],
            "elo_away": ratings[away],
            "elo_diff": ratings[home] - ratings[away],
        }
        for col in cols:
            if col.startswith("home_"):
                record[col] = state.at[home, col[5:]]
            elif col.startswith("away_"):
                record[col] = state.at[away, col[5:]]
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    bundle = joblib.load(MODEL_FILE)
    models = bundle["base_models"]
    cols = next(iter(models.values())).cols

    refresh_data()
    matches = load_history()
    season_rows = matches[matches["season"] == LIVE_SEASON]
    teams = sorted(set(season_rows["home"]) | set(season_rows["away"]))
    if len(teams) != 20:
        raise SystemExit(f"expected 20 teams this season, found {len(teams)}: {teams}")

    _, elo = add_elo(matches.copy())
    ratings = {team: elo.get(team, ELO_NEW_TEAM) for team in teams}
    state = team_state(matches, teams)

    pairs = build_pairs(state, ratings, teams, cols)
    probs = bundle["stacker"].predict_proba(stack_inputs([m.predict_proba(pairs) for m in models.values()]))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    predictions = pairs[["home", "away"]].copy()
    predictions[["p_home", "p_draw", "p_away"]] = probs.round(4)
    # Fixtures come from the FPL API; anything football-data already shows as played is dropped.
    scheduled = load_fixtures(teams)
    done = set(zip(season_rows["home"], season_rows["away"]))
    is_upcoming = [(h, a) not in done for h, a in zip(scheduled["home"], scheduled["away"])]
    upcoming = scheduled.loc[is_upcoming, ["home", "away", "fixture_date"]]
    predictions = predictions.merge(upcoming, on=["home", "away"], how="left")
    predictions["model"] = bundle["name"]
    predictions["model_version"] = bundle["version"]
    predictions["predicted_at"] = now

    team_stats = state[DISPLAY_STATS].round(3)
    team_stats.insert(0, "elo", pd.Series(ratings).round(0))
    team_stats = team_stats.sort_values("elo", ascending=False).reset_index()
    team_stats["updated_at"] = now

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT_DIR / "predictions.csv", index=False)
    team_stats.to_csv(OUT_DIR / "team_stats.csv", index=False)
    try:
        count = write_explain(bundle, pairs, now, OUT_DIR / "explain.json")
        print(f"Wrote explain.json ({count} pairings)")
    except Exception as error:
        print(f"WARNING: explain step failed, predictions are unaffected: {error}")
    played = season_rows[["home", "away", "date", "home_goals", "away_goals", "result"]].copy()
    played = played.rename(columns={"date": "fixture_date"})
    played["fixture_date"] = played["fixture_date"].dt.strftime("%Y-%m-%d")
    # Goals are ints, but the column is float64 (unplayed rows are NaN), so cast
    # back to int before writing, or Supabase's int column rejects "1.0".
    played["home_goals"] = played["home_goals"].astype(int)
    played["away_goals"] = played["away_goals"].astype(int)
    played.to_csv(OUT_DIR / "results.csv", index=False)
    fixtures = predictions.dropna(subset=["fixture_date"]).sort_values("fixture_date")
    history_file = OUT_DIR / "history.csv"
    fixtures.to_csv(history_file, mode="a", header=not history_file.exists(), index=False)     
    # How fresh is the xG? Understat's feed has been unreliable, so record it for the site.
    xg_through = matches.loc[matches["home_xg"].notna(), "date"].max()
    lag_days = (matches["date"].max() - xg_through).days
    xg_ok = lag_days <= XG_MAX_LAG_DAYS
    (OUT_DIR / "run_info.json").write_text(json.dumps({
        "xg_through": xg_through.strftime("%Y-%m-%d"),
        "xg_ok": xg_ok,
    }))
    if not xg_ok:
        print(f"WARNING: xG data stops at {xg_through.date()}, {lag_days} days behind the latest results.")

    print(f"Results through {matches['date'].max().date()}")
    print(f"{len(teams)} teams, {len(predictions)} pairings, {len(fixtures)} upcoming fixtures")

    if len(fixtures):
        first = fixtures["fixture_date"].min()
        next_round = fixtures[fixtures["fixture_date"] <= first + pd.Timedelta(days=4)]
        print("\nNext round:")
        print(next_round[["fixture_date", "home", "away", "p_home", "p_draw", "p_away"]].to_string(index=False))

    print("\nTeam stats:")
    print(team_stats.drop(columns="updated_at").to_string(index=False))
    print(f"\nSaved predictions.csv, team_stats.csv and history.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()