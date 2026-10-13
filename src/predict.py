import contextlib
import io
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import joblib
import pandas as pd

import download_data
import download_xg
from features import ELO_NEW_TEAM, add_elo, add_rolling, team_rows
from train_final import LIVE_SEASON, stack_inputs

MATCHES_FILE = Path("data/processed/matches_2005.csv")
XG_FILE = Path("data/processed/xg.csv")
MODEL_FILE = Path("models/lean_v1.joblib")
OUT_DIR = Path("data/predictions")

# Shown under the prediction on the website.
DISPLAY_STATS = [
    "points_last5", "goals_for_last5", "goals_against_last5",
    "xg_for_last5", "xg_against_last5", "xg_for_last10", "xg_against_last10",
    "xg_diff_ewm20", "xg_diff_ewm40",
]


def refresh_data() -> list[dict]:
    """Re-download the season in progress from both sources, quietly."""
    with contextlib.redirect_stdout(io.StringIO()):
        download_data.main(2005, MATCHES_FILE)
        download_xg.main()
    return download_xg.download_season(LIVE_SEASON)


def understat_table(dates: list[dict]) -> pd.DataFrame:
    """Every match of the current season, played or not, with our team names."""
    rename = download_xg.TEAM_NAMES
    rows = []
    for m in dates:
        played = m["xG"]["h"] is not None and m["goals"]["h"] is not None
        row = {
            "season": LIVE_SEASON,
            "date": pd.Timestamp(m["datetime"][:10]),
            "home": rename.get(m["h"]["title"], m["h"]["title"]),
            "away": rename.get(m["a"]["title"], m["a"]["title"]),
            "played": played,
        }
        if played:
            home_goals, away_goals = int(m["goals"]["h"]), int(m["goals"]["a"])
            row.update(
                home_goals=home_goals,
                away_goals=away_goals,
                result="H" if home_goals > away_goals else "A" if away_goals > home_goals else "D",
                home_xg=float(m["xG"]["h"]),
                away_xg=float(m["xG"]["a"]),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def load_history(current: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """All played matches with xG, topped up with any Understat has and football-data lacks."""
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["date"])
    matches = matches.merge(pd.read_csv(XG_FILE), on=["season", "home", "away"], how="left")

    this_season = matches[matches["season"] == LIVE_SEASON]
    have = set(zip(this_season["home"], this_season["away"]))
    played = current[current["played"]]
    is_new = [(h, a) not in have for h, a in zip(played["home"], played["away"])]
    extra = played[is_new].drop(columns="played")
    if len(extra):
        matches = pd.concat([matches, extra], ignore_index=True)
    return matches.sort_values("date", kind="stable").reset_index(drop=True), len(extra)


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

    current = understat_table(refresh_data())
    matches, added = load_history(current)
    teams = sorted(set(current["home"]) | set(current["away"]))

    _, elo = add_elo(matches.copy())
    ratings = {team: elo.get(team, ELO_NEW_TEAM) for team in teams}
    state = team_state(matches, teams)

    pairs = build_pairs(state, ratings, teams, cols)
    probs = bundle["stacker"].predict_proba(stack_inputs([m.predict_proba(pairs) for m in models.values()]))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    predictions = pairs[["home", "away"]].copy()
    predictions[["p_home", "p_draw", "p_away"]] = probs.round(4)
    upcoming = current.loc[~current["played"], ["home", "away", "date"]].rename(columns={"date": "fixture_date"})
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
    played = current.loc[current["played"], ["home", "away", "date", "home_goals", "away_goals", "result"]].copy()
    played = played.rename(columns={"date": "fixture_date"})
    played["fixture_date"] = played["fixture_date"].dt.strftime("%Y-%m-%d")
    played.to_csv(OUT_DIR / "results.csv", index=False)
    fixtures = predictions.dropna(subset=["fixture_date"]).sort_values("fixture_date")
    history_file = OUT_DIR / "history.csv"
    fixtures.to_csv(history_file, mode="a", header=not history_file.exists(), index=False)

    print(f"Results through {matches['date'].max().date()} ({added} matches added from Understat)")
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