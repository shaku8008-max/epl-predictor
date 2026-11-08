import contextlib
import io
import nn_explain
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
from fpl_fixtures import load_fixtures

import pandas as pd
import requests

import download_data
from nn_features import MATCHES_FILE, build_pairs, team_stat_table
from nn_infer import GoalsPredictor
from train_final import LIVE_SEASON

OUT_DIR = Path("data/predictions")
OUT_FILE = OUT_DIR / "nn_predictions.csv"
MODEL_NAME = "goals_nn"
MODEL_VERSION = "v1"
NEAR_DAYS = 10          # fixtures further out than this get a neutral rest-days value
ROUND_DAYS = 4          # how far past the first fixture counts as "the next round"

FPL_TEAMS_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_TO_OURS = {"Man Utd": "Man United", "Spurs": "Tottenham", "Coventry City": "Coventry",
               "Hull City": "Hull", "Ipswich Town": "Ipswich"}     # FPL name -> football-data name


def empty_fixtures() -> pd.DataFrame:
    return pd.DataFrame({
        "home": pd.Series(dtype="object"),
        "away": pd.Series(dtype="object"),
        "fixture_date": pd.Series(dtype="datetime64[ns]"),
    })


def get_json(url: str):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def load_fixtures(teams: list[str]) -> pd.DataFrame:
    """Unfinished, scheduled fixtures from the FPL API as home, away, fixture_date (UK date)."""
    try:
        names = {t["id"]: FPL_TO_OURS.get(t["name"], t["name"]) for t in get_json(FPL_TEAMS_URL)["teams"]}
        raw = pd.DataFrame(get_json(FPL_FIXTURES_URL))
        # FPL sets finished_provisional at full time, but finished only once the whole
        # gameweek is confirmed (often a day later), so treat either one as done.
        done = raw["finished"] | raw.get("finished_provisional", False)
        raw = raw[~done & raw["kickoff_time"].notna()]      # skip finished and unscheduled
        kickoff = pd.to_datetime(raw["kickoff_time"], utc=True).dt.tz_convert("Europe/London")
        fixtures = pd.DataFrame({
            "home": raw["team_h"].map(names),
            "away": raw["team_a"].map(names),
            "fixture_date": kickoff.dt.tz_localize(None).dt.normalize(),
        })
    except Exception as error:      # network error, changed format, ...
        print(f"WARNING: could not get fixtures ({type(error).__name__}: {error}).")
        print("         Predictions are still written, but without fixture dates.")
        return empty_fixtures()

    unknown = (set(fixtures["home"]) | set(fixtures["away"])) - set(teams)
    if unknown:
        print(f"WARNING: FPL team names not matched to ours, those fixtures are skipped: {sorted(map(str, unknown))}")
        fixtures = fixtures[fixtures["home"].isin(teams) & fixtures["away"].isin(teams)]
    return fixtures.reset_index(drop=True)


def main() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        download_data.main(2005, MATCHES_FILE)           # football-data results, quietly
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["date"])
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)

    model = GoalsPredictor()
    this_season = matches[matches["season"] == LIVE_SEASON]
    teams = sorted(set(this_season["home"]) | set(this_season["away"]))
    if len(teams) != 20:
        raise SystemExit(f"expected 20 teams this season, found {len(teams)}: {teams}")

    # Fixtures, minus any pairing football-data already shows as played this season.
    fixtures = load_fixtures(teams)
    played = this_season[["home", "away"]].assign(played=True)
    fixtures = fixtures.merge(played, on=["home", "away"], how="left")
    fixtures = fixtures[fixtures["played"].isna()].drop(columns="played")

    pairs = pd.DataFrame(list(permutations(teams, 2)), columns=["home", "away"])
    pairs = pairs.merge(fixtures, on=["home", "away"], how="left")
    pairs["fixture_date"] = pd.to_datetime(pairs["fixture_date"])      # real dates even when there are none

    last_result = matches["date"].max()
    near = pairs["fixture_date"].where(pairs["fixture_date"] <= last_result + pd.Timedelta(days=NEAR_DAYS))
    inputs = build_pairs(
        matches,
        pd.DataFrame({"home": pairs["home"], "away": pairs["away"], "date": near}),
        model.columns,
        teams,
    )

    probs = model.predict_probs(inputs)
    goals = model.expected_goals(inputs)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out = pairs[["home", "away"]].copy()
    out[["p_home", "p_draw", "p_away"]] = probs.round(4)
    out[["xg_home", "xg_away"]] = goals.round(3)
    out["fixture_date"] = pairs["fixture_date"].dt.strftime("%Y-%m-%d")
    out["model"] = MODEL_NAME
    out["model_version"] = MODEL_VERSION
    out["predicted_at"] = now

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False) 
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    stats = team_stat_table(matches, teams, today)
    stats["updated_at"] = now
    stats.round(3).to_csv(OUT_DIR / "nn_team_stats.csv", index=False)     
    explained = nn_explain.write_explain(
        model, inputs.assign(fixture_date=pairs["fixture_date"]), now,
        MODEL_NAME, MODEL_VERSION, OUT_DIR / "nn_explain.json",
    )
    top = stats.sort_values("elo", ascending=False).head(5)[["team", "elo"]].round(0)
    print(f"Saved nn_team_stats.csv ({len(stats)} teams). Top Elo: "
          + ", ".join(f"{r.team} {int(r.elo)}" for r in top.itertuples()))

    upcoming = out.dropna(subset=["fixture_date"]).sort_values("fixture_date")
    print(f"Results through {last_result.date()} | {len(teams)} teams, {len(out)} pairings, "
          f"{len(upcoming)} upcoming fixtures | blank model inputs: {int(inputs.isna().sum().sum())}")
    print(f"Saved {OUT_FILE}")

    if len(upcoming):
        first = pd.Timestamp(upcoming["fixture_date"].min())
        cutoff = (first + pd.Timedelta(days=ROUND_DAYS)).strftime("%Y-%m-%d")
        print("\nNext round:")         
        print(upcoming[upcoming["fixture_date"] <= cutoff][
            ["fixture_date", "home", "away", "p_home", "p_draw", "p_away", "xg_home", "xg_away"]
        ].to_string(index=False))
        first_game = upcoming.iloc[0]
        nn_explain.describe(explained, first_game["home"], first_game["away"])


if __name__ == "__main__":
    main()
