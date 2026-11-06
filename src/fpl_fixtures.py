"""Upcoming Premier League fixtures from the free FPL API."""
import pandas as pd
import requests

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
        raw = raw[~raw["finished"] & raw["kickoff_time"].notna()]      # skip finished and unscheduled
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