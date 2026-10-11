import json
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw")
MATCHES_FILE = Path("data/processed/matches_2005.csv")
OUT_FILE = Path("data/processed/xg.csv")
URL = "https://understat.com/getLeagueData/EPL/{year}"

# Season start years. Understat begins in 2014-15. The last one is the season in progress.
SEASONS = range(2014, 2027)

# Understat name -> name used in our match data.
TEAM_NAMES = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR",
}


def download_season(year: int, refresh: bool = False) -> list[dict]:
    """Return one season's match list, downloading it unless a raw copy exists."""
    path = RAW_DIR / f"understat_EPL_{year}.json"
    if refresh or not path.exists():
        headers = {
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://understat.com/league/EPL/{year}",
        }
        response = requests.get(URL.format(year=year), headers=headers, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
        print(f"downloaded {year}")
        time.sleep(2)
    return json.loads(path.read_text(encoding="utf-8"))["dates"]


def season_table(dates: list[dict], year: int) -> pd.DataFrame:
    """Keep played matches only and rename teams to match our data."""
    rows = []
    for m in dates:
        if m["xG"]["h"] is None or m["goals"]["h"] is None:
            continue   # not played yet
        rows.append({
            "season": year,
            "home": TEAM_NAMES.get(m["h"]["title"], m["h"]["title"]),
            "away": TEAM_NAMES.get(m["a"]["title"], m["a"]["title"]),
            "home_xg": float(m["xG"]["h"]),
            "away_xg": float(m["xG"]["a"]),
            "us_home_goals": int(m["goals"]["h"]),
            "us_away_goals": int(m["goals"]["a"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    current = max(SEASONS)
    xg = pd.concat(
        [season_table(download_season(y, refresh=(y == current)), y) for y in SEASONS],
        ignore_index=True,
    )

    matches = pd.read_csv(MATCHES_FILE, parse_dates=["date"])
    matches = matches[matches["season"] >= min(SEASONS)]
    merged = matches.merge(xg, on=["season", "home", "away"], how="left")

    print(f"\nUnderstat matches with xG: {len(xg)}")
    print(xg.groupby("season").size().to_string())

    missing = merged["home_xg"].isna()
    print(f"\nOur matches from {min(SEASONS)} on: {len(merged)}, without xG: {missing.sum()}")
    if missing.any():
        print(merged[missing].groupby("season").size().to_string())

    ours = set(matches["home"])
    theirs = set(xg["home"])
    print(f"\nTeam names only in Understat: {sorted(theirs - ours)}")
    print(f"Team names only in our data:  {sorted(ours - theirs)}")

    matched = merged[~missing]
    wrong = (matched["home_goals"] != matched["us_home_goals"]) | (matched["away_goals"] != matched["us_away_goals"])
    print(f"\nMatched games where the two sources disagree on the score: {wrong.sum()}")

    print("\nAverage xG per match by season:")
    print(xg.groupby("season")[["home_xg", "away_xg"]].mean().round(2).to_string())

    xg[["season", "home", "away", "home_xg", "away_xg"]].to_csv(OUT_FILE, index=False)
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()