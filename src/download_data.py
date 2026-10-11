from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import sys

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/matches.csv")
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# Season start years: 2015 means the 2015-16 season. The last one is the season in progress.
SEASONS = range(2015, 2027)

COLUMNS = {
    "Date": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",          # H, D or A
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
}


def season_code(start_year: int) -> str:
    """2015 -> '1516'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download_season(start_year: int, refresh: bool = False) -> Path:
    """Download one season's CSV unless a raw copy already exists."""
    code = season_code(start_year)
    path = RAW_DIR / f"E0_{code}.csv"
    if path.exists() and not refresh:
        return path
    response = requests.get(BASE_URL.format(code=code), timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"downloaded {code}")
    return path


def load_season(path: Path, start_year: int) -> pd.DataFrame:
    """Read one raw file and keep only the columns we use."""
    text = path.read_bytes().decode("latin-1")
    df = pd.read_csv(StringIO(text), on_bad_lines="skip")
    df = df[[c for c in COLUMNS if c in df.columns]].rename(columns=COLUMNS)
    df = df.dropna(subset=["date", "home", "away", "result"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    df["season"] = start_year
    return df


def main(first_season: int = min(SEASONS), out_file: Path = OUT_FILE) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    seasons = range(first_season, max(SEASONS) + 1)
    current = max(seasons)
    frames = []
    for year in seasons:
        # Always re-download the season in progress, since it gains new matches.
        path = download_season(year, refresh=(year == current))
        frames.append(load_season(path, year))

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values("date").reset_index(drop=True)
    matches.to_csv(out_file, index=False)

    print(f"\n{len(matches)} matches, {matches['date'].min().date()} to {matches['date'].max().date()}")
    print(matches.groupby("season").size())
    print("\nMissing values per column:")
    print(matches.isna().sum())


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(int(sys.argv[1]), Path(sys.argv[2]))
    else:
        main()