import sys
from pathlib import Path

import numpy as np
import pandas as pd

IN_FILE = Path("data/processed/matches.csv")
OUT_FILE = Path("data/processed/features.csv")

WINDOWS = (5, 10)        # rolling form over a team's last N matches
VENUE_WINDOW = 5         # last N home matches (home team) or away matches (away team)
EWM_SPANS = (20, 40)     # long-memory averages, recent matches weighted more
REST_CAP = 14            # cap rest days so the summer break is not a huge number

ELO_START = 1500.0       # teams in the first season
ELO_NEW_TEAM = 1400.0    # teams that first appear later (promoted)
ELO_K = 20.0
ELO_HOME_ADV = 60.0
ELO_CARRY = 1.0          # share of each team's gap to 1500 kept between seasons
ELO_MARGIN_POWER = 0.5   # 0 ignores the winning margin

CLOSED_START = pd.Timestamp("2020-06-17")
CLOSED_END = pd.Timestamp("2021-05-17")

STATS = [
    "points", "goals_for", "goals_against",
    "shots_for", "shots_against", "sot_for", "sot_against",
]
XG_STATS = ["xg_for", "xg_against"]     # only present when an xG file is given
VENUE_STATS = ["points", "goals_for", "goals_against"]

# Known only after the match, so never allowed into the output.
RAW_COLUMNS = [
    "home_shots", "away_shots", "home_shots_target", "away_shots_target",
    "odds_home", "odds_draw", "odds_away",
    "home_xg", "away_xg",
]


def team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Turn each match into two rows, one from each team's point of view."""
    home = pd.DataFrame({
        "match_id": matches["match_id"],
        "date": matches["date"],
        "team": matches["home"],
        "is_home": 1,
        "goals_for": matches["home_goals"],
        "goals_against": matches["away_goals"],
        "shots_for": matches["home_shots"],
        "shots_against": matches["away_shots"],
        "sot_for": matches["home_shots_target"],
        "sot_against": matches["away_shots_target"],
    })
    away = pd.DataFrame({
        "match_id": matches["match_id"],
        "date": matches["date"],
        "team": matches["away"],
        "is_home": 0,
        "goals_for": matches["away_goals"],
        "goals_against": matches["home_goals"],
        "shots_for": matches["away_shots"],
        "shots_against": matches["home_shots"],
        "sot_for": matches["away_shots_target"],
        "sot_against": matches["home_shots_target"],
    })
    if "home_xg" in matches.columns:
        home["xg_for"] = matches["home_xg"]
        home["xg_against"] = matches["away_xg"]
        away["xg_for"] = matches["away_xg"]
        away["xg_against"] = matches["home_xg"]

    rows = pd.concat([home, away], ignore_index=True)
    rows["points"] = np.select(
        [rows["goals_for"] > rows["goals_against"], rows["goals_for"] == rows["goals_against"]],
        [3, 1],
        default=0,
    )
    return rows.sort_values(["date", "match_id"]).reset_index(drop=True)


def add_rolling(rows: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match form. shift(1) keeps each match out of its own features."""
    stats = STATS + [s for s in XG_STATS if s in rows.columns]
    for window in WINDOWS:
        for stat in stats:
            rows[f"{stat}_last{window}"] = rows.groupby("team")[stat].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
    for stat in VENUE_STATS:
        rows[f"{stat}_venue{VENUE_WINDOW}"] = rows.groupby(["team", "is_home"])[stat].transform(
            lambda s: s.shift(1).rolling(VENUE_WINDOW, min_periods=1).mean()
        )
    rows["rest_days"] = rows.groupby("team")["date"].diff().dt.days.clip(upper=REST_CAP)
    rows["played"] = rows.groupby("team").cumcount()

    # Long-memory ratings: whole history, recent matches weighted more.
    rows["goal_diff"] = rows["goals_for"] - rows["goals_against"]
    long_stats = ["goal_diff"]
    if "xg_for" in rows.columns:
        rows["xg_diff"] = rows["xg_for"] - rows["xg_against"]
        long_stats.append("xg_diff")
    for span in EWM_SPANS:
        for stat in long_stats:
            rows[f"{stat}_ewm{span}"] = rows.groupby("team")[stat].transform(
                lambda s: s.shift(1).ewm(span=span, min_periods=1).mean()
            )
    return rows


def add_elo(
    matches: pd.DataFrame,
    k: float = ELO_K,
    home_adv: float = ELO_HOME_ADV,
    new_team: float = ELO_NEW_TEAM,
    carry: float = ELO_CARRY,
    margin_power: float = ELO_MARGIN_POWER,
) -> tuple[pd.DataFrame, dict]:
    """Record each team's Elo before the match, then update it with the result."""
    ratings: dict[str, float] = {}
    first_season = matches["season"].min()
    season = first_season
    elo_home, elo_away = [], []

    for m in matches.itertuples(index=False):
        if m.season != season:
            # New season: pull every rating part of the way back to average.
            season = m.season
            ratings = {t: ELO_START + carry * (r - ELO_START) for t, r in ratings.items()}

        default = ELO_START if m.season == first_season else new_team
        rating_home = ratings.setdefault(m.home, default)
        rating_away = ratings.setdefault(m.away, default)
        elo_home.append(rating_home)
        elo_away.append(rating_away)

        expected = 1 / (1 + 10 ** ((rating_away - rating_home - home_adv) / 400))
        actual = {"H": 1.0, "D": 0.5, "A": 0.0}[m.result]
        margin = max(abs(m.home_goals - m.away_goals), 1)
        change = k * margin ** margin_power * (actual - expected)
        ratings[m.home] = rating_home + change
        ratings[m.away] = rating_away - change

    matches["elo_home"] = elo_home
    matches["elo_away"] = elo_away
    matches["elo_diff"] = matches["elo_home"] - matches["elo_away"]
    return matches, ratings


def add_market_probs(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert odds to probabilities and remove the bookmaker's margin."""
    inverse = 1 / matches[["odds_home", "odds_draw", "odds_away"]]
    probs = inverse.div(inverse.sum(axis=1), axis=0)
    matches[["mkt_home", "mkt_draw", "mkt_away"]] = probs.to_numpy()
    return matches


def check_no_leak(matches: pd.DataFrame, features: pd.DataFrame) -> None:
    """Recompute one feature by hand from raw results and compare."""
    assert pd.isna(features.iloc[0]["home_goals_for_last5"]), "first match should have no history"
    assert not {"home_xg", "away_xg"} & set(features.columns), "raw xG must not be in the output"

    row = features.iloc[-1]
    team = row["home"]
    prior = matches[
        (matches["date"] < row["date"]) & ((matches["home"] == team) | (matches["away"] == team))
    ].tail(5)
    goals = np.where(prior["home"] == team, prior["home_goals"], prior["away_goals"])
    assert np.isclose(goals.mean(), row["home_goals_for_last5"]), "rolling feature does not match hand check"
    print(f"Leak check passed ({team}, {row['date'].date()})")


def main(in_file: Path = IN_FILE, out_file: Path = OUT_FILE, xg_file: Path | None = None) -> None:
    matches = pd.read_csv(in_file, parse_dates=["date"])
    if xg_file is not None:
        xg = pd.read_csv(xg_file)
        matches = matches.merge(xg, on=["season", "home", "away"], how="left")
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)
    matches["match_id"] = matches.index

    rows = add_rolling(team_rows(matches))
    feature_cols = [
        c for c in rows.columns
        if "_last" in c or "_venue" in c or "_ewm" in c or c in ("rest_days", "played")
    ]
    home = rows[rows["is_home"] == 1].set_index("match_id")[feature_cols].add_prefix("home_")
    away = rows[rows["is_home"] == 0].set_index("match_id")[feature_cols].add_prefix("away_")

    matches, ratings = add_elo(matches)
    matches = add_market_probs(matches)
    matches["home_win"] = (matches["result"] == "H").astype(int)
    matches["closed_doors"] = matches["date"].between(CLOSED_START, CLOSED_END, inclusive="left").astype(int)

    features = (
        matches.drop(columns=RAW_COLUMNS, errors="ignore")
        .join(home, on="match_id")
        .join(away, on="match_id")
    )
    features.to_csv(out_file, index=False)

    print(f"{features.shape[0]} rows, {features.shape[1]} columns")
    check_no_leak(matches, features)

    missing = features.isna().sum()
    print("\nColumns with missing values:")
    print(missing[missing > 0])

    print("\nTop 8 Elo ratings now:")
    print(pd.Series(ratings).sort_values(ascending=False).head(8).round(0))

    print("\nHome win rate by season:")
    print(features.groupby("season")["home_win"].mean().round(3))


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) > 2:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        main()