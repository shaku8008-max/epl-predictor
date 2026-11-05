from pathlib import Path

import numpy as np
import pandas as pd

from features import ELO_NEW_TEAM, REST_CAP, add_elo, add_rolling, team_rows
from nn_infer import GoalsPredictor
from train_final import LIVE_SEASON

MATCHES_FILE = Path("data/processed/matches_2005.csv")
FEATURES_FILE = Path("data/processed/features_2005.csv")
NEUTRAL_REST_DAYS = 7          # used when a fixture has no date: roughly the typical gap in training


def _placeholder_run(matches: pd.DataFrame, first: list[str], second: list[str]) -> pd.DataFrame:
    """Each team's form going into one imaginary match: `first` teams at home, `second` teams away."""
    placeholder = pd.DataFrame({
        "season": matches["season"].max(),
        "date": matches["date"].max() + pd.Timedelta(days=1),
        "home": first,
        "away": second,
    })
    both = pd.concat([matches, placeholder], ignore_index=True)
    both["match_id"] = both.index
    rows = add_rolling(team_rows(both))
    return rows[rows["match_id"] >= len(matches)]


def team_states(matches: pd.DataFrame, teams: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(state when playing at home, state when playing away), each indexed by team.

    Two runs with the roles swapped, so every team is the home side once and the away side once.
    """
    if len(teams) % 2:
        raise ValueError("need an even number of teams")
    run1 = _placeholder_run(matches, teams[0::2], teams[1::2])
    run2 = _placeholder_run(matches, teams[1::2], teams[0::2])
    home_state = pd.concat([run1[run1["is_home"] == 1], run2[run2["is_home"] == 1]]).set_index("team")
    away_state = pd.concat([run1[run1["is_home"] == 0], run2[run2["is_home"] == 0]]).set_index("team")
    return home_state, away_state


def last_match_dates(matches: pd.DataFrame) -> pd.Series:
    played = pd.concat([
        matches[["home", "date"]].rename(columns={"home": "team"}),
        matches[["away", "date"]].rename(columns={"away": "team"}),
    ])
    return played.groupby("team")["date"].max()


def build_pairs(matches: pd.DataFrame, fixtures: pd.DataFrame, columns: list[str],
                teams: list[str]) -> pd.DataFrame:
    """Model inputs for each fixture. `fixtures` has home, away and date (date may be blank).

    `matches` must hold only matches played BEFORE the fixtures, sorted by date.
    """
    home_state, away_state = team_states(matches, teams)
    _, ratings = add_elo(matches.copy())
    last = last_match_dates(matches)

    out = fixtures[["home", "away"]].reset_index(drop=True)
    out["elo_home"] = out["home"].map(lambda t: ratings.get(t, ELO_NEW_TEAM))
    out["elo_away"] = out["away"].map(lambda t: ratings.get(t, ELO_NEW_TEAM))
    out["elo_diff"] = out["elo_home"] - out["elo_away"]

    for col in columns:
        if col.startswith("home_"):
            out[col] = home_state.loc[out["home"], col[5:]].to_numpy()    # the home team's HOME-venue state
        elif col.startswith("away_"):
            out[col] = away_state.loc[out["away"], col[5:]].to_numpy()    # the away team's AWAY-venue state

    # Rest days depend on the fixture date: the real gap, or a neutral value if the date is unknown.
    dates = pd.to_datetime(fixtures["date"]).reset_index(drop=True)
    for side in ("home", "away"):
        gap = (dates - out[side].map(last)).dt.days.clip(upper=REST_CAP)
        out[f"{side}_rest_days"] = gap.fillna(NEUTRAL_REST_DAYS)

    return out[["home", "away"] + columns]


def parity_test(n_days: int = 6) -> None:
    """Rebuild past matches' features the webapp way and compare with the training rows."""
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["date"]).sort_values("date", kind="stable").reset_index(drop=True)
    feats = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    model = GoalsPredictor()
    columns = model.columns

    season = feats[feats["season"] == LIVE_SEASON - 1]
    teams = sorted(set(season["home"]) | set(season["away"]))
    days = np.sort(season["date"].unique())
    picks = days[np.linspace(10, len(days) - 1, n_days).astype(int)]

    worst = {c: 0.0 for c in columns}
    prob_diff, rows_checked, naive_wrong = 0.0, 0, 0
    venue_cols = [c for c in columns if "_venue5" in c]
    for day in picks:
        history = matches[matches["date"] < day]
        today = season[season["date"] == day]
        fixtures = today[["home", "away", "date"]]
        built = build_pairs(history, fixtures, columns, teams)
        truth = today[columns].reset_index(drop=True)

        for c in columns:
            both_nan = built[c].isna() & truth[c].isna()
            diff = (built[c] - truth[c]).abs().where(~both_nan, 0.0)
            if diff.isna().any():                      # one side blank, the other not
                diff = diff.fillna(np.inf)
            worst[c] = max(worst[c], float(diff.max()))
        prob_diff = max(prob_diff, np.abs(model.predict_probs(built) - model.predict_probs(truth)).max())
        rows_checked += len(today)

        # For contrast: the old single-placeholder approach (one state per team, whatever its venue).
        single = _placeholder_run(history, teams[0::2], teams[1::2]).set_index("team")
        wrong = np.zeros(len(today), dtype=bool)
        for c in venue_cols:
            state_col = c[5:]
            side_team = fixtures["home"] if c.startswith("home_") else fixtures["away"]
            naive = single.loc[side_team, state_col].to_numpy()
            wrong |= ~np.isclose(naive, truth[c].to_numpy(), atol=1e-9, equal_nan=True)
        naive_wrong += int(wrong.sum())

    bad = {c: d for c, d in worst.items() if d > 1e-9}
    print(f"Compared {rows_checked} matches on {len(picks)} match days, {len(columns)} features each.")
    print(f"Largest feature difference: {max(worst.values()):.2e}")
    print(f"Largest probability difference vs the training rows: {prob_diff:.2e}")
    print(f"For contrast, the old single-placeholder approach gets venue form wrong in "
          f"{naive_wrong} of {rows_checked} matches.")
    assert not bad, f"features differ from training: {bad}"
    print("Parity test passed.")


if __name__ == "__main__":
    parity_test()