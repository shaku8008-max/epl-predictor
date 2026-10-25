"""Daily Jev job: settle finished fixtures, then fetch FPL team news, ask Jev
for a shift and upload it to Supabase.

Run from the epl-predictor folder:
    python src/jev_job.py --dry-run
    python src/jev_job.py
    python src/jev_job.py --dry-run --today 2026-10-15
"""
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone

import requests

import fpl_news
from jev_shift import adjust, ask_jev, build_state
from jev_test import MODEL
from upload import Database, load_settings, read_rows

MAX_DAYS_AHEAD = 3
KICKOFF_LOCK_HOUR_UTC = 11
PROMPT_VERSION = "v1"
MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}
BACK_RE = re.compile(r"expected back (\d{1,2}) ([a-z]{3})", re.IGNORECASE)


def settle_results(db, dry_run):
    """Copy final scores from results.csv onto jev_predictions rows still waiting."""
    try:
        results = read_rows("results.csv")
    except FileNotFoundError:
        print("No results.csv found, skipping settling.")
        return
    finished = {}
    for r in results:
        if (
            r["result"] in ("H", "D", "A")
            and r["home_goals"] is not None
            and r["away_goals"] is not None
            and r["fixture_date"]
        ):
            finished[(r["home"], r["away"], r["fixture_date"][:10])] = r

    resp = requests.get(
        f"{db.base}/jev_predictions",
        headers=db.headers,
        params={"select": "home,away,fixture_date", "result": "is.null"},
        timeout=30,
    )
    if not resp.ok:
        print(f"Could not read unsettled rows: {resp.status_code} {resp.text}")
        return
    waiting = resp.json()

    settled = 0
    stamp = datetime.now(timezone.utc).isoformat()
    for row in waiting:
        score = finished.get((row["home"], row["away"], row["fixture_date"]))
        if score is None:
            continue
        body = {
            "home_goals": int(float(score["home_goals"])),
            "away_goals": int(float(score["away_goals"])),
            "result": score["result"],
            "settled_at": stamp,
        }
        label = f"{row['home']} v {row['away']} ({row['fixture_date']})"
        if dry_run:
            print(f"Would settle {label}: {body['home_goals']}-{body['away_goals']} {body['result']}")
            settled += 1
            continue
        patch = requests.patch(
            f"{db.base}/jev_predictions",
            headers=db.headers | {"Prefer": "return=minimal"},
            params={
                "home": f"eq.{row['home']}",
                "away": f"eq.{row['away']}",
                "fixture_date": f"eq.{row['fixture_date']}",
            },
            json=body,
            timeout=30,
        )
        if patch.ok:
            settled += 1
            print(f"Settled {label}: {body['home_goals']}-{body['away_goals']} {body['result']}")
        else:
            print(f"Could not settle {label}: {patch.status_code} {patch.text}")
    print(f"Settled {settled} of {len(waiting)} rows waiting for a result.")


def back_before_kickoff(news, fixture_date):
    """True if FPL says the player is back on or before the fixture date."""
    match = BACK_RE.search(news or "")
    if not match:
        return False
    month = MONTHS.get(match.group(2).lower())
    if not month:
        return False
    try:
        returns = date(fixture_date.year, month, int(match.group(1)))
    except ValueError:
        return False
    if returns > fixture_date + timedelta(days=180):
        returns = date(returns.year - 1, returns.month, returns.day)
    return returns <= fixture_date


def filtered_news(data, team, fixture_date):
    info = fpl_news.team_news(data, team)
    info["players"] = [
        p for p in info["players"] if not back_before_kickoff(p["news"], fixture_date)
    ]
    return info


def news_body(info):
    if not info["players"]:
        return "no reported injuries or suspensions among key players."
    parts = []
    for p in info["players"]:
        chance = (
            f", {p['chance_of_playing']}% chance of playing"
            if p["chance_of_playing"] is not None
            else ""
        )
        parts.append(f"{p['name']} ({p['position']}, {p['level']}{chance}): {p['news']}")
    return "; ".join(parts)


def parse_args(argv):
    dry_run = "--dry-run" in argv
    override = None
    if "--today" in argv:
        override = date.fromisoformat(argv[argv.index("--today") + 1])
    return dry_run, override


def main():
    dry_run, today_override = parse_args(sys.argv[1:])
    url, key = load_settings()
    db = Database(url, key)
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        sys.exit("TYPESAFE_API_KEY not found in epl-predictor/.env or the environment")

    settle_results(db, dry_run)

    now = datetime.now(timezone.utc)
    today = today_override or now.date()
    lock_today = today_override is None and now.hour >= KICKOFF_LOCK_HOUR_UTC

    predictions = read_rows("predictions.csv")
    candidates = []
    for row in predictions:
        if not row["fixture_date"]:
            continue
        fixture_date = date.fromisoformat(row["fixture_date"][:10])
        if fixture_date < today or fixture_date > today + timedelta(days=MAX_DAYS_AHEAD):
            continue
        if fixture_date == today and lock_today:
            continue
        candidates.append((row, fixture_date))

    print(
        f"\nToday (UTC): {today}. Fixtures in window: {len(candidates)}"
        + (" (today's fixtures locked)" if lock_today else "")
    )
    if not candidates:
        print("Nothing to do.")
        return

    data = fpl_news.get_data()
    generated_at = now.isoformat()
    rows = []
    skipped = 0
    failed = 0

    for row, fixture_date in candidates:
        home, away = row["home"], row["away"]
        home_team = fpl_news.find_team(data, home)
        away_team = fpl_news.find_team(data, away)
        if home_team is None or away_team is None:
            print(f"SKIP {home} v {away}: team not found in FPL data")
            skipped += 1
            continue

        home_info = filtered_news(data, home_team, fixture_date)
        away_info = filtered_news(data, away_team, fixture_date)
        home_text = news_body(home_info)
        away_text = news_body(away_info)
        model_probs = (
            float(row["p_home"]) * 100,
            float(row["p_draw"]) * 100,
            float(row["p_away"]) * 100,
        )
        state = build_state(home, away, model_probs, home_text, away_text)

        try:
            choice, confidence, shift, jev_probs = ask_jev(api_key, state)
        except (requests.RequestException, KeyError) as exc:
            print(f"FAIL {home} v {away}: {exc}")
            failed += 1
            continue

        adj_h, adj_d, adj_a = adjust(model_probs, shift)
        print(
            f"{home} v {away} ({fixture_date}): pick {choice}, shift {shift:+.1f} -> "
            f"home {adj_h:.1f}% (model {model_probs[0]:.1f}%)"
        )
        rows.append(
            {
                "home": home,
                "away": away,
                "fixture_date": fixture_date.isoformat(),
                "p_home": float(row["p_home"]),
                "p_draw": float(row["p_draw"]),
                "p_away": float(row["p_away"]),
                "news": {"home": home_info, "away": away_info},
                "news_text": f"{home}: {home_text}\n{away}: {away_text}",
                "jev_choice": choice,
                "jev_confidence": confidence,
                "jev_probabilities": jev_probs,
                "shift_points": round(shift, 2),
                "adj_home": round(adj_h / 100, 4),
                "adj_draw": round(adj_d / 100, 4),
                "adj_away": round(adj_a / 100, 4),
                "jev_model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "generated_at": generated_at,
            }
        )

    print(f"\nBuilt {len(rows)} rows, skipped {skipped}, failed {failed}.")
    if dry_run:
        print("Dry run: nothing uploaded.")
        return
    if rows:
        db.upsert("jev_predictions", rows, "home,away,fixture_date")
        print(f"Uploaded {len(rows)} rows to jev_predictions.")
    if failed and not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()