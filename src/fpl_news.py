"""FPL team-news fetcher: prints key injuries/suspensions for two teams."""
import sys

import requests

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
KEY_PLAYER_SHARE = 0.4  # regulars = at least this share of the season's max minutes
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
OUT_STATUSES = {"i", "s", "u", "n"}
STATUS_LABEL = {
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "not in squad",
    "d": "doubtful",
}
ALIASES = {
    "manchester city": "man city",
    "manchester united": "man utd",
    "man united": "man utd",
    "tottenham": "spurs",
    "tottenham hotspur": "spurs",
    "nottingham forest": "nott'm forest",
    "wolverhampton wanderers": "wolves",
    "newcastle united": "newcastle",
    "leeds united": "leeds",
    "west ham united": "west ham",
}


def get_data():
    resp = requests.get(
        URL, headers={"User-Agent": "Mozilla/5.0 (epl-predictor)"}, timeout=30
    )
    print(f"FPL API status: {resp.status_code}")
    resp.raise_for_status()
    return resp.json()


def find_team(data, name):
    wanted = ALIASES.get(name.lower(), name.lower())
    teams = data["teams"]
    for team in teams:
        if wanted in (team["name"].lower(), team["short_name"].lower()):
            return team
    loose = [
        t
        for t in teams
        if wanted in t["name"].lower() or t["name"].lower() in wanted
    ]
    if len(loose) == 1:
        print(f"  (matched '{name}' to FPL team '{loose[0]['name']}')")
        return loose[0]
    return None


def team_news(data, team):
    max_minutes = max(p["minutes"] for p in data["elements"])
    cutoff = KEY_PLAYER_SHARE * max_minutes
    players = []
    for p in data["elements"]:
        if p["team"] != team["id"] or p["minutes"] < cutoff:
            continue
        if p["status"] == "a":
            continue
        chance = p["chance_of_playing_next_round"]
        out = p["status"] in OUT_STATUSES or chance == 0
        players.append(
            {
                "name": p["web_name"],
                "position": POSITIONS.get(p["element_type"], "?"),
                "status": STATUS_LABEL.get(p["status"], p["status"]),
                "level": "out" if out else "doubtful",
                "chance_of_playing": chance,
                "news": p["news"],
                "minutes": p["minutes"],
            }
        )
    players.sort(key=lambda x: x["minutes"], reverse=True)
    return {"team": team["name"], "players": players, "min_minutes": round(cutoff)}


def news_text(info):
    if not info["players"]:
        return f"{info['team']}: no reported injuries or suspensions among key players."
    parts = []
    for p in info["players"]:
        chance = (
            f", {p['chance_of_playing']}% chance of playing"
            if p["chance_of_playing"] is not None
            else ""
        )
        parts.append(f"{p['name']} ({p['position']}, {p['level']}{chance}): {p['news']}")
    return f"{info['team']}: " + "; ".join(parts)


def main():
    home_name = sys.argv[1] if len(sys.argv) > 1 else "Arsenal"
    away_name = sys.argv[2] if len(sys.argv) > 2 else "Chelsea"
    data = get_data()
    teams = []
    for name in (home_name, away_name):
        team = find_team(data, name)
        if team is None:
            print(f"\nNo FPL team matches '{name}'. FPL names are:")
            print(", ".join(sorted(t["name"] for t in data["teams"])))
            sys.exit(1)
        teams.append(team)
    for team in teams:
        info = team_news(data, team)
        print(f"\n=== {info['team']} (regulars = {info['min_minutes']}+ minutes) ===")
        if not info["players"]:
            print("  nothing to report")
        for p in info["players"]:
            print(
                f"  [{p['level'].upper():8}] {p['name']} ({p['position']}, "
                f"{p['minutes']} min): {p['news'] or p['status']}"
            )
    print("\n--- text that would go to Jev ---")
    for team in teams:
        print(news_text(team_news(data, team)))


if __name__ == "__main__":
    main()