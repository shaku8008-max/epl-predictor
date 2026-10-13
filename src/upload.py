"""Upload the files predict.py wrote to Supabase."""
import csv
import os
from datetime import date, timedelta
from pathlib import Path

import requests

OUT_DIR = Path("data/predictions")
ENV_FILE = Path(".env")
ROUND_DAYS = 4


def load_settings() -> tuple[str, str]:
    """Read the URL and key from .env if it exists, otherwise from the environment."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    return os.environ["SUPABASE_URL"].rstrip("/"), os.environ["SUPABASE_SERVICE_KEY"]


def read_rows(name: str) -> list[dict]:
    """A CSV as a list of dicts, with blanks turned into nulls."""
    with open(OUT_DIR / name, newline="") as f:
        return [{k: (v if v != "" else None) for k, v in row.items()} for row in csv.DictReader(f)]


class Database:
    def __init__(self, url: str, key: str) -> None:
        self.base = f"{url}/rest/v1"
        self.headers = {"apikey": key, "Authorization": f"Bearer {key}"}

    def _send(self, method: str, table: str, prefer: str, **kwargs) -> None:
        response = requests.request(
            method,
            f"{self.base}/{table}",
            headers=self.headers | {"Prefer": prefer},
            timeout=30,
            **kwargs,
        )
        if not response.ok:
            raise SystemExit(f"{method} {table} failed: {response.status_code} {response.text}")

    def upsert(self, table: str, rows: list[dict], key_columns: str) -> None:
        self._send(
            "POST", table, "resolution=merge-duplicates,return=minimal",
            params={"on_conflict": key_columns}, json=rows,
        )

    def insert(self, table: str, row: dict) -> None:
        self._send("POST", table, "return=minimal", json=row)

    def delete_older(self, table: str, column: str, stamp: str) -> None:
        self._send("DELETE", table, "return=minimal", params={column: f"lt.{stamp}"})


def next_round(predictions: list[dict]) -> list[dict]:
    """Upcoming fixtures within a few days of the first one."""
    upcoming = [row for row in predictions if row["fixture_date"]]
    if not upcoming:
        return []
    first = min(date.fromisoformat(row["fixture_date"]) for row in upcoming)
    cutoff = first + timedelta(days=ROUND_DAYS)
    return [row for row in upcoming if date.fromisoformat(row["fixture_date"]) <= cutoff]


def main() -> None:
    db = Database(*load_settings())

    predictions = read_rows("predictions.csv")
    for row in predictions:
        if row["fixture_date"]:
            row["fixture_date"] = row["fixture_date"][:10]
    team_stats = read_rows("team_stats.csv")
    results = read_rows("results.csv")
    stamp = predictions[0]["predicted_at"]

    db.upsert("predictions", predictions, "home,away")
    db.delete_older("predictions", "predicted_at", stamp)

    db.upsert("team_stats", team_stats, "team")
    db.delete_older("team_stats", "updated_at", stamp)

    fixtures = next_round(predictions)
    if fixtures:
        db.upsert("prediction_history", fixtures, "home,away,fixture_date")

    db.upsert("results", results, "home,away,fixture_date")

    db.insert("runs", {
        "data_as_of": stamp,
        "model": predictions[0]["model"],
        "model_version": predictions[0]["model_version"],
        "xg_ok": True,
        "status": "ok",
    })

    print(f"Uploaded {len(predictions)} predictions, {len(team_stats)} teams, "
          f"{len(fixtures)} next-round fixtures, {len(results)} results (run stamped {stamp})")


if __name__ == "__main__":
    main()