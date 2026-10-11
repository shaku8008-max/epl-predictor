import json
from pathlib import Path

import requests

URL = "https://understat.com/getLeagueData/EPL/2024"
OUT_FILE = Path("data/raw/understat_EPL_2024.json")
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://understat.com/league/EPL/2024",
}


def describe(value, depth: int = 0, max_depth: int = 3) -> None:
    """Print the shape of nested JSON without dumping all of it."""
    pad = "  " * depth
    if isinstance(value, dict):
        keys = list(value)
        print(f"{pad}dict with {len(keys)} keys: {keys[:8]}")
        if depth < max_depth and keys:
            print(f"{pad}first key '{keys[0]}':")
            describe(value[keys[0]], depth + 1, max_depth)
    elif isinstance(value, list):
        print(f"{pad}list with {len(value)} items")
        if depth < max_depth and value:
            print(f"{pad}first item:")
            describe(value[0], depth + 1, max_depth)
    else:
        print(f"{pad}{type(value).__name__}: {str(value)[:80]}")


def main() -> None:
    response = requests.get(URL, headers=HEADERS, timeout=30)
    print(f"status {response.status_code}, type {response.headers.get('Content-Type')}, {len(response.content)} bytes")
    response.raise_for_status()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_bytes(response.content)

    try:
        data = response.json()
    except ValueError:
        print("Not JSON. First 500 characters:")
        print(response.text[:500])
        return

    print("\nTop level:")
    describe(data, max_depth=0)
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"\n'{key}':")
            describe(value, depth=1)


if __name__ == "__main__":
    main()