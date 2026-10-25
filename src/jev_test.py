"""Jev smoke test: one fixture, one Choice question, print the raw response."""
import json
import os
import sys
from pathlib import Path

import requests

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
LADDER = [-12, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12]


def load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def label(n):
    if n == 0:
        return "no_change"
    return f"{'up' if n > 0 else 'down'}_{abs(n)}"


def describe(n):
    if n == 0:
        return "The news does not change the home team's chance of winning."
    direction = "increases" if n > 0 else "decreases"
    return f"The news {direction} the home team's chance of winning by about {abs(n)} percentage points."


STATE = (
    "Premier League match: Arsenal (home) vs Chelsea (away).\n"
    "Statistical model probabilities: home win 52%, draw 25%, away win 23%.\n"
    "Team news for Arsenal: no reported injuries or suspensions.\n"
    "Team news for Chelsea: first-choice goalkeeper out for 4 weeks; "
    "top scorer a doubt (50% chance of playing).\n"
    "(This is invented test data.)"
)

INSTRUCTIONS = (
    "Given the team news, how should the home team's win probability be adjusted "
    "from the statistical model's figure? Pick the closest option."
)


def build_body(shape):
    question = {"type": "choice", "instructions": INSTRUCTIONS}
    if shape == "criteria":
        question["criteria"] = {label(n): describe(n) for n in LADDER}
    else:
        question["options"] = [label(n) for n in LADDER]
    return {"model": MODEL, "state": STATE, "questions": {"shift": question}}


def call(api_key, shape):
    resp = requests.post(
        URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=build_body(shape),
        timeout=30,
    )
    print(f"--- shape={shape} status={resp.status_code} ---")
    try:
        print(json.dumps(resp.json(), indent=2))
    except ValueError:
        print(resp.text)
    return resp


def main():
    load_env()
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        sys.exit("TYPESAFE_API_KEY not found in epl-predictor/.env")
    resp = call(api_key, "criteria")
    if resp.status_code in (400, 422):
        print("\nRetrying with the options-list shape...\n")
        call(api_key, "options")


if __name__ == "__main__":
    main()