"""Flatten the per-game event files into one big JSONL.

Reads data/processed/events/**/*.json (output of build_event_collection.py)
and writes data/processed/events_all.jsonl — one event per line, with the
game-level fields (match_id, teams, official split) copied onto each event.
The `split` column records SoccerNet's official assignment; resample as you
like, but always split at match level (all events of a game stay together).

Usage: python scripts/flatten_events.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENTS_DIR = PROJECT_ROOT / "data" / "processed" / "events"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "events_all.jsonl"


def main() -> None:
    count = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for game_file in sorted(EVENTS_DIR.rglob("*.json")):
            game = json.loads(game_file.read_text(encoding="utf-8"))
            for event in game["events"]:
                row = {
                    "match_id": game["match_id"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "split": game["split"],
                    **event,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(f"{OUT_PATH.name}: {count} events")


if __name__ == "__main__":
    main()
