"""Loading helpers for the event/query retrieval benchmark. Schema: handoff/DATA.md."""
import json
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_split(split: str) -> tuple[list[dict], list[dict]]:
    """Return (events, queries) for a split ('train' | 'valid' | 'test')."""
    events = load_jsonl(DATASET_DIR / split / "events.jsonl")
    queries = load_jsonl(DATASET_DIR / split / "queries.jsonl")
    return events, queries


def group_events_by_match(events: list[dict]) -> dict[str, list[dict]]:
    by_match: dict[str, list[dict]] = {}
    for e in events:
        by_match.setdefault(e["match_id"], []).append(e)
    return by_match


def event_text(event: dict) -> str:
    """Render an event as text, for the text-only baselines in src/baselines.py."""
    parts = [event["label"]]
    team = event.get("team")
    if team and team != "not applicable":
        parts.append(f"by the {team} team")
    player = event.get("player")
    if player:
        parts.append(f"({player})")
    caption = event.get("caption")
    if caption and caption.get("text"):
        parts.append(caption["text"])
    return " ".join(parts)


def query_text(query: dict) -> str:
    return query.get("text_for_training") or query["text"]
