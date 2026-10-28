"""Build per-match event collections from SoccerNet Labels-v2.json.

Each game produces one JSON in data/processed/events/<league>/<season>/<game>.json:
  - game-level metadata (teams, score, date, official split, source of events)
  - one entry per Labels-v2 annotation, with a context window and, when the
    game is covered by SoccerNet-Caption, the nearest human commentary text.

Usage (from project root):
    python scripts/build_event_collection.py
    python scripts/build_event_collection.py --window 10 --caption-tolerance 30
"""

import argparse
import json
import re
from pathlib import Path

from SoccerNet.utils import getListGames

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "soccernet"
CAPTION_DIR = DATA_DIR / "caption-2023"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "events"

# Of the 17 SoccerNet action-spotting classes, these are the ones worth asking
# about; the rest (Ball out of play, Throw-in, Clearance, Kick-off, Indirect
# free-kick) are high-frequency filler with no retrieval value and are
# dropped at the source. Keep in sync with generate_queries.SALIENT_LABELS.
SALIENT_LABELS = {
    "Goal", "Penalty", "Corner", "Direct free-kick", "Shots on target",
    "Shots off target", "Substitution", "Offside", "Yellow card",
    "Red card", "Yellow->red card", "Foul",
}

# Labels a caption sentence reliably names one actor for ("the foul by X",
# "the goal by X", "the yellow card X received") — the rest (Corner, Shots,
# Offside, Substitution, Penalty) either aren't asked about by player or the
# caption doesn't consistently single out one name for them.
PLAYER_LABELS = {"Goal", "Yellow card", "Red card", "Yellow->red card",
                 "Direct free-kick", "Foul"}


def game_time_to_seconds(game_time: str) -> tuple[int, int]:
    """'1 - 05:32' -> (half=1, t=332 seconds into that half)."""
    half_str, clock = game_time.split(" - ")
    minutes, seconds = clock.split(":")
    return int(half_str), int(minutes) * 60 + int(seconds)


def load_split_map() -> dict[str, str]:
    """Map 'league/season/game' -> official SoccerNet split."""
    mapping: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        for game in getListGames(split=split):
            mapping[game.replace("\\", "/")] = split
    return mapping


def load_captions(game_rel: str) -> list[dict]:
    """Return caption annotations for a game as [{half, t, text, ...}], or []."""
    path = CAPTION_DIR / game_rel / "Labels-caption.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    captions = []
    for i, ann in enumerate(data.get("annotations", [])):
        try:
            half, t = game_time_to_seconds(ann["gameTime"])
        except (ValueError, KeyError):
            continue
        captions.append(
            {
                "caption_id": f"c{i:03d}",
                "half": half,
                "t": t,
                "text": ann.get("description", ""),
                "anonymized": ann.get("anonymized", ""),
                "label": ann.get("label", ""),
                "important": ann.get("important", False),
            }
        )
    return captions


def load_roster(game_rel: str) -> list[dict]:
    """Return [{long_name, short_name, team}] for both squads (starters + bench)."""
    path = CAPTION_DIR / game_rel / "Labels-caption.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    roster = []
    for team in ("home", "away"):
        for p in data.get("lineup", {}).get(team, {}).get("players", []) or []:
            long_name = (p.get("long_name") or "").strip()
            short_name = (p.get("short_name") or "").strip()
            if long_name:
                roster.append({"long_name": long_name, "short_name": short_name, "team": team})
    return roster


def _name_hits(text: str, names: list[str]) -> set[str]:
    return {n for n in names if re.search(rf"\b{re.escape(n)}\b", text)}


def extract_player(caption_text: str, event_team: str, roster: list[dict]) -> str | None:
    """The one roster player (on `event_team`) named in this caption, if unambiguous.

    Captions consistently name the acting player, e.g. "The foul by John Terry
    (Chelsea) ..." or "... after John Terry (Chelsea) brought his opponent
    down." Filtering candidates to the event's own team resolves the common
    two-name case (fouler + the player fouled) without parsing grammar.
    """
    if event_team not in ("home", "away"):
        return None
    team_roster = [p for p in roster if p["team"] == event_team]

    long_hits = _name_hits(caption_text, [p["long_name"] for p in team_roster])
    if len(long_hits) == 1:
        return next(iter(long_hits))
    if len(long_hits) > 1:
        return None  # two same-team players named — genuinely ambiguous

    short_names = [p["short_name"] for p in team_roster if p["short_name"]]
    short_hits = _name_hits(caption_text, short_names)
    if len(short_hits) == 1:
        hit = next(iter(short_hits))
        return next(p["long_name"] for p in team_roster if p["short_name"] == hit)
    return None


# Event label -> caption labels that describe the same kind of moment.
CAPTION_LABEL_MAP = {
    "Corner": {"corner"},
    "Substitution": {"substitution"},
    "Yellow card": {"y-card"},
    "Red card": {"r-card"},
    "Yellow->red card": {"yr-card"},
    "Goal": {"soccer-ball", "soccer-ball-own"},
    "Penalty": {"penalty", "penalty-missed"},
}

# Fallback: keywords that must appear in the caption text (lowercase) for it
# to count as describing an event of this label. Labels absent here (and not
# in CAPTION_LABEL_MAP) never get a caption attached.
CAPTION_KEYWORDS = {
    "Corner": ["corner"],
    "Substitution": ["substitut", "replaces", "comes on for"],
    "Yellow card": ["yellow", "book"],
    "Red card": ["red card", "sent off", "dismiss"],
    "Yellow->red card": ["second yellow", "sent off"],
    "Goal": ["goal", "scores", "equalise", "equalize", "back of the net"],
    "Penalty": ["penalty"],
    "Foul": ["foul", "brings down", "free kick"],
    "Direct free-kick": ["free kick", "free-kick"],
    "Offside": ["offside"],
    "Shots on target": ["shot", "save", "effort", "attempt", "strike", "header"],
    "Shots off target": ["shot", "wide", "over the bar", "misses", "effort",
                         "attempt", "post", "crossbar"],
    "Throw-in": ["throw"],
    "Clearance": ["clear"],
    "Kick-off": ["kick off", "kick-off", "underway", "first half begins",
                 "second half begins"],
}


def caption_compatible(cap: dict, event_label: str) -> str | None:
    """Return 'label'/'keyword' if the caption plausibly describes this event."""
    if cap["label"] in CAPTION_LABEL_MAP.get(event_label, ()):
        return "label"
    text = cap["text"].lower()
    if any(kw in text for kw in CAPTION_KEYWORDS.get(event_label, ())):
        return "keyword"
    return None


def assign_captions(events: list[dict], captions: list[dict], tolerance: int,
                    keyword_tolerance: int) -> dict[int, dict]:
    """One-to-one event<->caption assignment for a whole match.

    A comment describes one moment, so each caption may be attached to at most
    one event (and each event gets at most one caption). Candidate pairs must
    be same-half and semantically compatible; label-tagged matches are allowed
    within `tolerance` s, weaker keyword matches only within the stricter
    `keyword_tolerance` s. Greedy assignment: label matches before keyword
    matches, closer in time first. Returns {event_index: caption_dict}.
    """
    candidates = []
    for ei, ev in enumerate(events):
        for cap in captions:
            if cap["half"] != ev["half"]:
                continue
            dt = abs(cap["t"] - ev["t"])
            how = caption_compatible(cap, ev["label"])
            if how == "label" and dt <= tolerance:
                candidates.append((0, dt, ei, cap))
            elif how == "keyword" and dt <= keyword_tolerance:
                candidates.append((1, dt, ei, cap))

    assigned: dict[int, dict] = {}
    used_captions: set[str] = set()
    for rank, dt, ei, cap in sorted(candidates, key=lambda c: (c[0], c[1])):
        if ei in assigned or cap["caption_id"] in used_captions:
            continue
        used_captions.add(cap["caption_id"])
        assigned[ei] = {
            "caption_id": cap["caption_id"],
            "text": cap["text"],
            "anonymized": cap["anonymized"],
            "important": cap["important"],
            "dt": dt,
            "match_type": "label" if rank == 0 else "keyword",
        }
    return assigned


def build_game(labels_path: Path, split_map: dict[str, str], window: int, tolerance: int) -> dict:
    game_rel = labels_path.parent.relative_to(DATA_DIR).as_posix()
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    captions = load_captions(game_rel)
    roster = load_roster(game_rel)

    events = []
    for ann in data["annotations"]:
        if ann["label"] not in SALIENT_LABELS:
            continue  # drop filler events
        if ann.get("visibility") != "visible":
            continue  # drop events not shown on screen
        half, t = game_time_to_seconds(ann["gameTime"])
        # `position` is the annotation time in ms — finer than the MM:SS string.
        position_ms = int(ann["position"]) if str(ann.get("position", "")).isdigit() else None
        if position_ms is not None:
            t = round(position_ms / 1000)
        events.append(
            {
                "event_id": f"e{len(events):03d}",
                "half": half,
                "t": t,
                "label": ann["label"],
                "team": ann.get("team", "not applicable"),
                "visibility": ann.get("visibility", ""),
                "window": [max(0, t - window), t + window],
                "caption": None,
                "player": None,
            }
        )

    for ei, cap in assign_captions(events, captions, tolerance,
                                   keyword_tolerance=min(15, tolerance)).items():
        events[ei]["caption"] = cap

    if roster:
        for ev in events:
            if ev["label"] in PLAYER_LABELS and ev["caption"]:
                ev["player"] = extract_player(ev["caption"]["text"], ev["team"], roster)

    return {
        "match_id": game_rel,
        "split": split_map.get(game_rel, "unknown"),
        "home_team": data.get("gameHomeTeam", ""),
        "away_team": data.get("gameAwayTeam", ""),
        "score": data.get("gameScore", ""),
        "date": data.get("gameDate", ""),
        "source": "gt",
        "has_captions": bool(captions),
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=10, help="context half-window in seconds")
    parser.add_argument("--caption-tolerance", type=int, default=30,
                        help="max |dt| in seconds to attach a caption to an event")
    args = parser.parse_args()

    split_map = load_split_map()
    label_files = sorted(p for p in DATA_DIR.rglob("Labels-v2.json")
                         if CAPTION_DIR not in p.parents)

    n_events = n_with_caption = n_games_captioned = 0
    n_player_eligible = n_with_player = 0
    per_split: dict[str, int] = {}
    for labels_path in label_files:
        game = build_game(labels_path, split_map, args.window, args.caption_tolerance)
        out_path = OUT_DIR / (game["match_id"] + ".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(game, indent=2, ensure_ascii=False), encoding="utf-8")

        n_events += len(game["events"])
        n_with_caption += sum(1 for e in game["events"] if e["caption"])
        n_games_captioned += game["has_captions"]
        n_player_eligible += sum(1 for e in game["events"] if e["label"] in PLAYER_LABELS)
        n_with_player += sum(1 for e in game["events"] if e["player"])
        per_split[game["split"]] = per_split.get(game["split"], 0) + 1

    print(f"Games processed:        {len(label_files)}  (per split: {per_split})")
    print(f"Games with captions:    {n_games_captioned}")
    print(f"Total events:           {n_events}")
    print(f"Events with a caption:  {n_with_caption} ({100 * n_with_caption / max(1, n_events):.1f}%)")
    print(f"Player-eligible events: {n_player_eligible} ({', '.join(sorted(PLAYER_LABELS))})")
    print(f"  attributed to a player: {n_with_player} "
          f"({100 * n_with_player / max(1, n_player_eligible):.1f}%)")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
