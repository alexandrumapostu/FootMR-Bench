"""Generate retrieval queries with exact relevance sets from the event collections.

Templates are restricted to what a fan would plausibly type when looking for
a highlight, not systematic enumeration (no "6th foul by Burnley"). Per game,
per salient label (+ the "Card" superlabel merging yellow/red/second-yellow):

  - first/last {event} in the first half   ("the last X before halftime")
  - first/last {event} in the second half  ("the first X after halftime")
  - first/last {event} by the home/away team
  - every {event} [in a half] [by a team]  (multi-answer, kept from before)

Plus two relational families anchored on the two moments fans actually
reference other events against:

  - "{event} before the goal" / "...before {team}'s goal" /
    "...before {team}'s {ordinal} goal" (disambiguated by how many goals
    exist in the match / for that team)
  - "{event} before the red card" / "...before {team}'s red card"
    (red-type cards — Red card or a second yellow — are rare enough that
    this is almost always unambiguous)

Plus player-attributed queries for the labels build_event_collection.py can
name an actor for (Goal, Foul, Direct free-kick, Yellow/Red/second-yellow
card), using the `player` field it extracts from caption text:

  - "the goal by Messi" / "the free kick by Messi" / "the foul by Ramos"
  - "the yellow card Ramos received" / "...red card..." (only these two read
    naturally the other way round)
  - first/last/every variants when a player has 2+ such events in a match

Relevance is computed by filtering/nearest-preceding-event, so it is correct
by construction.

Stage 2 (optional, --rephrase): send each query's constraints + caption context
to a local Ollama model, which writes a natural paraphrase into `text_llm`.
The LLM never changes `relevant` — only wording.

Usage:
    python scripts/generate_queries.py
    python scripts/generate_queries.py --rephrase --ollama-model qwen2.5:7b
"""

import argparse
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENTS_DIR = PROJECT_ROOT / "data" / "processed" / "events"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "queries.jsonl"
LEGEND_PATH = PROJECT_ROOT / "data" / "processed" / "template_legend.json"

# Labels worth asking about. Filler labels (Ball out of play, Throw-in,
# Clearance, Kick-off, Indirect free-kick) are dropped in build_event_collection.py
# and never reach this file. Keep in sync with build_event_collection.SALIENT_LABELS.
SALIENT_LABELS = {
    "Goal": "goal",
    "Penalty": "penalty",
    "Corner": "corner",
    "Direct free-kick": "free kick",
    "Shots on target": "shot on target",
    "Shots off target": "shot off target",
    "Substitution": "substitution",
    "Offside": "offside",
    "Yellow card": "yellow card",
    "Red card": "red card",
    "Yellow->red card": "second yellow card",
    "Foul": "foul",
}

# "Card" is a synthetic label merging the three card types: a fan asks about
# "the first booking of the match", not "the first yellow-card-or-red-card".
CARD_LABELS = {"Yellow card", "Red card", "Yellow->red card"}
RED_CARD_LABELS = {"Red card", "Yellow->red card"}

# Labels build_event_collection.py attaches a `player` name to (see its
# PLAYER_LABELS / extract_player). "the goal by Messi" / "the free kick by
# Messi" / "the foul by Ramos" use "by"; cards read more naturally the other
# way round: "the yellow card X received".
PLAYER_BY_STYLE = {"Goal": "goal", "Foul": "foul", "Direct free-kick": "free kick"}
PLAYER_RECEIVED_STYLE = {"Yellow card": "yellow card", "Red card": "red card",
                         "Yellow->red card": "second yellow card"}

ORDINAL_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
HALF_WORD = {1: "first", 2: "second"}

# Absolute match-minute markers "before/after minute X" queries anchor on
# (displayed-minute convention: half 2 continues from 45, not from 0).
MINUTE_MARKERS = (15, 30, 45, 60, 75)

# Structural shape of each query, independent of which label/team/player it
# names — the axis a train/test template-holdout split slices on (see
# HELDOUT_TEMPLATES below).
TEMPLATE_LEGEND = {
    "player_single": "the {event} by {player} / the {event} {player} received "
                     "— the player's only such event in the match",
    "player_first": "the first {event} by {player} — player had 2+ such events",
    "player_last": "the last {event} by {player} — player had 2+ such events",
    "player_every": "every {event} by {player} (multi-answer) — player had 2+ such events",
    "half_first": "the first {event} within a half (\"in the first half\" / \"after halftime\")",
    "half_last": "the last {event} within a half (\"before halftime\" / \"of the match\")",
    "half_every": "every {event} within a half (multi-answer)",
    "team_first": "the first {event} by a team",
    "team_last": "the last {event} by a team",
    "team_every": "every {event} by a team (multi-answer)",
    "team_half_every": "every {event} by a team within a half (multi-answer)",
    "label_every": "every {event} across the whole match (multi-answer)",
    "before_goal": "the {event} immediately preceding a goal "
                   "(unscoped / by team / by team+ordinal, whichever is unambiguous)",
    "before_red_card": "the {event} immediately preceding a red-type card "
                       "(unscoped / by team, whichever is unambiguous)",
    "before_minute": "the {event} immediately before match minute {minute} "
                     "(minute in {15,30,45,60,75})",
    "after_minute": "the {event} immediately after match minute {minute} "
                    "(minute in {15,30,45,60,75})",
    "every_event_before_minute": "every {event} before match minute {minute} (multi-answer)",
    "every_event_after_minute": "every {event} after match minute {minute} (multi-answer)",
}

# Templates never shown during training (match-level split == "train"); they
# only appear in valid/test, so eval measures generalization to unseen query
# *shapes*, not just unseen matches (2026-08-20 meeting: keep 2 of the 4 new
# minute-anchored templates in train so the model has SOME exposure to
# "before/after minute" phrasing; hold out the other 2 as the genuinely new
# shapes, plus 3 low-volume templates whose pattern is already covered by an
# in-train sibling — player_first/player_last (225 queries each; same
# "first/last within a group" shape as half_first/half_last and
# team_first/team_last, just grouped by player instead) and before_red_card
# (44 queries, too rare to teach the shape from anyway). Together these 3 are
# <1% of all queries, so excluding them from train costs ~nothing.
HELDOUT_TEMPLATES = {
    "after_minute", "every_event_before_minute",
    "player_first", "player_last", "before_red_card",
}


def absolute_minute(half: int, t: int) -> int:
    """Displayed match minute, e.g. half=2, t=61 -> minute 46 (not minute 2)."""
    return t // 60 + 1 if half == 1 else 45 + t // 60 + 1


def label_pool(events: list[dict], label: str) -> list[dict]:
    """This label's events (or, for 'Card', all card-type events), time-sorted."""
    if label == "Card":
        pool = [e for e in events if e["label"] in CARD_LABELS]
    else:
        pool = [e for e in events if e["label"] == label]
    return sorted(pool, key=lambda e: (e["half"], e["t"]))


def noun_for(label: str) -> str:
    return "card" if label == "Card" else SALIENT_LABELS[label]


def caption_context(events: list[dict], relevant: list[str]) -> str | None:
    """Human commentary of the relevant events (for the LLM pass), if any."""
    texts = [e["caption"]["text"] for e in events
             if e["event_id"] in relevant and e.get("caption")]
    return " | ".join(texts) if texts else None


def generate_for_game(game_file: Path, max_hits: int) -> list[dict]:
    game = json.loads(game_file.read_text(encoding="utf-8"))
    events = game["events"]
    home, away = game["home_team"], game["away_team"]
    queries: list[dict] = []
    seen_hit_sets: set[frozenset] = set()

    def emit(text: str, relevant: list[dict], constraints: dict, template_id: str) -> None:
        n = len(relevant)
        if not 1 <= n <= max_hits:
            return
        ids = [e["event_id"] for e in relevant]
        key = frozenset(ids)
        if key in seen_hit_sets:  # most natural phrasing wins (emitted first)
            return
        seen_hit_sets.add(key)
        queries.append({
            "match_id": game["match_id"],
            "split": game["split"],
            "text": text,
            "template_id": template_id,
            "constraints": constraints,
            "relevant": ids,
            "n_relevant": n,
            "caption_context": caption_context(events, ids),
            "template_seen_in_train": template_id not in HELDOUT_TEMPLATES,
        })

    # --- player-attributed: "the goal by Messi", "the yellow card Ramos received" ---
    # Emitted first: when a named-player query and a generic structural query
    # (e.g. "the last red card of the match") would resolve to the same event,
    # the named one is strictly more specific/natural, so it should claim the
    # hit-set and the generic phrasing should be the one skipped by dedup.
    for label, noun in {**PLAYER_BY_STYLE, **PLAYER_RECEIVED_STYLE}.items():
        pool = label_pool(events, label)
        by_player: dict[str, list[dict]] = defaultdict(list)
        for e in pool:
            if e.get("player"):
                by_player[e["player"]].append(e)

        received = label in PLAYER_RECEIVED_STYLE
        for player, evs in by_player.items():
            base = {"label": label, "player": player}
            if len(evs) == 1:
                text = (f"the {noun} {player} received" if received
                        else f"the {noun} by {player}")
                emit(text, evs, base, "player_single")
                continue
            first_text = (f"the first {noun} {player} received" if received
                          else f"the first {noun} by {player}")
            last_text = (f"the last {noun} {player} received" if received
                        else f"the last {noun} by {player}")
            every_text = (f"every {noun} {player} received" if received
                         else f"every {noun} by {player}")
            emit(first_text, [evs[0]], {**base, "which": "first"}, "player_first")
            emit(last_text, [evs[-1]], {**base, "which": "last"}, "player_last")
            emit(every_text, evs, base, "player_every")

    labels = list(SALIENT_LABELS) + ["Card"]

    for label in labels:
        noun = noun_for(label)
        pool = label_pool(events, label)
        if not pool:
            continue

        for half in (1, 2):
            half_pool = [e for e in pool if e["half"] == half]
            if not half_pool:
                continue
            first_text = (f"the first {noun} in the first half" if half == 1
                          else f"the first {noun} after halftime")
            last_text = (f"the last {noun} before halftime" if half == 1
                         else f"the last {noun} of the match")
            emit(first_text, [half_pool[0]],
                 {"label": label, "half": half, "which": "first"}, "half_first")
            emit(last_text, [half_pool[-1]],
                 {"label": label, "half": half, "which": "last"}, "half_last")
            if len(half_pool) > 1:
                emit(f"every {noun} in the {HALF_WORD[half]} half", half_pool,
                     {"label": label, "half": half}, "half_every")

        for team, team_name in (("home", home), ("away", away)):
            team_pool = [e for e in pool if e["team"] == team]
            if not team_pool:
                continue
            emit(f"the first {noun} by {team_name}", [team_pool[0]],
                 {"label": label, "team": team, "team_name": team_name, "which": "first"}, "team_first")
            emit(f"the last {noun} by {team_name}", [team_pool[-1]],
                 {"label": label, "team": team, "team_name": team_name, "which": "last"}, "team_last")
            if len(team_pool) > 1:
                emit(f"every {noun} by {team_name}", team_pool,
                     {"label": label, "team": team, "team_name": team_name}, "team_every")
            for half in (1, 2):
                th_pool = [e for e in team_pool if e["half"] == half]
                if len(th_pool) > 1:
                    emit(f"every {noun} by {team_name} in the {HALF_WORD[half]} half",
                         th_pool, {"label": label, "team": team,
                                   "team_name": team_name, "half": half}, "team_half_every")

        if len(pool) > 1:
            emit(f"every {noun}", pool, {"label": label}, "label_every")

        # --- minute-anchored: "{event} before/after minute X" (Vali's addition) ---
        for minute in MINUTE_MARKERS:
            before = [e for e in pool if absolute_minute(e["half"], e["t"]) < minute]
            after = [e for e in pool if absolute_minute(e["half"], e["t"]) >= minute]
            if before:
                emit(f"the {noun} before minute {minute}", [before[-1]],
                     {"label": label, "before_minute": minute}, "before_minute")
                if len(before) > 1:
                    emit(f"every {noun} before minute {minute}", before,
                         {"label": label, "before_minute": minute}, "every_event_before_minute")
            if after:
                emit(f"the {noun} after minute {minute}", [after[0]],
                     {"label": label, "after_minute": minute}, "after_minute")
                if len(after) > 1:
                    emit(f"every {noun} after minute {minute}", after,
                         {"label": label, "after_minute": minute}, "every_event_after_minute")

    # --- relational: "{event} before the goal" / "...before the red card" ---
    goals = label_pool(events, "Goal")
    reds = [e for e in events if e["label"] in RED_CARD_LABELS]
    reds.sort(key=lambda e: (e["half"], e["t"]))

    team_goal_totals = {"home": sum(1 for g in goals if g["team"] == "home"),
                        "away": sum(1 for g in goals if g["team"] == "away")}
    team_goal_seen = {"home": 0, "away": 0}

    for label, noun in SALIENT_LABELS.items():
        if label == "Goal":
            continue
        pool = label_pool(events, label)
        if not pool:
            continue

        for g in goals:
            preceding = [e for e in pool if (e["half"], e["t"]) < (g["half"], g["t"])]
            if not preceding:
                continue
            anchor = preceding[-1]
            team_goal_seen[g["team"]] += 1 if g["team"] in team_goal_seen else 0
            if len(goals) == 1:
                text = f"the {noun} before the goal"
            elif g["team"] not in ("home", "away"):
                continue
            elif team_goal_totals[g["team"]] == 1:
                team_name = home if g["team"] == "home" else away
                text = f"the {noun} before {team_name}'s goal"
            else:
                team_name = home if g["team"] == "home" else away
                ordinal = ORDINAL_WORDS.get(team_goal_seen[g["team"]])
                if not ordinal:
                    continue
                text = f"the {noun} before {team_name}'s {ordinal} goal"
            emit(text, [anchor], {"label": label, "before": "goal", "goal_event_id": g["event_id"]},
                 "before_goal")

        for card in reds:
            preceding = [e for e in pool if (e["half"], e["t"]) < (card["half"], card["t"])]
            if not preceding:
                continue
            anchor = preceding[-1]
            if len(reds) == 1:
                text = f"the {noun} before the red card"
            elif card["team"] in ("home", "away"):
                team_name = home if card["team"] == "home" else away
                text = f"the {noun} before {team_name}'s red card"
            else:
                text = f"the {noun} before the red card"
            emit(text, [anchor], {"label": label, "before": "red_card", "card_event_id": card["event_id"]},
                 "before_red_card")
        team_goal_seen = {"home": 0, "away": 0}

    return queries


def ollama_rephrase(query: dict, model: str, url: str) -> str:
    prompt = (
        "You write short search queries for moments in a soccer match.\n"
        f"Structured description: {json.dumps(query['constraints'])}\n"
        + (f"Commentary about the moment(s): {query['caption_context']}\n"
           if query["caption_context"] else "")
        + f"Current plain query: \"{query['text']}\"\n"
        "Rewrite it as ONE natural, specific search query a fan would type. "
        "Do not add facts that are not in the description or commentary. "
        "Answer with the query only."
    )
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(f"{url}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"].strip().strip('"')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-hits", type=int, default=6,
                        help="drop queries with more relevant events than this")
    parser.add_argument("--rephrase", action="store_true",
                        help="add LLM paraphrase (text_llm) via a local Ollama server")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    all_queries = []
    for game_file in sorted(EVENTS_DIR.rglob("*.json")):
        all_queries.extend(generate_for_game(game_file, args.max_hits))

    if args.rephrase:
        for i, q in enumerate(all_queries):
            q["text_llm"] = ollama_rephrase(q, args.ollama_model, args.ollama_url)
            if (i + 1) % 500 == 0:
                print(f"  rephrased {i + 1}/{len(all_queries)}")

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for i, q in enumerate(all_queries):
            out.write(json.dumps({"query_id": f"q{i:06d}", **q}, ensure_ascii=False) + "\n")
    LEGEND_PATH.write_text(json.dumps(TEMPLATE_LEGEND, indent=2), encoding="utf-8")

    single = sum(1 for q in all_queries if q["n_relevant"] == 1)
    print(f"Total queries: {len(all_queries)}")
    print(f"Single-answer: {single}   multi-answer: {len(all_queries) - single}")
    print(f"With caption context: {sum(1 for q in all_queries if q['caption_context'])}")
    by_template = Counter(q["template_id"] for q in all_queries)
    print("By template_id:")
    for tid, n in by_template.most_common():
        seen = "" if tid in HELDOUT_TEMPLATES else " (seen in train)"
        print(f"  {tid:<26} {n:6d}{seen}")
    n_heldout = sum(1 for q in all_queries if not q["template_seen_in_train"])
    print(f"Template holdout: {len(HELDOUT_TEMPLATES)}/{len(by_template)} template_ids "
          f"({100 * len(HELDOUT_TEMPLATES) / len(by_template):.0f}%) never appear when "
          f"split==train; {n_heldout} queries ({100 * n_heldout / len(all_queries):.1f}%) "
          f"carry template_seen_in_train=false")
    print(f"Output: {OUT_PATH}")
    print(f"Legend: {LEGEND_PATH}")


if __name__ == "__main__":
    main()
