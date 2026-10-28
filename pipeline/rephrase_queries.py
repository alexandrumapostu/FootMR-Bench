"""Rephrase every query into natural-sounding text via a local Ollama model.

Same two-style design as the earlier Anthropic-API version (now archived at
archive/rephrase_queries_anthropic.py — kept for reference, no longer run):

  A. Conversational — full sentences, mixed statements/questions/possessives
     ("the goal Messi scored", "when did Ramos get booked?").
  B. Search-bar — terse, keyword-forward, articles dropped where natural
     ("Messi goal second half", "Ramos yellow card").

All of TRAIN is style A. VALID and TEST are a 50/50 deterministic split of A
and B (by hashing query_id). Player names, teams, and constraints come
straight from generate_queries.py's `text` and are trusted; the attached
`caption_context` is not — only captions matched with match_type == "label"
(the reliable path, not the loose "keyword" fallback) are ever shown to the
model, and the prompt frames it as optional color, never a source of new
facts. See build_prompt() for the exact instructions.

This calls a **local** model through Ollama instead of a hosted API — no key,
no per-token cost, but also no cloud-scale throughput: expect hours to cover
the full ~54k-query dataset on a single consumer GPU. Progress is
checkpointed — output rows are appended as they complete and already-done
query_ids are skipped on restart — so it's safe to Ctrl-C and resume, or run
unattended in the background.

Usage:
    ollama pull llama3.1:8b        # once
    python scripts/rephrase_queries.py
    python scripts/rephrase_queries.py --limit 20 --workers 2   # cheap dry run
    python scripts/rephrase_queries.py                          # resumes automatically
"""

import argparse
import hashlib
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = PROJECT_ROOT / "data" / "processed" / "queries.jsonl"
EVENTS_DIR = PROJECT_ROOT / "data" / "processed" / "events"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "queries_rephrased.jsonl"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The rephrased, natural-sounding query"},
        "used_caption": {"type": "boolean",
                         "description": "Whether the commentary (not just the structured facts) shaped the wording"},
    },
    "required": ["query", "used_caption"],
    "additionalProperties": False,
}

STYLE_INSTRUCTION = {
    "A": (
        "Reword it as a plain, natural sentence or question a fan would "
        "actually type or ask — a light rephrase of the plain-template "
        "query's wording, not a verbatim copy, but stay understated and "
        "matter-of-fact. Do NOT add slang, nicknames, color-commentary "
        "flair, or exclamations (no \"naughty one\", no \"what a strike!\", "
        "etc.) — just say it the way someone would plainly ask for the "
        "clip, e.g. \"What foul did Diego Costa commit?\" or \"the yellow "
        "card Boyd received\". Vary the phrasing a little across queries, "
        "but keep the tone neutral."
    ),
    "B": (
        "Reword it as a short, keyword-forward search-bar query — the way "
        "someone quickly types into a video search box, not a full sentence. "
        "Drop filler words and articles where natural (e.g. \"Messi goal "
        "second half\", \"Ramos yellow card\", \"corner before Chelsea "
        "goal\"). Still sound like a person typed it in a hurry, not a "
        "database field."
    ),
}

_events_cache: dict[str, dict] = {}


def load_game(match_id: str) -> dict:
    if match_id not in _events_cache:
        path = EVENTS_DIR / (match_id + ".json")
        _events_cache[match_id] = json.loads(path.read_text(encoding="utf-8"))
    return _events_cache[match_id]


def trusted_caption_context(match_id: str, relevant: list[str]) -> str | None:
    """Caption text for `relevant`, but only from high-confidence matches."""
    game = load_game(match_id)
    events_by_id = {e["event_id"]: e for e in game["events"]}
    texts = []
    for eid in relevant:
        ev = events_by_id.get(eid)
        cap = ev.get("caption") if ev else None
        if cap and cap["match_type"] == "label":
            texts.append(cap["text"])
    return " | ".join(texts) if texts else None


def style_for(query: dict) -> str:
    """Train = always A. Valid/test = deterministic 50/50 split of A/B."""
    if query["split"] == "train":
        return "A"
    digest = hashlib.md5(query["query_id"].encode()).hexdigest()
    return "A" if int(digest, 16) % 2 == 0 else "B"


def build_prompt(query: dict, caption: str | None, style: str) -> str:
    # goal_event_id / card_event_id are internal bookkeeping (for the
    # "before X" relational templates) — not user-facing facts. Drop them
    # so the model can't echo a bare event ID into the rephrased text.
    facts = {k: v for k, v in query["constraints"].items() if not k.endswith("_event_id")}
    prompt = (
        "You write short search queries for moments in a soccer match, in "
        "natural human language — genuinely the way a real fan looking for "
        "a highlight would ask for it, not a robotic restatement of facts.\n"
        f"Structured description (ground truth — every fact here must be "
        f"preserved, and nothing outside it is a real fact): {json.dumps(facts)}\n"
    )
    if "which" in facts:
        prompt += (
            f"This is specifically the {facts['which'].upper()} one — the "
            f"rewrite MUST keep that ordinal (the word \"{facts['which']}\", "
            f"or a clear synonym: \"opening\"/\"earliest\" for first, "
            f"\"final\"/\"latest\" for last). Dropping it turns a query about "
            f"one specific event into one that could match several — that is "
            f"a factual error, not a style choice.\n"
        )

    minute_key = "before_minute" if "before_minute" in facts else (
        "after_minute" if "after_minute" in facts else None)
    if minute_key:
        minute = facts[minute_key]
        direction = "before" if minute_key == "before_minute" else "after"
        prompt += (
            f"\"{minute_key}\": {minute} means strictly {direction} the "
            f"{minute}-minute mark of the match — NOT at minute {minute}, "
            f"and NOT the match's final/last minute (a completely different, "
            f"wrong point in time). WRONG example you must not produce: for "
            f"\"after_minute: 60\", do NOT write \"in the last 10/20 minutes\" "
            f"— minute 60 is not \"near the end\", it's specifically minute 60, "
            f"a full 30 minutes before the match ends. The rewrite must keep "
            f"both the direction ({direction}) and the exact number {minute} "
            f"recognizable — never say \"at minute {minute}\", never say "
            f"\"final minute\" or \"last N minutes\", never drop or change "
            f"the number.\n"
        )

    if "label" in facts:
        prompt += (
            f"The event type is \"{facts['label']}\" — the rewrite MUST make "
            f"that recognizable (the word itself or a natural synonym/phrase "
            f"for it). WRONG example: turning \"the foul after minute 60\" "
            f"into \"What happened after minute 60?\" — that drops which "
            f"event type is being asked about entirely, leaving something "
            f"that could mean any event at all. Never generalize away the "
            f"event type like that, even in a short/terse rewrite.\n"
        )

    if facts.get("before") in ("goal", "red_card"):
        prompt += (
            "This query is about the event immediately BEFORE another one "
            "(a goal or red card), not the goal/red card itself — the "
            "rewrite must keep that \"before\" relationship clearly readable "
            "(\"before\", \"leading up to\", \"right before\", etc.). WRONG "
            "example: for a shot on target before a goal, do NOT write "
            "\"the shot on target that was the goal\" or just \"[team]'s "
            "goal shot\" — that describes the goal, not the shot preceding "
            "it, which is a different event entirely.\n"
        )

    if "team" not in facts and "team_name" not in facts and "player" not in facts:
        prompt += (
            "The structured description above does not name a team or "
            "player — do not invent one, even if the commentary below "
            "mentions someone; keep the rewrite generic on that point.\n"
        )

    n = query["n_relevant"]
    if n > 1:
        prompt += (
            f"This query matches {n} separate events in the match, not just "
            f"one — the rephrase must stay general enough to cover all {n} "
            f"of them collectively (e.g. \"every yellow card\", not one "
            f"specific card). Do NOT narrow it to a single instance: do not "
            f"name a specific player, team, or ordinal (\"first\", \"the "
            f"second one\", etc.) unless that exact detail is already in the "
            f"structured description above — even if the commentary below "
            f"happens to describe only one of the {n} events. The wording "
            f"itself must read as plural/general — use \"every\"/\"any\"/"
            f"\"all\" or a plain plural noun (\"fouls\", \"corners\", "
            f"\"yellow cards\"), never a singular noun or indefinite article "
            f"(\"a foul\", \"the corner\"). WRONG: \"Chelsea's foul in the "
            f"second half\" or \"Burnley fouled someone\" for a query that "
            f"matches several fouls — both read as one specific event. "
            f"RIGHT: \"Chelsea's fouls in the second half\".\n"
        )
    if caption:
        prompt += (
            f"Commentary about the moment (may be faulty), for vocabulary/style "
            f"only — do not pull in any fact from it that isn't already in the "
            f"structured description above{', and do not let it narrow a multi-event query to one instance' if n > 1 else ''}: {caption}\n"
        )
    prompt += (
        f"Current plain-template query (a mechanical placeholder, not the "
        f"target wording — don't just echo it back): \"{query['text']}\"\n"
        "The structured description above is not exhaustive — some facts "
        "(e.g. a team name, or which numbered goal/card, like \"first\" or "
        "\"second\") only appear in the plain-template query itself. Treat "
        "both as ground truth together: never drop an identifying detail "
        "that's present in either one, even to keep the rewrite short.\n"
        "Never imply a different, related-but-distinct outcome that isn't "
        "the stated label — e.g. a \"Foul\" is not necessarily a card, a "
        "\"Shot on target\" is not necessarily a goal; don't write \"got "
        "booked\" for a plain foul, or \"scored\" for a shot. Describe "
        "exactly what the label says, nothing implied beyond it.\n"
        f"{STYLE_INSTRUCTION[style]} Do not add any fact not present in "
        "either of the two above.\n"
        "Respond with JSON only: {\"query\": \"...\", \"used_caption\": true|false}."
    )
    return prompt


def ollama_rephrase(prompt: str, model: str, url: str) -> dict:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "format": RESPONSE_SCHEMA,
        "stream": False,
        "options": {"temperature": 0.7},
    }).encode()
    req = urllib.request.Request(f"{url}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(json.loads(resp.read())["response"])


ORDINAL_WORDS = ("first", "second", "third", "fourth", "fifth", "last")
FIRST_SYNONYMS = ("first", "opening", "earliest", "1st")
LAST_SYNONYMS = ("last", "final", "latest", "most recent", "recent")
BEFORE_CONNECTORS = ("before", "prior to", "leading up to", "led up to", "ahead of",
                     "preceding", "in the buildup to", "build-up to", "build up to",
                     "led to", "leads to", "resulted in", "resulting in", "caused")
GENERALITY_WORDS = ("every", "any", "all", "each", "multiple")
# Keywords used to check the label's own noun survived the rephrase at all,
# and (with an "s" appended) to check plurality on multi-answer queries.
# Deliberately generous with synonyms — this only ever triggers a flag (safe
# fallback to the plain template), never a silent accept, so recall matters
# more than precision here.
LABEL_KEYWORDS = {
    "Goal": ("goal",),
    "Penalty": ("penalty", "penalties"),
    "Corner": ("corner", "kick"),
    "Direct free-kick": ("free kick", "free-kick", "freekick", "kick"),
    "Shots on target": ("shot", "target"),
    "Shots off target": ("shot", "wide", "miss", "post", "crossbar"),
    "Substitution": ("substitution", "sub"),
    "Offside": ("offside", "incident"),
    "Yellow card": ("yellow", "card", "book"),
    "Red card": ("red card", "sent off", "dismiss"),
    "Yellow->red card": ("second yellow", "red card", "sent off"),
    "Foul": ("foul", "incident"),
    "Card": ("card", "book"),
}


def flag_narrowing(query: dict, text_llm: str) -> list[str]:
    """Cheap heuristics for the failure modes actually observed: dropping the
    "first"/"last" ordinal that makes a single-answer query unambiguous,
    misreading before_minute/after_minute as "at"/"final minute", inventing a
    team or player the structured facts don't name (checked for every query,
    not just multi-answer ones — the "Arsenal last substitution" bug showed
    up on an n_relevant==1 query), and narrowing a multi-answer query to one
    instance. Not exhaustive — flags rows worth a human look, not proof of
    correctness either way.
    """
    flags = []
    constraints = query["constraints"]
    n = query["n_relevant"]
    text_lower = text_llm.lower()

    label = constraints.get("label")
    keywords = LABEL_KEYWORDS.get(label, ())
    if keywords and not any(kw in text_lower for kw in keywords):
        flags.append("label_word_missing")

    if "which" in constraints:
        synonyms = FIRST_SYNONYMS if constraints["which"] == "first" else LAST_SYNONYMS
        if not any(re.search(rf"\b{w}\b", text_lower) for w in synonyms):
            flags.append("dropped_ordinal")

    minute_key = "before_minute" if "before_minute" in constraints else (
        "after_minute" if "after_minute" in constraints else None)
    if minute_key:
        minute = str(constraints[minute_key])
        at_patterns = (rf"\bat (the )?{minute}(st|nd|rd|th)?[\s-]?minute\b",
                      rf"\bat minute {minute}\b")
        if "final minute" in text_lower or any(re.search(p, text_lower) for p in at_patterns):
            flags.append("minute_misrepresented")
        if minute not in text_llm:
            flags.append("minute_number_dropped")

    if "team" not in constraints and "team_name" not in constraints:
        # Relational templates (before_goal/before_red_card) legitimately
        # carry a team name in the plain-template `text` itself even though
        # it's not in `constraints` (see build_prompt) — that's not injected.
        plain_text = query.get("text", "")
        game = load_game(query["match_id"])
        for team_name in (game.get("home_team"), game.get("away_team")):
            if not team_name:
                continue
            pattern = rf"\b{re.escape(team_name)}\b"
            if re.search(pattern, text_llm, re.I) and not re.search(pattern, plain_text, re.I):
                flags.append("team_name_injected")
                break

    if constraints.get("before") in ("goal", "red_card"):
        if not any(c in text_lower for c in BEFORE_CONNECTORS):
            flags.append("dropped_before_connector")

    if "player" not in constraints:
        game = load_game(query["match_id"])
        events_by_id = {e["event_id"]: e for e in game["events"]}
        players = {events_by_id[eid].get("player") for eid in query["relevant"]
                  if events_by_id.get(eid)}
        players.discard(None)
        if any(p and p.lower() in text_lower for p in players):
            flags.append("named_player_injected")

        # Structured `player` extraction is often None even when the caption
        # plainly names someone (e.g. two names in one sentence => ambiguous,
        # see extract_player() in build_event_collection.py) — catch those
        # too by looking for a "Firstname Lastname"-shaped phrase in the
        # rephrase that also appears in the (trusted-only) caption context.
        # A team name (e.g. "Real Madrid", "Maccabi Tel Aviv") is a legitimate
        # match for this same regex shape — strip both team names out of the
        # text *before* scanning so a 3+-word team name can't leave a
        # 2-word remainder (e.g. "Maccabi Tel") that then fails an exact-match
        # exclusion check.
        caption = query.get("caption_context") or ""
        text_without_teams = text_llm
        for team_name in (game.get("home_team"), game.get("away_team")):
            if team_name:
                text_without_teams = re.sub(rf"\b{re.escape(team_name)}\b", "",
                                            text_without_teams, flags=re.I)
        names_in_text = {re.sub(r"'s$", "", m) for m in
                         re.findall(r"\b[A-Z][a-zA-Z'-]+\s[A-Z][a-zA-Z'-]+\b", text_without_teams)}
        if any(name in caption for name in names_in_text):
            flags.append("possible_named_individual_from_caption")

    if n > 1 and "which" not in constraints and "half" not in constraints and any(
        re.search(rf"\b{w}\b", text_lower) for w in ORDINAL_WORDS
    ):
        flags.append("ordinal_in_multi_answer")

    if n > 1:
        has_generality_word = any(w in text_lower for w in GENERALITY_WORDS)
        has_pluralized_keyword = any(
            kw + "s" in text_lower for kw in keywords if kw.replace(" ", "").isalpha()
        )
        if not (has_generality_word or has_pluralized_keyword):
            flags.append("possible_singular_for_multi")

    return flags


def rephrase_one(model: str, url: str, query: dict) -> dict:
    style = style_for(query)
    caption = trusted_caption_context(query["match_id"], query["relevant"])
    prompt = build_prompt(query, caption, style)
    result = ollama_rephrase(prompt, model, url)
    return {
        **query,
        "text_llm": result["query"],
        "rephrase_style": style,
        "text_llm_used_caption": result["used_caption"],
        "text_llm_had_trusted_caption": caption is not None,
        "text_llm_flags": flag_narrowing(query, result["query"]),
    }


def already_done() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    ids = set()
    for line in OUT_PATH.open(encoding="utf-8"):
        line = line.strip()
        if line:
            ids.add(json.loads(line)["query_id"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--workers", type=int, default=2,
                        help="local GPU inference is the bottleneck, not I/O — "
                             "raising this rarely helps unless Ollama is configured "
                             "for multiple parallel model instances")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process this many queries total (for a cheap dry run)")
    args = parser.parse_args()

    all_queries = [json.loads(l) for l in QUERIES_PATH.open(encoding="utf-8")]
    done = already_done()
    todo = [q for q in all_queries if q["query_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(all_queries)} total queries, {len(done)} already done, "
          f"{len(todo)} to process this run.")
    if not todo:
        return

    write_lock = Lock()
    n_done = n_errors = n_flagged = 0
    style_counts = {"A": 0, "B": 0}

    with OUT_PATH.open("a", encoding="utf-8") as out, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(rephrase_one, args.model, args.ollama_url, q): q
                   for q in todo}
        for fut in as_completed(futures):
            q = futures[fut]
            try:
                result = fut.result()
                style_counts[result["rephrase_style"]] += 1
                if result["text_llm_flags"]:
                    n_flagged += 1
            except Exception as e:
                result = {**q, "text_llm": None, "text_llm_error": str(e)}
                n_errors += 1
            with write_lock:
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
            n_done += 1
            if n_done % 50 == 0:
                print(f"  {n_done}/{len(todo)}  (errors: {n_errors}, flagged: {n_flagged})")

    print(f"Done: {n_done} queries this run ({n_errors} errors, {n_flagged} flagged "
          f"for likely narrowing — check text_llm_flags before trusting those rows). "
          f"style A: {style_counts['A']}, style B: {style_counts['B']}")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
