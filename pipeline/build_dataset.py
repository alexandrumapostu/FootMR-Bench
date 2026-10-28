"""Materialize the clean, split dataset/ directory consumed by training/spotting/retrieval code.

Everything in data/processed/ is the working pipeline area (intermediate
per-game files, the full unsplit events_all.jsonl / queries.jsonl, in-progress
rephrasing checkpoints). This script reads the *finished* outputs from there
and writes a small, split-ready snapshot to dataset/<split>/{events,queries}.jsonl
— nothing else, so it's safe to point model code at dataset/ directly.

Rephrasing order doesn't matter for correctness — this script always reads
whatever's on disk right now:
  - If data/processed/queries_rephrased.jsonl exists (rephrase_queries.py's
    checkpointed output), its text_llm/rephrase_style/etc. fields are merged
    onto matching query_ids.
  - Queries not yet rephrased (or if the file doesn't exist at all) simply
    keep only the plain-template `text` field, with no `text_llm`.
Rerun this script any time after rephrase_queries.py makes further progress
to refresh dataset/ with newly-completed rephrasings — it always overwrites
with a full, consistent snapshot rather than patching in place.

Every merged row also gets `text_for_training`: `text_llm` when it exists and
rephrase_queries.flag_narrowing() finds nothing suspicious about it, else the
plain-template `text` (always factually correct by construction). This is
the field model code should actually train on — a small, known fraction of
rows (see the printed flagged-row count) fall back to the plain template
rather than risk training on a rephrase with a dropped ordinal, an invented
team/player, or a narrowed multi-answer query. `text_llm` itself is kept
as-is on every row for inspection even when flagged.

train/queries.jsonl is already filtered to template_seen_in_train == true
(the actual trainable set); valid/test keep every template, including the
held-out ones, by design — see generate_queries.py's HELDOUT_TEMPLATES.

Usage: python scripts/build_dataset.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
EVENTS_ALL = PROCESSED / "events_all.jsonl"
QUERIES = PROCESSED / "queries.jsonl"
QUERIES_REPHRASED = PROCESSED / "queries_rephrased.jsonl"
OUT_DIR = PROJECT_ROOT / "dataset"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rephrase_queries import flag_narrowing  # noqa: E402

SPLITS = ("train", "valid", "test")


def main() -> None:
    events = [json.loads(l) for l in EVENTS_ALL.open(encoding="utf-8")]
    queries = [json.loads(l) for l in QUERIES.open(encoding="utf-8")]

    n_merged = n_fallback = 0
    if QUERIES_REPHRASED.exists():
        rephrased_by_id = {}
        for line in QUERIES_REPHRASED.open(encoding="utf-8"):
            r = json.loads(line)
            rephrased_by_id[r["query_id"]] = r
        for q in queries:
            r = rephrased_by_id.get(q["query_id"])
            if r and r.get("text_llm"):
                q["text_llm"] = r["text_llm"]
                q["rephrase_style"] = r["rephrase_style"]
                q["text_llm_used_caption"] = r["text_llm_used_caption"]
                q["text_llm_had_trusted_caption"] = r["text_llm_had_trusted_caption"]
                n_merged += 1
                flags = flag_narrowing(q, q["text_llm"])
                if flags:
                    q["text_llm_flags"] = flags
                    q["text_for_training"] = q["text"]
                    n_fallback += 1
                else:
                    q["text_for_training"] = q["text_llm"]
            else:
                q["text_for_training"] = q["text"]
    else:
        for q in queries:
            q["text_for_training"] = q["text"]

    for split in SPLITS:
        split_dir = OUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_events = [e for e in events if e["split"] == split]
        with (split_dir / "events.jsonl").open("w", encoding="utf-8") as f:
            for e in split_events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        split_queries = [q for q in queries if q["split"] == split]
        if split == "train":
            split_queries = [q for q in split_queries if q["template_seen_in_train"]]
        with (split_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
            for q in split_queries:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        n_using_llm = sum(1 for q in split_queries if q["text_for_training"] == q.get("text_llm"))
        n_flagged = sum(1 for q in split_queries if q.get("text_llm_flags"))
        print(f"[{split}] {len(split_events):6d} events, {len(split_queries):6d} queries "
              f"({n_using_llm} using text_for_training=text_llm, "
              f"{n_flagged} flagged -> fell back to plain text)")

    print(f"Merged rephrasings for {n_merged}/{len(queries)} queries "
          f"({'from ' + QUERIES_REPHRASED.name if QUERIES_REPHRASED.exists() else 'none found'})")
    print(f"Of those, {n_fallback} were flagged by flag_narrowing() and fell back to "
          f"the plain template in text_for_training ({100 * n_fallback / max(1, n_merged):.2f}%)")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
