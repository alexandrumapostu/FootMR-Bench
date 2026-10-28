"""Standalone, generic scoring: given a predictions file and a dataset split,
compute every retrieval metric used in this project (R@1/R@5/R@10, MRR, mAP,
nDCG@10, AUC) - with no dependency on this repo's models, rankers, or GPU
code. This is the "if you have predictions for the test queries and the GT
list, get the metrics out of it" path - for scoring a method that was never
run through src/baselines.py at all (a different codebase, another paper's
model, a held-out submission, etc.).

Predictions file format: JSON Lines, one object per query:
    {"query_id": "q000001", "ranked_event_ids": ["e003", "e001", "e000", ...]}
`ranked_event_ids` should be that query's match's candidate event_ids
(dataset/<split>/events.jsonl, filtered to the same match_id), sorted by
predicted relevance, most relevant first - the same contract every rank_fn
in src/baselines.py already follows. Queries missing from the predictions
file are skipped (with a warning), not scored as zero.

Usage:
    python scripts/evaluate_predictions.py --predictions my_predictions.jsonl --split test

To produce a predictions file from this repo's own rankers (e.g. to sanity-check
this script against scripts/run_video_baselines.py's own numbers, which should
match exactly), use run_video_baselines.py --save-predictions-dir <dir>.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_split
from src.metrics import evaluate_queries


def load_predictions(path: str) -> dict[str, list[str]]:
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            preds[row["query_id"]] = row["ranked_event_ids"]
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                         help="JSONL file: one {'query_id', 'ranked_event_ids'} object per line.")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"],
                         help="Which split's queries.jsonl supplies the ground truth (relevant event_ids).")
    parser.add_argument("--train-only-templates", action="store_true",
                         help="Score only queries whose template_id was seen in training.")
    parser.add_argument("--unseen-only-templates", action="store_true",
                         help="Score only queries whose template_id was held out of training.")
    parser.add_argument("--output", default=None, help="Optional path to also write the metrics as JSON.")
    args = parser.parse_args()

    _, queries = load_split(args.split)
    if args.train_only_templates:
        queries = [q for q in queries if q.get("template_seen_in_train")]
    if args.unseen_only_templates:
        queries = [q for q in queries if not q.get("template_seen_in_train")]

    preds = load_predictions(args.predictions)

    scorable = [q for q in queries if q["query_id"] in preds]
    missing = len(queries) - len(scorable)
    if missing:
        print(f"WARNING: {missing}/{len(queries)} queries in split={args.split} have no prediction "
              f"in {args.predictions} - scoring only the {len(scorable)} that do.")
    if not scorable:
        raise SystemExit("No predicted queries matched this split's ground truth - nothing to score.")

    def rank_fn(q):
        return preds[q["query_id"]]

    metrics = evaluate_queries(scorable, rank_fn)
    print(f"\nscored {len(scorable)}/{len(queries)} queries from split={args.split}")
    print(f"R@1={metrics['recall@1']:.3f}  R@5={metrics['recall@5']:.3f}  R@10={metrics['recall@10']:.3f}  "
          f"MRR={metrics['mrr']:.3f}  mAP={metrics['map']:.3f}  nDCG@10={metrics['ndcg@10']:.3f}  "
          f"AUC={metrics['auc']:.3f}")

    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2))
        print(f"metrics written to {args.output}")


if __name__ == "__main__":
    main()
