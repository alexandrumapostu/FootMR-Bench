"""Run pretrained video/VLM retrieval baselines against the random-chance floor.

Usage:
    python scripts/run_video_baselines.py --split valid --limit-matches 5
    python scripts/run_video_baselines.py --split valid --model xclip --limit-matches 5
    python scripts/run_video_baselines.py --split valid --model all   # full run, no limit
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import LORA_FAMILIES, clip4clip_ranker, lora_ranker, random_ranker, siglip_ranker, xclip_ranker
from src.data import group_events_by_match, load_split
from src.metrics import evaluate_queries
from src.rerank import build_softmax_confidence_fn, ordinal_aware_rerank


def events_with_radius(events: list[dict], radius: float) -> list[dict]:
    out = []
    for e in events:
        e2 = dict(e)
        e2["window"] = [max(0, e["t"] - radius), e["t"] + radius]
        out.append(e2)
    return out

MODELS = {
    "clip4clip": ("clip4clip (openai/clip-vit-base-patch32)",
                  lambda eb, q, nf: clip4clip_ranker(eb, q, num_frames=nf)),
    "clip4clip-large": ("clip4clip (openai/clip-vit-large-patch14)",
                        lambda eb, q, nf: clip4clip_ranker(eb, q, model_name="openai/clip-vit-large-patch14", num_frames=nf)),
    "metaclip": ("clip4clip (facebook/metaclip-b32-400m)",
                 lambda eb, q, nf: clip4clip_ranker(eb, q, model_name="facebook/metaclip-b32-400m", num_frames=nf)),
    "siglip": ("siglip (google/siglip-base-patch16-224)",
               lambda eb, q, nf: siglip_ranker(eb, q, num_frames=nf)),
    "xclip": ("xclip (microsoft/xclip-base-patch32)",
              lambda eb, q, nf: xclip_ranker(eb, q, num_frames=nf)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--model", default="all", choices=[*MODELS, "all"])
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--window-radius", type=float, default=10,
                         help="Event window half-width in seconds (default: dataset's native +-10s). "
                              "Grid ablation found +-5s beats +-10s at equal frame count - use 5 "
                              "when evaluating a LoRA checkpoint trained at +-5s.")
    parser.add_argument("--limit-matches", type=int, default=5,
                         help="Only use the first N matches (0 = all). Keep this small "
                              "while iterating - each event needs a video decode + forward pass.")
    parser.add_argument("--limit-queries-per-match", type=int, default=0,
                         help="Cap queries per match (0 = no cap), for a quick sanity check.")
    parser.add_argument("--train-only-templates", action="store_true",
                         help="Keep only queries whose template_id was seen in training.")
    parser.add_argument("--unseen-only-templates", action="store_true",
                         help="Keep only queries whose template_id was held out of training - the "
                              "'new match, unseen phrasing' slice (see README.md's Dataset section).")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--lora-adapter", default=None,
                         help="Path to a LoRA adapter dir saved by scripts/train_lora.py. When set, "
                              "adds a <family>-lora ranker to the run - requires --lora-family.")
    parser.add_argument("--lora-family", default=None, choices=list(LORA_FAMILIES),
                         help="Which architecture the adapter was trained on (determines how it's loaded "
                              "and ranked - frame-pool for clip4clip/siglip, native video tower for xclip).")
    parser.add_argument("--lora-base", default=None,
                         help="Base checkpoint the --lora-adapter was trained on top of "
                              "(defaults to --lora-family's standard checkpoint).")
    parser.add_argument("--ordinal-rerank", action="store_true",
                         help="Also add a <...>+ordinal-rerank row next to --lora-adapter's: a "
                              "post-hoc, zero-retraining re-ranking of 'first X'/'last X' queries by "
                              "each candidate's own (known, not predicted) timestamp, weighted by the "
                              "base model's own confidence - see src/rerank.py. Reuses the same cached "
                              "embeddings, so this adds negligible time to the run.")
    parser.add_argument("--ordinal-rerank-top-k", type=int, default=10,
                         help="How many of the base ranking's top candidates get re-ranked.")
    parser.add_argument("--ordinal-rerank-softmax", action="store_true",
                         help="Also add a <...>+ordinal-rerank(softmax) row: same re-ranking as "
                              "--ordinal-rerank, but confidence is softmax(cosine_similarity x "
                              "logit_scale) - the model's actual trained-objective probability - "
                              "instead of the 1/rank proxy. Independent of --ordinal-rerank, so you "
                              "can pass either flag alone or both together for a 3-way comparison.")
    parser.add_argument("--save-predictions-dir", default=None,
                         help="Also write each ranker's predictions to <dir>/<ranker-name>.jsonl, in "
                              "the format scripts/evaluate_predictions.py expects - lets that generic "
                              "scoring script reproduce this run's numbers independently, and is the "
                              "format to hand off predictions from any other codebase for scoring here.")
    args = parser.parse_args()

    events, queries = load_split(args.split)
    if args.train_only_templates:
        queries = [q for q in queries if q.get("template_seen_in_train")]
    if args.unseen_only_templates:
        queries = [q for q in queries if not q.get("template_seen_in_train")]

    events_by_match = group_events_by_match(events)
    if args.limit_matches:
        match_ids = list(events_by_match.keys())[: args.limit_matches]
        events_by_match = {m: events_by_match[m] for m in match_ids}
        queries = [q for q in queries if q["match_id"] in events_by_match]

    if args.limit_queries_per_match:
        capped, per_match_count = [], {}
        for q in queries:
            c = per_match_count.get(q["match_id"], 0)
            if c < args.limit_queries_per_match:
                capped.append(q)
                per_match_count[q["match_id"]] = c + 1
        queries = capped

    if args.window_radius != 10:
        events_by_match = {m: events_with_radius(evs, args.window_radius) for m, evs in events_by_match.items()}

    n_events = sum(len(v) for v in events_by_match.values())
    print(f"split={args.split}  matches={len(events_by_match)}  events={n_events}  queries={len(queries)}"
          f"  window=+-{args.window_radius}s")
    print()

    builders = {}
    if not args.skip_random:
        builders["random"] = lambda: random_ranker(events_by_match)
    if args.lora_adapter:
        if not args.lora_family:
            parser.error("--lora-adapter requires --lora-family")
        lora_label = f"{args.lora_family}-lora ({args.lora_adapter})"
        builders[lora_label] = lambda: lora_ranker(
            events_by_match, queries, family=args.lora_family, adapter_dir=args.lora_adapter,
            base_model_name=args.lora_base, num_frames=args.num_frames)
        if args.ordinal_rerank:
            def build_reranked():
                base_rank_fn = lora_ranker(events_by_match, queries, family=args.lora_family,
                                            adapter_dir=args.lora_adapter, base_model_name=args.lora_base,
                                            num_frames=args.num_frames)
                return ordinal_aware_rerank(base_rank_fn, events_by_match, top_k=args.ordinal_rerank_top_k)
            builders[f"{lora_label} +ordinal-rerank"] = build_reranked
        if args.ordinal_rerank_softmax:
            def build_reranked_softmax():
                base_rank_fn = lora_ranker(events_by_match, queries, family=args.lora_family,
                                            adapter_dir=args.lora_adapter, base_model_name=args.lora_base,
                                            num_frames=args.num_frames)
                confidence_fn = build_softmax_confidence_fn(family=args.lora_family, adapter_dir=args.lora_adapter,
                                                             base_model_name=args.lora_base)
                return ordinal_aware_rerank(base_rank_fn, events_by_match, top_k=args.ordinal_rerank_top_k,
                                             confidence_fn=confidence_fn)
            builders[f"{lora_label} +ordinal-rerank(softmax)"] = build_reranked_softmax
    model_keys = list(MODELS) if args.model == "all" else [args.model]
    for key in model_keys:
        label, make = MODELS[key]
        builders[label] = lambda make=make: make(events_by_match, queries, args.num_frames)

    header = f"{'ranker':<42}" + "".join(f"{k:>10}" for k in
                                          ["R@1", "R@5", "R@10", "MRR", "mAP", "nDCG@10", "AUC"])
    print(header)
    print("-" * len(header))

    for name, build in builders.items():
        t0 = time.time()
        rank_fn = build()
        metrics = evaluate_queries(queries, rank_fn)
        dt = time.time() - t0
        row = f"{name:<42}"
        row += f"{metrics['recall@1']:>10.3f}"
        row += f"{metrics['recall@5']:>10.3f}"
        row += f"{metrics['recall@10']:>10.3f}"
        row += f"{metrics['mrr']:>10.3f}"
        row += f"{metrics['map']:>10.3f}"
        row += f"{metrics['ndcg@10']:>10.3f}"
        row += f"{metrics['auc']:>10.3f}"
        print(row + f"   ({dt:.1f}s)")

        if args.save_predictions_dir:
            out_dir = Path(args.save_predictions_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
            out_path = out_dir / f"{safe_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for q in queries:
                    f.write(json.dumps({"query_id": q["query_id"], "ranked_event_ids": rank_fn(q)}) + "\n")
            print(f"  predictions saved to {out_path}")


if __name__ == "__main__":
    main()
