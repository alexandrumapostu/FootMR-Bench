"""LoRA + contrastive (InfoNCE) fine-tuning on dataset/train, for any of
clip4clip (both backbones), SigLIP, or X-CLIP.

Recipe: LoRA adapters on a frozen checkpoint (both towers by default -
attention projections + the final text/visual embedding heads), trained with
an InfoNCE loss where positives = query <-> its relevant event(s), negatives
= the other events in that same match. This mirrors exactly how eval ranks
(src/baselines.py::_cosine_rank is always scoped to one match), so training
and eval see the same negative distribution. See README.md's "LoRA
fine-tuning" section for the full rationale and the ablation numbers behind
the defaults below.

Window defaults to +-5s/8 frames, not the dataset's native +-10s: a full grid
ablation over frame count x window radius on clip4clip ViT-B/32 (reproduce
with scripts/run_full_grid_ablation.py) found +-5s beats +-10s at every frame
count for the same compute cost - see README.md for the numbers. Not
re-verified on SigLIP/X-CLIP/ViT-L-14 - if you have time, re-running that
ablation per model is worth it before trusting the default here.

Two architecturally different training paths, picked automatically from
--model:
- clip4clip (either backbone) and SigLIP: "frame pool" - each event's frames
  go through the image tower independently and get mean-pooled into one
  vector (matching clip4clip_ranker/siglip_ranker in src/baselines.py).
- X-CLIP: "video native" - all of an event's frames go through the video
  tower together in one pass with temporal attention across frames (matching
  xclip_ranker) - no mean-pooling, the model does its own temporal fusion.

Saves a LoRA adapter (a few MB - NOT a full checkpoint) to --output-dir,
loadable at eval time with src.baselines.lora_ranker or directly:

    from peft import PeftModel
    from transformers import CLIPModel  # or SiglipModel / XCLIPModel
    base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = PeftModel.from_pretrained(base, "cache/lora/clip4clip-b32").merge_and_unload()

Usage:
    # smoke test - a few steps, confirms the pipeline runs end to end
    python scripts/train_lora.py --model siglip --output-dir cache/lora/smoke --max-steps 10 --eval-every 5 --eval-matches 2

    # real run
    python scripts/train_lora.py --model xclip --epochs 3 --output-dir cache/lora/xclip-b32
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import group_events_by_match, load_split, query_text
from src.metrics import evaluate_queries
from src.video import sample_frames

# model key -> (default base checkpoint, forward-pass kind, text padding
# strategy - matched to each family's existing eval-side ranker in
# src/baselines.py so training sees identical preprocessing to eval).
MODEL_NAMES = {
    "clip4clip": "openai/clip-vit-base-patch32",
    "clip4clip-large": "openai/clip-vit-large-patch14",
    "metaclip": "facebook/metaclip-b32-400m",
    "siglip": "google/siglip-base-patch16-224",
    "xclip": "microsoft/xclip-base-patch32",
}
MODEL_KIND = {
    "clip4clip": "frame_pool", "clip4clip-large": "frame_pool", "metaclip": "frame_pool",
    "siglip": "frame_pool", "xclip": "video_native",
}
TEXT_PADDING = {
    "clip4clip": "max_length", "clip4clip-large": "max_length", "metaclip": "max_length",
    "siglip": "max_length", "xclip": True,
}

# Attention projections + the final embedding heads that produce the vectors
# retrieval actually compares. Regexes (not plain lists) so text/vision-only
# presets can restrict to one tower without hand-listing every layer index,
# and so "both" can explicitly exclude architecture-specific extra modules
# that the training/eval forward pass never actually touches (verified per
# family - see REPORT.md: X-CLIP's prompts_generator looked relevant by name
# but get_video_features()/get_text_features() never call it, confirmed with
# forward hooks, so it's deliberately excluded - LoRA there would never get
# a gradient).
LORA_TARGET_PRESETS = {
    "clip4clip": {
        "both": r".*(q_proj|k_proj|v_proj|out_proj)$|^(text_projection|visual_projection)$",
        "text": r".*text_model.*(q_proj|k_proj|v_proj|out_proj)$|^text_projection$",
        "vision": r".*vision_model.*(q_proj|k_proj|v_proj|out_proj)$|^visual_projection$",
    },
    "siglip": {
        # SigLIP has no separate text_projection/visual_projection: the text
        # tower's final projection is text_model.head (a plain Linear); the
        # vision tower's is a whole attention-pooling head whose out_proj is
        # already covered by the generic *_proj suffix match below.
        "both": r".*(q_proj|k_proj|v_proj|out_proj)$|^text_model\.head$",
        "text": r".*text_model.*(q_proj|k_proj|v_proj|out_proj)$|^text_model\.head$",
        "vision": r".*vision_model.*(q_proj|k_proj|v_proj|out_proj)$",
    },
    "xclip": {
        # Scoped to text_model./vision_model. explicitly (not a bare suffix
        # match) because X-CLIP's prompts_generator.decoder.*.cross_attn also
        # has its own q_proj/k_proj/v_proj - an unscoped pattern silently
        # pulls those in too, even though they're dead weight for training
        # (see module docstring above). message_fc is X-CLIP's per-frame
        # cross-communication layer inside the vision tower - the mechanism
        # most directly responsible for the video-specific temporal mixing.
        "both": r".*(text_model|vision_model)\..*(q_proj|k_proj|v_proj|out_proj|message_fc)$"
                r"|^(text_projection|visual_projection)$",
        "text": r".*text_model.*(q_proj|k_proj|v_proj|out_proj)$|^text_projection$",
        "vision": r".*vision_model.*(q_proj|k_proj|v_proj|out_proj|message_fc)$|^visual_projection$",
    },
}
LORA_TARGET_PRESETS["clip4clip-large"] = LORA_TARGET_PRESETS["clip4clip"]
LORA_TARGET_PRESETS["metaclip"] = LORA_TARGET_PRESETS["clip4clip"]  # same CLIPModel class, just a different checkpoint


def events_with_radius(events: list[dict], radius: float) -> list[dict]:
    out = []
    for e in events:
        e2 = dict(e)
        e2["window"] = [max(0, e["t"] - radius), e["t"] + radius]
        out.append(e2)
    return out


def build_match_queries(queries: list[dict], events_by_match: dict) -> dict[str, list[dict]]:
    """Group queries by match, keeping only ones whose relevant event_ids
    actually exist in that match's event list (defensive against any data drift)."""
    by_match: dict[str, list[dict]] = {}
    for q in queries:
        m = q["match_id"]
        if m not in events_by_match:
            continue
        event_ids = {e["event_id"] for e in events_by_match[m]}
        if not set(q["relevant"]) <= event_ids:
            continue
        by_match.setdefault(m, []).append(q)
    return by_match


def build_curriculum_tiers(match_queries: dict[str, list[dict]], difficulty_path: str,
                            n_tiers: int = 3) -> list[set[str]]:
    """Stratified-by-event-type curriculum tiers, from a difficulty file written
    by scripts/score_curriculum_difficulty.py. Within each event-type label,
    queries are sorted by descending pretrained AP (easiest first) and split
    into n_tiers equal chunks - so tier 0 has the easiest queries from EVERY
    label (easy goals AND easy corners AND easy fouls...), not just whichever
    label the pretrained model happens to be best at overall. Returns n_tiers
    sets of query_id, cumulative: tier i already contains all of tiers 0..i-1,
    so "add data, don't walk through it" (REPORT.md curriculum notes) falls
    out of just picking tiers[epoch] directly as the active query set.
    """
    scores = {}
    with open(difficulty_path) as f:
        for line in f:
            rec = json.loads(line)
            scores[rec["query_id"]] = (rec["label"], rec["ap"])

    by_label: dict[str, list[tuple[str, float]]] = {}
    all_ids, scored_ids = set(), set()
    for qs in match_queries.values():
        for q in qs:
            qid = q["query_id"]
            all_ids.add(qid)
            if qid not in scores:
                continue
            scored_ids.add(qid)
            label, ap = scores[qid]
            by_label.setdefault(label, []).append((qid, ap))

    tiers: list[set[str]] = [set() for _ in range(n_tiers)]
    for items in by_label.values():
        items.sort(key=lambda x: -x[1])  # highest AP (easiest) first
        n = len(items)
        for tier_idx in range(n_tiers):
            lo, hi = n * tier_idx // n_tiers, n * (tier_idx + 1) // n_tiers
            for qid, _ in items[lo:hi]:
                tiers[tier_idx].add(qid)

    # queries with no difficulty score (e.g. file scored a different query
    # filter) default to the hardest tier, so nothing silently drops out.
    tiers[-1] |= (all_ids - scored_ids)

    cumulative, running = [], set()
    for t in tiers:
        running = running | t
        cumulative.append(set(running))
    return cumulative


def sample_pool(match_events: list[dict], batch_queries: list[dict], neg_pool_size: int, rng: random.Random):
    """Event pool for one training step: union of the batch's relevant
    events (positives), plus random negatives from the rest of the match,
    capped at neg_pool_size total - bounds step cost regardless of match size.
    """
    events_by_id = {e["event_id"]: e for e in match_events}
    pos_ids, seen = [], set()
    for q in batch_queries:
        for eid in q["relevant"]:
            if eid not in seen:
                seen.add(eid)
                pos_ids.append(eid)
    remaining = [eid for eid in events_by_id if eid not in seen]
    n_neg = max(0, neg_pool_size - len(pos_ids))
    neg_ids = rng.sample(remaining, min(n_neg, len(remaining)))
    pool_ids = pos_ids + neg_ids
    return pool_ids, [events_by_id[eid] for eid in pool_ids]


def embed_events_frame_pool(model, processor, device, match_id: str, pool_events: list[dict],
                             num_frames: int, cache_key: str | None) -> torch.Tensor:
    """clip4clip/SigLIP: each frame through the image tower independently,
    mean-pooled per event - matches clip4clip_ranker/siglip_ranker exactly."""
    flat_frames = []
    for e in pool_events:
        start, end = e["window"]
        flat_frames.extend(sample_frames(match_id, e["half"], start, end,
                                          num_frames=num_frames, cache_key=cache_key))
    inputs = processor(images=flat_frames, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs).pooler_output  # (n_events*num_frames, dim)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats = feats.view(len(pool_events), num_frames, -1).mean(dim=1)
    return feats / feats.norm(dim=-1, keepdim=True)


def embed_events_video_native(model, processor, device, match_id: str, pool_events: list[dict],
                               num_frames: int, cache_key: str | None) -> torch.Tensor:
    """X-CLIP: all of an event's frames through the video tower together in
    one pass (temporal attention across frames, no mean-pooling) - matches
    xclip_ranker exactly, just batched across events for training throughput."""
    videos = []
    for e in pool_events:
        start, end = e["window"]
        videos.append(sample_frames(match_id, e["half"], start, end,
                                     num_frames=num_frames, cache_key=cache_key))
    inputs = processor(videos=videos, return_tensors="pt").to(device)
    feats = model.get_video_features(**inputs).pooler_output  # (n_events, dim)
    return feats / feats.norm(dim=-1, keepdim=True)


EMBED_EVENTS = {"frame_pool": embed_events_frame_pool, "video_native": embed_events_video_native}


def embed_queries(model, processor, device, texts: list[str], text_padding) -> torch.Tensor:
    inputs = processor(text=texts, return_tensors="pt", padding=text_padding, truncation=True).to(device)
    feats = model.get_text_features(**inputs).pooler_output
    return feats / feats.norm(dim=-1, keepdim=True)


def multi_positive_info_nce(sims: torch.Tensor, pos_mask: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """sims: (B, M) cosine similarities, query x event pool.
    pos_mask: (B, M) bool, at least one True per row (queries can have up to 6
    relevant events - see the dataset README for how multi-answer queries are
    built). Standard multi-positive InfoNCE:
    -log( sum_pos exp(sim*scale) / sum_all exp(sim*scale) ), averaged over the batch.
    """
    logits = sims * logit_scale
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_log_prob = torch.where(pos_mask, log_prob, torch.full_like(log_prob, float("-inf")))
    return -torch.logsumexp(pos_log_prob, dim=1).mean()


@torch.no_grad()
def quick_eval(model, processor, device, embed_events_fn, text_padding, events_by_match: dict,
                queries: list[dict], num_frames: int, cache_key: str | None,
                eval_chunk_size: int = 16) -> dict:
    """eval_chunk_size bounds how many events go through one forward pass at a
    time - unlike training, a match here can have 100+ events (vs. the
    neg-pool-size cap on training steps), so embedding a whole match in one
    shot can spike VRAM far past what training itself needs.
    """
    model.eval()
    event_vecs = {}
    for match_id, evs in events_by_match.items():
        ids = [e["event_id"] for e in evs]
        chunks = []
        for i in range(0, len(evs), eval_chunk_size):
            chunk_feats = embed_events_fn(model, processor, device, match_id,
                                           evs[i:i + eval_chunk_size], num_frames, cache_key)
            chunks.append(chunk_feats.cpu().numpy())
        event_vecs[match_id] = (ids, np.concatenate(chunks, axis=0))
    query_vecs: dict[str, np.ndarray] = {}

    def rank_fn(q):
        ids, vecs = event_vecs[q["match_id"]]
        if q["query_id"] not in query_vecs:
            query_vecs[q["query_id"]] = embed_queries(
                model, processor, device, [query_text(q)], text_padding)[0].cpu().numpy()
        qvec = query_vecs[q["query_id"]]
        order = np.argsort(-(vecs @ qvec))
        return [ids[i] for i in order]

    metrics = evaluate_queries(queries, rank_fn)
    model.train()
    return metrics


def fmt_metrics(m: dict) -> str:
    return (f"R@1={m['recall@1']:.3f}  R@5={m['recall@5']:.3f}  R@10={m['recall@10']:.3f}  "
            f"MRR={m['mrr']:.3f}  mAP={m['map']:.3f}  nDCG@10={m['ndcg@10']:.3f}  AUC={m['auc']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="clip4clip", choices=list(MODEL_NAMES),
                         help="Which model to fine-tune: clip4clip/clip4clip-large/siglip use the same "
                              "frame-mean-pool forward pass; xclip uses its native video tower instead.")
    parser.add_argument("--window-radius", type=float, default=5,
                         help="Event window half-width in seconds. Default +-5s: an ablation over frame "
                              "count x window radius on clip4clip ViT-B/32 (reproduce with "
                              "scripts/run_full_grid_ablation.py) found +-5s beats the dataset's native "
                              "+-10s at every frame count tested - see README.md for the numbers. Not yet "
                              "re-verified on SigLIP/X-CLIP/ViT-L-14.")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8, help="Queries per training step.")
    parser.add_argument("--neg-pool-size", type=int, default=24,
                         help="Max events (positives + negatives) embedded per step - bounds VRAM/time "
                              "regardless of match size. Lower this for clip4clip-large on 8GB VRAM.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-target", default="both", choices=["both", "text", "vision"],
                         help="Which tower(s) LoRA adapts: 'both' (attention + projection heads in "
                              "both encoders), 'text' (text tower only - assumes the visual "
                              "representations are already good, only football-query phrasing needs "
                              "adapting), or 'vision' (vision tower only - assumes query language is "
                              "already understood, only broadcast-football visuals are the domain gap). "
                              "Untested empirically which wins here - a natural first hyperparameter to sweep.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                         help="Trade compute for VRAM: recompute activations during backward instead of "
                              "storing them. Same math, same result, just slower per step - turn this on "
                              "when a model is OOM-ing at a batch/neg-pool-size that's otherwise reasonable "
                              "(e.g. SigLIP's patch16 grid is ~4x more tokens/image than clip4clip's patch32, "
                              "so it hits VRAM limits much sooner on an 8GB card).")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = run full --epochs over all matches.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=200, help="Steps between quick valid-subset checks (0 = off).")
    parser.add_argument("--eval-matches", type=int, default=3, help="How many valid matches the quick check scores.")
    parser.add_argument("--eval-chunk-size", type=int, default=16,
                         help="Max events embedded in one forward pass during eval - matches can have "
                              "100+ events (vs. --neg-pool-size capping training steps), so this bounds "
                              "eval's VRAM spike independently of training batch settings.")
    parser.add_argument("--final-eval-matches", type=int, default=20,
                         help="How many valid matches the full end-of-training evaluation scores "
                              "(all 6 metrics: R@1/R@5/R@10/MRR/mAP/nDCG@10) - larger than --eval-matches "
                              "since it only runs once, at the end.")
    parser.add_argument("--curriculum-difficulty", default=None,
                         help="Path to a JSONL file from scripts/score_curriculum_difficulty.py (same "
                              "--model). When set, queries are stratified per event-type label into 3 "
                              "difficulty tiers (easy/medium/hard, by the pretrained model's per-query "
                              "AP) and added cumulatively one tier per epoch - epoch 1 trains on the "
                              "easiest tier only, epoch 2 adds medium, epoch 3+ trains on everything "
                              "(same regime as a non-curriculum run from epoch 3 on). Off by default.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=None, help="Adapter dir to continue training from.")
    parser.add_argument("--start-epoch", type=int, default=0,
                         help="0-based epoch index to resume into - pairs with --resume-from after an "
                              "interruption. Skips epochs before this one entirely (they're assumed baked "
                              "into the --resume-from weights) and, with --curriculum-difficulty on, makes "
                              "the curriculum stage/epoch-dir naming pick up at the right point instead of "
                              "restarting at the easiest tier. E.g. if epoch1 finished and saved but epoch2 "
                              "(medium tier) was interrupted before its own save, resume with "
                              "--resume-from <out_dir>/epoch1 --start-epoch 1 - epoch2 reruns from scratch "
                              "(no mid-epoch checkpoint exists to resume into more precisely) but at the "
                              "correct curriculum stage, and epoch1's completed training isn't redone.")
    args = parser.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model

    if args.model == "siglip":
        from transformers import SiglipModel as ModelCls
        from transformers import SiglipProcessor as ProcessorCls
    elif args.model == "xclip":
        from transformers import XCLIPModel as ModelCls
        from transformers import XCLIPProcessor as ProcessorCls
    else:
        from transformers import CLIPModel as ModelCls
        from transformers import CLIPProcessor as ProcessorCls

    embed_events_fn = EMBED_EVENTS[MODEL_KIND[args.model]]
    text_padding = TEXT_PADDING[args.model]
    lora_presets = LORA_TARGET_PRESETS[args.model]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = MODEL_NAMES[args.model]
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    events, queries = load_split("train")
    queries = [q for q in queries if q.get("template_seen_in_train")]
    events_by_match = group_events_by_match(events)
    if args.window_radius != 10:
        events_by_match = {m: events_with_radius(evs, args.window_radius) for m, evs in events_by_match.items()}
    match_queries = build_match_queries(queries, events_by_match)
    match_ids = list(match_queries)
    n_usable = sum(len(v) for v in match_queries.values())
    print(f"train: {len(match_ids)} matches, {n_usable} usable queries "
          f"(template_seen_in_train=true, relevant events present), window=+-{args.window_radius}s")

    processor = ProcessorCls.from_pretrained(model_name)
    if args.resume_from:
        base = ModelCls.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base, args.resume_from, is_trainable=True).to(device)
        print(f"resumed adapter from {args.resume_from}")
    else:
        base = ModelCls.from_pretrained(model_name)
        lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                               target_modules=lora_presets[args.lora_target], bias="none")
        model = get_peft_model(base, lora_cfg).to(device)
        model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # required with PEFT: base is frozen, so the
        # checkpointed frozen layers need their input to require grad, or the backward
        # pass stops dead at the first frozen layer and LoRA never sees a gradient.
        print("gradient checkpointing enabled")
    base_model = model.get_base_model()
    base_model.logit_scale.requires_grad_(True)  # standard CLIP-family fine-tuning practice
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    frame_cache_key = f"r{args.window_radius}_nf{args.num_frames}"

    valid_events, valid_queries = load_split("valid")
    valid_events_by_match = group_events_by_match(valid_events)
    if args.window_radius != 10:
        valid_events_by_match = {m: events_with_radius(evs, args.window_radius) for m, evs in valid_events_by_match.items()}
    eval_match_ids = list(valid_events_by_match)[: args.eval_matches]
    eval_events_by_match = {m: valid_events_by_match[m] for m in eval_match_ids}
    eval_queries = [q for q in valid_queries if q["match_id"] in eval_match_ids]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps({**vars(args), "base_model_name": model_name}, indent=2))

    curriculum_tiers = None
    if args.curriculum_difficulty:
        curriculum_tiers = build_curriculum_tiers(match_queries, args.curriculum_difficulty)
        print(f"curriculum on: cumulative tier sizes (easy->hard) = {[len(t) for t in curriculum_tiers]}")

    steps_per_epoch = sum(max(1, -(-len(qs) // args.batch_size)) for qs in match_queries.values())
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    print(f"~{steps_per_epoch} steps/epoch (full data - an upper bound while curriculum tiers are "
          f"still ramping up), {total_steps} planned total steps")

    step = 0
    running_loss, running_n = 0.0, 0
    t0 = time.time()
    stop = False
    for epoch in range(args.start_epoch, args.epochs):
        if stop:
            break
        if curriculum_tiers is not None:
            stage = min(epoch, len(curriculum_tiers) - 1)
            allowed = curriculum_tiers[stage]
            print(f"epoch {epoch + 1}: curriculum stage {stage} ({len(allowed)} queries active)", flush=True)
        else:
            allowed = None

        order = match_ids[:]
        rng.shuffle(order)
        for match_id in order:
            match_events = events_by_match[match_id]
            qs = match_queries[match_id][:]
            if allowed is not None:
                qs = [q for q in qs if q["query_id"] in allowed]
            rng.shuffle(qs)
            for i in range(0, len(qs), args.batch_size):
                batch_queries = qs[i:i + args.batch_size]
                pool_ids, pool_events = sample_pool(match_events, batch_queries, args.neg_pool_size, rng)
                if len(pool_events) < 2:
                    continue  # need at least one negative to learn anything

                event_feats = embed_events_fn(model, processor, device, match_id, pool_events,
                                               args.num_frames, frame_cache_key)
                text_feats = embed_queries(model, processor, device,
                                            [query_text(q) for q in batch_queries], text_padding)

                sims = text_feats @ event_feats.T
                pos_mask = torch.zeros_like(sims, dtype=torch.bool)
                for bi, q in enumerate(batch_queries):
                    rel = set(q["relevant"])
                    for mi, eid in enumerate(pool_ids):
                        if eid in rel:
                            pos_mask[bi, mi] = True

                logit_scale = base_model.logit_scale.exp().clamp(max=100.0)
                loss = multi_positive_info_nce(sims, pos_mask, logit_scale)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                with torch.no_grad():
                    base_model.logit_scale.clamp_(max=math.log(100.0))

                running_loss += loss.item()
                running_n += 1
                step += 1

                if step % args.log_every == 0:
                    dt = time.time() - t0
                    print(f"step {step}/{total_steps}  loss={running_loss / running_n:.4f}  "
                          f"({dt / step:.2f}s/step, {dt:.0f}s elapsed)", flush=True)
                    running_loss, running_n = 0.0, 0

                if args.eval_every and step % args.eval_every == 0:
                    m = quick_eval(model, processor, device, embed_events_fn, text_padding,
                                    eval_events_by_match, eval_queries, args.num_frames, frame_cache_key,
                                    args.eval_chunk_size)
                    print(f"  [quick eval @ step {step}, {args.eval_matches} valid matches] {fmt_metrics(m)}",
                          flush=True)

                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break
            if stop:
                break

        epoch_dir = out_dir / f"epoch{epoch + 1}"
        model.save_pretrained(str(epoch_dir))
        print(f"epoch {epoch + 1} done, adapter saved to {epoch_dir}", flush=True)

    model.save_pretrained(str(out_dir))
    print(f"final adapter saved to {out_dir}")

    final_match_ids = list(valid_events_by_match)[: args.final_eval_matches]
    final_events_by_match = {m: valid_events_by_match[m] for m in final_match_ids}
    final_queries = [q for q in valid_queries if q["match_id"] in final_match_ids]
    print(f"\nfinal eval ({len(final_match_ids)} valid matches, {len(final_queries)} queries, "
          f"window=+-{args.window_radius}s, {args.num_frames} frames):")
    t0 = time.time()
    final_metrics = quick_eval(model, processor, device, embed_events_fn, text_padding,
                                final_events_by_match, final_queries, args.num_frames, frame_cache_key,
                                args.eval_chunk_size)
    print(f"  {fmt_metrics(final_metrics)}   ({time.time() - t0:.0f}s)")
    print(f"eval it with: python scripts/run_video_baselines.py --split valid --model {args.model} "
          f"--lora-adapter {out_dir} --lora-family {args.model} --lora-base {model_name} "
          f"--window-radius {args.window_radius} --num-frames {args.num_frames} --limit-matches 20")


if __name__ == "__main__":
    main()
