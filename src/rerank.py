"""Eval-time re-ranking for ordinal ("first X" / "last X") queries - a
hybrid neuro-symbolic post-process, not a trained component.

Why this exists: every event is embedded independently (src/baselines.py),
with no architectural access to its timestamp or position relative to other
events - and empirically the fine-tuned models score at/below random chance
on queries that require picking the chronologically first/last of several
same-type candidates (see REPORT.md). But each candidate event's own
timestamp IS known at eval time - it's the fixed GT window this whole
benchmark is built on, not something predicted - so a cheap fix is
possible without retraining anything.

Leakage guard: ordinal intent ("first" vs "last") is detected from the
query's OWN TEXT only, via detect_ordinal_intent() - never from
query["constraints"]["which"] or template_id, which encode the
ground-truth-generation rule itself and would make this a lookup table,
not a real re-ranking signal. 97% of ordinal-template test queries still
contain the literal word after the LLM paraphrase pass, so text-only
detection is practical.

The combination is "the model's own confidence x how well this candidate
fits the requested ordinal position", not a hard filter+sort - a hard
sort-by-timestamp can be fooled by a wrong-type event that happens to have
an extreme timestamp (see ordinal_aware_rerank's docstring for a worked
example); multiplying by the base model's confidence keeps a low-similarity
wrong-type event from winning just because its timestamp fits.
"""
import math
import re
from pathlib import Path
from typing import Callable

# "first half"/"second half"/"last half" is a time-scope phrase, not an
# ordinal-of-event-type reference - strip it out before searching, or
# queries like "every corner in the first half" (multi-relevant - wants ALL
# corners in that half, not the earliest one) get misdetected as ordinal
# just because the word "first" appears. Checked empirically: after this
# exclusion, every query the detector fires on has exactly one relevant
# event (as it logically must - "the first X" can only mean one specific
# event), confirming the exclusion is doing its job.
_SCOPE_PHRASE_RE = re.compile(r"\b(first|second|last)\s+half\b", re.IGNORECASE)
_ORDINAL_RE = re.compile(r"\b(first|last)\b", re.IGNORECASE)


def detect_ordinal_intent(text: str) -> str | None:
    """'first', 'last', or None - decided from the query's own text only."""
    stripped = _SCOPE_PHRASE_RE.sub("", text)
    m = _ORDINAL_RE.search(stripped)
    return m.group(1).lower() if m else None


def detect_half_scope(text: str) -> int | None:
    """1 if the query names 'first half', 2 if 'second half' or 'last half',
    else None - the same phrase detect_ordinal_intent strips out (there, only
    to avoid a false ordinal trigger); here we keep which half it named, so
    ordinal_aware_rerank can restrict its timestamp comparison to candidates
    actually in that half instead of comparing across the whole match.
    """
    m = _SCOPE_PHRASE_RE.search(text)
    if not m:
        return None
    return 1 if m.group(1).lower() == "first" else 2


def build_softmax_confidence_fn(family: str, adapter_dir: str, base_model_name: str | None = None,
                                 device: str | None = None):
    """Real, trained-objective confidence instead of the 1/rank proxy:
    softmax(cosine_similarity x logit_scale) over the candidates being
    re-ranked - exactly the quantity multi_positive_info_nce (the training
    loss) optimizes, logit_scale (a learned temperature) included. Unlike
    1/rank, this is sensitive to HOW CLOSE the model's own similarity
    scores are, not just their order - e.g. a near-tie between the top two
    candidates yields near-equal confidence, letting the timestamp signal
    actually decide; 1/rank would instead always treat #1 as "worth 2x #2"
    regardless of how uncertain the model actually was.

    Loads each candidate's embedding from the on-disk cache (already
    computed by whichever eval run produced the base ranking - see
    src/embed_cache.py) and only runs a fresh forward pass for the query's
    own text (cheap - no video decode). Returns a confidence_fn(query,
    candidate_ids) -> dict[event_id, float] | None - None signals a cache
    miss, so the caller can fall back to the 1/rank proxy instead of
    silently re-embedding every candidate from scratch (slow, and this
    function is meant to reuse an eval that already ran, not redo it).
    """
    import numpy as np
    import torch
    from peft import PeftModel

    from src.baselines import LORA_FAMILIES, _pick_device
    from src.embed_cache import event_cache_path

    if family not in LORA_FAMILIES:
        raise ValueError(f"Unknown LoRA family {family!r} - choose one of {list(LORA_FAMILIES)}")
    default_base, _kind = LORA_FAMILIES[family]
    base_model_name = base_model_name or default_base
    device = _pick_device(device)
    adapter_key = Path(adapter_dir).resolve().name
    model_key = f"{base_model_name.replace('/', '__')}__lora_{adapter_key}"

    text_padding = {"clip4clip": "max_length", "clip4clip-large": "max_length",
                     "metaclip": "max_length", "siglip": "max_length", "xclip": True}[family]
    if family == "siglip":
        from transformers import SiglipModel as ModelCls
        from transformers import SiglipProcessor as ProcessorCls
    elif family == "xclip":
        from transformers import XCLIPModel as ModelCls
        from transformers import XCLIPProcessor as ProcessorCls
    else:
        from transformers import CLIPModel as ModelCls
        from transformers import CLIPProcessor as ProcessorCls

    base = ModelCls.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload().to(device).eval()
    processor = ProcessorCls.from_pretrained(base_model_name)
    logit_scale = model.logit_scale.exp().clamp(max=100.0).item()

    @torch.no_grad()
    def embed_query(text: str):
        inputs = processor(text=[text], return_tensors="pt", padding=text_padding, truncation=True).to(device)
        feat = model.get_text_features(**inputs).pooler_output[0]
        return (feat / feat.norm()).cpu().numpy()

    def confidence_fn(query: dict, candidate_ids: list[str]) -> dict[str, float] | None:
        from src.data import query_text
        vecs = {}
        for eid in candidate_ids:
            path = event_cache_path(model_key, query["match_id"], eid)
            if not path.exists():
                return None  # cache miss - caller falls back to the rank-based proxy
            vecs[eid] = np.load(path)
        qvec = embed_query(query_text(query))
        scaled = {eid: float(qvec @ v) * logit_scale for eid, v in vecs.items()}
        m = max(scaled.values())
        exps = {eid: math.exp(s - m) for eid, s in scaled.items()}
        z = sum(exps.values())
        return {eid: e / z for eid, e in exps.items()}

    return confidence_fn


def ordinal_aware_rerank(rank_fn: Callable[[dict], list[str]], events_by_match: dict[str, list[dict]],
                          top_k: int = 10, query_text_fn=None,
                          confidence_fn: Callable[[dict, list[str]], dict[str, float] | None] | None = None
                          ) -> Callable[[dict], list[str]]:
    """Wrap any rank_fn(query) -> list[event_id] so ordinal queries get
    re-ranked within their own top_k; every other query passes through
    completely unchanged.

    For each of the top_k candidates:
      confidence = 1 / rank_in_original_list      (the base model's own guess, as a 0-1 proxy)
      time_fit   = 1 / rank_by_timestamp           (best-fitting timestamp = 1, among just these top_k)
      combined   = confidence * time_fit
    Re-sorted by combined, descending; events beyond top_k are appended
    after, in their original order.

    Example of why this beats a hard filter+sort: top_k has a Shots-on-target
    event that snuck in weakly (rank 8, confidence 0.125) with the earliest
    raw timestamp in the group, and the true Corner answer sits at rank 1
    (confidence 1.0) with the 3rd-earliest timestamp. A hard "just take the
    earliest timestamp" rule would wrongly promote the off-type event to #1.
    Here: Shots-on-target gets confidence 0.125 x time_fit 1.0 = 0.125;
    the correct Corner gets confidence 1.0 x time_fit 0.33 = 0.33 - it wins,
    because its much higher base confidence outweighs its worse (but not
    terrible) timestamp fit.

    Half-scoped queries ("the first X in the second half"): time_fit is
    computed only among top_k candidates in the named half - a candidate from
    the other half cannot be the answer regardless of confidence, so it gets
    time_fit=0 rather than competing on a match-wide timestamp comparison it
    can never legitimately win. Falls back to the match-wide comparison if
    none of top_k happen to be in the named half.
    """
    from src.data import query_text as default_query_text
    query_text_fn = query_text_fn or default_query_text

    def wrapped(query: dict) -> list[str]:
        ranked = rank_fn(query)
        text = query_text_fn(query)
        which = detect_ordinal_intent(text)
        if which is None:
            return ranked

        events_by_id = {e["event_id"]: e for e in events_by_match[query["match_id"]]}
        head, tail = ranked[:top_k], ranked[top_k:]

        confidence = confidence_fn(query, head) if confidence_fn is not None else None
        if confidence is None:  # no confidence_fn given, or it hit a cache miss
            confidence = {eid: 1.0 / (i + 1) for i, eid in enumerate(head)}

        half_scope = detect_half_scope(text)
        scoped_head = [eid for eid in head if events_by_id[eid]["half"] == half_scope] if half_scope else []
        pool = scoped_head or head

        time_order = sorted(pool, key=lambda eid: (events_by_id[eid]["half"], events_by_id[eid]["t"]),
                             reverse=(which == "last"))
        ranked_time_fit = {eid: 1.0 / (i + 1) for i, eid in enumerate(time_order)}
        time_fit = {eid: ranked_time_fit.get(eid, 0.0) for eid in head}

        reranked_head = sorted(head, key=lambda eid: -(confidence[eid] * time_fit[eid]))
        return reranked_head + tail

    return wrapped
