"""Retrieval baselines: build(events_by_match, queries) -> rank_fn(query).

rank_fn(query) returns event_ids for query["match_id"], sorted by predicted
relevance, scored only against events in that same match.
"""
import random

import numpy as np
import torch

from src.data import event_text, query_text
from src.embed_cache import event_cache_path, load_or_compute
from src.video import sample_frames


def random_ranker(events_by_match: dict[str, list[dict]], seed: int = 0):
    rng = random.Random(seed)

    def rank_fn(query: dict) -> list[str]:
        ids = [e["event_id"] for e in events_by_match[query["match_id"]]]
        rng.shuffle(ids)
        return ids

    return rank_fn


def _cosine_rank(query_vec: np.ndarray, event_vecs: np.ndarray, event_ids: list[str]) -> list[str]:
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    en = event_vecs / (np.linalg.norm(event_vecs, axis=1, keepdims=True) + 1e-9)
    order = np.argsort(-(en @ qn))
    return [event_ids[i] for i in order]


def tfidf_ranker(events_by_match: dict[str, list[dict]], queries: list[dict]):
    """Lexical text-only baseline, predates having the videos - kept for reference."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_events = [e for evs in events_by_match.values() for e in evs]
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    vectorizer.fit([event_text(e) for e in all_events] + [query_text(q) for q in queries])

    event_vecs_by_match = {
        match_id: ([e["event_id"] for e in evs], vectorizer.transform([event_text(e) for e in evs]).toarray())
        for match_id, evs in events_by_match.items()
    }

    def rank_fn(query: dict) -> list[str]:
        ids, vecs = event_vecs_by_match[query["match_id"]]
        qvec = vectorizer.transform([query_text(query)]).toarray()[0]
        return _cosine_rank(qvec, vecs, ids)

    return rank_fn


def sbert_ranker(events_by_match: dict[str, list[dict]], queries: list[dict],
                  model_name: str = "all-MiniLM-L6-v2"):
    """Text-only embedding baseline, predates having the videos - kept for reference."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    event_vecs_by_match = {
        match_id: ([e["event_id"] for e in evs],
                   model.encode([event_text(e) for e in evs], convert_to_numpy=True, show_progress_bar=False))
        for match_id, evs in events_by_match.items()
    }
    query_vec_by_id = {q["query_id"]: v for q, v in zip(
        queries, model.encode([query_text(q) for q in queries], convert_to_numpy=True, show_progress_bar=False))}

    def rank_fn(query: dict) -> list[str]:
        ids, vecs = event_vecs_by_match[query["match_id"]]
        return _cosine_rank(query_vec_by_id[query["query_id"]], vecs, ids)

    return rank_fn


def _pick_device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _embed_events_cached(events_by_match: dict[str, list[dict]], model_key: str, embed_fn, label: str,
                          progress: bool) -> dict[str, tuple[list[str], np.ndarray]]:
    """Embed every event once; reuse cache/embeddings/<model_key>/... across runs."""
    total = sum(len(evs) for evs in events_by_match.values())
    done = 0
    result = {}
    for match_id, evs in events_by_match.items():
        vecs = []
        for e in evs:
            path = event_cache_path(model_key, match_id, e["event_id"])
            vecs.append(load_or_compute(path, lambda e=e: embed_fn(e, match_id)))
            done += 1
            if progress and done % 200 == 0:
                print(f"  {label} embed: {done}/{total} events", flush=True)
        result[match_id] = ([e["event_id"] for e in evs], np.stack(vecs))
    return result


def _frame_pool_ranker_from_model(events_by_match: dict[str, list[dict]], queries: list[dict],
                                   model, processor, model_key: str, label: str,
                                   num_frames: int, device: str, progress: bool):
    """Shared core of the CLIP-style rankers, given an already-loaded model +
    processor (either a plain HF checkpoint, or a LoRA-adapted one - see
    lora_ranker). `model_key` controls the embedding cache path.
    """

    @torch.no_grad()
    def embed_event(event: dict, match_id: str) -> np.ndarray:
        start, end = event["window"]
        frames = sample_frames(match_id, event["half"], start, end, num_frames=num_frames)
        inputs = processor(images=frames, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vec = feats.mean(dim=0)
        return (vec / vec.norm()).cpu().numpy().astype("float32")

    event_vecs_by_match = _embed_events_cached(events_by_match, model_key, embed_event, label, progress)

    @torch.no_grad()
    def embed_query(text: str) -> np.ndarray:
        inputs = processor(text=[text], return_tensors="pt", padding="max_length", truncation=True).to(device)
        feats = model.get_text_features(**inputs).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().numpy().astype("float32")

    query_vec_cache: dict[str, np.ndarray] = {}

    def rank_fn(query: dict) -> list[str]:
        ids, vecs = event_vecs_by_match[query["match_id"]]
        qid = query["query_id"]
        if qid not in query_vec_cache:
            query_vec_cache[qid] = embed_query(query_text(query))
        return _cosine_rank(query_vec_cache[qid], vecs, ids)

    return rank_fn


def _frame_pool_ranker(events_by_match: dict[str, list[dict]], queries: list[dict],
                        model_cls, processor_cls, model_name: str,
                        num_frames: int, device: str | None, progress: bool,
                        cache_variant: str | None = None):
    """CLIP-style ranker: encode num_frames per event independently with the
    image tower and mean-pool into one event vector ("Clip for Clip"); encode
    the query with the same model's text tower. Works for any HF model whose
    get_image_features/get_text_features return .pooler_output (CLIP, SigLIP).

    `cache_variant`, when set, isolates the on-disk embedding cache under its
    own key (e.g. for ablations with a different window/frame count) so it
    can never collide with or overwrite the default cache for `model_name`.
    """
    device = _pick_device(device)
    model = model_cls.from_pretrained(model_name).to(device).eval()
    processor = processor_cls.from_pretrained(model_name)
    model_key = model_name.replace("/", "__") + (f"__{cache_variant}" if cache_variant else "")

    return _frame_pool_ranker_from_model(events_by_match, queries, model, processor, model_key,
                                          model_name.split("/")[-1], num_frames, device, progress)


def clip4clip_ranker(events_by_match: dict[str, list[dict]], queries: list[dict],
                      model_name: str = "openai/clip-vit-base-patch32",
                      num_frames: int = 8, device: str | None = None, progress: bool = True,
                      cache_variant: str | None = None):
    from transformers import CLIPModel, CLIPProcessor

    return _frame_pool_ranker(events_by_match, queries, CLIPModel, CLIPProcessor, model_name,
                               num_frames, device, progress, cache_variant)


def clip4clip_latefusion_ranker(events_by_match: dict[str, list[dict]], queries: list[dict],
                                 model_name: str = "openai/clip-vit-base-patch32",
                                 num_frames: int = 8, agg: str = "max",
                                 device: str | None = None, progress: bool = True,
                                 cache_variant: str | None = None):
    """Like clip4clip_ranker, but never pools frame embeddings into one vector.
    Each frame is compared to the query independently (N similarities), then
    the scores themselves are aggregated (agg="max" or "mean") - "late fusion"
    vs. clip4clip's "early fusion" (pool embeddings, then compare once).
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = _pick_device(device)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    model_key = model_name.replace("/", "__") + "__latefusion" + (f"__{cache_variant}" if cache_variant else "")

    @torch.no_grad()
    def embed_event(event: dict, match_id: str) -> np.ndarray:
        start, end = event["window"]
        frames = sample_frames(match_id, event["half"], start, end, num_frames=num_frames)
        inputs = processor(images=frames, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")  # (num_frames, dim) - not pooled

    event_vecs_by_match = _embed_events_cached(
        events_by_match, model_key, embed_event, model_name.split("/")[-1] + " (late fusion)", progress)

    @torch.no_grad()
    def embed_query(text: str) -> np.ndarray:
        inputs = processor(text=[text], return_tensors="pt", padding="max_length", truncation=True).to(device)
        feats = model.get_text_features(**inputs).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().numpy().astype("float32")

    query_vec_cache: dict[str, np.ndarray] = {}

    def rank_fn(query: dict) -> list[str]:
        ids, vecs = event_vecs_by_match[query["match_id"]]  # vecs: (n_events, num_frames, dim)
        qid = query["query_id"]
        if qid not in query_vec_cache:
            query_vec_cache[qid] = embed_query(query_text(query))
        qvec = query_vec_cache[qid]
        qn = qvec / (np.linalg.norm(qvec) + 1e-9)
        vn = vecs / (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-9)
        sims = vn @ qn  # (n_events, num_frames)
        scores = sims.max(axis=1) if agg == "max" else sims.mean(axis=1)
        order = np.argsort(-scores)
        return [ids[i] for i in order]

    return rank_fn


def siglip_ranker(events_by_match: dict[str, list[dict]], queries: list[dict],
                   model_name: str = "google/siglip-base-patch16-224",
                   num_frames: int = 8, device: str | None = None, progress: bool = True):
    from transformers import SiglipModel, SiglipProcessor

    return _frame_pool_ranker(events_by_match, queries, SiglipModel, SiglipProcessor, model_name,
                               num_frames, device, progress)


def _xclip_ranker_from_model(events_by_match: dict[str, list[dict]], queries: list[dict],
                              model, processor, model_key: str, label: str,
                              num_frames: int, device: str, progress: bool):
    """Shared core for X-CLIP-family rankers, given an already-loaded model +
    processor (pretrained or LoRA-adapted). X-CLIP encodes all num_frames of a
    clip together in one native video-tower pass (temporal attention across
    frames), unlike clip4clip/SigLIP's independent-frame mean-pooling.
    """

    @torch.no_grad()
    def embed_event(event: dict, match_id: str) -> np.ndarray:
        start, end = event["window"]
        frames = sample_frames(match_id, event["half"], start, end, num_frames=num_frames)
        inputs = processor(videos=[frames], return_tensors="pt").to(device)
        vec = model.get_video_features(**inputs).pooler_output[0]
        return (vec / vec.norm()).cpu().numpy().astype("float32")

    event_vecs_by_match = _embed_events_cached(events_by_match, model_key, embed_event, label, progress)

    @torch.no_grad()
    def embed_query(text: str) -> np.ndarray:
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        feats = model.get_text_features(**inputs).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().numpy().astype("float32")

    query_vec_cache: dict[str, np.ndarray] = {}

    def rank_fn(query: dict) -> list[str]:
        ids, vecs = event_vecs_by_match[query["match_id"]]
        qid = query["query_id"]
        if qid not in query_vec_cache:
            query_vec_cache[qid] = embed_query(query_text(query))
        return _cosine_rank(query_vec_cache[qid], vecs, ids)

    return rank_fn


def xclip_ranker(events_by_match: dict[str, list[dict]], queries: list[dict],
                  model_name: str = "microsoft/xclip-base-patch32",
                  num_frames: int = 8, device: str | None = None, progress: bool = True):
    from transformers import XCLIPModel, XCLIPProcessor

    device = _pick_device(device)
    model = XCLIPModel.from_pretrained(model_name).to(device).eval()
    processor = XCLIPProcessor.from_pretrained(model_name)

    return _xclip_ranker_from_model(events_by_match, queries, model, processor,
                                     model_name.replace("/", "__"), "xclip", num_frames, device, progress)


# family -> (default base checkpoint, "frame_pool" | "video_native") - the exact
# model/processor classes are resolved explicitly below, never via Auto* classes,
# so the loaded architecture is guaranteed identical to what scripts/train_lora.py
# trained the adapter on (Auto* class resolution isn't guaranteed to match).
LORA_FAMILIES = {
    "clip4clip": ("openai/clip-vit-base-patch32", "frame_pool"),
    "clip4clip-large": ("openai/clip-vit-large-patch14", "frame_pool"),
    "metaclip": ("facebook/metaclip-b32-400m", "frame_pool"),
    "siglip": ("google/siglip-base-patch16-224", "frame_pool"),
    "xclip": ("microsoft/xclip-base-patch32", "video_native"),
}


def lora_ranker(events_by_match: dict[str, list[dict]], queries: list[dict],
                 family: str, adapter_dir: str, base_model_name: str | None = None,
                 num_frames: int = 8, device: str | None = None, progress: bool = True,
                 cache_variant: str | None = None):
    """Loads a LoRA adapter trained by scripts/train_lora.py on top of its
    frozen base checkpoint, merges it in, and reuses the matching pretrained
    ranker's eval core (frame-pool for clip4clip/SigLIP, native video-tower
    for X-CLIP) so results land in the same embedding-cache shape and are
    directly comparable to the pretrained-baseline rows.
    """
    from pathlib import Path

    from peft import PeftModel

    if family not in LORA_FAMILIES:
        raise ValueError(f"Unknown LoRA family {family!r} - choose one of {list(LORA_FAMILIES)}")
    default_base, kind = LORA_FAMILIES[family]
    base_model_name = base_model_name or default_base
    device = _pick_device(device)
    adapter_key = Path(adapter_dir).resolve().name
    model_key = f"{base_model_name.replace('/', '__')}__lora_{adapter_key}" + \
                (f"__{cache_variant}" if cache_variant else "")

    if kind == "frame_pool":
        if family == "siglip":
            from transformers import SiglipModel as ModelCls
            from transformers import SiglipProcessor as ProcessorCls
        else:
            from transformers import CLIPModel as ModelCls
            from transformers import CLIPProcessor as ProcessorCls

        base = ModelCls.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload().to(device).eval()
        processor = ProcessorCls.from_pretrained(base_model_name)
        return _frame_pool_ranker_from_model(events_by_match, queries, model, processor, model_key,
                                              f"lora:{adapter_key}", num_frames, device, progress)
    else:
        from transformers import XCLIPModel, XCLIPProcessor

        base = XCLIPModel.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload().to(device).eval()
        processor = XCLIPProcessor.from_pretrained(base_model_name)
        return _xclip_ranker_from_model(events_by_match, queries, model, processor, model_key,
                                         f"lora:{adapter_key}", num_frames, device, progress)
