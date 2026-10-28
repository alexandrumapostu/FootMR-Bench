"""On-disk cache for per-event embedding vectors, keyed by (model, match_id, event_id)."""
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "embeddings"


def event_cache_path(model_key: str, match_id: str, event_id: str) -> Path:
    return CACHE_DIR / model_key / match_id / f"{event_id}.npy"


def load_or_compute(path: Path, compute_fn) -> np.ndarray:
    if path.exists():
        return np.load(path)
    vec = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vec)
    return vec
