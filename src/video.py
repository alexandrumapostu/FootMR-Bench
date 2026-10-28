"""Sample frames from data/soccernet/<match_id>/<half>_224p.mkv for a
given event window [start_sec, end_sec]."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SOCCERNET_DIR = Path(__file__).resolve().parent.parent / "data" / "soccernet"
FRAME_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "frames"


def video_path(match_id: str, half: int) -> Path:
    return SOCCERNET_DIR / match_id / f"{half}_224p.mkv"


def sample_frames(match_id: str, half: int, start_sec: float, end_sec: float,
                   num_frames: int = 8, cache_key: str | None = None) -> list[Image.Image]:
    """Return `num_frames` PIL RGB frames evenly spaced within [start_sec, end_sec].

    `cache_key`, when set, caches the decoded pixels on disk under
    cache/frames/<cache_key>/... keyed by (match_id, half, window, num_frames).
    Only worth it for multi-epoch training (LoRA re-reads the same events every
    epoch) - one-pass eval runs leave this unset, since the embedding cache
    already avoids recompute across separate runs.
    """
    if cache_key is not None:
        cache_path = (FRAME_CACHE_DIR / cache_key / match_id /
                       f"{half}_{start_sec:.2f}_{end_sec:.2f}_{num_frames}.npy")
        if cache_path.exists():
            return [Image.fromarray(f) for f in np.load(cache_path)]

    frames = _decode_frames(match_id, half, start_sec, end_sec, num_frames)

    if cache_key is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np.stack([np.array(f) for f in frames]))
    return frames


def _decode_frames(match_id: str, half: int, start_sec: float, end_sec: float,
                    num_frames: int) -> list[Image.Image]:
    path = video_path(match_id, half)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_ms = (frame_count / fps) * 1000.0 if frame_count > 0 and fps > 0 else None

    start_ms = max(0.0, start_sec * 1000.0)
    end_ms = max(start_ms + 1.0, end_sec * 1000.0)
    if duration_ms is not None:
        start_ms = min(start_ms, duration_ms - 1)
        end_ms = min(end_ms, duration_ms - 1)
        end_ms = max(end_ms, start_ms + 1.0)

    if num_frames == 1:
        timestamps = np.array([(start_ms + end_ms) / 2])
    else:
        timestamps = np.linspace(start_ms, end_ms, num=num_frames)
    frames: list[Image.Image] = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded for {match_id} half={half} window=[{start_sec},{end_sec}]")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]
