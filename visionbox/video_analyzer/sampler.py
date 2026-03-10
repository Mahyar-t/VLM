"""
visionbox/video_analyzer/sampler.py
Smart video sampler — splits video into ~64-frame clips with adaptive
frame stepping based on motion, using overlapping windows at scene transitions.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def _motion_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Mean absolute difference between two RGB frames (uint8 → float)."""
    return float(np.mean(np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32))))


def _decode_all_frames(path: str) -> tuple[List[np.ndarray], float]:
    """Decode every frame from the video; return (frames, fps)."""
    import av  # lazy; keeps sampler importable even without av

    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 30)
        frames: List[np.ndarray] = []
        for packet in container.demux(stream):
            for frame in packet.decode():
                frames.append(frame.to_rgb().to_ndarray())
    return frames, fps


def _pick_step(motion: float, fps: float,
               high_thresh: float, low_thresh: float) -> int:
    """Return adaptive sampling step (2, 4, or 8) from a per-frame motion score."""
    if fps <= 15:
        # Low-fps source — already temporally sparse
        base = 2
    elif fps >= 50:
        # High-fps source — can afford an extra reduction
        base = 8
    else:
        base = 4

    if motion > high_thresh:
        return max(2, base // 2)  # Dense sampling for high motion
    elif motion < low_thresh:
        return min(8, base * 2)  # Coarse sampling for static shots
    return base


def sample_video(
    path: str,
    clip_len: int = 64,
    high_motion_threshold: float = 15.0,
    low_motion_threshold: float = 5.0,
    use_adaptive_step: bool = True,
    use_overlap: bool = True,
) -> List[List[np.ndarray]]:
    """
    Decode a video and return a list of clips, each a list of `clip_len` frames.
    """
    raw_frames, fps = _decode_all_frames(path)
    n = len(raw_frames)
    if n == 0:
        raise ValueError(f"Could not read any frames from {path}")

    logger.info("Sampler: %d raw frames @ %.1f fps from %s", n, fps, path)

    # --- Compute per-frame motion scores ---
    motion_scores: List[float] = [0.0]  # index 0 has no predecessor
    for i in range(1, n):
        motion_scores.append(_motion_score(raw_frames[i - 1], raw_frames[i]))

    # Smooth over a 5-frame window to reduce noise
    kernel = np.ones(5) / 5
    smoothed = np.convolve(motion_scores, kernel, mode="same")

    # --- Walk through frames, building clips ---
    clips: List[List[np.ndarray]] = []
    current_clip: List[np.ndarray] = []
    i = 0
    prev_motion = smoothed[0]

    while i < n:
        motion = smoothed[i]
        
        if use_adaptive_step:
            step = _pick_step(motion, fps, high_motion_threshold, low_motion_threshold)
        else:
            # Default fixed step if adaptive is off (e.g. step 4 for 30fps)
            step = 1 if fps < 10 else (2 if fps < 20 else 4)

        current_clip.append(raw_frames[i])

        # Detect transition: sudden large motion spike after relative calm
        transition = (
            use_overlap
            and len(current_clip) >= clip_len // 2
            and motion > high_motion_threshold * 1.5
            and prev_motion < low_motion_threshold
        )

        if len(current_clip) >= clip_len or transition:
            # Pad short clips
            while len(current_clip) < clip_len:
                current_clip.append(current_clip[-1])
            clips.append(current_clip[:clip_len])

            if transition:
                # Overlapping window: rewind by 50% of clip_len
                overlap_start = max(0, i - clip_len // 2)
                i = overlap_start
                current_clip = []
            else:
                current_clip = []

        prev_motion = motion
        i += step

    # Flush any remaining frames as a final (padded) clip
    if current_clip:
        while len(current_clip) < clip_len:
            current_clip.append(current_clip[-1])
        clips.append(current_clip[:clip_len])

    if not clips:
        # Edge case: very short video — just return all frames as one padded clip
        chunk = list(raw_frames)
        while len(chunk) < clip_len:
            chunk.append(chunk[-1])
        clips.append(chunk[:clip_len])

    logger.info(
        "Sampler produced %d clip(s) of %d frames from %d raw frames",
        len(clips), clip_len, n,
    )
    return clips
