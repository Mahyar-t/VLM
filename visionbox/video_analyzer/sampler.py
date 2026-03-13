"""
visionbox/video_analyzer/sampler.py

Optimized video sampler — splits video into ~64-frame clips with adaptive
frame stepping based on motion. Use streaming decoding and low-res motion
analysis to minimize memory and CPU overhead.
"""
from __future__ import annotations

import logging
from typing import Generator, List, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _get_low_res_thumbnail(frame: np.ndarray, size=(64, 64)) -> np.ndarray:
    """Create a small grayscale thumbnail for fast motion scoring."""
    img = Image.fromarray(frame).convert("L")
    img = img.resize(size, Image.NEAREST)
    return np.array(img, dtype=np.float32)


def _motion_score(thumb_a: np.ndarray, thumb_b: np.ndarray) -> float:
    """Mean absolute difference between two low-res grayscale thumbnails."""
    return float(np.mean(np.abs(thumb_a - thumb_b)))


def _stream_frames(path: str) -> Generator[Tuple[np.ndarray, float], None, None]:
    """Generator that yields (RGB_frame, fps) one by one."""
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        # Use a reasonable default if rates are missing
        fps = float(stream.average_rate or stream.base_rate or 30)
        for packet in container.demux(stream):
            for frame in packet.decode():
                yield frame.to_rgb().to_ndarray(), fps


def _pick_step(motion: float, fps: float,
               high_thresh: float, low_thresh: float) -> int:
    """Return adaptive sampling step (2, 4, or 8) based on motion and fps."""
    if fps <= 15:
        base = 2
    elif fps >= 50:
        base = 8
    else:
        base = 4

    if motion > high_thresh:
        return max(2, base // 2)
    elif motion < low_thresh:
        return min(8, base * 2)
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
    Optimized one-pass video sampling with streaming decoding and low-res motion scoring.
    """
    logger.info("Sampler: Starting optimized one-pass sampling of %s", path)

    clips: List[List[np.ndarray]] = []
    current_clip: List[np.ndarray] = []
    
    last_thumb: np.ndarray | None = None
    frame_idx = 0
    target_idx = 0
    fps = 30.0  # default
    
    # Simple recursive smoothing for motion scores
    smooth_motion = 0.0
    alpha = 0.4  # smoothing factor
    
    # Track previous smoothed motion for transition detection
    prev_smooth_motion = 0.0

    for frame, stream_fps in _stream_frames(path):
        fps = stream_fps
        
        # Only process frames we actually need based on the previous step decision
        if frame_idx == target_idx:
            # 1. Update motion score using low-res thumbnails
            current_thumb = _get_low_res_thumbnail(frame)
            if last_thumb is not None:
                raw_motion = _motion_score(last_thumb, current_thumb)
                smooth_motion = alpha * raw_motion + (1 - alpha) * smooth_motion
            else:
                smooth_motion = 0.0
            
            # 2. Add frame to clip
            current_clip.append(frame)
            
            # 3. Decision logic: adaptive step and transition detection
            if use_adaptive_step:
                step = _pick_step(smooth_motion, fps, high_motion_threshold, low_motion_threshold)
            else:
                step = 1 if fps < 10 else (2 if fps < 20 else 4)
            
            # Transition: sudden motion spike after calm
            transition = (
                use_overlap
                and len(current_clip) >= clip_len // 2
                and smooth_motion > high_motion_threshold * 1.5
                and prev_smooth_motion < low_motion_threshold
            )
            
            if len(current_clip) >= clip_len or transition:
                # Pad if necessary
                while len(current_clip) < clip_len:
                    current_clip.append(current_clip[-1])
                clips.append(current_clip[:clip_len])
                
                # Reset or Overlap
                if transition:
                    # Note: We can't easily "rewind" a stream without seeking or buffering.
                    # For simplicity in this streaming v1, we'll just start a new clip immediately
                    # and accept slightly less overlap than the previous full-seek version.
                    current_clip = current_clip[-(clip_len // 2):]
                else:
                    current_clip = []
            
            prev_smooth_motion = smooth_motion
            last_thumb = current_thumb
            target_idx += step
            
        frame_idx += 1

    # Flush final clip
    if current_clip:
        while len(current_clip) < clip_len:
            current_clip.append(current_clip[-1])
        clips.append(current_clip[:clip_len])

    if not clips:
        logger.warning("Sampler: No clips were produced. Check video duration or format.")

    logger.info("Sampler: Produced %d clip(s) from %d frames processed.", len(clips), frame_idx)
    return clips
