"""
visionbox/video_analyzer/latent_change_detector.py

Detects meaningful semantic transitions in a stream of V-JEPA clip embeddings
using cosine distance with adaptive thresholding.

Design principle:
   V-JEPA decides WHEN something changed — not a language model.
   We measure directional shift (cosine distance) in the latent space,
   then fire triggers only when the shift exceeds a running adaptive threshold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ChangePoint:
    """A single detected semantic change in the embedding stream."""
    clip_index: int
    distance: float       # cosine distance at this point
    threshold: float      # adaptive threshold that was exceeded


def _cosine_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine distances between consecutive embeddings.

    Args:
        embeddings: [N, D] tensor of clip embeddings.

    Returns:
        [N-1] tensor where d[i] = 1 - cos(e[i], e[i+1]).
    """
    normed = F.normalize(embeddings, p=2, dim=1)
    # dot product between consecutive normalised vectors
    cosine_sim = (normed[:-1] * normed[1:]).sum(dim=1)
    return 1.0 - cosine_sim


def detect_latent_changes(
    embeddings: torch.Tensor,
    window_size: int = 5,
    sensitivity: float = 0.8,
    cooldown: int = 1,
    min_distance: float = 0.01,
) -> Tuple[List[ChangePoint], torch.Tensor]:
    """
    Detect points where the V-JEPA latent representation shifts meaningfully.

    Algorithm:
        1. Compute cosine distance between each pair of consecutive clip embeddings.
        2. Maintain a running window of size `window_size` over past distances.
        3. Fire a trigger when d(t) > running_mean + sensitivity * running_std.
        4. Enforce a `cooldown` of clips between consecutive triggers.
        5. Guarantee at least 1 trigger — the clip with the largest delta —
           so short or visually uniform videos still produce narration output.

    Args:
        embeddings:   [N, D] tensor from embed.py
        window_size:  number of past distances for running stats (smaller = faster adaptation)
        sensitivity:  multiplier on std for threshold (lower = more triggers)
        cooldown:     minimum gap between triggers (in clips)
        min_distance: absolute minimum distance to consider as a trigger

    Returns:
        change_points: list of ChangePoint dataclasses
        distances:     [N-1] raw cosine distance tensor
    """
    N = embeddings.shape[0]
    if N < 2:
        logger.warning("detect_latent_changes: need at least 2 clips, got %d", N)
        return [], torch.tensor([])

    distances = _cosine_distances(embeddings)
    logger.info("Latent change detector: %d pairwise distances computed", len(distances))

    change_points: List[ChangePoint] = []
    last_trigger = -cooldown - 1

    # ── Always narrate the opening scene (clip 0) ────────────────────────
    opening = ChangePoint(
        clip_index=0,
        distance=0.0,
        threshold=0.0,
    )
    change_points.append(opening)
    logger.info("Injected opening scene at clip 0")

    for i in range(len(distances)):
        d = distances[i].item()

        # Build running stats from preceding window
        start = max(0, i - window_size)
        window = distances[start:i + 1]
        mu = window.mean().item()
        sigma = window.std().item() if len(window) > 1 else 0.0

        threshold = mu + sensitivity * sigma
        threshold = max(threshold, min_distance)

        if d > threshold and (i - last_trigger) >= cooldown:
            cp = ChangePoint(
                clip_index=i + 1,
                distance=d,
                threshold=threshold,
            )
            change_points.append(cp)
            last_trigger = i
            logger.debug(
                "TRIGGER at clip %d: distance=%.4f > threshold=%.4f (mu=%.4f sigma=%.4f)",
                cp.clip_index, d, threshold, mu, sigma,
            )

    # ── Fallback: always produce at least 1 narrative ──────────────────────
    # If nothing was detected (e.g. very uniform video), pick the single
    # clip with the largest cosine delta so the user always gets output.
    if not change_points and len(distances) > 0:
        peak_idx = int(distances.argmax().item())
        cp = ChangePoint(
            clip_index=peak_idx + 1,
            distance=float(distances[peak_idx]),
            threshold=0.0,
        )
        change_points.append(cp)
        logger.info(
            "No triggers above threshold — using peak delta at clip %d (distance=%.4f) as fallback",
            cp.clip_index, cp.distance,
        )

    logger.info(
        "Latent change detector: %d event(s) detected out of %d clip transitions",
        len(change_points), len(distances),
    )
    return change_points, distances
