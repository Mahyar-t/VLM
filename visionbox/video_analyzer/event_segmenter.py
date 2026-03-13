"""
visionbox/video_analyzer/event_segmenter.py

Groups raw change-detector triggers into coherent event windows.
Adjacent triggers that fire within a small gap are merged into a single event.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import torch

from .latent_change_detector import ChangePoint

logger = logging.getLogger(__name__)


@dataclass
class EventWindow:
    """A coherent event spanning one or more clips."""
    start_idx: int          # first clip index (inclusive)
    end_idx: int            # last clip index (inclusive)
    trigger_idx: int        # the primary trigger clip
    peak_delta: float       # largest cosine distance within this event
    event_id: int           # sequential event number


def segment_events(
    change_points: List[ChangePoint],
    distances: torch.Tensor,
    total_clips: int,
    merge_gap: int = 3,
    context_before: int = 1,
    context_after: int = 1,
) -> List[EventWindow]:
    """
    Convert raw trigger points into structured event windows.

    Algorithm:
        1. Sort triggers by clip index.
        2. Merge triggers within `merge_gap` clips into one event group.
        3. Expand each group with `context_before` / `context_after` clips.
        4. Clamp boundaries to [0, total_clips - 1].

    Args:
        change_points:  list of ChangePoint from the detector
        distances:      [N-1] cosine distance tensor
        total_clips:    total number of clips in the video
        merge_gap:      max gap between triggers to merge into one event
        context_before: extra clips before the trigger
        context_after:  extra clips after the trigger

    Returns:
        List of EventWindow, chronologically ordered.
    """
    if not change_points:
        logger.info("No change points provided — no events to segment.")
        return []

    # Sort by clip index (should already be, but be safe)
    sorted_cps = sorted(change_points, key=lambda cp: cp.clip_index)

    # --- Group adjacent triggers ---
    groups: List[List[ChangePoint]] = []
    current_group: List[ChangePoint] = [sorted_cps[0]]

    for cp in sorted_cps[1:]:
        if cp.clip_index - current_group[-1].clip_index <= merge_gap:
            current_group.append(cp)
        else:
            groups.append(current_group)
            current_group = [cp]
    groups.append(current_group)

    # --- Build EventWindows ---
    events: List[EventWindow] = []

    for event_id, group in enumerate(groups):
        # Primary trigger = the one with the largest distance in the group
        primary = max(group, key=lambda cp: cp.distance)

        raw_start = min(cp.clip_index for cp in group) - context_before
        raw_end = max(cp.clip_index for cp in group) + context_after

        start = max(0, raw_start)
        end = min(total_clips - 1, raw_end)

        # Peak delta within the event window (from the distances tensor)
        dist_start = max(0, start)
        dist_end = min(len(distances), end)
        if dist_start < dist_end:
            peak = distances[dist_start:dist_end].max().item()
        else:
            peak = primary.distance

        events.append(EventWindow(
            start_idx=start,
            end_idx=end,
            trigger_idx=primary.clip_index,
            peak_delta=peak,
            event_id=event_id,
        ))

    logger.info("Event segmenter: %d event(s) from %d trigger(s)", len(events), len(change_points))
    return events
