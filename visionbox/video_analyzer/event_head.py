"""
visionbox/video_analyzer/event_head.py

Structured Event Head — extracts a semantic descriptor from the pooled
V-JEPA embedding of an event window.

v1 (heuristic):  No learned parameters. Uses argmax over logit-space
                  embeddings to get the action class, plus L2 norm for
                  motion magnitude.  Gracefully handles empty or cryptic
                  id2label mappings by focusing on change magnitude.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class EventDescriptor:
    """Structured description of a detected event — no natural language yet."""
    event_id: int
    top_actions: List[Dict[str, float]]   # [{"label": str, "score": float}, ...]
    motion_magnitude: float               # L2 norm of embedding delta
    confidence: float                     # max softmax probability
    change_intensity: str                 # "high" | "medium" | "low"
    start_idx: int
    end_idx: int
    trigger_idx: int
    is_opening: bool = False              # True for the injected clip-0 event


def _clean_label(raw_label: str) -> str:
    """
    Clean up a raw id2label string into something human-readable.
    Handles SSv2 labels like 'Approaching [something] with your camera'
    and cryptic labels like '['Inward', '35som', 'NoTwis', 'TUCK']'.
    """
    # Check if it looks like a raw list/array dumped as string
    if raw_label.startswith("[") or raw_label.startswith("("):
        return ""  # unusable
    if raw_label.startswith("class_"):
        return ""  # generic fallback, not useful
    # Check if it's very short and looks like an abbreviation
    if len(raw_label) <= 3 and not raw_label.isalpha():
        return ""  # e.g. '35som'
    # Clean up SSv2-style labels
    cleaned = raw_label.replace("[something]", "an object")
    cleaned = cleaned.replace("[Something]", "An object")
    cleaned = cleaned.strip()
    return cleaned


def _classify_intensity(motion_mag: float) -> str:
    """Classify the motion magnitude into human-readable intensity."""
    if motion_mag > 80.0:
        return "high"
    elif motion_mag > 30.0:
        return "medium"
    return "low"


def describe_event(
    embeddings: torch.Tensor,
    start_idx: int,
    end_idx: int,
    trigger_idx: int,
    event_id: int,
    id2label: Dict[int, str],
    top_k: int = 5,
    is_opening: bool = False,
) -> EventDescriptor:
    """
    Produce a structured descriptor for a single event window.

    Heuristic v1: argmax + norms, with label cleaning.
    """
    # Pool: mean of embeddings within the event window
    window_emb = embeddings[start_idx:end_idx + 1]
    pooled = window_emb.mean(dim=0)   # [D]

    # Softmax over the pooled logits → action probabilities
    probs = torch.softmax(pooled, dim=0)
    top_probs, top_indices = probs.topk(min(top_k, probs.numel()))

    top_actions = []
    for prob, idx in zip(top_probs, top_indices):
        raw_label = id2label.get(int(idx), "")
        cleaned = _clean_label(raw_label)
        if cleaned:
            top_actions.append({"label": cleaned, "score": float(prob)})

    # Motion magnitude: L2 norm of the delta between trigger and the clip before it
    if trigger_idx > 0 and trigger_idx < embeddings.shape[0]:
        delta = embeddings[trigger_idx] - embeddings[trigger_idx - 1]
        motion = float(torch.norm(delta, p=2))
    else:
        motion = float(torch.norm(pooled, p=2))

    confidence = float(top_probs[0]) if len(top_probs) > 0 else 0.0
    intensity = _classify_intensity(motion)

    return EventDescriptor(
        event_id=event_id,
        top_actions=top_actions,
        motion_magnitude=motion,
        confidence=confidence,
        change_intensity=intensity,
        start_idx=start_idx,
        end_idx=end_idx,
        trigger_idx=trigger_idx,
        is_opening=is_opening,
    )
