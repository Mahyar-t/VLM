"""
visionbox/video_analyzer/predict.py
Video classification inference using V-JEPA 2 and the smart sampler.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List

import numpy as np

from .sampler import sample_video

logger = logging.getLogger(__name__)


def classify_video(
    video_path: str,
    model_name: str,
    device: str,
    load_model_fn,       # callable: (model_name, device) → (processor, model)
    get_device_fn,       # callable: (device: str) → torch.device
    top_k: int = 5,
    clip_len: int = 64,
    high_motion_threshold: float = 15.0,
    low_motion_threshold: float = 5.0,
    aggregate_clips: bool = True,
    use_adaptive_step: bool = True,
    use_overlap: bool = True,
) -> List[Dict[str, Any]]:
    """
    Classify actions in a video file.
    """
    import torch  # local import so the module is importable without torch

    dev = get_device_fn(device)
    processor, model = load_model_fn(model_name, device)

    # ── 1. Sample clips ───────────────────────────────────────────────────────
    clips = sample_video(
        video_path,
        clip_len=clip_len,
        high_motion_threshold=high_motion_threshold,
        low_motion_threshold=low_motion_threshold,
        use_adaptive_step=use_adaptive_step,
        use_overlap=use_overlap,
    )
    logger.info("classify_video: %d clip(s) to process", len(clips))

    # ── 2. Per-clip inference ─────────────────────────────────────────────────
    all_logits: List[torch.Tensor] = []

    for clip_idx, clip_frames in enumerate(clips):
        try:
            inputs = processor(clip_frames, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, "to")}

            with torch.no_grad():
                outputs = model(**inputs)

            all_logits.append(outputs.logits.squeeze(0).cpu())
        except Exception as exc:
            logger.warning("Clip %d failed: %s", clip_idx, exc)
            continue

    if not all_logits:
        raise RuntimeError("All clips failed during inference.")

    # ── 3. Aggregate ──────────────────────────────────────────────────────────
    if aggregate_clips:
        # Aggregation: mean logits across all clips
        stacked = torch.stack(all_logits, dim=0)   # (num_clips, num_classes)
        final_logits = stacked.mean(dim=0)          # (num_classes,)
    else:
        # Just use the first successful clip
        final_logits = all_logits[0]

    probs = final_logits.softmax(dim=-1)

    # ── 4. Top-k ──────────────────────────────────────────────────────────────
    top_probs, top_idx = probs.topk(min(top_k, probs.numel()))
    id2label: Dict[int, str] = model.config.id2label

    return [
        {"class": id2label[int(i)], "probability": float(p)}
        for p, i in zip(top_probs, top_idx)
    ]
