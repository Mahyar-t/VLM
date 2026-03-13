"""
visionbox/video_analyzer/narration_decoder.py

Event-triggered VLM narration decoder.

Architecture:
    V-JEPA 2 decides WHEN to narrate (opening scene + detected changes).
    Qwen2.5-VL describes WHAT is visible at each triggered event.

This is far cheaper than captioning every frame because the VLM
only fires at event boundaries — typically 2–5 calls total per video
vs. dozens for dense captioning.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image

from .event_head import EventDescriptor

logger = logging.getLogger(__name__)


def _caption_frame(
    frame_np,
    processor,
    model,
    device: str,
    prompt: str,
    max_tokens: int = 120,
) -> str:
    """Run Qwen2.5-VL on a single frame with a given prompt."""
    from qwen_vl_utils import process_vision_info

    img_pil = Image.fromarray(frame_np)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_pil},
            {"type": "text", "text": prompt},
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    img_inputs, vid_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text], images=img_inputs, videos=vid_inputs,
        padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens)

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    caption = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return caption


def narrate_event(
    descriptor: EventDescriptor,
    clips: list,
    qwen_processor,
    qwen_model,
    qwen_device: str,
    total_clips: int = 0,
    prev_caption: str = "",
) -> str:
    """
    Use Qwen2.5-VL to produce a rich narration for a single event.

    For the opening scene: "Describe in detail what you see."
    For change events: "Describe what changed compared to the previous scene."
    """
    # Pick the representative frame: middle frame of the trigger clip
    trigger = min(descriptor.trigger_idx, len(clips) - 1)
    clip_frames = clips[trigger]
    mid_frame = clip_frames[len(clip_frames) // 2]

    clip_range = f"clips {descriptor.start_idx}–{descriptor.end_idx}"

    if descriptor.is_opening:
        prompt = (
            "Describe in detail what you see in this video frame. "
            "Focus on the main subjects, their actions, and the setting. "
            "Write 1-2 concise sentences."
        )
        caption = _caption_frame(
            mid_frame, qwen_processor, qwen_model, qwen_device, prompt
        )
        return f"Opening ({clip_range}): {caption}"
    else:
        if prev_caption:
            prompt = (
                f"The previous scene showed: \"{prev_caption}\"\n\n"
                "Now describe what you see in this new frame. "
                "Focus on what is different or has changed. "
                "Write 1-2 concise sentences."
            )
        else:
            prompt = (
                "Describe in detail what you see in this video frame. "
                "Focus on the main subjects, their actions, and the setting. "
                "Write 1-2 concise sentences."
            )

        caption = _caption_frame(
            mid_frame, qwen_processor, qwen_model, qwen_device, prompt
        )
        return f"Event {descriptor.event_id} ({clip_range}): {caption}"


def narrate_events(
    descriptors: List[EventDescriptor],
    clips: list,
    qwen_processor,
    qwen_model,
    qwen_device: str,
    total_clips: int = 0,
    progress_fn: Optional[Callable[[str, int], None]] = None,
) -> dict:
    """
    Narrate all events using Qwen2.5-VL.

    Returns:
        {
            "events": ["Opening: ...", "Event 1: ...", ...],
            "summary": "full chronological narration",
            "event_count": int,
        }
    """
    if not descriptors:
        return {
            "events": [],
            "summary": "No significant events were detected in this video.",
            "event_count": 0,
        }

    narrations = []
    prev_caption = ""

    for i, desc in enumerate(descriptors):
        pct = 88 + int(10 * i / len(descriptors))
        if progress_fn:
            label = "Describing opening scene..." if desc.is_opening else f"Describing event {desc.event_id}..."
            progress_fn(label, pct)

        narration = narrate_event(
            descriptor=desc,
            clips=clips,
            qwen_processor=qwen_processor,
            qwen_model=qwen_model,
            qwen_device=qwen_device,
            total_clips=total_clips,
            prev_caption=prev_caption,
        )
        narrations.append(narration)

        # Extract just the caption part (after the prefix) for context chaining
        if ": " in narration:
            prev_caption = narration.split(": ", 1)[1]
        else:
            prev_caption = narration

    summary = " ".join(narrations)

    logger.info("VLM narration: produced %d event caption(s)", len(narrations))

    return {
        "events": narrations,
        "summary": summary,
        "event_count": len(narrations),
    }
