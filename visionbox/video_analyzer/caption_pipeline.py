"""
visionbox/video_analyzer/caption_pipeline.py
Orchestrates the frame extraction, individual captioning, and final summarization using Qwen.
"""

import logging
import torch
from PIL import Image
from typing import Dict, Any, Callable, Optional

from .embed import extract_video_embeddings
from .key_selector import select_key_clips

logger = logging.getLogger(__name__)


def generate_video_summary(
    video_path: str,
    vjepa_model_name: str,
    qwen_device: str,
    vjepa_device: str,
    load_vjepa_fn: Callable,
    load_qwen_fn: Callable,
    get_device_fn: Callable,
    clip_len: int = 64,
    k_clips: int = 3,
    progress_fn: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """
    End-to-end pipeline for Video Summarization.
    progress_fn(stage: str, percent: int): optional callback to report progress.
    """
    def _progress(stage: str, pct: int):
        logger.info("[SUMMARIZE %d%%] %s", pct, stage)
        if progress_fn:
            progress_fn(stage, pct)

    _progress("Extracting V-JEPA embeddings from video clips...", 5)

    # 1. Extract embeddings (returns model object too — avoids a second load call)
    clips, embeddings, _ = extract_video_embeddings(
        video_path=video_path,
        model_name=vjepa_model_name,
        device=vjepa_device,
        load_model_fn=load_vjepa_fn,
        get_device_fn=get_device_fn,
        clip_len=clip_len,
        progress_fn=_progress,
    )

    _progress("Selecting representative key scenes...", 52)

    # 2. Select key clips
    key_indices = select_key_clips(embeddings, k=k_clips)

    # 3. Extract 1 representative frame per key clip (the middle frame)
    key_frames = []
    for idx in key_indices:
        clip_frames = clips[idx]
        mid_idx = len(clip_frames) // 2
        key_frames.append(clip_frames[mid_idx])  # np.ndarray (RGB)

    _progress("Loading Qwen2.5-VL model for captioning...", 55)

    # 4. Load Qwen2.5-VL
    processor, model = load_qwen_fn(qwen_device, "4")

    from qwen_vl_utils import process_vision_info

    # 5. Caption each frame individually
    captions = []
    n_frames = len(key_frames)
    logger.info("Generating captions for %d key scenes...", n_frames)

    for i, frame_np in enumerate(key_frames):
        pct = 60 + int(25 * i / n_frames)
        _progress(f"Captioning key scene {i + 1} of {n_frames}...", pct)

        img_pil = Image.fromarray(frame_np)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_pil},
                {"type": "text",
                 "text": "Describe this key frame from a video in detail. Focus on the main action, subjects, and setting."},
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text], images=img_inputs, videos=vid_inputs,
            padding=True, return_tensors="pt"
        ).to(qwen_device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150)

        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        caption = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        # Approximate timestamp from clip index
        approx_frame = key_indices[i] * clip_len
        stage_desc = f"Scene {i + 1} (~frame {approx_frame}): {caption}"
        captions.append(stage_desc)
        logger.info(stage_desc)

    _progress("Generating final video summary...", 88)

    # 6. Final Summary
    all_captions_text = "\n".join(captions)
    summary_prompt = (
        f"Here are descriptions of key scenes from a video in chronological order:\n\n"
        f"{all_captions_text}\n\n"
        f"Based on these scenes, write a concise, cohesive paragraph summarizing "
        f"the overall event or action in the video."
    )

    summary_messages = [{
        "role": "user",
        "content": [{"type": "text", "text": summary_prompt}]
    }]

    summary_text = processor.apply_chat_template(summary_messages, tokenize=False, add_generation_prompt=True)
    summary_inputs = processor(
        text=[summary_text], padding=True, return_tensors="pt"
    ).to(qwen_device)

    with torch.no_grad():
        summary_out = model.generate(**summary_inputs, max_new_tokens=256)

    summary_trimmed = [o[len(i):] for i, o in zip(summary_inputs.input_ids, summary_out)]
    final_summary = processor.batch_decode(summary_trimmed, skip_special_tokens=True)[0].strip()

    _progress("Done!", 100)
    logger.info("Summary pipeline complete.")

    return {
        "scenes": captions,
        "summary": final_summary
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Event-Driven Narration Pipeline
#   V-JEPA 2 decides WHEN → Qwen2.5-VL describes WHAT
# ═══════════════════════════════════════════════════════════════════════════════

def narrate_video(
    video_path: str,
    vjepa_model_name: str,
    vjepa_device: str,
    load_vjepa_fn: Callable,
    load_qwen_fn: Callable,
    get_device_fn: Callable,
    qwen_device: str = "cuda",
    clip_len: int = 64,
    sensitivity: float = 0.8,
    cooldown: int = 1,
    merge_gap: int = 3,
    progress_fn: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """
    Event-driven video narration.

    Pipeline:
        1. Extract V-JEPA embeddings (reuses embed.py)
        2. Detect latent changes (cosine distance + adaptive threshold)
        3. Segment into event windows
        4. Build structured descriptors
        5. Load Qwen2.5-VL and caption only at event triggers

    Returns:
        {
            "events":       ["Opening: ...", "Event 1: ...", ...],
            "summary":      "Full chronological narration",
            "event_count":  int,
            "total_clips":  int,
        }
    """
    from .latent_change_detector import detect_latent_changes
    from .event_segmenter import segment_events
    from .event_head import describe_event
    from .narration_decoder import narrate_events

    def _progress(stage: str, pct: int):
        logger.info("[NARRATE %d%%] %s", pct, stage)
        if progress_fn:
            progress_fn(stage, pct)

    # 1. Extract embeddings (also returns the loaded model to avoid a second load)
    clips, embeddings, vjepa_model = extract_video_embeddings(
        video_path=video_path,
        model_name=vjepa_model_name,
        device=vjepa_device,
        load_model_fn=load_vjepa_fn,
        get_device_fn=get_device_fn,
        clip_len=clip_len,
        progress_fn=_progress,
    )

    total_clips = len(clips)

    # ── 2. Detect latent changes ─────────────────────────────────────────────
    _progress("Detecting latent scene changes...", 55)

    change_points, distances = detect_latent_changes(
        embeddings,
        sensitivity=sensitivity,
        cooldown=cooldown,
    )

    _progress(f"Detected {len(change_points)} change point(s) — segmenting events...", 65)

    # ── 3. Segment into events ───────────────────────────────────────────────
    events = segment_events(
        change_points=change_points,
        distances=distances,
        total_clips=total_clips,
        merge_gap=merge_gap,
    )

    _progress(f"Segmented into {len(events)} event(s) — building descriptors...", 75)

    # ── 4. Get id2label from already-loaded model — no second load needed ────
    id2label = getattr(vjepa_model.config, "id2label", {})

    descriptors = []
    for event in events:
        desc = describe_event(
            embeddings=embeddings,
            start_idx=event.start_idx,
            end_idx=event.end_idx,
            trigger_idx=event.trigger_idx,
            event_id=event.event_id,
            id2label=id2label,
            is_opening=(event.event_id == 0),
        )
        descriptors.append(desc)

    # ── 5. Free V-JEPA from VRAM before loading Qwen ─────────────────────────
    # We must delete the local reference so `clear_other_models` can gc it.
    del vjepa_model
    
    _progress("Loading Qwen2.5-VL for event captioning...", 82)
    qwen_processor, qwen_model = load_qwen_fn(qwen_device, "4")

    result = narrate_events(
        descriptors=descriptors,
        clips=clips,
        qwen_processor=qwen_processor,
        qwen_model=qwen_model,
        qwen_device=qwen_device,
        total_clips=total_clips,
        progress_fn=_progress,
    )
    result["total_clips"] = total_clips

    _progress("Done!", 100)

    return result

