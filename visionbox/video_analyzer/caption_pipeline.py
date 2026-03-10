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

    # 1. Extract embeddings
    clips, embeddings = extract_video_embeddings(
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
