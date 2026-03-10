"""
visionbox/video_analyzer/embed.py
Extract per-clip semantic embeddings from V-JEPA 2 for use in key-clip selection.

IMPLEMENTATION NOTE:
  We intentionally use the raw CLASSIFICATION LOGITS as clip embeddings rather than
  the hidden states. This approach:
   1. Avoids the CUDA kernel assertion that fires when output_hidden_states=True is
      passed to VJEPA2ForVideoClassification (the classification head triggers an
      index-assertion on the label space).
   2. Works identically for ALL V-JEPA model variants (ViT-L, ViT-G, etc.)
      because the logits are always shape [1, num_classes].
   3. Still captures full semantic content — the logit vector IS a linear projection
      of the pooled encoder features, so clips with similar semantics will have
      similar logit vectors.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from .sampler import sample_video

logger = logging.getLogger(__name__)


def extract_video_embeddings(
    video_path: str,
    model_name: str,
    device: str,
    load_model_fn: Callable,   # (model_name, device) → (processor, model)
    get_device_fn: Callable,   # (device: str) → torch.device
    clip_len: int = 64,
    use_adaptive_step: bool = True,
    use_overlap: bool = True,
    progress_fn: Optional[Callable[[str, int], None]] = None,
) -> Tuple[List[list], "torch.Tensor"]:
    """
    Sample a video into clips and extract a logit-space embedding for each clip.

    Returns:
        clips: List of clip frames (each clip is a list of np.ndarray).
        embeddings: Tensor of shape [N, C] where N=num_clips, C=num_classes.
    """
    import torch

    dev = get_device_fn(device)

    if progress_fn:
        progress_fn("Loading V-JEPA model...", 5)

    processor, model = load_model_fn(model_name, device)
    model.eval()

    if progress_fn:
        progress_fn("Sampling video into clips...", 15)

    # 1. Sample clips
    clips = sample_video(
        video_path,
        clip_len=clip_len,
        use_adaptive_step=use_adaptive_step,
        use_overlap=use_overlap,
    )
    logger.info("extract_video_embeddings: %d clip(s) sampled", len(clips))

    if not clips:
        raise RuntimeError("No clips could be sampled from the video.")

    # 2. Extract per-clip logit vectors
    all_embeddings: List[torch.Tensor] = []

    for clip_idx, clip_frames in enumerate(clips):
        pct = 15 + int(35 * (clip_idx + 1) / len(clips))
        if progress_fn:
            progress_fn(f"Extracting embedding — clip {clip_idx + 1}/{len(clips)}...", pct)

        try:
            inputs = processor(clip_frames, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, "to")}

            with torch.no_grad():
                # Standard forward pass — NO output_hidden_states, just logits
                outputs = model(**inputs)

            # logits shape: [1, num_classes]  — a lightweight semantic embedding
            logit_vec = outputs.logits.squeeze(0).cpu().float()
            all_embeddings.append(logit_vec)
            logger.info("Clip %d: logit embedding shape %s", clip_idx, logit_vec.shape)

        except Exception as exc:
            logger.warning("Clip %d embedding failed: %s", clip_idx, exc)
            # Skip failed clips rather than crashing everything
            continue

    if not all_embeddings:
        raise RuntimeError("Embedding extraction failed for all clips.")

    embeddings_tensor = torch.stack(all_embeddings, dim=0)  # [N, C]

    if progress_fn:
        progress_fn("Embeddings extracted — selecting key scenes...", 50)

    return clips, embeddings_tensor
