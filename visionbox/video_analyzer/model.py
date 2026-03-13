"""
visionbox/video_analyzer/model.py
Loads V-JEPA 2 models — checks local V_JEPA2/ directory first,
falls back to Hugging Face download.
"""
from __future__ import annotations

import contextlib
import io
import os
import logging
from typing import Tuple, Any

logger = logging.getLogger(__name__)

# Project root relative to this file (visionbox/video_analyzer/model.py → ../../)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOCAL_MODEL_DIR = os.path.join(_PROJECT_ROOT, "V_JEPA2")


def load_vjepa_model(
    model_name: str,
    device: str,
    models: dict,
    loading_states: dict,
    get_device,
    clear_other_models,
) -> Tuple[Any, Any]:
    """
    Load a V-JEPA 2 processor + model pair.

    Resolution order:
    1. Return from in-process cache (``models`` dict) if already loaded.
    2. Load from ``<project_root>/V_JEPA2/`` if the directory exists.
    3. Download from Hugging Face Hub.

    Parameters mirror the shared state maintained by ``backend.py``.
    """
    key = f"vjepa_{model_name}_{device}"
    state_key = f"{model_name}::{device}"

    # Always clear other models to ensure only the active one is in VRAM
    clear_other_models(key)

    if key in models:
        return models[key]

    # ── Resolution order ──────────────────────────────────────────────────────
    # 1. LOCAL: model_name matches a subdirectory inside V_JEPA2/
    #    e.g. "V-JEPA2-ViT-L-fpc16-300M" → loads from V_JEPA2/V-JEPA2-ViT-L-fpc16-300M/
    # 2. LOCAL ROOT: HF default ID passed but V_JEPA2/ root dir exists (legacy compat)
    # 3. HUGGING FACE: download from hub (requires internet)

    specific_local_path = os.path.join(_LOCAL_MODEL_DIR, model_name)

    if os.path.isdir(specific_local_path):
        logger.info("✅ Local model found — loading from: %s", specific_local_path)
        model_to_load = specific_local_path
    elif os.path.isdir(_LOCAL_MODEL_DIR) and not os.path.sep in model_name:
        # V_JEPA2/ root exists but the subdir wasn't found — try each subdir with a config.json
        # This handles the case where HF ID is passed but a local equivalent exists
        for entry in sorted(os.listdir(_LOCAL_MODEL_DIR)):
            entry_path = os.path.join(_LOCAL_MODEL_DIR, entry)
            if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "config.json")):
                logger.info("✅ HF ID '%s' not found as local dir, using first available local model: %s",
                            model_name, entry_path)
                model_to_load = entry_path
                break
        else:
            logger.info("⬇️  No local model found for '%s', downloading from Hugging Face...", model_name)
            model_to_load = model_name  # let HF handle it
    else:
        logger.info("⬇️  Downloading model '%s' from Hugging Face...", model_name)
        model_to_load = model_name

    dev = get_device(device)
    loading_states[state_key] = "loading_processor"

    import torch

    # Reset any stale CUDA state from a previous crashed process
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass  # Swallow any lingering CUDA errors from previous runs

    from transformers import VJEPA2VideoProcessor, VJEPA2ForVideoClassification

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            processor = VJEPA2VideoProcessor.from_pretrained(model_to_load)
        except Exception:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_to_load)

        loading_states[state_key] = "loading_model"

        # Try dtype fallback chain: float16 → bfloat16 → float32
        # Some model variants (ViT-G, 1B) may not be stable with float16 on all CUDA versions
        model = None
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            try:
                logger.info("Trying to load V-JEPA with dtype=%s ...", dtype)
                model = VJEPA2ForVideoClassification.from_pretrained(
                    model_to_load, torch_dtype=dtype
                )
                loading_states[state_key] = "moving_to_device"
                
                # Sync CUDA before moving to detect any residual errors
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                model = model.to(dev)
                model.eval()
                
                # Quick GPU sync after move to verify we're clean
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                logger.info("V-JEPA model loaded successfully with dtype=%s on %s", dtype, dev)
                break
            except Exception as e:
                logger.warning("Failed to load with dtype=%s: %s. Trying next...", dtype, e)
                model = None
                # Empty CUDA cache and try to clear any error state
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        if model is None:
            raise RuntimeError(f"Could not load V-JEPA model '{model_to_load}' with any supported dtype.")

        models[key] = (processor, model)

    loading_states[state_key] = "done"
    return models[key]
