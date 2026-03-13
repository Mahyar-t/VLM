# Video Analyzer Pipeline

This directory contains the core logic for the **Video Summarization Pipeline**, a state-of-the-art approach to processing long-form videos. The pipeline intelligently samples frames, extracts lightweight semantic embeddings using **V-JEPA 2**, selects the most representative scenes, and generates a cohesive natural language summary using **Qwen2.5-VL**.

## High-Level Architecture

The summarization process flows through the following stages:

1. **Smart Sampling (`sampler.py`)**  
   The video is not just blindly split. It uses motion-adaptive stepping (MAD score) to sample dynamic scenes more densely, while sparsely sampling static shots. It also supports overlapping clip sampling so actions across cuts aren't lost.
2. **Feature Extraction (`embed.py` & `model.py`)**  
   The sampled clips are passed through a Meta **V-JEPA 2** encoder. Instead of heavy hidden states, we extract the **raw classification logits** of the `[CLS]` token. This provides a highly discriminative, lightweight semantic vector for every clip, completely avoiding `float16` CUDA assertions under `output_hidden_states=True` in standard pipelines.
3. **Key Scene Selection (`key_selector.py`)**  
   Using the extracted embeddings, the pipeline selects $k$ "Key Scenes". It computes temporal differences (L2 norm) between adjacent clips to find moments of peak semantic change (action) while penalizing selecting clips that are temporally too close to one another.
4. **Captioning & Summarization (`caption_pipeline.py`)**  
   A central frame from each chosen Key Scene is passed to **Qwen2.5-VL** (loaded dynamically in `bfloat16` to prevent `NaN` generation). Each frame is individually captioned, and finally, all the text descriptions are aggregated and passed to the Language Model once more to generate a coherent, chronological final video summary.

## Key Files & Responsibilities

| File                  | Responsibility                                                                                                                                                                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `caption_pipeline.py` | The main orchestrator. Wires up extraction, selection, and the Qwen LMM for the two-stage caption-then-summarize routine.                                                                                                                         |
| `embed.py`            | Extracts the logit-space embeddings from the V-JEPA model for each clip. Gracefully handles `OOM` and `device-side assertions` during forward passes.                                                                                             |
| `key_selector.py`     | Implements the math for selecting diverse, high-action clips (temporal boundaries) from continuous embedding streams.                                                                                                                             |
| `model.py`            | Responsible for robustly loading the V-JEPA components. Implements fallback from `float16` → `bfloat16` → `float32`, sanitizes stale CUDA contexts (`torch.cuda.synchronize()`), and prioritizes local model weights over Hugging Face downloads. |
| `sampler.py`          | Decodes the source video via PyAV or OpenCV. Dynamically shifts the frame-step (FPS decimation) based on local motion complexity.                                                                                                                 |

## Model Resolution Strategy

To ensure offline availability and speed, `model.py` follows a strict resolution hierarchy for V-JEPA models:

1. **Specific Local Folder Match**: E.g., `V_JEPA2/V-JEPA2-ViT-L-fpc16-300M`
2. **Fallback to First Available**: If the UI requests the HF default, but the local `V_JEPA2/` root contains an alternative valid model, it picks the first local option automatically to save bandwidth.
3. **Hugging Face Hub**: Only downloads if the local `V_JEPA2` directory is empty or missing.

## Memory & Hardware Stability

Video Language Models (VLMs) and advanced Vision Transformers push the edge of available VRAM. This module implements several defensive paradigms:

- **bfloat16 Fallback**: `Qwen2.5-VL` scaling logic defaults to `bfloat16` since standard IEEE `float16` overflows and creates `NaN` tensors during generation (`torch.multinomial` failure).
- **Stale CUDA Protection**: If a previous pipeline crashes, the GPU context may hold a pending error flag. The backend triggers `torch.cuda.synchronize()` immediately prior to loading or cache-clearing memory to silently flush and reset, allowing seamless recovery without a process restart.
- **Batched Exectution**: We don't hold the video and language models simultaneously active unless absolutely required. (Also managed heavily by the `/api/free-memory` backend API).
