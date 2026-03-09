# `visionbox/smart_detect` — Smart Object Detection Pipeline

## Overview

The `smart_detect` module implements a **state-of-the-art, zero-shot object detection and instance segmentation pipeline** by chaining three powerful deep learning models in sequence:

1. **Qwen2.5-VL-3B** — Multimodal Large Language Model (language supervision)
2. **Grounding DINO** — Open-vocabulary, zero-shot object detector
3. **SAM 2 (Segment Anything Model 2)** — Instance segmentation from bounding box prompts

Unlike classic object detectors (such as YOLO) that can only detect a fixed set of pre-trained classes, this pipeline accepts **any natural language query** from the user and detects objects described in free-form text, without retraining.

---

## Why This Architecture?

Traditional object detection models (YOLO, Faster R-CNN) are closed-vocabulary: they can only detect a finite list of classes they were trained on (e.g., 80 COCO classes). This makes them unsuitable for queries like:

- _"Find the blue mug on the left side of the table"_
- _"Detect any damaged areas on the car body"_
- _"Highlight the largest tree in the background"_

This pipeline solves that by using a **language-grounded, open-vocabulary** approach, where the query is first interpreted by an LLM with visual understanding, and then handed off to a detector that can find **anything described in natural language**.

---

## Pipeline Architecture

```
User Query + Image
       │
       ▼
┌─────────────────────┐
│  Qwen2.5-VL-3B      │  LLM Language Supervision
│  (Multimodal LLM)   │  ─── Understands query in context of image
└────────┬────────────┘  ─── Outputs precise noun phrases: e.g. "red car, stop sign"
         │
         ▼ text_prompt (Grounding DINO format: "red car . stop sign .")
┌─────────────────────┐
│  Grounding DINO     │  Zero-Shot Open-Vocabulary Detection
│  (ViT + BERT)       │  ─── Cross-attends image patches to text tokens
└────────┬────────────┘  ─── Outputs bounding boxes + confidence scores + labels
         │
         ▼ boxes [[x1,y1,x2,y2], ...]
┌─────────────────────┐
│  SAM 2              │  Instance Segmentation from Box Prompts
│  (Segment Anything) │  ─── Takes each bounding box as spatial prompt
└────────┬────────────┘  ─── Outputs per-instance boolean masks [H, W]
         │
         ▼
  JSON response with bounding boxes, labels, confidence scores,
  and base64-encoded RGBA PNG segmentation masks
```

---

## Stage-by-Stage Explanation

### Stage 1 — Language Supervision with Qwen2.5-VL

**Function:** `generate_search_queries(img, user_query, qwen_processor, qwen_model)`

The first stage converts a potentially vague, high-level user query into **precise, grounding-ready noun phrases** that Grounding DINO can anchor to visual regions.

**Why this is necessary:**
Grounding DINO expects short, unambiguous object phrases separated by `.` (e.g. `"cat . red car . person ."`). A raw user query like _"find whatever looks suspicious on the shelf"_ cannot be directly fed into a detector — it is too ambiguous and contextual.

**How it works:**

1. The full image is passed to Qwen2.5-VL alongside a carefully engineered system prompt that instructs the model to act as a "computer vision assistant."
2. The model is asked to output **only** a comma-separated list of object names relevant to the query, without any surrounding explanation or preamble.
3. The output is post-processed with a regex to strip quotes and brackets, then split by comma into a clean list of phrases.
4. These phrases are joined in Grounding DINO's expected format: `"red car . stop sign . pedestrian ."`.

**Example:**

| User Query                          | Qwen Output          | DINO Prompt               |
| ----------------------------------- | -------------------- | ------------------------- |
| `"find all the red vehicles"`       | `red car, red truck` | `"red car . red truck ."` |
| `"what's the animal in the scene?"` | `cat`                | `"cat ."`                 |
| `"find interesting objects"`        | `cat, lamp, mug`     | `"cat . lamp . mug ."`    |

**Inference settings:** Images are downscaled to a max of `1280×720` pixels for this stage to limit memory usage, since full-resolution inputs are not needed for query interpretation. `max_new_tokens=50` is used to keep the output short and structured.

After use, the Qwen processor and model are **explicitly deleted** and `torch.cuda.empty_cache()` is called before the next stage loads.

---

### Stage 2 — Zero-Shot Detection with Grounding DINO

**Function:** `detect_with_grounding_dino(img, text_prompt, gdino_processor, gdino_model, threshold)`

**Model:** `IDEA-Research/grounding-dino-tiny` (via Hugging Face `transformers`)

Grounding DINO is a **language-conditioned object detector** that fuses a BERT-based text encoder with a ViT-based image encoder using cross-attention. It learns to align image regions with arbitrary text phrases without seeing those specific classes during training.

**How it works:**

1. The processor tokenizes the text prompt and extracts image patch features simultaneously.
2. The model performs cross-modal attention between text tokens and image patch tokens, allowing each text phrase to "attend" to the image regions that match it semantically.
3. The `post_process_grounded_object_detection()` method decodes the output into pixel-space bounding boxes (in `[x1, y1, x2, y2]` format), confidence scores, and the matched text label for each box.
4. Both the visual detection threshold and text matching threshold are set to the user-specified `threshold` (default: `0.3`).

**Output:** A dictionary containing `boxes` (float tensor, pixel coordinates), `scores` (float tensor), and `labels` (list of matched strings).

After use, the Grounding DINO model is explicitly released and GPU cache is cleared before SAM loads.

---

### Stage 3 — Instance Segmentation with SAM 2

**Function:** `segment_with_sam(img, boxes, sam_processor, sam_model)`

**Model:** `facebook/sam2-hiera-small` (via Hugging Face `transformers` — native SAM 2 support)

SAM 2 (Segment Anything Model 2) is Meta's second-generation segmentation model. It is prompt-based: given a spatial hint (e.g., a point, a box, or a mask), it delineates the full object at high resolution. Here, the bounding boxes from Grounding DINO serve as **box prompts**.

**How it works:**

1. The processor accepts the input image and the set of bounding boxes (one per detected object) as `input_boxes`.
2. The encoder computes high-resolution image embeddings using a hierarchical ViT (Hiera) backbone.
3. For each bounding box, the decoder generates **three mask candidates** and a corresponding IoU prediction score for each. This multi-mask output accounts for ambiguity (e.g., when the box covers two touching objects).
4. The mask with the **highest predicted IoU score** is selected as the best mask for each detection.
5. `post_process_masks()` maps the masks back to the original image resolution using the stored `original_sizes`.

**Output:** A list of `[H, W]` boolean NumPy arrays — one per detected bounding box.

---

### Result Encoding

After all three models run, each detection is assembled into a Python dict:

```python
{
    "box":        [x1, y1, x2, y2],  # pixel coords in original image space
    "label":      "cat",              # matched label from Grounding DINO
    "score":      0.93,               # confidence score (0–1)
    "mask_base64": "<...>"            # base64-encoded RGBA PNG
}
```

**Why RGBA PNG?** The segmentation mask is encoded as an **RGBA PNG** (not grayscale), where:

- Pixels **inside** the segmented object → `alpha = 255` (fully opaque)
- Pixels **outside** the object → `alpha = 0` (fully transparent)
- All RGB channels are set to white (`255, 255, 255`)

This design choice enables the frontend JavaScript canvas to use `globalCompositeOperation = 'source-in'` to paint the mask in any desired color, affecting only the pixels within the object's shape without bleeding over the background.

---

## VRAM Management Strategy

All three models (Qwen2.5-VL-3B, Grounding DINO, SAM 2) require significant VRAM individually. Loading all three simultaneously would exceed 8 GB VRAM, which is the limit of most consumer-grade GPUs (e.g., RTX 3070/4070).

The pipeline implements a **sequential loading strategy**:

```
Load Qwen → Run → DELETE Qwen → gc.collect() + empty_cache()
Load DINO → Run → DELETE DINO → gc.collect() + empty_cache()
Load SAM  → Run → DELETE SAM  → gc.collect() + empty_cache()
```

Each model is garbage-collected before the next one is loaded. This ensures the total peak VRAM never simultaneously holds more than one large model.

| Model                         | Approx. VRAM (4-bit Qwen / full DINO / SAM-small) |
| ----------------------------- | ------------------------------------------------- |
| Qwen2.5-VL-3B (4-bit)         | ~1.5 GB                                           |
| Grounding DINO (tiny)         | ~0.7 GB                                           |
| SAM 2 (hiera-small)           | ~0.9 GB                                           |
| **Peak (sequential)**         | **~1.5 GB**                                       |
| **Hypothetical (concurrent)** | **~3.1 GB — would OOM on 4-bit**                  |

---

## Real-Time Progress Feedback

The pipeline supports an optional `status_callback` argument:

```python
run_smart_detect(..., status_callback=fn)
```

Where `fn(stage: str, percent: int, label: str)` is called at each pipeline milestone:

| Stage                | %   | Label                                    |
| -------------------- | --- | ---------------------------------------- |
| `loading_qwen`       | 10  | Loading Qwen2.5-VL...                    |
| `analyzing_query`    | 25  | Analyzing query with Qwen2.5-VL...       |
| `loading_gdino`      | 40  | Loading Grounding DINO...                |
| `detecting_objects`  | 55  | Detecting objects with Grounding DINO... |
| `loading_sam`        | 70  | Loading SAM 2...                         |
| `segmenting_objects` | 85  | Segmenting objects with SAM 2...         |
| `processing_results` | 95  | Processing visualization...              |
| `done`               | 100 | Ready!                                   |

The backend stores the current stage in a global `smart_detect_state` dict and exposes it via the `GET /api/smart-detect-status` endpoint. The frontend JavaScript polls this endpoint every second during inference and renders a live progress bar.

---

## API Integration

The pipeline is exposed via the FastAPI backend (`visionbox/backend.py`) and proxied through the Java Spring Boot application:

| Layer            | Path                                               |
| ---------------- | -------------------------------------------------- |
| Python FastAPI   | `POST /api/smart-detect`                           |
| Python FastAPI   | `GET /api/smart-detect-status`                     |
| Java Spring Boot | `POST /api/smart-detect` → proxies to Python       |
| Java Spring Boot | `GET /api/smart-detect-status` → proxies to Python |

**Request format** (`POST /api/smart-detect`, JSON body):

```json
{
  "image_base64": "<base64-encoded image>",
  "user_query": "find the cat and the blue chair",
  "threshold": 0.3,
  "device": "cuda",
  "precision": "4"
}
```

**Response format**:

```json
{
  "queries_used": ["cat", "blue chair"],
  "detections": [
    {
      "box": [391.3, 268.4, 1657.6, 2289.6],
      "label": "cat",
      "score": 0.93,
      "mask_base64": "iVBORw0KGgo..."
    }
  ]
}
```

---

## Files

| File          | Purpose                                    |
| ------------- | ------------------------------------------ |
| `pipeline.py` | Full detection pipeline: Qwen → DINO → SAM |
| `README.md`   | This file                                  |

The module has no `__init__.py` — it is imported directly by the backend:

```python
from visionbox.smart_detect.pipeline import run_smart_detect
```

---

## Dependencies

| Package         | Purpose                                                        |
| --------------- | -------------------------------------------------------------- |
| `transformers`  | Hosts Grounding DINO, SAM 2, and Qwen2.5-VL via HuggingFace    |
| `torch`         | Model inference and GPU tensor operations                      |
| `Pillow`        | Image loading, resizing, and RGBA mask encoding                |
| `numpy`         | Mask array manipulation                                        |
| `qwen_vl_utils` | Utility for processing vision inputs for Qwen VL models        |
| `bitsandbytes`  | Quantized 4-bit/8-bit loading of Qwen to reduce VRAM footprint |
