import io
import contextlib
import asyncio
import base64
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional

import os
import logging
# Suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    Blip2Processor, Blip2ForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    BitsAndBytesConfig,
)

# Local Qwen2.5-VL model path (sentinel value used as model_name by the frontend)
QWEN_LOCAL_ID  = "local::Qwen2.5-VL-3B"
_QWEN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "Qwen/Qwen2.5-VL-3B-Instruct"
)
QWEN_LOCAL_PATH = _QWEN_DIR if os.path.exists(_QWEN_DIR) else "Qwen/Qwen2.5-VL-3B-Instruct"

from visionbox.utils import get_device

# Global dictionary to hold lazy-loaded models in VRAM
models = {}

# Tracks real-time loading state per (model_name, device) key
loading_states: dict = {}

STAGE_PERCENTAGES = {
    "queued":            5,
    "loading_processor": 20,
    "loading_model":     65,
    "moving_to_device":  88,
    "done":             100,
    "error":            100,
}

app = FastAPI(title="VisionBox API Server")

def _is_blip2(model_name: str) -> bool:
    return "blip2" in model_name.lower()

def _is_qwen(model_name: str) -> bool:
    return model_name == QWEN_LOCAL_ID

import gc

def clear_other_models(key_to_keep: str):
    """
    Clears all models from VRAM except `key_to_keep` and the small CLIP model.
    Allows V-JEPA and Qwen to coexist since they fit in 8GB VRAM together.
    """
    keys_to_delete = []
    for k in models.keys():
        if k == key_to_keep or k.startswith("clip_"):
            continue
            
        # Allow Qwen and V-JEPA to coexist
        if "Qwen" in key_to_keep and k.startswith("vjepa_"):
            continue
        if key_to_keep.startswith("vjepa_") and "Qwen" in k:
            continue
            
        keys_to_delete.append(k)
        
    if keys_to_delete:
        for k in keys_to_delete:
            print(f"Clearing {k} from VRAM to free space...")
            del models[k]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def load_qwen_caption_model(device: str, precision: str = "4"):
    key = f"caption_{QWEN_LOCAL_ID}_{device}_{precision}"
    state_key = f"{QWEN_LOCAL_ID}::{device}"
    
    # Always clear other models to ensure only the active one is in VRAM
    clear_other_models(key)

    if key not in models:
        print(f"Loading Qwen2_5-VL-3B ({precision}-bit) from {QWEN_LOCAL_PATH} ...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        # Avoid NaN probabilities during generation by using bfloat16 if hardware supports it
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        quant_cfg = None
        if precision == "4":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif precision == "8":
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        loading_states[state_key] = "loading_processor"
        processor = AutoProcessor.from_pretrained(QWEN_LOCAL_PATH)
        loading_states[state_key] = "loading_model"
        
        model_kwargs = {
            "device_map": "auto",
            # Dropped attn_implementation="sdpa" because it causes NaN logits in float16
        }
        if quant_cfg:
            model_kwargs["quantization_config"] = quant_cfg
        else:
            model_kwargs["torch_dtype"] = compute_dtype
            
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_LOCAL_PATH,
            **model_kwargs
        )
        model.eval()
        models[key] = (processor, model)
        loading_states[state_key] = "done"
        
        # Free any intermediate weights/tensors created during loading
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return models[key]

def load_caption_model(model_name: str, device: str):
    key = f"caption_{model_name}_{device}"
    state_key = f"{model_name}::{device}"
    
    # Always clear other models to ensure only the active one is in VRAM
    clear_other_models(key)

    if key not in models:
        print(f"Loading {model_name} into VRAM...")
        dev = get_device(device)
        loading_states[state_key] = "loading_processor"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if _is_blip2(model_name):
                processor = Blip2Processor.from_pretrained(model_name)
                loading_states[state_key] = "loading_model"
                model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch.float32
                )
            else:
                processor = BlipProcessor.from_pretrained(model_name)
                loading_states[state_key] = "loading_model"
                model = BlipForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch.float16
                )
            loading_states[state_key] = "moving_to_device"
            model = model.to(dev)
            models[key] = (processor, model)
        loading_states[state_key] = "done"
    return models[key]

def load_vqa_model(model_name: str, device: str):
    key = f"vqa_{model_name}_{device}"
    state_key = f"{model_name}::{device}"
    
    # Always clear other models to ensure only the active one is in VRAM
    clear_other_models(key)

    if key not in models:
        print(f"Loading VQA model {model_name} into VRAM...")
        dev = get_device(device)
        loading_states[state_key] = "loading_processor"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            processor = BlipProcessor.from_pretrained(model_name)
            loading_states[state_key] = "loading_model"
            model = BlipForQuestionAnswering.from_pretrained(model_name, torch_dtype=torch.float16)
            loading_states[state_key] = "moving_to_device"
            model = model.to(dev)
            models[key] = (processor, model)
        loading_states[state_key] = "done"
    return models[key]

def load_clip_model(device: str):
    key = f"clip_patch32_{device}"
    if key not in models:
        print("Loading CLIP model into VRAM...")
        dev = get_device(device)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16).to(dev)
            models[key] = (processor, model)
    return models[key]



def load_yolo_model(model_name: str, device: str):
    key = f"yolo_{model_name}_{device}"
    state_key = f"{model_name}::{device}"
    
    # Always clear other models to ensure only the active one is in VRAM
    clear_other_models(key)
    
    if key not in models:
        print(f"Loading YOLO model {model_name} into VRAM...")
        dev = get_device(device)
        loading_states[state_key] = "loading_model"
        
        # Load Ultralytics YOLO model
        from ultralytics import YOLO
        import io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = YOLO(model_name)
            model.to(dev)
            models[key] = (None, model)
        loading_states[state_key] = "done"
    return models[key]

def load_grounding_dino_model(model_name: str, device: str):
    key = f"gdino_{model_name}_{device}"
    state_key = f"{model_name}::{device}"
    
    # Always clear other models to ensure only the active one is in VRAM
    clear_other_models(key)

    if key not in models:
        print(f"Loading Grounding DINO {model_name} into VRAM...")
        dev = get_device(device)
        loading_states[state_key] = "loading_processor"
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            processor = AutoProcessor.from_pretrained(model_name)
            loading_states[state_key] = "loading_model"
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(dev)
            models[key] = (processor, model)
        loading_states[state_key] = "done"
    return models[key]

def load_sam_model(model_name: str, device: str):
    key = f"sam_{model_name}_{device}"
    state_key = f"{model_name}::{device}"
    
    # Always clear other models to ensure only the active one is in VRAM
    clear_other_models(key)

    if key not in models:
        print(f"Loading SAM {model_name} into VRAM...")
        dev = get_device(device)
        loading_states[state_key] = "loading_processor"
        from transformers import AutoProcessor, AutoModelForMaskGeneration
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            processor = AutoProcessor.from_pretrained(model_name)
            loading_states[state_key] = "loading_model"
            model = AutoModelForMaskGeneration.from_pretrained(model_name).to(dev)
            models[key] = (processor, model)
        loading_states[state_key] = "done"
    return models[key]

def load_vjepa_model(model_name: str, device: str):
    """Thin wrapper — delegates to visionbox.video_analyzer.model."""
    from visionbox.video_analyzer.model import load_vjepa_model as _load
    return _load(
        model_name=model_name,
        device=device,
        models=models,
        loading_states=loading_states,
        get_device=get_device,
        clear_other_models=clear_other_models,
    )

from pydantic import BaseModel

class CaptionRequest(BaseModel):
    image_base64: str
    condition: Optional[str] = None
    model_name: str = "Salesforce/blip-image-captioning-large"
    device: str = "cuda"
    max_pixels: Optional[int] = None
    precision: str = "4"

class VQARequest(BaseModel):
    image_base64: str
    question: str
    model_name: str = "Salesforce/blip-vqa-base"
    device: str = "cuda"

class VideoClassifyRequest(BaseModel):
    video_base64: str
    model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"
    device: str = "cuda"
    clip_len: int = 64
    use_adaptive_step: bool = True
    use_overlap: bool = True
    aggregate_clips: bool = True

class VideoSummarizeRequest(BaseModel):
    video_base64: str
    # V-JEPA settings
    vjepa_model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"
    vjepa_device: str = "cuda"
    clip_len: int = 64
    k_clips: int = 3
    # Qwen settings
    qwen_device: str = "cuda"

class VideoNarrateRequest(BaseModel):
    video_base64: str
    vjepa_model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"
    vjepa_device: str = "cuda"
    qwen_device: str = "cuda"
    clip_len: int = 64
    sensitivity: float = 0.8
    cooldown: int = 1
    merge_gap: int = 3

# State for Smart Detection progress
smart_detect_state = {}

# State for Video Summarization progress
summarize_state: dict = {}

# State for Event Narration progress
narrate_state: dict = {}

class PredictRequest(BaseModel):
    image_base64: str
    candidate_classes: Optional[str] = None
    topk: int = 5
    device: str = "cuda"

class DetectRequest(BaseModel):
    image_base64: str
    model_name: str = "yolo11n.pt"
    threshold: float = 0.5
    device: str = "cuda"

class SmartDetectRequest(BaseModel):
    image_base64: str
    user_query: str
    qwen_model_name: str = QWEN_LOCAL_ID
    gdino_model_name: str = "IDEA-Research/grounding-dino-tiny"
    sam_model_name: str = "facebook/sam2-hiera-tiny"
    device: str = "cuda"
    threshold: float = 0.3
    precision: str = "4"

class PreloadRequest(BaseModel):
    model_name: str
    task: str = "caption"
    device: str = "cuda"
    precision: str = "4"

@app.post("/api/preload")
async def preload_model(req: PreloadRequest):
    state_key = f"{req.model_name}::{req.device}"
    loading_states[state_key] = "queued"
    try:
        if req.task == "vqa":
            await asyncio.to_thread(load_vqa_model, req.model_name, req.device)
        elif req.task == "video":
            await asyncio.to_thread(load_vjepa_model, req.model_name, req.device)
        elif _is_qwen(req.model_name):
            await asyncio.to_thread(load_qwen_caption_model, req.device, req.precision)
        else:
            await asyncio.to_thread(load_caption_model, req.model_name, req.device)
        return {"status": "ok", "message": f"Successfully loaded {req.model_name} to {req.device}"}
    except Exception as e:
        loading_states[state_key] = "error"
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/free-memory")
async def free_memory():
    # Explicitly delete all references to models
    for k in list(models.keys()):
        val = models.pop(k)
        del val
        
    models.clear()
    loading_states.clear()
    
    # Force garbage collection multiple times if needed
    import gc
    gc.collect()
    gc.collect()
    
    import torch
    if torch.cuda.is_available():
        # Flush any pending assertion errors so they don't blow up empty_cache
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
            
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(f"Warning: Failed to empty CUDA cache (likely stale context): {e}")

    return {"status": "ok", "message": "GPU memory freed successfully."}

@app.get("/api/preload-status")
async def preload_status(model_name: str, device: str, precision: str = "4"):
    state_key = f"{model_name}::{device}"

    # Check if model is already cached under any key pattern
    if _is_qwen(model_name):
        already_cached = f"caption_{model_name}_{device}_{precision}" in models
    else:
        already_cached = (
            f"caption_{model_name}_{device}" in models
            or f"vqa_{model_name}_{device}" in models
            or f"vjepa_{model_name}_{device}" in models
        )

    if already_cached:
        return {"stage": "done", "percent": 100, "label": "Ready!"}

    stage = loading_states.get(state_key, "idle")
    percent = STAGE_PERCENTAGES.get(stage, 0)
    labels = {
        "idle":              "Waiting...",
        "queued":            "Queued...",
        "loading_processor": "Loading processor...",
        "loading_model":     "Loading model weights...",
        "moving_to_device":  "Transferring to GPU...",
        "done":              "Ready!",
        "error":             "Error!",
    }
    return {"stage": stage, "percent": percent, "label": labels.get(stage, stage)}

@app.get("/api/gpu-stats")
async def get_gpu_stats():
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / 1024**2
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    # Ensure memory is somewhat accurate to NVIDIA-SMI by returning reserved memory,
    # as PyTorch caches memory that contributes to the nvidia-smi total.
    return {
        "total": round(total, 2),
        "used": round(reserved, 2),
        "free": round(total - reserved, 2),
        "device_name": props.name
    }

@app.post("/api/caption")
async def generate_caption(req: CaptionRequest):
    try:
        contents = base64.b64decode(req.image_base64)
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        dev = get_device(req.device)

        if _is_qwen(req.model_name):
            # ── Qwen2.5-VL path ──
            processor, model = await asyncio.to_thread(load_qwen_caption_model, req.device, req.precision)
            prompt_text = req.condition if req.condition else "Describe this image in detail."
            image_content = {"type": "image", "image": img}
            if req.max_pixels is not None:
                image_content["max_pixels"] = req.max_pixels
                
            messages = [{
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text",  "text": prompt_text},
                ],
            }]
            from qwen_vl_utils import process_vision_info
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / 1024**2
                print(f"[Qwen Memory] Before generation: {mem_before:.2f} MB used")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256)
                
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / 1024**2
                print(f"[Qwen Memory] After generation: {mem_after:.2f} MB used (diff: {mem_after-mem_before:+.2f} MB)")

            # Trim the input tokens from the output
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
            caption = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            
            # Explicit cleanup after generation
            del inputs, out, trimmed, image_inputs, video_inputs, messages
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[Qwen Memory] After GC/empty_cache: {torch.cuda.memory_allocated()/1024**2:.2f} MB used")
            return {"caption": caption}

        processor, model = await asyncio.to_thread(load_caption_model, req.model_name, req.device)

        if _is_blip2(req.model_name):
            # BLIP-2 inference: pass optional text prompt, decode dropping the prompt prefix
            if req.condition:
                inputs = processor(images=img, text=req.condition, return_tensors="pt").to(dev)
            else:
                inputs = processor(images=img, return_tensors="pt").to(dev)
            out = model.generate(**inputs, max_new_tokens=100)
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
            if req.condition and caption.lower().startswith(req.condition.lower()):
                caption = caption[len(req.condition):].strip()
        else:
            # BLIP-1 inference — use beam search to avoid greedy-decoding artifact
            if req.condition:
                inputs = processor(img, req.condition, return_tensors="pt").to(dev, torch.float16)
                out = model.generate(**inputs, max_new_tokens=50, num_beams=5, early_stopping=True)
            else:
                inputs = processor(img, return_tensors="pt").to(dev, torch.float16)
                out = model.generate(**inputs, max_new_tokens=50, num_beams=5, early_stopping=True)
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
            caption = caption[:1].upper() + caption[1:] if caption else caption
            if req.condition and caption.lower().startswith(req.condition.lower()):
                caption = caption[len(req.condition):].strip()
            if not caption and req.condition:
                caption = req.condition

        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vqa")
async def answer_question(req: VQARequest):
    try:
        contents = base64.b64decode(req.image_base64)
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        dev = get_device(req.device)
        
        processor, model = load_vqa_model(req.model_name, req.device)
        
        inputs = processor(img, req.question, return_tensors="pt").to(dev, torch.float16)
        out = model.generate(**inputs, max_new_tokens=50)
        answer = processor.decode(out[0], skip_special_tokens=True)
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict_image(req: PredictRequest):
    try:
        contents = base64.b64decode(req.image_base64)
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        dev = get_device(req.device)
        
        processor, model = load_clip_model(req.device)
        
        if req.candidate_classes:
            classes = [c.strip() for c in req.candidate_classes.split(",") if c.strip()]
        else:
            classes = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
            
        text_inputs = [f"a photo of a {c}" if not c.startswith("a photo") else c for c in classes]
        
        inputs = processor(text=text_inputs, images=img, return_tensors="pt", padding=True)
        inputs["pixel_values"] = inputs["pixel_values"].to(dev, torch.float16)
        inputs["input_ids"] = inputs["input_ids"].to(dev)
        inputs["attention_mask"] = inputs["attention_mask"].to(dev)
        
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze(0)
        
        top_probs, top_idx = probs.topk(min(req.topk, probs.numel()))
        
        out = [{"class": classes[int(i)], "probability": float(p)} for p, i in zip(top_probs.cpu(), top_idx.cpu())]
        
        return {"predictions": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/video/classify")
async def classify_video_endpoint(req: VideoClassifyRequest):
    """
    Decode the incoming base64 video, write it to a temp file,
    then delegate to visionbox.video_analyzer.predict.classify_video
    which uses the smart sampler and aggregated V-JEPA 2 inference.
    """
    import tempfile
    from visionbox.video_analyzer.predict import classify_video as _classify

    try:
        contents = base64.b64decode(req.video_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            predictions = await asyncio.to_thread(
                _classify,
                tmp_path,
                req.model_name,
                req.device,
                load_vjepa_model,    # inject shared model cache
                get_device,          # inject device helper
                clip_len=req.clip_len,
                use_adaptive_step=req.use_adaptive_step,
                use_overlap=req.use_overlap,
                aggregate_clips=req.aggregate_clips
            )
        finally:
            os.remove(tmp_path)

        return {"predictions": predictions}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/video/summarize")
async def summarize_video_endpoint(req: VideoSummarizeRequest):
    """
    Decode the incoming base64 video, pick key clips using V-JEPA embeddings,
    and then use Qwen2.5-VL to summarize those frames.
    Reports progress live via the summarize_state dict.
    """
    import tempfile
    from visionbox.video_analyzer.caption_pipeline import generate_video_summary

    # Reset progress state
    summarize_state.clear()
    summarize_state["stage"] = "starting"
    summarize_state["label"] = "Starting summarization pipeline..."
    summarize_state["percent"] = 0

    def _progress(label: str, pct: int):
        summarize_state["label"] = label
        summarize_state["percent"] = pct

    try:
        contents = base64.b64decode(req.video_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            result = await asyncio.to_thread(
                generate_video_summary,
                video_path=tmp_path,
                vjepa_model_name=req.vjepa_model_name,
                qwen_device=req.qwen_device,
                vjepa_device=req.vjepa_device,
                load_vjepa_fn=load_vjepa_model,
                load_qwen_fn=load_qwen_caption_model,
                get_device_fn=get_device,
                clip_len=req.clip_len,
                k_clips=req.k_clips,
                progress_fn=_progress,
            )
        finally:
            os.remove(tmp_path)

        summarize_state["stage"] = "done"
        summarize_state["label"] = "Done!"
        summarize_state["percent"] = 100
        return result
    except Exception as e:
        summarize_state["stage"] = "error"
        summarize_state["label"] = str(e)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video/summarize-status")
async def summarize_video_status():
    """Returns the current progress status of the active summarization."""
    if not summarize_state:
        return {"stage": "idle", "percent": 0, "label": "Waiting..."}
    return summarize_state

@app.post("/api/video/narrate")
async def narrate_video_endpoint(req: VideoNarrateRequest):
    """
    Event-driven video narration:
    V-JEPA 2 detects WHEN → Qwen2.5-VL describes WHAT.
    """
    import tempfile
    from visionbox.video_analyzer.caption_pipeline import narrate_video

    narrate_state.clear()
    narrate_state["stage"] = "starting"
    narrate_state["label"] = "Starting event narration pipeline..."
    narrate_state["percent"] = 0

    def _progress(label: str, pct: int):
        narrate_state["label"] = label
        narrate_state["percent"] = pct

    try:
        contents = base64.b64decode(req.video_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            result = await asyncio.to_thread(
                narrate_video,
                video_path=tmp_path,
                vjepa_model_name=req.vjepa_model_name,
                vjepa_device=req.vjepa_device,
                load_vjepa_fn=load_vjepa_model,
                load_qwen_fn=load_qwen_caption_model,
                get_device_fn=get_device,
                qwen_device=req.qwen_device,
                clip_len=req.clip_len,
                sensitivity=req.sensitivity,
                cooldown=req.cooldown,
                merge_gap=req.merge_gap,
                progress_fn=_progress,
            )
        finally:
            os.remove(tmp_path)

        narrate_state["stage"] = "done"
        narrate_state["label"] = "Done!"
        narrate_state["percent"] = 100
        return {"result": result}
    except Exception as e:
        narrate_state["stage"] = "error"
        narrate_state["label"] = str(e)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video/narrate-status")
async def narrate_video_status():
    """Returns the current progress status of the active narration."""
    if not narrate_state:
        return {"stage": "idle", "percent": 0, "label": "Waiting..."}
    return narrate_state

@app.post("/api/detect")
async def detect_objects(req: DetectRequest):
    try:
        contents = base64.b64decode(req.image_base64)
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        dev = get_device(req.device)
        
        _, model = load_yolo_model(req.model_name, req.device)
        from visionbox.yolo.predict import get_prediction
        result = get_prediction(model, img, req.threshold, dev)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/smart-detect")
async def smart_detect_objects(req: SmartDetectRequest):
    try:
        dev = get_device(req.device)
        
        req_params = {
            "device": req.device,
            "precision": req.precision,
            "gdino_name": req.gdino_model_name,
            "sam_name": req.sam_model_name
        }
        load_fns = (load_qwen_caption_model, load_grounding_dino_model, load_sam_model)
        
        from visionbox.smart_detect.pipeline import run_smart_detect
        contents = base64.b64decode(req.image_base64)
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        def update_status(stage, percent, label):
            smart_detect_state["stage"] = stage
            smart_detect_state["percent"] = percent
            smart_detect_state["label"] = label
            
        smart_detect_state.clear()
        update_status("queued", 5, "Queued...")
        
        result = await asyncio.to_thread(
            run_smart_detect,
            img, req.user_query, req.threshold,
            req_params, load_fns, update_status
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        smart_detect_state["stage"] = "error"
        smart_detect_state["label"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/smart-detect-status")
async def get_smart_detect_status():
    if not smart_detect_state:
        return {"stage": "idle", "percent": 0, "label": "Waiting..."}
    return smart_detect_state

@app.get("/api/video/available-models")
async def list_available_video_models():
    """
    Scans the V_JEPA2/ directory for subdirectories containing config.json.
    Returns a list of model objects for the frontend.
    The 'id' field is exactly what will be passed to load_vjepa_model() as model_name.
    """
    from visionbox.video_analyzer.model import _LOCAL_MODEL_DIR

    results = []

    if os.path.isdir(_LOCAL_MODEL_DIR):
        for item in sorted(os.listdir(_LOCAL_MODEL_DIR)):
            item_path = os.path.join(_LOCAL_MODEL_DIR, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                # id = the folder name, which is what model.py uses to find it in V_JEPA2/
                results.append({
                    "id": item,
                    "name": f"Local: {item}",
                    "is_local": True
                })

    # If no local models found, offer the default HF model as a fallback
    if not results:
        results.append({
            "id": "facebook/vjepa2-vitl-fpc16-256-ssv2",
            "name": "V-JEPA 2 ViT-L (Hugging Face)",
            "is_local": False
        })

    return {"models": results}
