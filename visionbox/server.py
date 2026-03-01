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
QWEN_LOCAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models", "Qwen2_5_3B"
)

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
    This prevents OOM errors when switching between large models on 8GB GPUs.
    """
    keys_to_delete = [k for k in models.keys() if k != key_to_keep and not k.startswith("clip_")]
    if keys_to_delete:
        for k in keys_to_delete:
            print(f"Clearing {k} from VRAM to free space...")
            del models[k]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def load_qwen_caption_model(device: str):
    key = f"caption_{QWEN_LOCAL_ID}_{device}"
    state_key = f"{QWEN_LOCAL_ID}::{device}"
    if key not in models:
        clear_other_models(key)
        print(f"Loading Qwen2.5-VL-3B (4-bit) from {QWEN_LOCAL_PATH} ...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        loading_states[state_key] = "loading_processor"
        processor = AutoProcessor.from_pretrained(QWEN_LOCAL_PATH, local_files_only=True)
        loading_states[state_key] = "loading_model"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_LOCAL_PATH,
            quantization_config=quant_cfg,
            device_map="auto",
            attn_implementation="sdpa",
            local_files_only=True,
        )
        model.eval()
        models[key] = (processor, model)
        loading_states[state_key] = "done"
    return models[key]

def load_caption_model(model_name: str, device: str):
    key = f"caption_{model_name}_{device}"
    state_key = f"{model_name}::{device}"
    if key not in models:
        clear_other_models(key)
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
    if key not in models:
        clear_other_models(key)
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

from pydantic import BaseModel

class CaptionRequest(BaseModel):
    image_base64: str
    condition: Optional[str] = None
    model_name: str = "Salesforce/blip-image-captioning-large"
    device: str = "cuda"

class VQARequest(BaseModel):
    image_base64: str
    question: str
    model_name: str = "Salesforce/blip-vqa-base"
    device: str = "cuda"

class PredictRequest(BaseModel):
    image_base64: str
    candidate_classes: Optional[str] = None
    topk: int = 5
    device: str = "cuda"

class PreloadRequest(BaseModel):
    model_name: str
    task: str = "caption"
    device: str = "cuda"

@app.post("/api/preload")
async def preload_model(req: PreloadRequest):
    state_key = f"{req.model_name}::{req.device}"
    loading_states[state_key] = "queued"
    try:
        if req.task == "vqa":
            await asyncio.to_thread(load_vqa_model, req.model_name, req.device)
        elif _is_qwen(req.model_name):
            await asyncio.to_thread(load_qwen_caption_model, req.device)
        else:
            await asyncio.to_thread(load_caption_model, req.model_name, req.device)
        return {"status": "ok", "message": f"Successfully loaded {req.model_name} to {req.device}"}
    except Exception as e:
        loading_states[state_key] = "error"
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/free-memory")
async def free_memory():
    try:
        models.clear() # Clear everything unconditionally
        loading_states.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "ok", "message": "GPU memory freed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/preload-status")
async def preload_status(model_name: str, device: str):
    state_key = f"{model_name}::{device}"
    # If already in models cache, it's done regardless of state
    caption_cached = f"caption_{model_name}_{device}" in models
    vqa_cached = f"vqa_{model_name}_{device}" in models
    if caption_cached or vqa_cached:
        return {"stage": "done", "percent": 100, "label": "Ready!"}
    stage = loading_states.get(state_key, "idle")
    percent = STAGE_PERCENTAGES.get(stage, 0)
    labels = {
        "idle":             "Waiting...",
        "queued":           "Queued...",
        "loading_processor":"Loading processor...",
        "loading_model":    "Loading model weights...",
        "moving_to_device": "Transferring to GPU...",
        "done":             "Ready!",
        "error":            "Error!",
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
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    # 'Used' in nvidia-smi sense is more like 'reserved' or a mix, 
    # but for LLMs, 'allocated' is what the model actually holds.
    # We'll show total and allocated.
    return {
        "total": round(total, 2),
        "used": round(allocated, 2),
        "free": round(total - allocated, 2),
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
            processor, model = await asyncio.to_thread(load_qwen_caption_model, req.device)
            prompt_text = req.condition if req.condition else "Describe this image in detail."
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
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
