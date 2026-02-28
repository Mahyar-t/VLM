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
    CLIPProcessor, CLIPModel
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

def load_caption_model(model_name: str, device: str):
    key = f"caption_{model_name}_{device}"
    state_key = f"{model_name}::{device}"
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
        else:
            await asyncio.to_thread(load_caption_model, req.model_name, req.device)
        return {"status": "ok", "message": f"Successfully loaded {req.model_name} to {req.device}"}
    except Exception as e:
        loading_states[state_key] = "error"
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

@app.post("/api/caption")
async def generate_caption(req: CaptionRequest):
    try:
        contents = base64.b64decode(req.image_base64)
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        dev = get_device(req.device)

        processor, model = await asyncio.to_thread(load_caption_model, req.model_name, req.device)

        if _is_blip2(req.model_name):
            # BLIP-2 inference: pass optional text prompt, decode dropping the prompt prefix
            if req.condition:
                inputs = processor(images=img, text=req.condition, return_tensors="pt").to(dev)
            else:
                inputs = processor(images=img, return_tensors="pt").to(dev)
            out = model.generate(**inputs, max_new_tokens=100)
            # Decode full output then strip the prompt if it was echoed
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
