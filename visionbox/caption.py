import os
import io
import contextlib
import torch
from PIL import Image

from .utils import get_device

# Suppress library warnings that might interfere with stdout parsing
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

@torch.no_grad()
def generate_caption(image_path: str, condition: str = None, device: str = "cuda", model_name: str = "Salesforce/blip-image-captioning-large") -> str:
    dev = get_device(device)
    
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    # Deep silencing during model load to prevent stdout pollution
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(dev)
    
    image = Image.open(image_path).convert("RGB")
    
    if condition:
        # Conditional image captioning
        inputs = processor(image, condition, return_tensors="pt").to(dev)
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # BLIP typically echoes the condition at the start of the output. 
        # Strip it out if it exists so the UI only gets the generated continuation.
        if caption.lower().startswith(condition.lower()):
            caption = caption[len(condition):].strip()
        
        # If the model didn't add anything new, or just generated punctuation
        if not caption or all(c in "?.! " for c in caption):
            caption = condition
    else:
        # Unconditional image captioning
        inputs = processor(image, return_tensors="pt").to(dev)
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption.strip()
