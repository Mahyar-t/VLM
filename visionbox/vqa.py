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
def answer_question(image_path: str, question: str, device: str = "cuda") -> str:
    dev = get_device(device)
    
    from transformers import BlipProcessor, BlipForQuestionAnswering
    
    # Deep silencing during model load to prevent stdout pollution
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(dev)
    
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(image, question, return_tensors="pt").to(dev)
    
    out = model.generate(**inputs, max_new_tokens=50)
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    return answer
