
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Optional
from pathlib import Path

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Suppress library warnings that might interfere with stdout parsing
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from .utils import load_checkpoint, get_device
from .model import create_model


def build_inference_transform(
    img_size: Tuple[int, int] = (224, 224),
    mean: Sequence[float] = (0.4368, 0.4336, 0.3294),
    std: Sequence[float] = (0.2457, 0.2413, 0.2447),
):
    return transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@torch.no_grad()
def predict_image(
    image_path: str,
    weights_path: Optional[str] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
    candidate_classes: Optional[str] = None,
    model_name: str = "mobilenet_v3_small",
    img_size: Tuple[int, int] = (224, 224),
    mean: Sequence[float] = (0.4368, 0.4336, 0.3294),
    std: Sequence[float] = (0.2457, 0.2413, 0.2447),
    device: str = "cuda",
    topk: int = 5,
):
    """Predict classes for a single image. Returns list of (class_name, prob)."""
    dev = get_device(device)

    if model_name == "clip-vit-base-patch32":
        from transformers import CLIPProcessor, CLIPModel
        import contextlib
        import io

        # Deep silencing during model load to prevent stdout pollution
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16).to(dev)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if candidate_classes:
            classes = [c.strip() for c in candidate_classes.split(",") if c.strip()]
        else:
            classes = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
            
        text_inputs = [f"a photo of a {c}" if not c.startswith("a photo") else c for c in classes]
        
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True).to(dev)
        
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze(0)
        
        top_probs, top_idx = probs.topk(min(topk, probs.numel()))
        out = [(classes[int(i)], float(p)) for p, i in zip(top_probs.cpu(), top_idx.cpu())]
        return out

    from torchvision.models import get_model_weights
    
    dev = get_device(device)
    
    if weights_path and class_to_idx:
        # Custom fine-tuned weights
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        model = create_model(model_name=model_name, num_classes=num_classes, pretrained=False)
        state = load_checkpoint(weights_path, map_location=dev)
        model.load_state_dict(state)
    else:
        # Pretrained ImageNet weights out-of-the-box
        from .model import _SUPPORTED
        if model_name not in _SUPPORTED:
            raise ValueError(f"Unsupported model_name={model_name}")
            
        import torchvision.models as models
        fn = getattr(models, _SUPPORTED[model_name])
        model = fn(weights="DEFAULT")
        
        # Get ImageNet categories
        w = get_model_weights(fn).DEFAULT
        categories = w.meta["categories"]
        idx_to_class = {i: categories[i] for i in range(len(categories))}

    model.eval().to(dev)
    model.eval().to(dev)

    tf = build_inference_transform(img_size=img_size, mean=mean, std=std)

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(dev)

    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)
    top_probs, top_idx = probs.topk(min(topk, probs.numel()))

    out = [(idx_to_class[int(i)], float(p)) for p, i in zip(top_probs.cpu(), top_idx.cpu())]
    return out
