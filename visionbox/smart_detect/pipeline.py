import base64
import io
import torch
import numpy as np
from PIL import Image

def generate_search_queries(img: Image.Image, user_query: str, qwen_processor, qwen_model):
    """
    Given a user query, use Qwen2.5-VL to extract exact object names to detect.
    """
    prompt = (
        f"You are a computer vision assistant. The user wants to detect object(s): '{user_query}'. "
        "Extract a comma-separated list of precise object names or phrases to search for in this image. "
        "Only output the comma-separated phrases, nothing else. "
        "For example, if the query is 'find the red car and the biggest tree', output 'red car, biggest tree'."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img, "max_pixels": 1280 * 720},
        {"type": "text", "text": prompt}
    ]}]
    from qwen_vl_utils import process_vision_info
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    device = next(qwen_model.parameters()).device
    inputs = qwen_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = qwen_model.generate(**inputs, max_new_tokens=50)
    
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    result_text = qwen_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    
    # Process text into a list of queries
    import re
    result_text = re.sub(r'[\'\"\[\]]', '', result_text)
    queries = [q.strip() for q in result_text.split(',') if q.strip()]
    if not queries:
        queries = [user_query]
    
    # Grounding DINO format: "red car . biggest tree ."
    text_prompt = " . ".join(queries) + " ."
    return text_prompt, queries

def detect_with_grounding_dino(img: Image.Image, text_prompt: str, gdino_processor, gdino_model, threshold: float = 0.3):
    device = next(gdino_model.parameters()).device
    inputs = gdino_processor(images=img, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gdino_model(**inputs)
    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[img.size[::-1]] # (height, width)
    )[0]
    return results # dict with 'boxes', 'scores', 'labels'

def segment_with_sam(img: Image.Image, boxes: torch.Tensor, sam_processor, sam_model):
    if len(boxes) == 0:
        return []
        
    device = next(sam_model.parameters()).device
    
    # Prepare inputs for SAM/SAM2
    # box needs to be [[[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]]
    boxes_list = [boxes.cpu().tolist()] 
    inputs = sam_processor(images=img, input_boxes=boxes_list, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = sam_model(**inputs)
    
    # post_process_masks returns a list of masks per image. 
    # For one image, it's a tensor of shape [num_boxes, 3 (masks per box), H, W]
    from transformers import __version__
    
    masks_list = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu()
    )
    # the first image masks: shape [num_boxes, 3, H, W]
    masks = masks_list[0] 
    
    scores = outputs.iou_scores[0].cpu() # Shape [num_boxes, 3]
    
    batched_masks = []
    # get the best mask for each box
    for i in range(len(boxes)):
        best_mask_idx = torch.argmax(scores[i])
        best_mask = masks[i, best_mask_idx]
        batched_masks.append(best_mask.numpy())
        
    return batched_masks

def run_smart_detect(img: Image.Image, user_query: str, threshold: float, req_params: dict, load_fns: tuple, status_callback=None):
    load_qwen, load_gdino, load_sam = load_fns
    import gc
    
    # 1. Qwen language supervision
    if status_callback: status_callback("loading_qwen", 10, "Loading Qwen2.5-VL...")
    qwen_processor, qwen_model = load_qwen(req_params["device"], req_params["precision"])
    
    if status_callback: status_callback("analyzing_query", 25, "Analyzing query with Qwen2.5-VL...")
    text_prompt, queries = generate_search_queries(img, user_query, qwen_processor, qwen_model)
    del qwen_processor, qwen_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. Grounding DINO detection
    if status_callback: status_callback("loading_gdino", 40, "Loading Grounding DINO...")
    gdino_processor, gdino_model = load_gdino(req_params["gdino_name"], req_params["device"])
    
    if status_callback: status_callback("detecting_objects", 55, "Detecting objects with Grounding DINO...")
    detections = detect_with_grounding_dino(img, text_prompt, gdino_processor, gdino_model, threshold)
    boxes = detections["boxes"] # [num_boxes, 4]
    del gdino_processor, gdino_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 3. SAM 2 segmentation
    if status_callback: status_callback("loading_sam", 70, "Loading SAM 2...")
    sam_processor, sam_model = load_sam(req_params["sam_name"], req_params["device"])
    
    if status_callback: status_callback("segmenting_objects", 85, "Segmenting objects with SAM 2...")
    masks = segment_with_sam(img, boxes, sam_processor, sam_model)
    del sam_processor, sam_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if status_callback: status_callback("processing_results", 95, "Processing visualization...")
    
    results = []
    import base64, io
    for i in range(len(boxes)):
        box = boxes[i].cpu().tolist()
        label = detections["labels"][i]
        score = detections["scores"][i].item()
        mask = masks[i] # [H, W] bool array
        
        # Convert mask to RGBA PNG: alpha=255 where mask is True, 0 elsewhere.
        # This lets the JS canvas source-in composite colorize ONLY the masked
        # region without bleeding a semi-transparent fill over the whole image.
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = 255  # white foreground (recolored in JS)
        rgba[:, :, 1] = 255
        rgba[:, :, 2] = 255
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)   # alpha = mask
        mask_img = Image.fromarray(rgba, mode="RGBA")
        buffered = io.BytesIO()
        mask_img.save(buffered, format="PNG")
        mask_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        
        results.append({
            "box": box,
            "label": label,
            "score": score,
            "mask_base64": mask_b64
        })
        
    if status_callback: status_callback("done", 100, "Ready!")
    return {"queries_used": queries, "detections": results}
