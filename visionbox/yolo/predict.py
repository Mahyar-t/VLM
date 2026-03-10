import torch
from PIL import Image

def get_prediction(model, image: Image.Image, threshold: float, device: torch.device):
    """
    Get bounding box predictions for an image using Ultralytics YOLO.
    Returns:
        dict containing 'boxes' (list of [x1, y1, x2, y2]),
        'classes' (list of strings), and 'scores' (list of floats).
    """

    # Run inference on the provided image
    results = model(image, verbose=False)

    pred_boxes = []
    pred_class = []
    pred_scores = []

    # Process results
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        if boxes is None:
            continue
            
        # Get properties as cpu numpy arrays
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            score = float(conf[i])
            # Filter by confidence threshold
            if score >= threshold:
                pred_boxes.append([float(x) for x in xyxy[i]])
                pred_class.append(result.names[int(cls[i])])
                pred_scores.append(score)

    return {
        "boxes": pred_boxes,
        "classes": pred_class,
        "scores": pred_scores
    }
