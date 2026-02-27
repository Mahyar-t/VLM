
from __future__ import annotations
import argparse
import json

from visionbox.predict import predict_image


def parse_args():
    p = argparse.ArgumentParser(description="Run inference on one image using a trained checkpoint.")
    p.add_argument("--image", required=True, help="Path to an image file.")
    p.add_argument("--weights", default=None, help="Path to model weights (best.pt). Leave blank for ImageNet.")
    p.add_argument("--class-map", default=None, help="Path to class_to_idx.json. Leave blank for ImageNet.")
    p.add_argument("--classes", default=None, help="Comma-separated list of classes for zero-shot (e.g., 'cat,dog').")
    p.add_argument("--model", default="mobilenet_v3_small")
    p.add_argument("--device", default="cuda")
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    class_to_idx = None
    if args.class_map:
        class_to_idx = json.loads(open(args.class_map, "r", encoding="utf-8").read())
    out = predict_image(
        image_path=args.image,
        weights_path=args.weights,
        class_to_idx=class_to_idx,
        candidate_classes=args.classes,
        model_name=args.model,
        device=args.device,
        topk=args.topk,
    )
    for cls, prob in out:
        print(f"{cls}\t{prob:.4f}")


if __name__ == "__main__":
    main()
