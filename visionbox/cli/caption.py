import argparse
import sys
from visionbox.caption import generate_caption

def parse_args():
    p = argparse.ArgumentParser(description="Run Image Captioning on an image.")
    p.add_argument("--image", required=True, help="Path to an image file.")
    p.add_argument("--condition", default=None, help="Text condition to start the caption with (optional).")
    p.add_argument("--model", default="Salesforce/blip-image-captioning-large", help="The BLIP model variant to use.")
    p.add_argument("--device", default="cuda", help="Device to run on (cpu or cuda).")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        caption = generate_caption(args.image, args.condition, args.device, args.model)
        # Simply print the caption so it can be easily parsed by the Java bridge
        print(caption)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
