import argparse
import sys
from visionbox.vqa import answer_question

def parse_args():
    p = argparse.ArgumentParser(description="Run Visual Question Answering on an image.")
    p.add_argument("--image", required=True, help="Path to an image file.")
    p.add_argument("--question", required=True, help="Question to ask about the image.")
    p.add_argument("--device", default="cuda", help="Device to run on (cpu or cuda).")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        answer = answer_question(args.image, args.question, args.device)
        # Simply print the answer so it can be easily parsed by the Java bridge
        print(answer)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
