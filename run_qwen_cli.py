import os
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import argparse

# Suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Path to local model based on your project structure
QWEN_LOCAL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

def main(image_path, prompt, max_tokens):
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(QWEN_LOCAL_PATH)

    print("Loading 4-bit quantized model (this may take a minute)...")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_LOCAL_PATH,
        quantization_config=quant_cfg,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    print(f"Loading image from {image_path}...")
    img = Image.open(image_path).convert("RGB")
    
    # Resize image to prevent massive patch lists and float16 NaNs
    max_size = 1024
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size))
        print(f"Resized image to {img.width}x{img.height}")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": prompt},
        ],
    }]

    print("Processing inputs...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    print("Generating caption...")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens)

    # Trim input tokens
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    caption = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    print("\n--- Result ---")
    print(caption)
    print("--------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL model from the terminal")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="Text prompt for the image")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    
    args = parser.parse_args()
    main(args.image_path, args.prompt, args.max_tokens)
