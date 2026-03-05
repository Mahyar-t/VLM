# NiluLab

The **NiluLab** repository contains a Python API and a beautiful web application for interacting with Computer Vision models, including **Image Captioning (with Multimodal LLMs)**, **Visual Question Answering (VQA)**, and **Image Classification**. It also provides a CLI package for fine-tuning pretrained torchvision models.

## Install (editable)

```bash
pip install -e .
```

Or build a wheel:

```bash
pip wheel . -w dist
pip install dist/*.whl
```

## Image Classification Fine-Tuning

NiluLab provides built-in Command Line Interfaces (CLIs) to easily fine-tune PyTorch image classification models (like MobileNet, ResNet, etc.) on your own custom datasets. This is useful when you want to train a lightweight model to recognize specific categories (e.g., classifying types of plants, separating good vs. defective manufacturing parts, etc.) rather than relying on a heavy General Purpose Vision Model.

### Dataset format (ImageFolder)

To train a model, your image dataset must be organized in the standard PyTorch `ImageFolder` structure.
The directory name forms the class label that the model will learn to predict.

Expected directory structure:

```text
data_dir/
  train/
    class_a/xxx.jpg     # All training images for class_a go here
    class_b/yyy.jpg     # All training images for class_b go here
  val/                  # Validation set to measure accuracy (or "valid/")
    class_a/zzz.jpg
    class_b/www.jpg
```

### Train (CLI)

The `imgcls-train` command initiates the fine-tuning process. It reads your dataset, initializes the specified base model with pretrained weights, and trains it to recognize your custom classes.

```bash
imgcls-train --data-dir /path/to/data_dir \
  --model mobilenet_v3_small \
  --epochs 20 --batch-size 32 --lr 1e-4 \
  --checkpoint best.pt --save-class-map class_to_idx.json
```

- `--model`: The torchvision architecture to use (e.g., `mobilenet_v3_small`, `resnet18`).
- `--checkpoint`: Where the trained model weights will be saved.
- `--save-class-map`: Generates a JSON file mapping folder names (e.g., `class_a`) to numeric IDs, which is required for making predictions later.

**Monitoring with TensorBoard:**
You can track training accuracy and loss in real-time by directing the logs to a folder:

```bash
imgcls-train --data-dir /path/to/data_dir --log-dir runs/exp1
tensorboard --logdir runs
```

### Predict (CLI)

Once training is complete, use `imgcls-predict` to test your trained model on a new, unseen image.

```bash
imgcls-predict --image /path/to/image.jpg \
  --weights best.pt \
  --class-map class_to_idx.json
```

**Output:** The command will print a ranked list of the predicted class names along with their confidence probabilities.

## Python usage

```python
from imgcls_ft.data import build_dataloaders
from imgcls_ft.model import create_model
from imgcls_ft.engine import fit
from imgcls_ft.utils import get_device, set_seed

set_seed(42)
train_loader, val_loader, class_to_idx = build_dataloaders("/path/to/data_dir")

model = create_model("mobilenet_v3_small", num_classes=len(class_to_idx), pretrained=True)
optimizer = __import__("torch").optim.AdamW(model.parameters(), lr=1e-4)

hist = fit(model, train_loader, val_loader, optimizer, device=get_device("cuda"), num_epochs=10)
```

## Notes

- Default normalization mean/std match the notebook's Monkey Species dataset. If your dataset differs, pass `--mean` and `--std`, or change `DataConfig`.
- The package supports a few common torchvision backbones; extend `imgcls_ft/model.py` if needed.

## Run Qwen2.5-VL-3B via Terminal

You can run the local Qwen2.5-VL-3B model directly from the command line without starting the web application.

```bash
python run_qwen_cli.py /path/to/image.jpg
```

**Custom Prompt:**

```bash
python run_qwen_cli.py /path/to/image.jpg --prompt "Extract all the text present in this image."
```

**Custom Output Length:**

```bash
python run_qwen_cli.py /path/to/image.jpg --max_tokens 128
```

## Reset the Cached Models (Web UI)

The **"Reset the cached models"** button on the Image Captioning page is used to free up GPU Memory (VRAM).

NiluLab supports large multimodal models (such as Qwen2.5-VL and BLIP) which are kept cached in VRAM after being loaded to ensure fast subsequent generations. However, loading multiple large models can lead to Out of Memory (OOM) errors on systems with limited GPU memory.

Clicking this button sends a request from the web interface to the backend server to entirely unload all cached models, perform garbage collection, and explicitly clear the CUDA cache (`torch.cuda.empty_cache()`). This fully releases the VRAM back to the system, allowing you to load different models from scratch without memory crashes.

**How VRAM Monitoring Works:**
The web interface features a real-time VRAM usage indicator. This works via a polling mechanism:

1. Every 5 seconds, the frontend (`caption.html`) hits the `/api/gpu-stats` endpoint on the Java Spring Boot backend.
2. The Java backend relays this fetch to the Python FastAPI backend, which relies on `torch.cuda.memory_allocated()` to report exactly how much VRAM is currently held by loaded model weights and PyTorch tensors.
3. The UI updates the usage pill based on this data (showing the allocated VRAM vs the GPU's total VRAM).
4. When you click **"Reset the cached models"**, the VRAM is cleared as explained above. On the very next 5-second polling tick, the `torch.cuda.memory_allocated()` call accurately reads near 0 MB (or significantly less) since memory was released, turning the UI indicator green to confirm to the user that the GPU is ready for the next model.
