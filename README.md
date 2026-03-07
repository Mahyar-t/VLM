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

The `visionbox-train` command initiates the fine-tuning process. It reads your dataset, initializes the specified base model with pretrained weights, and trains it to recognize your custom classes.

```bash
visionbox-train --data-dir /path/to/data_dir \
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
visionbox-train --data-dir /path/to/data_dir --log-dir runs/exp1
tensorboard --logdir runs
```

### Predict (CLI)

Once training is complete, use `visionbox-predict` to test your trained model on a new, unseen image.

```bash
visionbox-predict --image /path/to/image.jpg \
  --weights best.pt \
  --class-map class_to_idx.json
```

**Output:** The command will print a ranked list of the predicted class names along with their confidence probabilities.

## Python usage

```python
from visionbox.classification.data import build_dataloaders
from visionbox.classification.model import create_model
from visionbox.classification.engine import fit
from visionbox.utils import get_device, set_seed

set_seed(42)
train_loader, val_loader, class_to_idx = build_dataloaders("/path/to/data_dir")

model = create_model("mobilenet_v3_small", num_classes=len(class_to_idx), pretrained=True)
optimizer = __import__("torch").optim.AdamW(model.parameters(), lr=1e-4)

hist = fit(model, train_loader, val_loader, optimizer, device=get_device("cuda"), num_epochs=10)
```

## Notes

- Default normalization mean/std match the notebook's Monkey Species dataset. If your dataset differs, pass `--mean` and `--std`, or change `DataConfig`.
- The package supports a few common torchvision backbones; extend `visionbox/classification/model.py` if needed.
