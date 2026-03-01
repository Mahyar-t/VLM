# VisionBox API

A small Python package for **image classification** by **fine-tuning pretrained torchvision models**, based on the notebook you provided.

## Install (editable)

```bash
pip install -e .
```

Or build a wheel:

```bash
pip wheel . -w dist
pip install dist/imgcls_ft-0.1.0-py3-none-any.whl
```

## Dataset format (ImageFolder)

Expected directory structure:

```
data_dir/
  train/
    class_a/xxx.jpg
    class_b/yyy.jpg
  val/            # or "valid/"
    class_a/zzz.jpg
    class_b/www.jpg
```

## Train (CLI)

```bash
imgcls-train --data-dir /path/to/data_dir \
  --model mobilenet_v3_small \
  --epochs 20 --batch-size 32 --lr 1e-4 \
  --checkpoint best.pt --save-class-map class_to_idx.json
```

TensorBoard:

```bash
imgcls-train --data-dir /path/to/data_dir --log-dir runs/exp1
tensorboard --logdir runs
```

## Predict (CLI)

```bash
imgcls-predict --image /path/to/image.jpg \
  --weights best.pt \
  --class-map class_to_idx.json
```

Output is a ranked list of class names + probabilities.

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
