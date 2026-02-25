
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from imgcls_ft.config import DataConfig, TrainConfig
from imgcls_ft.data import build_dataloaders
from imgcls_ft.engine import fit
from imgcls_ft.model import create_model, freeze_backbone, num_trainable_params
from imgcls_ft.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a pretrained torchvision model for image classification (ImageFolder).")
    p.add_argument("--data-dir", required=True, help="Dataset root containing train/ and val/ (or valid/).")
    p.add_argument("--model", default="mobilenet_v3_small", help="Model name (torchvision): mobilenet_v3_small, resnet18, ...")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, nargs=2, default=(224, 224), metavar=("H","W"))
    p.add_argument("--mean", type=float, nargs=3, default=(0.4368, 0.4336, 0.3294))
    p.add_argument("--std", type=float, nargs=3, default=(0.2457, 0.2413, 0.2447))
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--train-classifier-only", action="store_true", help="Freeze backbone and train only classification head.")
    p.add_argument("--log-dir", default=None, help="If set, write TensorBoard logs here.")
    p.add_argument("--checkpoint", default="best.pt", help="Path to save best weights.")
    p.add_argument("--save-class-map", default="class_to_idx.json", help="Where to save class_to_idx mapping.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    train_loader, val_loader, class_to_idx = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
        mean=tuple(args.mean),
        std=tuple(args.std),
        augment=not args.no_augment,
    )

    num_classes = len(class_to_idx)
    model = create_model(args.model, num_classes=num_classes, pretrained=True)
    freeze_backbone(model, train_classifier_only=args.train_classifier_only)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    print(f"Classes: {num_classes} | Trainable params: {num_trainable_params(model):,} | Device: {device}")

    hist = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        log_dir=args.log_dir,
    )

    # Save class map next to checkpoint by default
    out_path = Path(args.save_class_map)
    out_path.write_text(json.dumps(class_to_idx, indent=2), encoding="utf-8")
    print(f"Saved class_to_idx to {out_path.resolve()}")
    print(f"Saved best weights to {Path(args.checkpoint).resolve()}")


if __name__ == "__main__":
    main()
