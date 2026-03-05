
from __future__ import annotations
import os
from pathlib import Path
from typing import Sequence, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(
    img_size: Tuple[int, int] = (224, 224),
    mean: Sequence[float] = (0.4368, 0.4336, 0.3294),
    std: Sequence[float] = (0.2457, 0.2413, 0.2447),
    augment: bool = True,
):
    preprocess = transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
        ]
    )

    common = transforms.Compose(
        [
            preprocess,
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    if not augment:
        return common, common

    train_tf = transforms.Compose(
        [
            preprocess,
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(p=0.4),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))],
                p=0.1,
            ),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, common


def _resolve_split_dirs(data_dir: str, train_dir: Optional[str] = None, val_dir: Optional[str] = None):
    base = Path(data_dir)
    if train_dir is None:
        train_dir = str(base / "train")
    if val_dir is None:
        # accept val or valid
        cand1 = base / "val"
        cand2 = base / "valid"
        val_dir = str(cand1 if cand1.exists() else cand2)
    return train_dir, val_dir


def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (224, 224),
    mean: Sequence[float] = (0.4368, 0.4336, 0.3294),
    std: Sequence[float] = (0.2457, 0.2413, 0.2447),
    augment: bool = True,
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
):
    train_tf, val_tf = build_transforms(img_size=img_size, mean=mean, std=std, augment=augment)
    train_dir, val_dir = _resolve_split_dirs(data_dir, train_dir=train_dir, val_dir=val_dir)

    train_ds = datasets.ImageFolder(root=train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(root=val_dir, transform=val_tf)

    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader, train_ds.class_to_idx
