
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import save_checkpoint


@dataclass
class History:
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_acc: List[float]


def train_one_epoch(model, loader: DataLoader, optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    model.to(device)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += int((preds == labels).sum().item())

    return running_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    model.to(device)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        running_loss += float(loss.item())
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += int((preds == labels).sum().item())

    return running_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    device: torch.device,
    num_epochs: int,
    checkpoint_path: str = "best.pt",
    log_dir: Optional[str] = None,
) -> History:
    writer = None
    if log_dir:
        from torch.utils.tensorboard import SummaryWriter  # optional dependency via torch
        writer = SummaryWriter(log_dir=log_dir)

    tr_losses: List[float] = []
    tr_accs: List[float] = []
    va_losses: List[float] = []
    va_accs: List[float] = []

    best_val_acc = -1.0
    best_state = None

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, device)

        tr_losses.append(tr_loss); tr_accs.append(tr_acc)
        va_losses.append(va_loss); va_accs.append(va_acc)

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
            f"val loss {va_loss:.4f} acc {va_acc:.2f}%"
        )

        if writer:
            writer.add_scalar("Loss/train", tr_loss, epoch)
            writer.add_scalar("Loss/val", va_loss, epoch)
            writer.add_scalar("Accuracy/train", tr_acc, epoch)
            writer.add_scalar("Accuracy/val", va_acc, epoch)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            save_checkpoint(best_state, checkpoint_path)

    if writer:
        writer.close()

    return History(tr_losses, tr_accs, va_losses, va_accs)
