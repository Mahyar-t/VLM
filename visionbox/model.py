
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models


_SUPPORTED = {
    "mobilenet_v3_small": "mobilenet_v3_small",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
    "clip-vit-base-patch32": "clip-vit-base-patch32",
}


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a torchvision classification model and replace its classification head."""
    if model_name not in _SUPPORTED:
        raise ValueError(f"Unsupported model_name={model_name}. Supported: {sorted(_SUPPORTED)}")

    weights = "DEFAULT" if pretrained else None

    if model_name.startswith("mobilenet_v3"):
        fn = getattr(models, _SUPPORTED[model_name])
        m = fn(weights=weights)
        # torchvision mobilenet_v3 classifier is Sequential(..., Linear(in=1024,out=1000), ...)
        # In the notebook they replace classifier[3] (the final Linear) with out_features=num_classes.
        if isinstance(m.classifier, nn.Sequential):
            # find last Linear
            last_linear_idx = None
            in_features = None
            for i in range(len(m.classifier) - 1, -1, -1):
                if isinstance(m.classifier[i], nn.Linear):
                    last_linear_idx = i
                    in_features = m.classifier[i].in_features
                    break
            if last_linear_idx is None or in_features is None:
                raise RuntimeError("Could not locate final Linear layer in MobileNetV3 classifier.")
            m.classifier[last_linear_idx] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        else:
            raise RuntimeError("Unexpected MobileNetV3 classifier type.")
        return m

    if model_name.startswith("resnet"):
        fn = getattr(models, _SUPPORTED[model_name])
        m = fn(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if model_name.startswith("densenet"):
        fn = getattr(models, _SUPPORTED[model_name])
        m = fn(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    if model_name.startswith("efficientnet"):
        fn = getattr(models, _SUPPORTED[model_name])
        m = fn(weights=weights)
        # classifier is Sequential(Dropout, Linear)
        if isinstance(m.classifier, nn.Sequential):
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        else:
            m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    raise ValueError(f"Unhandled model_name={model_name}")


def freeze_backbone(model: nn.Module, train_classifier_only: bool = False) -> None:
    """Optionally freeze all layers except the classification head."""
    if not train_classifier_only:
        return
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze common heads
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True


def num_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
