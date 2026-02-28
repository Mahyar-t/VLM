
from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


def get_device(prefer: str = "cuda") -> torch.device:
    """Return an available torch.device.

    prefer: "cuda" or "cpu" (anything else treated as "cpu").
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set seeds for reproducibility.

    Note: deterministic=True can reduce performance; keep it on for repeatability.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def save_checkpoint(state_dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state_dict, path)


def load_checkpoint(path: str, map_location: Optional[str | torch.device] = None):
    return torch.load(path, map_location=map_location)
