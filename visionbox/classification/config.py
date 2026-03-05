
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional


@dataclass
class DataConfig:
    """Dataset + preprocessing configuration.

    Directory layout expected by default:
      data_dir/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg

    If your folder is named 'valid' instead of 'val', this package will detect it.
    """
    data_dir: str
    img_size: Tuple[int, int] = (224, 224)
    mean: Sequence[float] = (0.4368, 0.4336, 0.3294)  # from the notebook (Monkey Species dataset)
    std: Sequence[float] = (0.2457, 0.2413, 0.2447)
    augment: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
    log_dir: Optional[str] = None  # if set, TensorBoard logs are written here
    checkpoint_path: str = "best.pt"  # best weights by val accuracy
