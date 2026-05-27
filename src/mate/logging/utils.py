from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch


PathLike = Union[str, os.PathLike]


def ensure_dir(path: PathLike) -> Path:
    path_obj = Path(path)
    if path_obj.suffix:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    else:
        path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def make_run_id(condition: str, seed: int) -> str:
    return f"{condition}_seed_{seed}"


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
