"""
Dataset helpers for fine-tuning.

Expects an ImageFolder layout (one sub-directory per class)::

    data/plants/
    ├── tomato/        img001.jpg ...
    ├── potato/        img101.jpg ...
    └── basil/         img201.jpg ...

This avoids any dependency on the HuggingFace ``datasets`` library — we use
torchvision's ImageFolder, which is already installed via ultralytics/torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


def _make_transform(processor):
    """Build a transform that turns a PIL image into a model-ready tensor."""
    size = processor.size.get("shortest_edge", processor.size.get("height", 224))
    mean = processor.image_mean
    std = processor.image_std

    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_dataloaders(
    data_dir: str,
    processor,
    batch_size: int = 16,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[int, str], Dict[str, int]]:
    """Return (train_loader, val_loader, id2label, label2id)."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            "Create an ImageFolder layout (one sub-directory per class)."
        )

    transform = _make_transform(processor)
    dataset = ImageFolder(str(root), transform=transform)

    id2label = {i: c for i, c in enumerate(dataset.classes)}
    label2id = {c: i for i, c in enumerate(dataset.classes)}

    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, id2label, label2id
