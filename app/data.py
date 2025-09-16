from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_val_loader(batch_size: int, num_workers: int = 2) -> Tuple[DataLoader, int]:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    num_classes = 1000
    ds = datasets.FakeData(
        size=100,
        image_size=(3, 224, 224),
        num_classes=num_classes,
        transform=transform,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader, num_classes
