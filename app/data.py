import os
import shutil
import os.path as osp
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from loguru import logger
import zipfile

from utils import download_file, unzip_file


def get_val_loader(batch_size: int, num_workers: int = 2, data_path: str = "data") -> Tuple[DataLoader, int]:
    """
    Get ImageNet validation data loader.
    
    Args:
        batch_size: Batch size for the data loader
        num_workers: Number of worker processes for data loading
        data_path: Path to ImageNet dataset. If None, will use default torchvision path
    
    Returns:
        Tuple of (DataLoader, num_classes)
    """
    num_classes = 1000
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Use ImageNet validation dataset
    os.makedirs(data_path, exist_ok=True)
    val_dir = "data/tiny-imagenet-200/val"
    if not osp.isfile(f"{data_path}/tiny-imagenet-200.zip"):
        download_file("http://cs231n.stanford.edu/tiny-imagenet-200.zip", f"{data_path}/tiny-imagenet-200.zip")
        unzip_file("data/tiny-imagenet-200.zip", "data")
        prepare_mini_imagenet()

    val_dataset = ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader, num_classes


def prepare_mini_imagenet(val_dir: str = "data/tiny-imagenet-200/val") -> None:
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")
    val_images_dir = os.path.join(val_dir, "images")

    # Create subdirectories for each class in val/
    with open(val_annotations_file, 'r') as f:
        for line in tqdm(f.readlines()):
            img_name, class_id, *_ = line.strip().split()
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(val_images_dir, img_name)
            dst = os.path.join(class_dir, img_name)
            # TODO
            # shutil.move(src, dst)
            shutil.copy(src, dst)

