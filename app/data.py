import os
import shutil
import glob
import os.path as osp
from typing import Tuple, List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from loguru import logger
import zipfile

from utils import download_file, unzip_file


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def get_val_images(data_path: str = "data", num_images: int = 100) -> List:
    """
    Get ImageNet validation data loader.

    Args:
        batch_size: Batch size for the data loader
        num_workers: Number of worker processes for data loading
        data_path: Path to ImageNet dataset. If None, will use default torchvision path
    """
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
    val_dir = "data/tiny-imagenet-200/val/images"
    if not osp.isfile(f"{data_path}/tiny-imagenet-200.zip"):
        logger.info("Downloading mini ImageNet Dataset")
        download_file("http://cs231n.stanford.edu/tiny-imagenet-200.zip", f"{data_path}/tiny-imagenet-200.zip")
        unzip_file("data/tiny-imagenet-200.zip", "data")

    img_path_list = sorted(glob.glob(os.path.join(val_dir, "*JPEG"))[:num_images])
    img_list = [Image.open(img_path).convert("RGB") for img_path in img_path_list]
    img_tensor_list = [transform(img) for img in img_list]
    return img_tensor_list

dataset = get_val_images()

def get_val_loader(batch_size: int, num_workers: int = 2, data_path: str = "data") -> DataLoader:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader
