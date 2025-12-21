import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ACDCSliceDataset(Dataset):
    """ACDC 2D 切片数据集，读取 PNG 图像与对应 mask。"""

    def __init__(self, root: Path, split: str, input_size: Tuple[int, int], augment: bool) -> None:
        self.root = Path(root)
        self.split = split
        self.input_size = tuple(input_size)
        self.augment = augment

        self.image_dir = self.root / "image" / split
        self.mask_dir = self.root / "mask" / split
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Missing split folders under {self.root} (expected image/mask/{split}).")

        self.samples: List[Path] = sorted(self.image_dir.glob("*.png"))
        if not self.samples:
            raise RuntimeError(f"No PNG slices found in {self.image_dir}.")

        self.ignore_label = 255
        self.num_classes = 4

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mask(self, path: Path) -> Image.Image:
        mask = Image.open(path).convert("L")
        if self.input_size:
            mask = mask.resize(self.input_size, Image.NEAREST)
        return mask

    def _load_image(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("RGB")
        if self.input_size:
            image = image.resize(self.input_size, Image.BILINEAR)
        return image

    def __getitem__(self, index: int):
        img_path = self.samples[index]
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {img_path.name} under {self.mask_dir}.")

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)

        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image_tensor, mask_tensor, img_path.name
