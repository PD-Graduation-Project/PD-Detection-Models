from torch.utils.data import Dataset
from pathlib import Path
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

class SpectrogramDataset(Dataset):
    """
    PyTorch dataset class for spectrogram images for Parkinson’s Disease (PD) classification.

    Key points:
        - Loads RGB spectrogram images from two directories (Healthy and PD).
        - Assigns labels automatically (0: Healthy, 1: PD).
        - Shuffles data reproducibly using a fixed random seed.
        - No data augmentation applied to preserve spectrogram information.
        - Only resizing, normalization, and conversion to tensor are performed.

    Args:
        healthy_dir (str): Path to the directory containing healthy spectrogram images.
        pd_dir (str): Path to the directory containing PD spectrogram images.
        img_size (tuple): Target image size (height, width) after resizing.
        random_seed (int): Random seed for reproducible shuffling.
    """
    def __init__(
        self,
        healthy_dir: str,
        pd_dir: str,
        img_size: tuple = (512, 512),
        random_seed: int = 42,
    ):
        super().__init__()

        # Initialize paths and parameters
        self.healthy_dir = Path(healthy_dir)
        self.pd_dir = Path(pd_dir)
        self.img_size = img_size

        # Define transforms: resize, normalize, convert to tensor
        self.transforms = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2()
        ])

        # Load all image paths and labels
        healthy_files = list(self.healthy_dir.glob("*.jpg"))
        pd_files = list(self.pd_dir.glob("*.jpg"))

        self.img_files = healthy_files + pd_files
        self.labels = [0] * len(healthy_files) + [1] * len(pd_files)  # 0=Healthy, 1=PD

        # Shuffle data with fixed random seed
        combined = list(zip(self.img_files, self.labels))
        random.seed(random_seed)
        random.shuffle(combined)
        self.img_files, self.labels = zip(*combined)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image and label
        img_path = self.img_files[idx]
        label = self.labels[idx]

        # Load RGB image
        img = Image.open(img_path).convert('RGB')

        # Convert to numpy array
        img = np.array(img)

        # Apply transforms (resize, normalize, to tensor)
        img = self.transforms(image=img)["image"]

        return {
            "image": img,
            "label": label
        }
