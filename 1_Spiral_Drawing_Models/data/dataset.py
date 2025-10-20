from torch.utils.data import Dataset
from pathlib import Path
import random

import albumentations as A
from PIL import Image
import numpy as np

# Dataset class
# ---------------
class DrawingDataset(Dataset):
    """
    PyTorch dataset class for spiral & wave drawing images for Parkinson’s Disease (PD) classification.

    The dataset:
        - Loads grayscale images from two directories (Healthy and PD).
        - Assigns labels automatically (0: Healthy, 1: PD).
        - Shuffles data reproducibly using a fixed random seed.
        - Applies Albumentations transformations (including optional user augmentations, mainly for training dataset).
        - Converts images to tensors for PyTorch models.

    Args:
        healthy_dir (str): Path to the directory containing healthy subject images.
        pd_dir (str): Path to the directory containing PD subject images.
        img_size (tuple): Target image size (height, width) after resizing.
        transforms (list, optional): Custom list of Albumentations transformations 
                                    to apply before the default preprocessing.
        random_seed (int): Random seed to ensure reproducible shuffling.

    Returns:
        dict: A dictionary with:
            - "image": transformed image tensor
            - "label": integer label (0 for healthy, 1 for PD)
    """
    def __init__(self,
                healthy_dir:str,
                pd_dir:str,
                img_size:tuple = (512, 512),
                transforms:list = None,
                random_seed :int = 42,):
        super().__init__()
        
        # 1. init paths and params
        self.healthy_dir = Path(healthy_dir)
        self.pd_dir = Path(pd_dir)
        self.img_size = img_size
        
        # 2. init default transforms
        default_transforms = [
            # 2.1. resize image
            A.Resize(height= img_size[0],
                    width= img_size[1]),
            
            # 2.2. use CLAHE (for histogram equalization) 
            # NOTE: TEST WITHOUT IT LATER
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8,8),
                p= 1.0,
            ),
            
            # 2.3. normalize images to grayscale 
            A.Normalize(mean=[0.5],
                        std=[0.5],
                        max_pixel_value=255.0),
            
            # 2.4. convert to pytorch tensor
            A.ToTensorV2(),
        ]
        
        # 3. concat user's transforms with default transforms (if given)
        if transforms is not None:
            self.transforms = A.Compose([
                *transforms,
                *default_transforms,
            ])
        else:
            self.transforms = A.Compose(default_transforms)
            
        # 4. get all data, and create labels
        healthy_files = list( self.healthy_dir.glob("*.png") )
        pd_files = list ( self.pd_dir.glob("*.png") )
        
        self.img_files = healthy_files + pd_files
        self.labels = [0]*len(healthy_files) + [1]*len(pd_files) # 0:for healthy, 1:for pd
        
        # 5. shuffle data (with random seed)
        # 5.1. combine them so that the labels and the images positions are still alligned
        combined = list(zip(self.img_files, self.labels)) 
        
        random.seed(random_seed)
        random.shuffle(combined) # shuffle is an inplace (dw)
        
        # 5.2. split them again
        self.img_files, self.labels = zip(*combined)
        
    # get len function
    # -----------------
    def __len__(self):
        return (len(self.img_files))
    
    # get item function
    # -------------------
    def __getitem__(self, idx):
        # 1. load the image's path, and its label
        img_path = self.img_files[idx]
        label = self.labels[idx]
        
        # 2. load the image as grayscale
        img = Image.open(img_path).convert('L')
        
        # 3. convert to numpy array (for albumentation transforms to work)
        img = np.array(img)
        
        # 4. apply transforms
        img = self.transforms(image=img)["image"]
        
        # 5. return item as dict
        return{
            "image": img,
            "label": label, # 0: for healthy, 1: for pd
        }
        
# ======================================================================= #

# Data Augmentation
# -------------------
from cv2 import BORDER_CONSTANT
def train_transforms():
    """
    Defines data augmentation for the training dataset to improve model generalization.
    The validation dataset uses only the default transformations.

    Most parameters use default values but are explicitly written for clarity and practice.
    """
    
    return [
        # 1. geometric trans
        # --------------------
        A.RandomRotate90(p=0.4),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.3),
        A.Transpose(p=0.3),
        A.Affine( # new version of 'ShiftScaleRotate'
            translate_percent=0.05,     # same as shift_limit
            scale=(0.9, 1.1),           # equivalent to scale_limit=0.1
            rotate=(-10, 10),           # same as rotate_limit
            fit_output=False,           # same as keeping image size fixed
            p=0.4
        ),
        A.Perspective( # change in camera viewpoint or angle
            scale = (0.05, 0.1),
            p=0.25
        ),

        
        # 2. non-rigid (elastic) trans
        # ------------------------------
        A.ElasticTransform(
            alpha= 1,
            sigma=50,
            border_mode= BORDER_CONSTANT,
            p = 0.25
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.25,
            border_mode= BORDER_CONSTANT,
            p = 0.25
        ),
        
        # 3. intensity and color trans
        # ------------------------------
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.RandomGamma(
            gamma_limit=(85, 115),
            p=0.2
        ),
        
        # 4. noise and blurring
        # ----------------------
        A.GaussNoise(
            std_range=(0.03, 0.08),  # equivalent strength; range as a fraction of max value
            mean_range=(0.0, 0.0),  # keep mean centered
            p=0.05
        ),
        A.GaussianBlur(
            blur_limit=3,
            p=0.15
        ),
        A.CoarseDropout(
            num_holes_range=(3, 6),
            hole_height_range=(10, 20),
            hole_width_range=(10, 20),
            fill="random_uniform",
            p=0.15
        )
    ]