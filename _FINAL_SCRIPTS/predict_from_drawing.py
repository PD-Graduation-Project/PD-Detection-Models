"""
Parkinson's Disease Drawing Prediction from Spiral/Wave Images

This module predicts Parkinson's Disease probability from hand-drawn spirals or waves.
Images are preprocessed with CLAHE enhancement and normalized before inference.
"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

from models.mobilenetV3 import MobileNetV3LargeBinary


# -------------------------
# Preprocessing
# -------------------------
def preprocess_drawing(image_or_path):
    """
    Load and preprocess a drawing image.
    
    Args:
        image_or_path: str (file path) or PIL.Image.Image
    
    Returns:
        np.ndarray: grayscale image array
    """
    # Load image
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path).convert("L")  # grayscale
    elif isinstance(image_or_path, Image.Image):
        image = image_or_path.convert("L")
    else:
        raise TypeError("Input must be a file path (str) or PIL Image")

    return np.array(image)


def get_transforms():
    """
    Define image preprocessing transforms.
    
    Returns:
        albumentations.Compose: transform pipeline
    """
    return A.Compose([
        # Resize to model input size
        A.Resize(height=256, width=256),

        # Contrast Limited Adaptive Histogram Equalization
        # Enhances local contrast to highlight drawing features
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=1.0
        ),

        # Normalize pixel values to [-1, 1]
        A.Normalize(
            mean=[0.5],
            std=[0.5],
            max_pixel_value=255.0
        ),

        # Convert to PyTorch tensor
        ToTensorV2(),
    ])


# -------------------------
# Prediction
# -------------------------
def predict(image_or_path, device=None):
    """
    Predict Parkinson's probability from a drawing image.
    
    Args:
        image_or_path: str (path to image file) or PIL.Image.Image
        device: optional torch.device (defaults to CUDA if available, else CPU)
    
    Returns:
        float: probability (0–1) where higher values indicate higher PD likelihood
    
    Example:
        >>> prob = predict("spiral_drawing.png")
        >>> print(f"Parkinson's probability: {prob:.4f}")
    """
    
    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess image
    image = preprocess_drawing(image_or_path)
    
    # Apply transforms
    transforms = get_transforms()
    image = transforms(image=image)["image"]
    image = image.unsqueeze(0).to(device)  # add batch dimension: (1, C, H, W)

    # Load model
    model = MobileNetV3LargeBinary().to(device)
    checkpoint = torch.load(
        "weights/Spiral_Drawing_Model.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Inference
    with torch.inference_mode():
        logits = model(image)
        prob = torch.sigmoid(logits).item()

    return prob
