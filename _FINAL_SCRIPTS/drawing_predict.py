import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

from FINAL_MODELS.mobilenetV3 import MobileNetV3LargeBinary


def predict(image_or_path, 
            device=None):
    """
    Run inference on a single image.

    Args:
        image_or_path (str | PIL.Image.Image): Path to image or PIL image.
        device (torch.device, optional): CPU or CUDA device.

    Returns:
        float: Prediction probability (0–1)
    """

    # 1. Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load image
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path).convert("L")  # grayscale
    elif isinstance(image_or_path, Image.Image):
        image = image_or_path.convert("L")
    else:
        raise TypeError("Input must be a file path or a PIL Image")

    image = np.array(image)

    # 3. Define transforms
    transforms = A.Compose([
        A.Resize(height=256, width=256),

        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=1.0
        ),

        A.Normalize(
            mean=[0.5],
            std=[0.5],
            max_pixel_value=255.0
        ),

        ToTensorV2(),
    ])

    # 4. Apply transforms
    image = transforms(image=image)["image"]
    image = image.unsqueeze(0).to(device)  # add batch dimension

    # 5. Load model
    model = MobileNetV3LargeBinary().to(device)
    checkpoint = torch.load(
        "FINAL_MODELS/Spiral_Drawing_Model.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 6. Inference
    with torch.inference_mode():
        logits = model(image)
        probability = torch.sigmoid(logits).item()

    return probability
