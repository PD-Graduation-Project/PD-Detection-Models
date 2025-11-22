import cv2
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from PIL import Image

# --------------------------------------------------------
# Load image (grayscale)
# --------------------------------------------------------
img_path = "Parkinson7.png"  # change to your image
img = Image.open(img_path).convert("L")
img = np.array(img)

# --------------------------------------------------------
# CLAHE transform (same as your dataset)
# --------------------------------------------------------
clahe_transform = A.CLAHE(
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    p=1.0
)

img_clahe = clahe_transform(image=img)["image"]

# --------------------------------------------------------
# Show both images
# --------------------------------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("After CLAHE")
plt.imshow(img_clahe, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
