"""
Expected input (1D list / np.array / torch tensor):

[
    meanF0Hz, stdevF0Hz, 
    HNR, 
    localJitter, localabsoluteJitter,
    rapJitter, ppq5Jitter, ddpJitter, 
    localShimmer, localdbShimmer,
    apq3Shimmer, apq5Shimmer, 
    apq11Shimmer, ddaShimmer,
    f1_mean, f2_mean, f3_mean, f4_mean,
    f1_stdev, f2_stdev, f3_stdev, f4_stdev,
    gender, f0min, f0max
]
"""

import torch
import numpy as np
import joblib
from FINAL_MODELS.densenet169 import DenseNet1691D


# -------------------------
# Preprocessing
# -------------------------
def preprocess_audio_input(x, scaler):
    """
    Preprocess a single tabular audio sample.
    Args:
        x: list / np.ndarray / torch.Tensor (raw features)
        scaler: fitted sklearn MinMaxScaler
    Returns:
        torch.Tensor: shape (1, num_features)
    """

    # Convert to numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, list):
        x = np.array(x, dtype=np.float32)
    elif isinstance(x, np.ndarray):
        x = x.astype(np.float32)
    else:
        raise TypeError("Input must be list, numpy array, or torch tensor")

    # Scale
    x_scaled = scaler.transform(x.reshape(1, -1))

    # To tensor
    return torch.tensor(x_scaled, dtype=torch.float32)


# -------------------------
# Prediction
# -------------------------
def predict(x, device=None):
    """
    Predict Parkinson's probability for a single audio sample.
    Args:
        x: raw feature vector (list / np.ndarray / torch.Tensor)
        device: optional torch.device
    Returns:
        float: probability (0–1)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenseNet1691D().to(device)
    checkpoint = torch.load(
        "FINAL_MODELS/FINAL_PTH/Audio_Tabular_Model.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = joblib.load("FINAL_MODELS/FINAL_PTH/audio_scaler.save")
    
    X = preprocess_audio_input(x, scaler).to(device)

    with torch.inference_mode():
        logits = model(X)
        prob = torch.sigmoid(logits).item()

    return prob
