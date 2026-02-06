"""
Parkinson's Disease Audio Prediction from WAV files

Expected model input order (25 features):
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
from models.densenet169 import DenseNet1691D
from utils.uams_feature_extractor import UAMSFeatureExtractor


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


def extract_features_from_wav(audio_path, gender=None):
    """
    Extract features from a WAV file using UAMS feature extractor.
    
    Args:
        audio_path: path to .wav file
        gender: 'M' or 'F' or None (for auto-detect)
    
    Returns:
        np.ndarray: feature vector in correct model order (25 features)
    """
    # Initialize feature extractor
    extractor = UAMSFeatureExtractor(target_sr=8000, segment_duration=1.5)
    
    # Extract features with metadata
    features_dict = extractor.extract_features_with_metadata(
        audio_path=audio_path,
        gender=gender
    )
    
    # Determine gender encoding (0 = male, 1 = female)
    if 'is_female' in features_dict:
        gender_encoded = 1 if features_dict['is_female'] else 0
    else:
        # Default to female if not specified
        gender_encoded = 1
    
    # Extract features in the correct order for the model
    feature_vector = [
        features_dict['meanF0Hz'],
        features_dict['stdevF0Hz'],
        features_dict['HNR'],
        features_dict['localJitter'],
        features_dict['localabsoluteJitter'],
        features_dict['rapJitter'],
        features_dict['ppq5Jitter'],
        features_dict['ddpJitter'],
        features_dict['localShimmer'],
        features_dict['localdbShimmer'],
        features_dict['apq3Shimmer'],
        features_dict['apq5Shimmer'],
        features_dict['apq11Shimmer'],
        features_dict['ddaShimmer'],
        features_dict['f1_mean'],
        features_dict['f2_mean'],
        features_dict['f3_mean'],
        features_dict['f4_mean'],
        features_dict['f1_stdev'],
        features_dict['f2_stdev'],
        features_dict['f3_stdev'],
        features_dict['f4_stdev'],
        gender_encoded,
        features_dict['f0min'],
        features_dict['f0max']
    ]
    
    return np.array(feature_vector, dtype=np.float32)


# -------------------------
# Prediction
# -------------------------
def predict(audio_path, gender=None, device=None):
    """
    Predict Parkinson's probability from a WAV file.
    
    Args:
        audio_path: path to .wav file
        gender: 'M' or 'F' or None (for auto-detect)
        device: optional torch.device
    
    Returns:
        float: probability (0–1)
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DenseNet1691D().to(device)
    checkpoint = torch.load(
        "weights/Audio_Tabular_Model.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load scaler
    scaler = joblib.load("weights/audio_scaler.save")
    
    # Extract features from WAV file
    feature_vector = extract_features_from_wav(audio_path, gender)
    
    # Preprocess features
    X = preprocess_audio_input(feature_vector, scaler).to(device)

    # Predict
    with torch.inference_mode():
        logits = model(X)
        prob = torch.sigmoid(logits).item()

    return prob