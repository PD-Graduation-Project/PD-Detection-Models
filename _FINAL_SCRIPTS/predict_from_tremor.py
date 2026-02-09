"""
Parkinson's Disease Tremor Prediction from Wrist IMU (.txt) files

Expected input:
- Directory containing TWO .txt files:
    - *_LeftWrist.txt
    - *_RightWrist.txt

Pipeline (must match training):
1. Remove timestamp column
2. Keep accelerometer only (X, Y, Z)
3. Convert to vector magnitude
4. Segment signal
5. Extract catch22 features
6. Concatenate:
    [left_features, right_features, asymmetry_features]
7. Append metadata:
    - handedness
    - movement index

Movements indeces:
    - 'CrossArms': 0,
    - 'DrinkGlas': 1,
    - 'Entrainment': 2,
    - 'HoldWeight': 3,
    - 'LiftHold': 4,
    - 'PointFinger': 5,
    - 'Relaxed': 6,
    - 'RelaxedTask': 7,
    - 'StretchHold': 8,
    - 'TouchIndex': 9,
    - 'TouchNose': 10
    
"""

import torch
import numpy as np
import joblib
from pathlib import Path

import pycatch22

# -------------------------
# Model
# -------------------------
from models.tremorNet import TremorClassifier 


# -------------------------
# Constants (MUST MATCH TRAINING)
# -------------------------
WINDOW_SIZE = 1024
OVERLAP = 0.5
NUM_CATCH22 = 22


# -------------------------
# Preprocessing utilities
# -------------------------
def _remove_timestamp_column(data):
    if data.shape[1] == 7:
        return data[:, 1:]
    return data


def _handle_missing_values(data):
    mask = np.isnan(data)
    idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return data[idx, np.arange(data.shape[1])]


def _compute_vector_magnitude(data):
    return np.sqrt(np.sum(data ** 2, axis=1, keepdims=True))


def _segment_signal(data, window_size, overlap):
    step = int(window_size * (1 - overlap))
    segments = []

    for start in range(0, len(data) - window_size + 1, step):
        segments.append(data[start:start + window_size])

    if not segments:
        pad = window_size - len(data)
        data = np.pad(data, ((0, pad), (0, 0)), mode="edge")
        segments.append(data)

    return segments


def _extract_features_from_segment(segment):
    all_features = []
    
    # Extract features from EACH channel
    for channel_idx in range(segment.shape[1]):
        channel_signal = segment[:, channel_idx]
        
        # catch22 returns 22 features per channel
        features = pycatch22.catch22_all(channel_signal)['values']
        all_features.extend(features)
    
    return np.array(all_features, dtype=np.float32)


# -------------------------
# Feature extraction
# -------------------------
def extract_features_from_txt(left_path, right_path):
    left = np.loadtxt(left_path, delimiter=",", dtype=np.float32)
    right = np.loadtxt(right_path, delimiter=",", dtype=np.float32)

    left = _remove_timestamp_column(left)
    right = _remove_timestamp_column(right)

    left = _handle_missing_values(left)
    right = _handle_missing_values(right)

    # accelerometer only
    left = left[:, :3]
    right = right[:, :3]

    # vector magnitude
    left = _compute_vector_magnitude(left)
    right = _compute_vector_magnitude(right)

    left_segs = _segment_signal(left, WINDOW_SIZE, OVERLAP)
    right_segs = _segment_signal(right, WINDOW_SIZE, OVERLAP)

    all_features = []

    for l, r in zip(left_segs, right_segs):
        lf = _extract_features_from_segment(l)
        rf = _extract_features_from_segment(r)
        asym = np.abs(lf - rf)
        all_features.append(np.concatenate([lf, rf, asym]))

    # average over segments
    return np.mean(all_features, axis=0)


# -------------------------
# Prediction
# -------------------------
def predict(
    txt_dir,
    movement: int,
    handedness: str,
    device=None
):
    """
    Args:
        txt_dir: directory containing Left & Right wrist .txt files
        movement: movement index (int)
        handedness: "left" or "right"
    Returns:
        Parkinson probability (float)
    """

    txt_dir = Path(txt_dir)
    left = list(txt_dir.glob("*LeftWrist*.txt"))
    right = list(txt_dir.glob("*RightWrist*.txt"))

    if len(left) != 1 or len(right) != 1:
        raise ValueError("Directory must contain exactly one LeftWrist and one RightWrist file")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # features
    features = extract_features_from_txt(left[0], right[0])

    # metadata
    handedness_val = 0 if handedness.lower() == "left" else 1
    movement_val = movement

    # convert to pytorch tensor
    x = torch.tensor(features, dtype=torch.float32).to(device)
    handedness_val = torch.tensor(handedness_val, dtype=torch.long).to(device)
    movement_val = torch.tensor(movement_val, dtype=torch.long).to(device)
    
    # model expects batch dimension
    x = x.unsqueeze(0)
    handedness_val = handedness_val.unsqueeze(0)
    movement_val = movement_val.unsqueeze(0)

    # model
    model = TremorClassifier().to(device)
    checkpoint = torch.load(
        "weights/Tremor_Model.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        logits = model(x, handedness_val, movement_val)
        prob = torch.sigmoid(logits).item()

    return prob
