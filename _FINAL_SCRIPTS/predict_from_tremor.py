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
5. Extract catch22 features (22 features +2 catch24 features)
6. Compute short time fourier transform (STFT) (+3 features)
6. Concatenate:
    [left_features, right_features, asymmetry_features]
7. Append metadata:
    - handedness
    - movement index

Movements indeces:
    - 'CrossArms'  : 0 |  10  sec
    - 'DrinkGlas'  : 1 |  10  sec
    - 'Entrainment': 2 | *20* sec
    - 'HoldWeight' : 3 |  10  sec
    - 'LiftHold'   : 4 |  10  sec
    - 'PointFinger': 5 |  10  sec
    - 'Relaxed'    : 6 | *20* sec 
    - 'RelaxedTask': 7 | *20* sec
    - 'StretchHold': 8 |  10  sec
    - 'TouchIndex' : 9 |  10  sec
    - 'TouchNose'  : 10|  10  sec
    
"""

import torch
import numpy as np
import joblib
from pathlib import Path

from scipy.signal import stft
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
SCALER_PATH = "weights/tremor_scaler.pkl" 

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
    else:
        # pad last segment if it's not long enough
        last_start = (len(segments) - 1) * step + window_size
        remaining = data[last_start:]
        if len(remaining) > 0 and len(remaining) < window_size:
            pad = window_size - len(remaining)
            remaining = np.pad(remaining, ((0, pad), (0, 0)), mode="edge")
            segments.append(remaining)

    return segments

def _compute_stft_pd_feature(
    segment: np.ndarray,
    fs: float = 100.0,
    tremor_band=(3.0, 8.0)
):
    """
    Compute STFT-based PD features:
    - Tremor band power ratio
    - Tremor stability (std over time)
    - Peak frequency

    Returns:
        (ratio, stability, peak_freq)
    """

    # Extract 1D signal (handle single or multi-channel input)
    signal_1d = segment[:, 0] if segment.ndim == 2 else segment
    signal_1d = np.asarray(signal_1d, dtype=np.float32)

    # Return zeros if signal too short for meaningful STFT
    if signal_1d.size < 8:
        return 0.0, 0.0, 0.0

    # Normalize signal to remove amplitude bias across subjects
    signal_1d = (signal_1d - np.mean(signal_1d)) / (np.std(signal_1d) + 1e-8)

    # Set STFT window length (~2 seconds for good low-frequency resolution)
    nperseg = int(fs * 2)

    # Ensure window is not longer than signal
    nperseg = min(nperseg, signal_1d.size)

    # Use high overlap to improve temporal smoothness
    noverlap = int(nperseg * 0.75)

    # Compute STFT using Hann window without padding artifacts
    f, _, Zxx = stft(
        signal_1d,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )

    # Compute spectrogram power from complex STFT output
    power = np.abs(Zxx) ** 2

    # Return zeros if STFT failed or is empty
    if power.size == 0:
        return 0.0, 0.0, 0.0

    # Create mask for tremor frequency band (3–8 Hz)
    band_mask = (f >= tremor_band[0]) & (f <= tremor_band[1])

    # Compute total power across all frequencies and time
    total_power = power.sum() + 1e-12

    # Compute total tremor-band power across all time frames
    band_power = power[band_mask, :].sum() if np.any(band_mask) else 0.0

    # Compute tremor power ratio (how dominant tremor band is) -> 1
    tremor_ratio = band_power / total_power

    # Compute time-varying tremor power per frame (captures intermittency)
    band_power_t = (
        power[band_mask, :].mean(axis=0)
        if np.any(band_mask)
        else np.zeros(power.shape[1])
    )

    # Compute tremor stability as std over time (higher = more fluctuation) -> 2
    tremor_stability = np.std(band_power_t)

    # Compute average spectrum across time
    avg_spectrum = power.mean(axis=1)

    # Extract dominant (peak) frequency from average spectrum -> 3
    peak_freq = f[np.argmax(avg_spectrum)] if len(f) > 0 else 0.0

    return float(tremor_ratio), float(tremor_stability), float(peak_freq)

def _extract_features_from_segment(segment):
    """
    Extract statistical, temporal, and spectral features from a signal segment.
    Uses catch22 library to extract features from each channel independently,
    then concatenates all features.
    
    Args:
        segment: np.ndarray of shape (window_size, num_channels)
                Time series data for a single window across channels
    
    Returns:
        np.ndarray of shape (num_features * num_channels,)
                Concatenated features from all channels
    """
    all_features = []
    
    # Extract features from EACH channel
    for channel_idx in range(segment.shape[1]):
        channel_signal = segment[:, channel_idx]
        
        # Force catch24=True so we include mean and std (24 total catch features).
        features = pycatch22.catch22_all(channel_signal, catch24=True)['values']
        all_features.extend(features)
        
    # NEW: append STFT tremor-band power ratio as LAST feature
    stft_ratio, stft_stability, peak_freq = _compute_stft_pd_feature(
            segment,
            fs=100,
            tremor_band=(3.0, 7.0)
        )
    all_features.extend([stft_ratio, stft_stability, peak_freq])
    
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
        all_features.append(np.concatenate([lf, rf]))

    # return array of features (one per segment) with padding if needed
    features_array = np.array(all_features, dtype=np.float32)
    
    # pad if only one segment
    if len(features_array) == 1:
        features_array = np.pad(features_array, ((0, 1), (0, 0)), mode='edge')
    
    return features_array


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

    # features - returns array of shape (num_segments, feature_dim)
    features_array = extract_features_from_txt(left[0], right[0])
    
    # Load and apply scaler (must match training)
    scaler = joblib.load(SCALER_PATH)
    features_array = scaler.transform(features_array)

    # metadata
    handedness_val = 0 if handedness.lower() == "left" else 1
    movement_val = movement

    # model
    model = TremorClassifier().to(device)
    checkpoint = torch.load(
        "weights/Tremor_Model.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # make predictions for each segment and average
    predictions = []
    with torch.inference_mode():
        for features in features_array:
            x = torch.tensor(features, dtype=torch.float32).to(device)
            handedness_tensor = torch.tensor(handedness_val, dtype=torch.long).to(device)
            movement_tensor = torch.tensor(movement_val, dtype=torch.long).to(device)
            
            # add batch dimension
            x = x.unsqueeze(0)
            handedness_tensor = handedness_tensor.unsqueeze(0)
            movement_tensor = movement_tensor.unsqueeze(0)
            
            logits = model(x, handedness_tensor, movement_tensor)
            prob = torch.sigmoid(logits).item()
            predictions.append(prob)

    return np.mean(predictions)
