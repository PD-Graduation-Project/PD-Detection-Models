import numpy as np
from scipy.fft import fft, fftfreq

# ============================================================================
# PART 1: Feature Extraction Functions
# ============================================================================

def extract_frequency_features(signal_data, sampling_rate=100):
    """
    Extract frequency domain features from acceleration signal.
    
    Args:
        signal_data: numpy array of shape (T, 3) - single hand xyz data
        sampling_rate: Hz
    
    Returns:
        dict of features
    """
    features = {}
    
    # Compute magnitude (combines x, y, z)
    magnitude = np.sqrt(np.sum(signal_data**2, axis=1))
    
    # FFT
    fft_vals = np.abs(fft(magnitude))
    freqs = fftfreq(len(magnitude), 1/sampling_rate)
    
    # Only positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = fft_vals[:len(fft_vals)//2]
    
    # PD tremor band (3-6 Hz)
    pd_band_mask = (positive_freqs >= 3) & (positive_freqs <= 6)
    features['pd_band_power'] = positive_fft[pd_band_mask].sum()
    features['pd_band_peak'] = positive_fft[pd_band_mask].max() if pd_band_mask.any() else 0
    
    # Find dominant frequency
    if len(positive_fft) > 0:
        features['dominant_freq'] = positive_freqs[np.argmax(positive_fft)]
        features['dominant_power'] = positive_fft.max()
    else:
        features['dominant_freq'] = 0
        features['dominant_power'] = 0
    
    # Spectral centroid (frequency "center of mass")
    features['spectral_centroid'] = np.sum(positive_freqs * positive_fft) / (np.sum(positive_fft) + 1e-10)
    
    # Spectral spread (frequency variance)
    features['spectral_spread'] = np.sqrt(
        np.sum(((positive_freqs - features['spectral_centroid'])**2) * positive_fft) / 
        (np.sum(positive_fft) + 1e-10)
    )
    
    # Total power
    features['total_power'] = positive_fft.sum()
    
    # Power ratio (tremor band / total)
    features['pd_power_ratio'] = features['pd_band_power'] / (features['total_power'] + 1e-10)
    
    return features


def extract_temporal_features(signal_data):
    """
    Extract time domain features.
    
    Args:
        signal_data: numpy array of shape (T, 3) - single hand xyz data
    
    Returns:
        dict of features
    """
    features = {}
    
    # Magnitude
    magnitude = np.sqrt(np.sum(signal_data**2, axis=1))
    
    # Basic statistics
    features['mean'] = magnitude.mean()
    features['std'] = magnitude.std()
    features['max'] = magnitude.max()
    features['min'] = magnitude.min()
    features['range'] = features['max'] - features['min']
    
    # Percentiles
    features['p25'] = np.percentile(magnitude, 25)
    features['p75'] = np.percentile(magnitude, 75)
    features['iqr'] = features['p75'] - features['p25']
    
    # Root mean square (RMS)
    features['rms'] = np.sqrt(np.mean(magnitude**2))
    
    # Zero crossing rate (indicator of oscillation)
    centered = magnitude - magnitude.mean()
    features['zero_crossing_rate'] = np.sum(np.diff(np.sign(centered)) != 0) / len(magnitude)
    
    # Peak count (tremor episodes)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(magnitude, distance=10)
    features['peak_count'] = len(peaks)
    features['peak_rate'] = len(peaks) / (len(magnitude) / 100)  # peaks per second
    
    # Autocorrelation (periodicity indicator)
    autocorr = np.correlate(centered, centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Find first significant peak in autocorrelation (excluding lag 0)
    if len(autocorr) > 20:
        features['autocorr_peak'] = np.max(autocorr[10:100]) if len(autocorr) > 100 else np.max(autocorr[10:])
    else:
        features['autocorr_peak'] = 0
    
    # Jerk (rate of change of acceleration)
    jerk = np.diff(magnitude)
    features['jerk_mean'] = np.abs(jerk).mean()
    features['jerk_std'] = jerk.std()
    
    return features


def extract_bilateral_features(left_data, right_data):
    """
    Extract features comparing left and right hands.
    
    Args:
        left_data: numpy array of shape (T, 3)
        right_data: numpy array of shape (T, 3)
    
    Returns:
        dict of features
    """
    features = {}
    
    # Magnitudes
    left_mag = np.sqrt(np.sum(left_data**2, axis=1))
    right_mag = np.sqrt(np.sum(right_data**2, axis=1))
    
    # Power asymmetry
    left_power = np.mean(left_mag**2)
    right_power = np.mean(right_mag**2)
    features['power_asymmetry'] = abs(left_power - right_power) / (left_power + right_power + 1e-10)
    
    # Amplitude asymmetry
    features['amplitude_asymmetry'] = abs(left_mag.std() - right_mag.std()) / (left_mag.std() + right_mag.std() + 1e-10)
    
    # Cross-correlation (synchrony between hands)
    left_centered = left_mag - left_mag.mean()
    right_centered = right_mag - right_mag.mean()
    cross_corr = np.correlate(left_centered, right_centered, mode='valid')
    features['bilateral_correlation'] = cross_corr[0] / (left_mag.std() * right_mag.std() * len(left_mag))
    
    # Difference signal features
    diff = left_mag - right_mag
    features['bilateral_diff_mean'] = diff.mean()
    features['bilateral_diff_std'] = diff.std()
    
    return features


def extract_all_features(signal_tensor):
    """
    Extract all features from a single sample.
    
    Args:
        signal_tensor: torch.Tensor of shape (2, T, 6)
                      [0] = left hand (T, 6), [1] = right hand (T, 6)
    
    Returns:
        numpy array of all features concatenated
    """
    # Convert to numpy
    signal_np = signal_tensor.cpu().numpy()
    
    left_hand = signal_np[0]   # (T, 6) - accel_xyz, gyro_xyz
    right_hand = signal_np[1]  # (T, 6)
    
    # Split accel and gyro
    left_accel = left_hand[:, :3]
    right_accel = right_hand[:, :3]
    left_gyro = left_hand[:, 3:]
    right_gyro = right_hand[:, 3:]
    
    all_features = {}
    
    # LEFT HAND FEATURES
    left_freq = extract_frequency_features(left_accel)
    left_temp = extract_temporal_features(left_accel)
    all_features.update({f'left_accel_{k}': v for k, v in left_freq.items()})
    all_features.update({f'left_accel_{k}': v for k, v in left_temp.items()})
    
    left_gyro_freq = extract_frequency_features(left_gyro)
    left_gyro_temp = extract_temporal_features(left_gyro)
    all_features.update({f'left_gyro_{k}': v for k, v in left_gyro_freq.items()})
    all_features.update({f'left_gyro_{k}': v for k, v in left_gyro_temp.items()})
    
    # RIGHT HAND FEATURES
    right_freq = extract_frequency_features(right_accel)
    right_temp = extract_temporal_features(right_accel)
    all_features.update({f'right_accel_{k}': v for k, v in right_freq.items()})
    all_features.update({f'right_accel_{k}': v for k, v in right_temp.items()})
    
    right_gyro_freq = extract_frequency_features(right_gyro)
    right_gyro_temp = extract_temporal_features(right_gyro)
    all_features.update({f'right_gyro_{k}': v for k, v in right_gyro_freq.items()})
    all_features.update({f'right_gyro_{k}': v for k, v in right_gyro_temp.items()})
    
    # BILATERAL FEATURES
    bilateral_accel = extract_bilateral_features(left_accel, right_accel)
    bilateral_gyro = extract_bilateral_features(left_gyro, right_gyro)
    all_features.update({f'bilateral_accel_{k}': v for k, v in bilateral_accel.items()})
    all_features.update({f'bilateral_gyro_{k}': v for k, v in bilateral_gyro.items()})
    
    # Convert to numpy array (consistent ordering)
    feature_vector = np.array(list(all_features.values()), dtype=np.float32)
    
    return feature_vector, list(all_features.keys())