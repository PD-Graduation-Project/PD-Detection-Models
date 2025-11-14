import numpy as np
import torch
from scipy import signal as sp
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import xgboost as xgb


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


# ============================================================================
# PART 2: DataLoader to Sklearn Conversion
# ============================================================================

def dataloader_to_sklearn(dataloader, binary_classification=True):
    """
    Convert PyTorch DataLoader to sklearn-compatible format.
    
    Args:
        dataloader: PyTorch DataLoader
        binary_classification: If True, filter out class 2 (Other) and only keep 0 (Healthy) and 1 (PD)
    
    Returns:
        X: numpy array of shape (N, num_features)
        y: numpy array of shape (N,)
        handedness: numpy array of shape (N,)
        movements: numpy array of shape (N,)
        feature_names: list of feature names
    """
    X_list = []
    y_list = []
    handedness_list = []
    movements_list = []
    feature_names = None
    
    print("Converting DataLoader to sklearn format...")
    for batch in dataloader:
        signals, handedness, movements, labels = batch
        
        # Process each sample in batch
        for i in range(len(signals)):
            # Skip "Other" class if binary classification
            if binary_classification and labels[i].item() == 2:
                continue
            
            # Extract features
            features, names = extract_all_features(signals[i])
            
            if feature_names is None:
                feature_names = names
            
            X_list.append(features)
            y_list.append(labels[i].item())
            handedness_list.append(handedness[i].item())
            movements_list.append(movements[i].item())
    
    X = np.array(X_list)
    y = np.array(y_list)
    handedness = np.array(handedness_list)
    movements = np.array(movements_list)
    
    print(f"Converted {len(X)} samples with {X.shape[1]} features")
    
    return X, y, handedness, movements, feature_names


def dataloader_dict_to_sklearn(dataloader_dict, binary_classification=True):
    """
    Convert per-movement dataloader dictionary to sklearn format.
    
    Args:
        dataloader_dict: dict of {movement_name: {"train": DataLoader, "val": DataLoader}}
        binary_classification: If True, filter out class 2 (Other)
    
    Returns:
        train_data: dict with X, y, handedness, movements
        val_data: dict with X, y, handedness, movements
        feature_names: list of feature names
    """
    # Collect all training data from all movements
    X_train_list = []
    y_train_list = []
    handedness_train_list = []
    movements_train_list = []
    
    X_val_list = []
    y_val_list = []
    handedness_val_list = []
    movements_val_list = []
    
    feature_names = None
    
    for movement_name, loaders in dataloader_dict.items():
        print(f"\nProcessing movement: {movement_name}")
        
        # Train data
        X_train, y_train, h_train, m_train, names = dataloader_to_sklearn(
            loaders["train"], binary_classification
        )
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        handedness_train_list.append(h_train)
        movements_train_list.append(m_train)
        
        # Val data
        X_val, y_val, h_val, m_val, names = dataloader_to_sklearn(
            loaders["val"], binary_classification
        )
        X_val_list.append(X_val)
        y_val_list.append(y_val)
        handedness_val_list.append(h_val)
        movements_val_list.append(m_val)
        
        if feature_names is None:
            feature_names = names
    
    # Concatenate all movements
    train_data = {
        'X': np.vstack(X_train_list),
        'y': np.concatenate(y_train_list),
        'handedness': np.concatenate(handedness_train_list),
        'movements': np.concatenate(movements_train_list)
    }
    
    val_data = {
        'X': np.vstack(X_val_list),
        'y': np.concatenate(y_val_list),
        'handedness': np.concatenate(handedness_val_list),
        'movements': np.concatenate(movements_val_list)
    }
    
    print(f"\n=== FINAL DATASET ===")
    print(f"Train: {train_data['X'].shape[0]} samples")
    print(f"Val: {val_data['X'].shape[0]} samples")
    print(f"Features: {train_data['X'].shape[1]}")
    
    return train_data, val_data, feature_names


# ============================================================================
# PART 3: Training Classical ML Models
# ============================================================================

def train_classical_ml(train_data, val_data, feature_names):
    """
    Train multiple classical ML models and compare results.
    
    Args:
        train_data: dict with X, y, handedness, movements
        val_data: dict with X, y, handedness, movements
        feature_names: list of feature names
    
    Returns:
        results: dict of model performances
    """
    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    results = {}
    
    # ===== 1. Random Forest =====
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf.fit(X_train_scaled, y_train)
    
    y_pred_rf = rf.predict(X_val_scaled)
    acc_rf = balanced_accuracy_score(y_val, y_pred_rf)
    
    print(f"\nRandom Forest Results:")
    print(f"Balanced Accuracy: {acc_rf:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_rf, target_names=['Healthy', 'Parkinson']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred_rf))
    
    # Feature importance
    feature_importance = rf.feature_importances_
    top_indices = np.argsort(feature_importance)[-20:][::-1]
    print("\nTop 20 Most Important Features:")
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    results['random_forest'] = {
        'model': rf,
        'accuracy': acc_rf,
        'predictions': y_pred_rf
    }
    
    # ===== 2. XGBoost =====
    print("\n" + "="*50)
    print("Training XGBoost...")
    print("="*50)
    
    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    xgb_model.fit(X_train_scaled, y_train)
    
    y_pred_xgb = xgb_model.predict(X_val_scaled)
    acc_xgb = balanced_accuracy_score(y_val, y_pred_xgb)
    
    print(f"\nXGBoost Results:")
    print(f"Balanced Accuracy: {acc_xgb:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_xgb, target_names=['Healthy', 'Parkinson']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred_xgb))
    
    results['xgboost'] = {
        'model': xgb_model,
        'accuracy': acc_xgb,
        'predictions': y_pred_xgb
    }
    
    # ===== 3. Gradient Boosting =====
    print("\n" + "="*50)
    print("Training Gradient Boosting...")
    print("="*50)
    
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42
    )
    
    gb.fit(X_train_scaled, y_train)
    
    y_pred_gb = gb.predict(X_val_scaled)
    acc_gb = balanced_accuracy_score(y_val, y_pred_gb)
    
    print(f"\nGradient Boosting Results:")
    print(f"Balanced Accuracy: {acc_gb:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_gb, target_names=['Healthy', 'Parkinson']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred_gb))
    
    results['gradient_boosting'] = {
        'model': gb,
        'accuracy': acc_gb,
        'predictions': y_pred_gb
    }
    
    # ===== Summary =====
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Random Forest:      {results['random_forest']['accuracy']:.4f}")
    print(f"XGBoost:            {results['xgboost']['accuracy']:.4f}")
    print(f"Gradient Boosting:  {results['gradient_boosting']['accuracy']:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest Model: {best_model[0].upper()} ({best_model[1]['accuracy']:.4f})")
    
    return results, scaler


# ============================================================================
# PART 4: Usage Example
# ============================================================================

def main_example(your_dataloader_dict):
    """
    Example usage with your per-movement dataloader dictionary.
    
    Args:
        your_dataloader_dict: dict of {movement_name: {"train": DataLoader, "val": DataLoader}}
    """
    
    # Convert dataloaders to sklearn format
    train_data, val_data, feature_names = dataloader_dict_to_sklearn(
        your_dataloader_dict,
        binary_classification=True
    )
    
    # Train classical ML models
    results, scaler = train_classical_ml(train_data, val_data, feature_names)
    
    # Save best model (optional)
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_model_name]['model']
    
    import joblib
    joblib.dump(best_model, f'best_model_{best_model_name}.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    print(f"\nSaved best model: {best_model_name}")
    
    return results


# ============================================================================
# If you have single train/val dataloaders (not per-movement):
# ============================================================================

def main_example_single_dataloader(train_dataloader, val_dataloader):
    """
    Example usage with single train/val dataloaders.
    
    Args:
        train_dataloader: PyTorch DataLoader for training
        val_dataloader: PyTorch DataLoader for validation
    """
    
    # Convert to sklearn format
    X_train, y_train, h_train, m_train, feature_names = dataloader_to_sklearn(
        train_dataloader, binary_classification=True
    )
    
    X_val, y_val, h_val, m_val, _ = dataloader_to_sklearn(
        val_dataloader, binary_classification=True
    )
    
    train_data = {
        'X': X_train,
        'y': y_train,
        'handedness': h_train,
        'movements': m_train
    }
    
    val_data = {
        'X': X_val,
        'y': y_val,
        'handedness': h_val,
        'movements': m_val
    }
    
    # Train models
    results, scaler = train_classical_ml(train_data, val_data, feature_names)
    
    return results