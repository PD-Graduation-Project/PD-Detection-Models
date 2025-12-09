import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import signal, stats
from scipy.fft import fft, fftfreq


def extract_statistical_features(window, prefix=""):
    """Extract statistical features from a time window."""
    features = {}
    
    # Basic statistics
    features[f'{prefix}_mean'] = np.mean(window)
    features[f'{prefix}_std'] = np.std(window)
    features[f'{prefix}_var'] = np.var(window)
    features[f'{prefix}_min'] = np.min(window)
    features[f'{prefix}_max'] = np.max(window)
    features[f'{prefix}_range'] = np.ptp(window)  # peak-to-peak
    features[f'{prefix}_median'] = np.median(window)
    features[f'{prefix}_mad'] = np.median(np.abs(window - np.median(window)))  # median absolute deviation
    
    # Higher-order moments
    features[f'{prefix}_skewness'] = stats.skew(window)
    features[f'{prefix}_kurtosis'] = stats.kurtosis(window)
    
    # Energy and power
    features[f'{prefix}_rms'] = np.sqrt(np.mean(window**2))
    features[f'{prefix}_energy'] = np.sum(window**2)
    
    # Percentiles
    features[f'{prefix}_q25'] = np.percentile(window, 25)
    features[f'{prefix}_q75'] = np.percentile(window, 75)
    features[f'{prefix}_iqr'] = features[f'{prefix}_q75'] - features[f'{prefix}_q25']
    
    # Zero crossing rate
    features[f'{prefix}_zcr'] = np.sum(np.diff(np.sign(window)) != 0) / len(window)
    
    return features


def extract_frequency_features(window, fs=102.4, prefix=""):
    """Extract frequency-domain features from a time window."""
    features = {}
    
    # Compute FFT
    n = len(window)
    yf = fft(window)
    xf = fftfreq(n, 1/fs)[:n//2]
    power = np.abs(yf[:n//2])**2
    
    # Normalize power
    total_power = np.sum(power)
    if total_power > 0:
        norm_power = power / total_power
    else:
        norm_power = power
    
    # Frequency bands relevant to tremor (Hz)
    # Rest tremor: 4-6 Hz
    # Postural tremor: 4-9 Hz
    # Action tremor: 3-12 Hz
    bands = {
        'delta': (0.5, 3),
        'tremor_low': (3, 6),
        'tremor_mid': (6, 9),
        'tremor_high': (9, 12),
        'beta': (12, 30),
        'gamma': (30, 50)
    }
    
    for band_name, (low, high) in bands.items():
        idx = np.where((xf >= low) & (xf < high))[0]
        if len(idx) > 0:
            features[f'{prefix}_power_{band_name}'] = np.sum(power[idx])
            features[f'{prefix}_power_ratio_{band_name}'] = np.sum(norm_power[idx])
        else:
            features[f'{prefix}_power_{band_name}'] = 0
            features[f'{prefix}_power_ratio_{band_name}'] = 0
    
    # Dominant frequency
    if len(power) > 0:
        dom_idx = np.argmax(power)
        features[f'{prefix}_dominant_freq'] = xf[dom_idx]
        features[f'{prefix}_dominant_power'] = power[dom_idx]
    else:
        features[f'{prefix}_dominant_freq'] = 0
        features[f'{prefix}_dominant_power'] = 0
    
    # Spectral features
    if total_power > 0:
        features[f'{prefix}_spectral_centroid'] = np.sum(xf * power) / total_power
        features[f'{prefix}_spectral_spread'] = np.sqrt(np.sum(((xf - features[f'{prefix}_spectral_centroid'])**2) * power) / total_power)
        features[f'{prefix}_spectral_entropy'] = -np.sum(norm_power * np.log2(norm_power + 1e-10))
    else:
        features[f'{prefix}_spectral_centroid'] = 0
        features[f'{prefix}_spectral_spread'] = 0
        features[f'{prefix}_spectral_entropy'] = 0
    
    # Spectral rolloff (frequency below which 85% of energy is contained)
    cumsum = np.cumsum(norm_power)
    rolloff_idx = np.where(cumsum >= 0.85)[0]
    if len(rolloff_idx) > 0:
        features[f'{prefix}_spectral_rolloff'] = xf[rolloff_idx[0]]
    else:
        features[f'{prefix}_spectral_rolloff'] = xf[-1] if len(xf) > 0 else 0
    
    return features


def extract_temporal_features(window, prefix=""):
    """Extract temporal/time-domain features."""
    features = {}
    
    # Signal complexity
    features[f'{prefix}_entropy'] = -np.sum((window**2) * np.log(np.abs(window) + 1e-10))
    
    # Autocorrelation at lag 1
    if len(window) > 1:
        features[f'{prefix}_autocorr_lag1'] = np.corrcoef(window[:-1], window[1:])[0, 1]
    else:
        features[f'{prefix}_autocorr_lag1'] = 0
    
    # First and second derivatives (velocity and acceleration)
    velocity = np.diff(window)
    if len(velocity) > 0:
        features[f'{prefix}_velocity_mean'] = np.mean(velocity)
        features[f'{prefix}_velocity_std'] = np.std(velocity)
        features[f'{prefix}_velocity_max'] = np.max(np.abs(velocity))
    else:
        features[f'{prefix}_velocity_mean'] = 0
        features[f'{prefix}_velocity_std'] = 0
        features[f'{prefix}_velocity_max'] = 0
    
    if len(velocity) > 1:
        acceleration = np.diff(velocity)
        features[f'{prefix}_acceleration_mean'] = np.mean(acceleration)
        features[f'{prefix}_acceleration_std'] = np.std(acceleration)
        features[f'{prefix}_acceleration_max'] = np.max(np.abs(acceleration))
    else:
        features[f'{prefix}_acceleration_mean'] = 0
        features[f'{prefix}_acceleration_std'] = 0
        features[f'{prefix}_acceleration_max'] = 0
    
    # Peak detection
    peaks, _ = signal.find_peaks(window)
    features[f'{prefix}_num_peaks'] = len(peaks)
    features[f'{prefix}_peak_rate'] = len(peaks) / (len(window) / 102.4)  # peaks per second
    
    return features


def extract_bimanual_features(left_window, right_window, prefix=""):
    """Extract features comparing left and right hand signals."""
    features = {}
    
    # Correlation between hands
    if len(left_window) == len(right_window) and len(left_window) > 1:
        features[f'{prefix}_lr_correlation'] = np.corrcoef(left_window, right_window)[0, 1]
    else:
        features[f'{prefix}_lr_correlation'] = 0
    
    # Difference statistics
    diff = left_window - right_window
    features[f'{prefix}_lr_diff_mean'] = np.mean(diff)
    features[f'{prefix}_lr_diff_std'] = np.std(diff)
    features[f'{prefix}_lr_diff_max'] = np.max(np.abs(diff))
    
    # Ratio of energies
    left_energy = np.sum(left_window**2)
    right_energy = np.sum(right_window**2)
    if right_energy > 0:
        features[f'{prefix}_lr_energy_ratio'] = left_energy / right_energy
    else:
        features[f'{prefix}_lr_energy_ratio'] = 0
    
    # Asymmetry index
    total_energy = left_energy + right_energy
    if total_energy > 0:
        features[f'{prefix}_asymmetry_index'] = (left_energy - right_energy) / total_energy
    else:
        features[f'{prefix}_asymmetry_index'] = 0
    
    return features


def process_npz_file(npz_path, fs=102.4):
    """Process a single .npz file and extract all features per second."""
    
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    left_signal, right_signal = data['signal']  # (1024, 6) each
    label = int(data['label'])
    handedness = int(data['wrist'])
    subject_id = int(data['subject_id'])
    metadata = data['metadata'].item()
    
    # Determine window size (1 second = 102.4 samples, rounded to 102)
    window_size = int(fs)
    num_windows = 10  # We have ~10 seconds of data
    
    all_features = []
    
    for sec in range(num_windows):
        start_idx = sec * window_size
        end_idx = min(start_idx + window_size, len(left_signal))
        
        if end_idx - start_idx < window_size // 2:  # Skip if less than half window
            continue
        
        features = {
            'subject_id': subject_id,
            'label': label,
            'label_name': {0: 'Healthy', 1: 'Parkinson', 2: 'Other'}[label],
            'handedness': handedness,
            'handedness_name': {0: 'Left', 1: 'Right'}[handedness],
            'second': sec,
            'movement': npz_path.parent.parent.name,
        }
        
        # Add metadata
        for key, val in metadata.items():
            features[f'meta_{key}'] = val
        
        # Extract features for each channel
        channels = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        for hand in ['left', 'right']:
            signal_data = left_signal if hand == 'left' else right_signal
            window = signal_data[start_idx:end_idx]
            
            # Per-channel features
            for ch_idx, ch_name in enumerate(channels):
                ch_window = window[:, ch_idx]
                prefix = f'{hand}_{ch_name}'
                
                features.update(extract_statistical_features(ch_window, prefix))
                features.update(extract_frequency_features(ch_window, fs, prefix))
                features.update(extract_temporal_features(ch_window, prefix))
            
            # Magnitude features (3D acceleration and gyro magnitudes)
            accel_mag = np.linalg.norm(window[:, :3], axis=1)
            gyro_mag = np.linalg.norm(window[:, 3:], axis=1)
            
            features.update(extract_statistical_features(accel_mag, f'{hand}_accel_mag'))
            features.update(extract_frequency_features(accel_mag, fs, f'{hand}_accel_mag'))
            features.update(extract_temporal_features(accel_mag, f'{hand}_accel_mag'))
            
            features.update(extract_statistical_features(gyro_mag, f'{hand}_gyro_mag'))
            features.update(extract_frequency_features(gyro_mag, fs, f'{hand}_gyro_mag'))
            features.update(extract_temporal_features(gyro_mag, f'{hand}_gyro_mag'))
        
        # Bimanual features (comparing left and right)
        for ch_idx, ch_name in enumerate(channels):
            left_ch = left_signal[start_idx:end_idx, ch_idx]
            right_ch = right_signal[start_idx:end_idx, ch_idx]
            features.update(extract_bimanual_features(left_ch, right_ch, f'bimanual_{ch_name}'))
        
        # Bimanual magnitude features
        left_accel_mag = np.linalg.norm(left_signal[start_idx:end_idx, :3], axis=1)
        right_accel_mag = np.linalg.norm(right_signal[start_idx:end_idx, :3], axis=1)
        left_gyro_mag = np.linalg.norm(left_signal[start_idx:end_idx, 3:], axis=1)
        right_gyro_mag = np.linalg.norm(right_signal[start_idx:end_idx, 3:], axis=1)
        
        features.update(extract_bimanual_features(left_accel_mag, right_accel_mag, 'bimanual_accel_mag'))
        features.update(extract_bimanual_features(left_gyro_mag, right_gyro_mag, 'bimanual_gyro_mag'))
        
        all_features.append(features)
    
    return all_features


def create_feature_csv(
    preprocessed_dir: Path = Path("../../../project_datasets/tremor/"),
    output_csv: Path = Path("../../../project_datasets/tremor/features_flattened.csv"),
    movements: list = None,
    save_every: int = 1  # Save after every N npz files (1 = save after each npz file)
):
    """
    Create a FLATTENED CSV file where each row is one subject-movement pair,
    with features from all 10 seconds as separate columns (e.g., feature_sec0, feature_sec1, ...).
    
    Note: Each npz file represents ONE subject doing ONE movement (e.g., subject 001 doing CrossArms).
    The script saves after processing every 'save_every' npz files.
    
    Parameters
    ----------
    preprocessed_dir : Path
        Directory containing the preprocessed .npz files organized by movement/label/
    output_csv : Path
        Path to save the output CSV file
    movements : list, optional
        List of specific movements to process. If None, process all movements.
    save_every : int, optional
        Save CSV after processing every N npz files (default=1, save after each npz file)
    """
    
    all_subject_rows = []
    npz_counter = 0  # Counter for NPZ files processed
    
    # Find all .npz files
    movement_dirs = sorted([d for d in preprocessed_dir.iterdir() if d.is_dir()])
    
    if movements:
        movement_dirs = [d for d in movement_dirs if d.name in movements]
    
    # Create output directory if it doesn't exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    for movement_dir in movement_dirs:
        for label_dir in movement_dir.iterdir():
            if not label_dir.is_dir():
                continue
            
            npz_files = list(label_dir.glob("*.npz"))
            print (f" Processing -> {movement_dir.name}/{label_dir.name} ...")
            
            for npz_file in tqdm(npz_files, desc=f"  {movement_dir.name}/{label_dir.name}", leave=False):
                try:
                    # Get all features per second for this subject
                    features_per_second = process_npz_file(npz_file)
                    
                    if len(features_per_second) == 0:
                        continue
                    
                    # Create flattened row
                    flattened_row = {}
                    
                    # Add metadata (same for all seconds, take from first)
                    first_sec = features_per_second[0]
                    flattened_row['subject_id'] = first_sec['subject_id']
                    flattened_row['label'] = first_sec['label']
                    flattened_row['label_name'] = first_sec['label_name']
                    flattened_row['handedness'] = first_sec['handedness']
                    flattened_row['handedness_name'] = first_sec['handedness_name']
                    flattened_row['movement'] = first_sec['movement']
                    
                    # Add metadata fields
                    for key in first_sec.keys():
                        if key.startswith('meta_'):
                            flattened_row[key] = first_sec[key]
                    
                    # Flatten features across seconds
                    for sec_data in features_per_second:
                        sec = sec_data['second']
                        for key, value in sec_data.items():
                            # Skip metadata and identifiers
                            if key in ['subject_id', 'label', 'label_name', 'handedness', 
                                      'handedness_name', 'movement', 'second'] or key.startswith('meta_'):
                                continue
                            
                            # Add feature with second suffix
                            flattened_row[f'{key}_sec{sec}'] = value
                    
                    all_subject_rows.append(flattened_row)
                    npz_counter += 1  # Increment NPZ counter
                    
                    # Save CSV after every 'save_every' NPZ files
                    if npz_counter % save_every == 0:
                        df_temp = pd.DataFrame(all_subject_rows)
                        
                        # Reorder columns: metadata first, then features
                        meta_cols = ['subject_id', 'label', 'label_name', 'handedness', 'handedness_name', 'movement']
                        meta_cols += [c for c in df_temp.columns if c.startswith('meta_')]
                        feature_cols = sorted([c for c in df_temp.columns if c not in meta_cols])
                        df_temp = df_temp[meta_cols + feature_cols]
                        
                        # Save to CSV
                        df_temp.to_csv(output_csv, index=False)
                        print(f"\n  ✓ Checkpoint saved: {npz_counter} npz files / {len(df_temp)} subject-movement pairs processed")
                    
                except Exception as e:
                    print(f"Error processing {npz_file}: {e}")
                    continue
    
    # Final save (if not already saved in last iteration)
    df = pd.DataFrame(all_subject_rows)
    
    # Reorder columns: metadata first, then features
    meta_cols = ['subject_id', 'label', 'label_name', 'handedness', 'handedness_name', 'movement']
    meta_cols += [c for c in df.columns if c.startswith('meta_')]
    feature_cols = sorted([c for c in df.columns if c not in meta_cols])
    df = df[meta_cols + feature_cols]
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*70}")
    print(f"Feature extraction complete (FLATTENED)!")
    print(f"{'='*70}")
    print(f"Total npz files processed: {npz_counter}")
    print(f"Total rows (one per subject-movement): {len(df)}")
    print(f"Total features per row: {len(df.columns)}")
    print(f"Output saved to: {output_csv.resolve()}")
    print(f"\nFeature breakdown:")
    print(f"  - Metadata columns: {len(meta_cols)}")
    print(f"  - Feature columns (across all seconds): {len(feature_cols)}")
    print(f"\nLabel distribution:")
    print(df['label_name'].value_counts())
    print(f"\nMovement distribution:")
    print(df['movement'].value_counts())
    print(f"\nExample feature columns (first 10):")
    print(feature_cols[:10])
    
    return df


if __name__ == "__main__":
    # Example usage
    df = create_feature_csv(
        preprocessed_dir=Path("../../../project_datasets/tremor/"),
        output_csv=Path("../../../project_datasets/tremor/features_flattened.csv"),
        movements=None,  # Process all movements, or specify list like ['CrossArms', 'DrawSpiral']
        save_every=1  # Save after each file (change to 10 to save every 10 files, etc.)
    )
    
    print(f"\nFirst few columns: {list(df.columns[:20])}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nSample of flattened features:")
    sample_cols = [c for c in df.columns if 'left_accel_x_mean' in c][:5]
    print(f"Example: {sample_cols}")
    if len(sample_cols) > 0:
        print(df[['subject_id', 'movement'] + sample_cols].head(2))