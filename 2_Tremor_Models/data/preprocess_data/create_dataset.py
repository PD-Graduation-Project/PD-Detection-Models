import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import stft

# NEW: Feature extraction library
import pycatch22

# ------------------------
# NEW: Preprocessing (from paper)
# ------------------------

def _compute_vector_magnitude(data):
    """
    Convert tri-axial accelerometer (X, Y, Z) to vector magnitude.
    Paper: "vector amplitude sum across the three available axes"
    
    Args:
        data: shape (T, 3) - accelerometer X, Y, Z
    
    Returns:
        magnitude: shape (T, 1) - single channel
    """
    # Vector magnitude = sqrt(x² + y² + z²)
    magnitude = np.sqrt(np.sum(data**2, axis=1, keepdims=True))
    return magnitude

# ------------------------
# NEW: Compute STFT
# ------------------------
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


# ------------------------
# NEW: Feature Extraction
# ------------------------

def _extract_features_from_segment(segment, fs: float = 100.0, tremor_band=(3.0, 7.0)):
    """
    Extract time-series features from a segment using catch22.
    then append STFT tremor-band feature as the last feature.
    
    Args:
        segment: numpy array of shape (window_size, num_channels)
        
    Returns:
        features: 1D array of all features concatenated
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
            fs=fs,
            tremor_band=tremor_band
        )
    all_features.extend([stft_ratio, stft_stability, peak_freq])
    
    return np.array(all_features, dtype=np.float32)

# ------------------------
#  preprocessing utilities
# ------------------------
def _remove_timestamp_column(data):
    """Remove timestamp column if present (column 0)."""
    if data.shape[1] == 7 or data.shape[1] == 4: # 7 -> 1 time, 3 accel, 3 gyro || 4 -> 1 time, 3 accel
        return data[:, 1:]
    return data

def _handle_missing_values(data):
    """Handle missing values by forward fill then backward fill."""
    df = pd.DataFrame(data)
    df = df.ffill().bfill()
    return df.values

def _segment_signal(data, window_size=1024, overlap=0.5):
    """Segment long signals into windows with overlap."""
    step_size = int(window_size * (1 - overlap))
    segments = []
    
    for start_idx in range(0, len(data) - window_size + 1, step_size):
        segment = data[start_idx:start_idx + window_size]
        segments.append(segment)
    
    if len(segments) == 0 and len(data) > 0:
        if len(data) < window_size:
            pad_length = window_size - len(data)
            segment = np.pad(data, ((0, pad_length), (0, 0)), mode='edge')
            segments.append(segment)
        else:
            segments.append(data[:window_size])
    
    return segments


# ------------------------
# MODIFIED: Main dataset creation function
# ------------------------

def create_clean_dataset(
    root_dir: Path = Path("../../../project_datasets/tremor/Tremor_dataset"),
    time_series_subdir: str = "movement/timeseries",
    file_list_subdir: str = "preprocessed/file_list.csv",
    
    output_csv: Path = Path("../../../project_datasets/tremor/tremor_features.csv"),
    
    window_size: int = 1024,
    overlap: float = 0.5,
    
    include_other: bool = False, 
    sampling_rate: float = 100.0,         
    pd_tremor_band=(3.0, 7.0),            
    ):
    
    TIME_SERIES_DIR = root_dir / time_series_subdir
    FILE_LIST = root_dir / file_list_subdir
    OUTPUT_CSV = Path(output_csv)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load labels
    print("Loading labels...")
    labels_df = pd.read_csv(FILE_LIST)
    id_to_label = dict(zip(labels_df['id'], labels_df['label']))
    id_to_handedness = {
        row['id']: 0 if row['handedness'].lower() == 'left' else 1
        for _, row in labels_df.iterrows()
    }

    # 2. Collect and group files
    print("Collecting movement files...")
    movement_files = sorted(TIME_SERIES_DIR.glob("*.txt"))
    
    def parse_filename(fname: str):
        stem = Path(fname).stem
        subject_id, movement_full = stem.split("_", 1)
        wrist = 0 if "Left" in movement_full else 1
        movement_name = movement_full.replace("_LeftWrist", "").replace("_RightWrist", "")
        return int(subject_id), movement_name, wrist

    grouped_files = {}
    for f in movement_files:
        sid, mv, wrist = parse_filename(f)
        grouped_files.setdefault((sid, mv), {})[wrist] = f

    # Stable movement encoding (0..N-1) for downstream tabular models.
    movement_names = sorted({mv for (_, mv) in grouped_files.keys()})
    movement_map = {name: idx for idx, name in enumerate(movement_names)}

    all_rows = []
    num_left_features = None
    num_right_features = None

    total_segments = 0
    skipped_recordings = 0
    
    # 3. Process paired recordings
    for (subject_id, movement_name), wrist_files in tqdm(grouped_files.items(), desc="Processing"):

        # Require both wrists
        if 0 not in wrist_files or 1 not in wrist_files:
            skipped_recordings += 1
            continue

        # Load data
        try:
            left_data = np.loadtxt(wrist_files[0], delimiter=',', dtype=np.float32)
            right_data = np.loadtxt(wrist_files[1], delimiter=',', dtype=np.float32)
        except Exception:
            skipped_recordings += 1
            continue

        # Basic preprocessing
        left_data = _remove_timestamp_column(left_data)
        right_data = _remove_timestamp_column(right_data)
        
        label = id_to_label.get(subject_id)
        handedness = id_to_handedness.get(subject_id)
        if label is None or handedness is None:
            skipped_recordings += 1
            continue

        if label == 2 and not include_other:
            skipped_recordings += 1
            continue

        # Handle missing values
        left_data = _handle_missing_values(left_data)
        right_data = _handle_missing_values(right_data)

        # Convert accelerometer (X,Y,Z) → vector magnitude
        left_data = _compute_vector_magnitude(left_data[:, :3])   # (T, 1)
        right_data = _compute_vector_magnitude(right_data[:, :3])  # (T, 1)

        # Segment both hands
        left_segments = _segment_signal(left_data, window_size, overlap)
        right_segments = _segment_signal(right_data, window_size, overlap)

        # Extract features and add to CSV rows
        for seg_idx, (left_seg, right_seg) in enumerate(zip(left_segments, right_segments)):

            left_features = _extract_features_from_segment(
                left_seg, fs=sampling_rate, tremor_band=pd_tremor_band
            )
            right_features = _extract_features_from_segment(
                right_seg, fs=sampling_rate, tremor_band=pd_tremor_band
            )
            
            if num_left_features is None:
                num_left_features = len(left_features)
            if num_right_features is None:
                num_right_features = len(right_features)

            if len(left_features) != num_left_features or len(right_features) != num_right_features:
                skipped_recordings += 1
                continue

            row = {
                "movement": movement_map[movement_name],
                "handedness": int(handedness),
                "label": int(label),
                "segment_idx": int(seg_idx),
            }

            for i, value in enumerate(left_features, start=1):
                row[f"lh_{i}"] = float(value)

            for i, value in enumerate(right_features, start=1):
                row[f"rh_{i}"] = float(value)

            all_rows.append(row)
            total_segments += 1

    print("\nCreating CSV...")
    df = pd.DataFrame(all_rows)

    if df.empty:
        metadata_cols = ["movement", "handedness", "label", "segment_idx"]
        df = pd.DataFrame(columns=metadata_cols)
    else:
        metadata_cols = ["movement", "handedness", "label", "segment_idx"]
        left_cols = [f"lh_{i}" for i in range(1, num_left_features + 1)]
        right_cols = [f"rh_{i}" for i in range(1, num_right_features + 1)]
        df = df[metadata_cols + left_cols + right_cols]

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"Dataset creation complete!")
    print(f"CSV saved: {OUTPUT_CSV.resolve()}")
    print(f"Total segments: {total_segments}")
    print(f"Skipped (others): {skipped_recordings}")
    print("Movement mapping:")
    for name, idx in movement_map.items():
        print(f"  {idx}: {name}")
    print(f"{'='*60}")

    return df