import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# NEW: Feature extraction library
import pycatch22
from tsfresh.feature_extraction import extract_features


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


def _select_more_affected_hand(left_data, right_data):
    """
    Detect which hand has larger tremor amplitude (like paper).
    Paper: "detect the more-affected hand using AUC as proxy for tremor power"
    
    Args:
        left_data: shape (T, num_channels)
        right_data: shape (T, num_channels)
    
    Returns:
        more_affected_data: the hand with larger tremor amplitude
    """
    # Calculate AUC (Area Under Curve) = sum of absolute values
    left_auc = np.sum(np.abs(left_data))
    right_auc = np.sum(np.abs(right_data))
    
    # Return the hand with larger tremor power
    if left_auc > right_auc:
        return left_data
    else:
        return right_data


# ------------------------
# NEW: Feature Extraction
# ------------------------

def _extract_features_from_segment(segment):
    """
    Extract time-series features from a segment using catch22.
    
    Args:
        segment: numpy array of shape (window_size, num_channels)
        
    Returns:
        features: 1D array of all features concatenated
    """
    all_features = []
    
    # Extract features from EACH channel
    for channel_idx in range(segment.shape[1]):
        channel_signal = segment[:, channel_idx]
        
        # catch22 returns 22 features per channel
        features = pycatch22.catch22_all(channel_signal)['values']
        all_features.extend(features)
    
    return np.array(all_features, dtype=np.float32)

def _extract_features_from_segment(segment):
    # tsfresh extracts ~790 features
    df = pd.DataFrame(segment)
    features = extract_features(df, column_id=0, column_sort=0)
    return features.values.flatten()

# ------------------------
# Keep existing preprocessing utilities
# ------------------------

def _remove_timestamp_column(data):
    """Remove timestamp column if present (column 0)."""
    if data.shape[1] == 7:
        return data[:, 1:]
    return data

def _handle_missing_values(data):
    """Handle missing values by forward fill then backward fill."""
    df = pd.DataFrame(data)
    df = df.fillna(method='ffill').fillna(method='bfill')
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
    output_dir: Path = Path("../../../project_datasets/tremor/movements"),
    window_size: int = 1024,
    overlap: float = 0.5,
    include_other: bool = False,
    
    extract_features: bool = True,  # NEW: Toggle feature extraction
    use_vector_magnitude: bool = True,  # NEW: Use magnitude instead of X,Y,Z
    use_more_affected_hand: bool = False  # NEW: Select hand with larger tremor (not better in healthy vs pd)
):
    """
    Create dataset with FEATURE EXTRACTION (like the paper).
    
    Args:
        extract_features: If True, saves features instead of raw signals
        use_vector_magnitude: If True, converts (X,Y,Z) to single magnitude
        use_more_affected_hand: If True, uses only the hand with larger tremor
    """

    TIME_SERIES_DIR = root_dir / time_series_subdir
    FILE_LIST = root_dir / file_list_subdir
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load labels
    print("Loading labels...")
    labels_df = pd.read_csv(FILE_LIST)
    id_to_label = dict(zip(labels_df['id'], labels_df['label']))
    id_to_handedness = {
        row['id']: 0 if row['handedness'].lower() == 'left' else 1
        for _, row in labels_df.iterrows()
    }

    # 2. Collect files
    print("Collecting movement files...")
    movement_files = sorted(TIME_SERIES_DIR.glob("*.txt"))
    
    def parse_filename(fname: str):
        stem = Path(fname).stem
        subject_id, movement_full = stem.split("_", 1)
        wrist = 0 if "Left" in movement_full else 1
        movement_name = movement_full.replace("_LeftWrist", "").replace("_RightWrist", "")
        return int(subject_id), movement_name, wrist

    # 3. Group files
    grouped_files = {}
    for f in movement_files:
        sid, mv, wrist = parse_filename(f)
        grouped_files.setdefault((sid, mv), {})[wrist] = f

    total_segments = 0
    skipped_recordings = 0
    
    # 4. Process paired recordings
    for (subject_id, movement_name), wrist_files in tqdm(grouped_files.items(), desc="Processing"):

        if 0 not in wrist_files or 1 not in wrist_files:
            skipped_recordings += 1
            continue

        # 4.1. Load data
        try:
            left_data = np.loadtxt(wrist_files[0], delimiter=',', dtype=np.float32)
            right_data = np.loadtxt(wrist_files[1], delimiter=',', dtype=np.float32)
        except Exception as e:
            skipped_recordings += 1
            continue

        # 4.2. Preprocess
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

        # 4.3. Handle missing values
        left_data = _handle_missing_values(left_data)
        right_data = _handle_missing_values(right_data)

        # NEW: Apply paper's preprocessing
        # ---------------------------------
        
        # 5. Convert to vector magnitude (optional, like paper)
        if use_vector_magnitude:
            # Use only accelerometer channels (first 3)
            left_data = _compute_vector_magnitude(left_data[:, :3])
            right_data = _compute_vector_magnitude(right_data[:, :3])
            # Now: (T, 1) instead of (T, 3)
        
        # 5.1. Select more affected hand (optional, like paper)
        if use_more_affected_hand:
            # Use single hand with larger tremor
            selected_data = _select_more_affected_hand(left_data, right_data)
            # Segment only this hand
            segments = _segment_signal(selected_data, window_size, overlap)
        else:
            # Use both hands separately (original behavior)
            left_segments = _segment_signal(left_data, window_size, overlap)
            right_segments = _segment_signal(right_data, window_size, overlap)

        # 6. Create output directory
        label_name = {0: "Healthy", 1: "Parkinson", 2: "Other"}.get(label, "Unknown")
        out_dir = OUTPUT_DIR / movement_name / label_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 7. NEW: Process and save each segment
        if use_more_affected_hand:
            # Single hand case (like paper)
            for seg_idx, segment in enumerate(segments):
                
                if extract_features:
                    # 7.1. Extract features from single hand
                    features = _extract_features_from_segment(segment)
                    
                    save_dict = {
                        'features': features,
                        'label': label,
                        'handedness': handedness,
                        'subject_id': subject_id,
                        'movement_name': movement_name,
                        'segment_idx': seg_idx
                    }
                else:
                    # 7.2. Save raw signal from single hand
                    save_dict = {
                        'signal': segment.astype(np.float32),
                        'label': label,
                        'handedness': handedness,
                        'subject_id': subject_id,
                        'movement_name': movement_name,
                        'segment_idx': seg_idx
                    }
                
                filename = f"{subject_id}_seg{seg_idx:03d}.npz"
                np.savez_compressed(out_dir / filename, **save_dict)
                total_segments += 1
        
        else:
            # 7.3. Both hands case (original behavior)
            for seg_idx, (left_seg, right_seg) in enumerate(zip(left_segments, right_segments)):
                
                if extract_features:
                    # Extract features from both hands
                    left_features = _extract_features_from_segment(left_seg)
                    right_features = _extract_features_from_segment(right_seg)
                    
                    # NEW: Asymmetry features
                    """
                    Healthy = low amplitude + low asymmetry
                    PD = high amplitude + high asymmetry
                    """
                    asymmetry_features = np.abs(left_features - right_features)
                    
                    combined_features = np.concatenate([left_features, right_features, asymmetry_features])
                    
                    save_dict = {
                        'features': combined_features,
                        'label': label,
                        'handedness': handedness,
                        'subject_id': subject_id,
                        'movement_name': movement_name,
                        'segment_idx': seg_idx
                    }
                else:
                    # 7.4. Save raw signals from both hands
                    save_dict = {
                        'signal': (left_seg.astype(np.float32), right_seg.astype(np.float32)),
                        'label': label,
                        'handedness': handedness,
                        'subject_id': subject_id,
                        'movement_name': movement_name,
                        'segment_idx': seg_idx
                    }
                
                filename = f"{subject_id}_seg{seg_idx:03d}.npz"
                np.savez_compressed(out_dir / filename, **save_dict)
                total_segments += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Dataset creation complete!")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Total segments: {total_segments}")
    print(f"Mode: {'FEATURES' if extract_features else 'RAW SIGNALS'}")
    print(f"Vector magnitude: {use_vector_magnitude}")
    print(f"More-affected hand: {use_more_affected_hand}")
    print(f"{'='*60}")