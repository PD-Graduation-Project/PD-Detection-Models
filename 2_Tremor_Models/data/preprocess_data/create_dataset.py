import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# NEW: Feature extraction library
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters, ComprehensiveFCParameters


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
    Extract time-series features from a segment using tsfresh.
    
    Args:
        segment: numpy array of shape (window_size, num_channels)
        
    Returns:
        features: 1D array of all features concatenated (~780 per channel)
    
    Feature settings options:
        - MinimalFCParameters:      ~10 features per channel  (fastest)
        - EfficientFCParameters:    ~780 features per channel (recommended)
        - ComprehensiveFCParameters: ~4000 features per channel (slowest)
    """
    all_features = []
    
    for channel_idx in range(segment.shape[1]):
        channel_signal = segment[:, channel_idx]
        
        # tsfresh requires a specific DataFrame format:
        # - 'id'    : which time series (we only have 1 segment = id 0)
        # - 'time'  : time index
        # - 'value' : signal values
        df = pd.DataFrame({
            'id':    0,
            'time':  np.arange(len(channel_signal)),
            'value': channel_signal
        })
        
        # Extract features (~780 features per channel with EfficientFCParameters)
        features_df = extract_features(
            df,
            column_id='id',
            column_sort='time',
            column_value='value',
            default_fc_parameters=EfficientFCParameters(),
            disable_progressbar=True  # Suppress per-channel progress bar
        )
        
        # Convert to numpy and add to list
        all_features.extend(features_df.values.flatten())
    
    return np.array(all_features, dtype=np.float32)

# ------------------------
#  preprocessing utilities
# ------------------------
def _remove_timestamp_column(data):
    """Remove timestamp column if present (column 0)."""
    if data.shape[1] == 7:
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
    output_dir: Path = Path("../../../project_datasets/tremor/movements"),
    window_size: int = 1024,
    overlap: float = 0.5,
    include_other: bool = False, 
    ):
    
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

        # Create output directory
        label_name = {0: "Healthy", 1: "Parkinson", 2: "Other"}.get(label, "Unknown")
        out_dir = OUTPUT_DIR / movement_name / label_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Extract features and save
        for seg_idx, (left_seg, right_seg) in enumerate(zip(left_segments, right_segments)):

            left_features = _extract_features_from_segment(left_seg)    # (~780,)
            right_features = _extract_features_from_segment(right_seg)  # (~780,)
            
            # Combine: Left + Right 
            combined_features = np.concatenate([left_features, right_features])

            np.savez_compressed(
                out_dir / f"{subject_id}_seg{seg_idx:03d}.npz",
                features=combined_features,
                label=label,
                handedness=handedness,
                subject_id=subject_id,
                movement_name=movement_name,
                segment_idx=seg_idx
            )
            total_segments += 1

    print(f"\n{'='*60}")
    print(f"Dataset creation complete!")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Total segments: {total_segments}")
    print(f"Skipped: {skipped_recordings}")
    print(f"{'='*60}")