import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ------------------------
# Minimal preprocessing utilities
# ------------------------

def _remove_timestamp_column(data):
    """Remove timestamp column if present (column 0)."""
    if data.shape[1] == 7:
        return data[:, 1:]
    return data

def _handle_missing_values(data):
    """
    Handle missing values by forward fill then backward fill.
    More principled than replacing with zeros.
    """
    # Convert to pandas for easier handling
    df = pd.DataFrame(data)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df.values

def _segment_signal(data, window_size=1024, overlap=0.5):
    """
    Segment long signals into windows with overlap.
    Returns list of segments.
    """
    step_size = int(window_size * (1 - overlap))
    segments = []
    
    for start_idx in range(0, len(data) - window_size + 1, step_size):
        segment = data[start_idx:start_idx + window_size]
        segments.append(segment)
    
    # If signal is shorter than window, pad it
    if len(segments) == 0 and len(data) > 0:
        if len(data) < window_size:
            pad_length = window_size - len(data)
            segment = np.pad(data, ((0, pad_length), (0, 0)), mode='edge')
            segments.append(segment)
        else:
            segments.append(data[:window_size])
    
    return segments

def _compute_basic_stats(data):
    """
    Compute basic statistics for each channel for optional normalization.
    Returns mean and std for each channel.
    """
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)  # avoid division by zero
    return mean, std


# ------------------------
# Main dataset creation function
# ------------------------

def create_clean_dataset(
    root_dir: Path = Path("../../../project_datasets/tremor/Tremor_dataset"),
    time_series_subdir: str = "movement/timeseries",
    file_list_subdir: str = "preprocessed/file_list.csv",
    output_dir: Path = Path("../../../project_datasets/tremor/movements"),
    window_size: int = 1024,
    overlap: float = 0.5,
    include_other: bool = False,
    save_normalization_stats: bool = False,
    channels_to_use: str = 'accel'  # 'all', 'accel', 'gyro'
):
    """
    Create clean, minimally preprocessed dataset from Parkinson's Smartwatch data.
    
    Key improvements:
    - No aggressive filtering that destroys signal characteristics
    - No arbitrary clipping
    - No forced normalization (preserves raw signal magnitudes)
    - Proper handling of missing values
    - Segmentation for variable-length signals
    - Saves both raw data and normalization stats (for optional use)
    
    Args:
        root_dir: Root directory of the dataset
        time_series_subdir: Subdirectory containing time series data
        file_list_subdir: Path to file list CSV with labels
        output_dir: Output directory for processed data
        window_size: Length of signal windows (samples)
        overlap: Overlap between consecutive windows (0 to 1)
        include_other: Whether to include label 2 ("Other")
        save_normalization_stats: Save mean/std for optional normalization
        channels_to_use: 'all' (6 channels), 'accel' (3 channels), or 'gyro' (3 channels)
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

    # 2. Collect movement files
    print("Collecting movement files...")
    movement_files = sorted(TIME_SERIES_DIR.glob("*.txt"))
    print(f"Found {len(movement_files)} files")

    # 3. Parse filename
    def parse_filename(fname: str):
        stem = Path(fname).stem
        subject_id, movement_full = stem.split("_", 1)
        wrist = 0 if "Left" in movement_full else 1
        movement_name = movement_full.replace("_LeftWrist", "").replace("_RightWrist", "")
        return int(subject_id), movement_name, wrist

    # 4. Group files by (subject, movement)
    grouped_files = {}
    for f in movement_files:
        sid, mv, wrist = parse_filename(f)
        grouped_files.setdefault((sid, mv), {})[wrist] = f

    print(f"Grouped into {len(grouped_files)} (subject, movement) pairs")

    # Statistics tracking
    total_segments = 0
    skipped_recordings = 0
    
    # 5. Process paired recordings
    for (subject_id, movement_name), wrist_files in tqdm(grouped_files.items(), desc="Processing"):

        # Require both wrists
        if 0 not in wrist_files or 1 not in wrist_files:
            skipped_recordings += 1
            continue

        # Load data
        try:
            left_data = np.loadtxt(wrist_files[0], delimiter=',', dtype=np.float32)
            right_data = np.loadtxt(wrist_files[1], delimiter=',', dtype=np.float32)
        except Exception as e:
            print(f"\nError loading files for subject {subject_id}, movement {movement_name}: {e}")
            skipped_recordings += 1
            continue

        # Remove timestamp column if present
        left_data = _remove_timestamp_column(left_data)
        right_data = _remove_timestamp_column(right_data)

        # Get metadata
        label = id_to_label.get(subject_id)
        handedness = id_to_handedness.get(subject_id)
        if label is None or handedness is None:
            skipped_recordings += 1
            continue

        # Skip "Other" recordings if requested
        if label == 2 and not include_other:
            skipped_recordings += 1
            continue

        # Select channels
        if channels_to_use == 'accel':
            left_data = left_data[:, :3]
            right_data = right_data[:, :3]
        elif channels_to_use == 'gyro':
            left_data = left_data[:, 3:6]
            right_data = right_data[:, 3:6]
        # else: use all 6 channels

        # Handle missing values
        left_data = _handle_missing_values(left_data)
        right_data = _handle_missing_values(right_data)

        # Compute normalization stats (optional for user)
        left_mean, left_std = _compute_basic_stats(left_data)
        right_mean, right_std = _compute_basic_stats(right_data)

        # Segment signals
        left_segments = _segment_signal(left_data, window_size, overlap)
        right_segments = _segment_signal(right_data, window_size, overlap)

        # Create output directory
        label_name = {0: "Healthy", 1: "Parkinson", 2: "Other"}.get(label, "Unknown")
        out_dir = OUTPUT_DIR / movement_name / label_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save each segment pair
        for seg_idx, (left_seg, right_seg) in enumerate(zip(left_segments, right_segments)):
            
            save_dict = {
                'signal': (left_seg.astype(np.float32), right_seg.astype(np.float32)),
                'label': label,
                'handedness': handedness,
                'subject_id': subject_id,
                'movement_name': movement_name,
                'segment_idx': seg_idx
            }
            
            # Optionally save normalization stats
            if save_normalization_stats:
                save_dict.update({
                    'left_mean': left_mean.astype(np.float32),
                    'left_std': left_std.astype(np.float32),
                    'right_mean': right_mean.astype(np.float32),
                    'right_std': right_std.astype(np.float32)
                })
            
            filename = f"{subject_id}_seg{seg_idx:03d}.npz"
            np.savez_compressed(out_dir / filename, **save_dict)
            total_segments += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset creation complete!")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Total segments created: {total_segments}")
    print(f"Skipped recordings: {skipped_recordings}")
    print(f"Window size: {window_size} samples")
    print(f"Overlap: {overlap*100}%")
    print(f"Channels used: {channels_to_use}")
    print(f"{'='*60}")
