import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import butter, filtfilt, resample

# ------------------------
# Preprocessing utilities
# ------------------------

def _butter_lowpass_filter(data, cutoff=10, fs=50, order=4):
    """Apply a low-pass Butterworth filter to each column of IMU data."""
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data, axis=0)

def _normalize_signal(data):
    """Z-score normalize each channel independently."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1.0  # avoid division by zero
    return (data - mean) / std

def _preprocess_signal(data, target_len=1024):
    """
    Apply preprocessing pipeline:
    1. Remove NaNs
    2. Clip outliers
    3. Low-pass filter
    4. Keep only accelerometer channels (first 3)
    5. Resample to fixed length
    6. Normalize
    """
    # 1. Replace NaNs with 0
    data = np.nan_to_num(data, nan=0.0)

    # 2. Clip outliers
    data = np.clip(data, -50, 50)

    # 3. Filter noise
    try:
        data = _butter_lowpass_filter(data)
    except ValueError:
        pass  # skip short signals that can't be filtered

    # 4. Keep only accelerometer channels
    data = data[:, :3]

    # 5. Resample to target length
    if data.shape[0] != target_len:
        data = resample(data, target_len, axis=0)

    # 6. Normalize per channel
    data = _normalize_signal(data)

    return data


# ------------------------
# Main dataset creation function
# ------------------------

def create_preprocessed_dataset(
    root_dir: Path = Path("../../../project_datasets/tremor/Tremor_dataset"),
    time_series_subdir: str = "movement/timeseries",
    file_list_subdir: str = "preprocessed/file_list.csv",
    output_dir: Path = Path("../../../project_datasets/tremor/movemnets"),
    target_len: int = 1024,
    include_other: bool = False
):
    """
    Preprocess Parkinson's Smartwatch Dataset:
    - Only accelerometer channels (3 axes)
    - Removes metadata
    - Applies noise reduction (low-pass filter, clipping, normalization)
    - Optionally include or skip label 2 ("Other")
    """

    TIME_SERIES_DIR = root_dir / time_series_subdir
    FILE_LIST = root_dir / file_list_subdir
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load labels
    labels_df = pd.read_csv(FILE_LIST)
    id_to_label = dict(zip(labels_df['id'], labels_df['label']))
    id_to_handedness = {row['id']: 0 if row['handedness'].lower() == 'left' else 1
                        for _, row in labels_df.iterrows()}

    # 2. Collect movement files
    movement_files = sorted(TIME_SERIES_DIR.glob("*.txt"))

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

    # 5. Process paired recordings
    for (subject_id, movement_name), wrist_files in tqdm(grouped_files.items(), desc="Creating dataset..."):

        if 0 not in wrist_files or 1 not in wrist_files:
            continue

        left_data = np.loadtxt(wrist_files[0], delimiter=',', dtype=np.float32)
        right_data = np.loadtxt(wrist_files[1], delimiter=',', dtype=np.float32)

        if left_data.shape[1] == 7: left_data = left_data[:, 1:]
        if right_data.shape[1] == 7: right_data = right_data[:, 1:]

        label = id_to_label.get(subject_id)
        handedness = id_to_handedness.get(subject_id)
        if label is None or handedness is None:
            continue

        # Skip "Other" recordings if include_other=False
        if label == 2 and not include_other:
            continue

        left_data = _preprocess_signal(left_data, target_len=target_len)
        right_data = _preprocess_signal(right_data, target_len=target_len)

        label_name = {0: "Healthy", 1: "Parkinson", 2: "Other"}.get(label, "Unknown")
        out_dir = OUTPUT_DIR / movement_name / label_name
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            out_dir / f"{subject_id}.npz",
            signal=(left_data, right_data),
            label=label,
            wrist=handedness,
            subject_id=subject_id,
        )

    print(f"\nFinished preprocessing. Saved dataset to: {OUTPUT_DIR.resolve()}")