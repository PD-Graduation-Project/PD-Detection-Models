import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import butter, filtfilt, resample


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
    4. Resample to fixed length
    5. Normalize
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

    # 4. Resample (upsample (data < 1024) or downsample (data > 1024) to target length)
    if data.shape[0] != target_len:
        data = resample(data, target_len, axis=0)

    # 5. Normalize per channel
    data = _normalize_signal(data)

    return data


# Main dataset creation function
# ----------------------------------
def create_preprocessed_dataset(
    root_dir: Path = Path("../../../project_datasets/tremor/pads-parkinsons-disease-smartwatch-dataset-1.0.0"),
    time_series_subdir: str = "movement/timeseries",
    file_list_subdir: str = "preprocessed/file_list.csv",
    output_dir: Path = Path("../../../project_datasets/tremor/") ,
    target_len: int = 1024, # as used in the official .bin files
    ):
    """
    Preprocesses the *Parkinson's Disease Smartwatch Dataset (PADS)* to create 
    **11 merged movement datasets**, where both Left and Right wrist recordings 
    for the same subject and movement are combined into one entry.

    Each dataset will contain all subjects' paired recordings for that specific movement,
    stored as compressed `.npz` files. Each file includes:
        - signal: tuple of two np.ndarray ((1024, 6), (1024, 6))
                    where (left_signal, right_signal) - ALWAYS in this order
        - label: integer (0=Healthy, 1=Parkinson, 2=Other)
        - wrist: integer (0=Left-handed, 1=Right-handed)
        - subject_id: integer (3-digit subject identifier)
        - metadata: dict containing age, height, weight, gender from CSV

    Parameters
    ----------
    root_dir : Path, optional
        Path to the root dataset directory containing `movement/timeseries/` and metadata.
    time_series_subdir : str, optional
        Subdirectory (relative to `root_dir`) where the raw `.txt` movement files are stored.
    file_list_subdir : str, optional
        Subdirectory (relative to `root_dir`) where the `file_list.csv` file containing 
        subject IDs, handedness, and health labels is stored.
    output_dir : Path, optional
        Directory to save the preprocessed `.npz` datasets.

    Returns
    -------
    None
        Creates and saves 11 directories (one per merged movement type),
        each containing subject-level `.npz` files in subfolders by label.
    """
    
    # 0. Config and paths setup
    # --------------------------
    # path to movement data (.txt)
    TIME_SERIES_DIR = root_dir / time_series_subdir
    # path to csv containing labels
    FILE_LIST = root_dir / file_list_subdir
    
    # our final preprocessed data save location
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    # 1. Load patient metadata (labels + handedness + clinical info)
    # ----------------------------------------------
    # 1.1. read csv file and get patients' ids and labels
    labels_df = pd.read_csv(FILE_LIST)
    
    # 1.2. create a dict with id:label & id:handedness
    id_to_label = dict(zip(labels_df['id'], labels_df['label']))
    id_to_handedness = {row['id']: 0 if row['handedness'].lower() == 'left' else 1
                        for _, row in labels_df.iterrows()}
    
    # 1.3. create metadata dict for each subject (all available fields)
    id_to_metadata = {}
    for _, row in labels_df.iterrows():
        # Encode categorical variables numerically
        gender_map = {'male': 0, 'female': 1}
        handedness_map = {'left': 0, 'right': 1}
        effect_alcohol_map = {'Unknown': 0, 'No effect': 1, 'Reduced': 2, 'Increased': 3}
        
        id_to_metadata[row['id']] = {
            'age_at_diagnosis': row['age_at_diagnosis'] if pd.notna(row['age_at_diagnosis']) else -1,
            'age': row['age'] if pd.notna(row['age']) else -1,
            'height': row['height'] if pd.notna(row['height']) else -1,
            'weight': row['weight'] if pd.notna(row['weight']) else -1,
            'gender': gender_map.get(row['gender'], -1),
            'appearance_in_kinship': 
                    1 if row['appearance_in_kinship'] == True else 0 if row['appearance_in_kinship'] == False else -1,
            'appearance_in_first_grade_kinship':
                    1 if row['appearance_in_first_grade_kinship'] == True else 0 if row['appearance_in_first_grade_kinship'] == False else -1,
            'effect_of_alcohol_on_tremor': 
                    effect_alcohol_map.get(row['effect_of_alcohol_on_tremor'], -1) if pd.notna(row['effect_of_alcohol_on_tremor']) else -1,
        }

    # 2. Final all movement files
    # ----------------------------
    movement_files  = sorted(TIME_SERIES_DIR.glob("*.txt"))

    # 3. Parse filename helper
    # --------------------------
    def parse_filename(fname:str):
        # Example: "001_CrossArms_LeftWrist.txt"
        # 3.1. remove the extention from name
        stem = Path(fname).stem
        
        # 3.2. split id from movement name
        subject_id, movement_full  = stem.split("_", 1) # 1: only the first instance of '_'
        
        # 3.3 detrmine wrist (0=Left, 1=Right)
        if "Left" in movement_full:
            wrist = 0
        elif "Right" in movement_full:
            wrist = 1
            
        # 3.4. remove _LeftWrist or _RightWrist suffix to merge into 11 movements
        movement_name = movement_full.replace("_LeftWrist", "").replace("_RightWrist", "")
        
        # 3.5. return id, movement name, and wrist
        return int(subject_id), movement_name, wrist
    
    # 4. Group files by (subject, movement)
    # --------------------------------------
    grouped_files = {}
    for f in movement_files:
        sid, mv, wrist = parse_filename(f)
        grouped_files.setdefault((sid, mv), {})[wrist] = f


    # 5. Process paired recordings
    # -----------------------------
    for (subject_id, movement_name), wrist_files in tqdm(grouped_files.items(), desc="Creating dataset..."):
        
        # 5.1. skip if one hand is missing
        if 0 not in wrist_files or 1 not in wrist_files:
            continue
        
        # 5.2. Load BOTH signals - explicitly by wrist position (0=Left, 1=Right)
        left_data = np.loadtxt(wrist_files[0], delimiter=',', dtype=np.float32)   # wrist=0 is LEFT
        right_data = np.loadtxt(wrist_files[1], delimiter=',', dtype=np.float32)  # wrist=1 is RIGHT

        # 5.3. remove time column if present
        if left_data.shape[1] == 7: left_data = left_data[:, 1:]
        if right_data.shape[1] == 7: right_data = right_data[:, 1:]

        # 5.4. get label, handedness, and metadata
        label = id_to_label.get(subject_id)
        handedness = id_to_handedness.get(subject_id)
        metadata = id_to_metadata.get(subject_id)

        if label is None or handedness is None or metadata is None:
            continue  # skip if metadata missing

        # 5.5. preprocess both signals
        left_data = _preprocess_signal(left_data, target_len=target_len)
        right_data = _preprocess_signal(right_data, target_len=target_len)

        # 5.6. build save path
        label_name = {
            0: "Healthy",
            1: "Parkinson",
            2: "Other"
        }.get(label, "Unknown")

        out_dir = OUTPUT_DIR / movement_name / label_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 5.7. save tupled signals + metadata - ALWAYS (left, right) order
        np.savez_compressed(
            out_dir / f"{subject_id}.npz",
            signal=(left_data, right_data),  # ALWAYS: (left wrist, right wrist)
            label=label,
            wrist=handedness,     # handedness: 0=Left-handed, 1=Right-handed
            subject_id=subject_id,
            metadata=metadata,    # age, height, weight, gender
        )

    print(f"\nFinished preprocessing. Saved dataset to: {OUTPUT_DIR.resolve()}")