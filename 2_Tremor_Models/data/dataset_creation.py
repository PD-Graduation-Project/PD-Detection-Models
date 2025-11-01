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
    root_dir: Path = Path("../project_datasets/tremor/pads-parkinsons-disease-smartwatch-dataset-1.0.0"),
    time_series_subdir: str = "movement/timeseries",
    file_list_subdir: str = "preprocessed/file_list.csv",
    output_dir: Path = Path("../project_datasets/tremor/") ,
    target_len: int = 1024, # as used in the official .bin files
    ):
    """
    Preprocesses the *Parkinson's Disease Smartwatch Dataset (PADS)* to create 
    **11 merged movement datasets**, where Left and Right wrist recordings 
    are combined under one movement type.

    Each dataset will contain all subjects' recordings for that specific movement,
    stored as compressed `.npz` files. Each file includes:
        - signal: np.ndarray of shape (N, 6), representing IMU sensor values
        - label: integer (0=Healthy, 1=Parkinson, 2=Other)
        - wrist: integer (0=Left, 1=Right)
        - subject_id: integer (3-digit subject identifier)

    Parameters
    ----------
    root_dir : Path, optional
        Path to the root dataset directory containing `movement/timeseries/` and metadata.
    time_series_subdir : str, optional
        Subdirectory (relative to `root_dir`) where the raw `.txt` movement files are stored.
    file_list_subdir : str, optional
        Subdirectory (relative to `root_dir`) where the `file_list.csv` file containing 
        subject IDs and health labels is stored.
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


    # 1. Load patient labels (healthy, PD, others)
    # ----------------------------------------------
    # 1.1. read csv file and get patients' ids and labels
    labels_df = pd.read_csv(FILE_LIST)
    labels_df = labels_df[['id', 'label']]

    # 1.2. create a dict with id:label
    id_to_label = dict(zip(labels_df['id'], labels_df['label']))


    # 2. Final all movement files
    # ----------------------------
    movement_files  = sorted(TIME_SERIES_DIR.glob("*.txt"))


    # 3. Group files by movement
    # ----------------------------

    # where files are named: SubjectID_MovementName_Wrist.txt
    # Example: "001_CrossArms_LeftWrist.txt" -> subject_id="001", movement_name="CrossArms", wrist = "LeftWrist"

    def parse_filename(fname:str):
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


    # 4. Build dataset per merged movement
    # --------------------------------------
    for movement_file in tqdm(movement_files , desc="Creating dataset..."):
        
        # 4.1. get info from file name
        subject_id, movement_name, wrist = parse_filename(movement_file)
        
        # 4.2. load data
        data = np.loadtxt(
            movement_file,
            delimiter=',',
            dtype=np.float32,
        )
        
        # 4.3. remove the first column (time)
        if data.shape[1] == 7:
            data = data[:, 1:]
            
        # 4.4. get patient's label (health status) form id
        label = id_to_label.get(subject_id) # (0=Healthy, 1=PD, 2=Other)
        if label is None:
            continue # skip if not in file_list.csv
        
        # 4.5. [NEW] Apply preprocessing
        data = _preprocess_signal(data, target_len=target_len)
        
        # 4.6. save info in the output dir
        label_name = {
            0: "Healthy",
            1: "Parkinson",
            2: "Other"
        }.get(label, "Unkown")
        
        # full path: output_dir/movement_name/label_name/
        out_dir = OUTPUT_DIR / movement_name / label_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 4.7. save preprocessed .npz file
        np.savez_compressed(
            out_dir / f"{subject_id}_{'L' if wrist==0 else 'R'}.npz",
            signal = data,
            label = label,
            wrist=wrist,
            subject_id = subject_id
        )
        
    print(f"\nFinished preprocessing. Saved dataset to: {OUTPUT_DIR.resolve()}")
        
    # 5. To load any of them:
    # --------------------------

    # npz = np.load("preprocessed_datasets/CrossArms/Healthy/001_L.npz")
    # X = npz["signal"]
    # y = npz["label"]
    # wrist = npz["wrist"]
