import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_preprocessed_dataset(
    root_dir: Path = Path("../../project_datasets/tremor/pads-parkinsons-disease-smartwatch-dataset-1.0.0"),
    time_series_subdir: str = "movement/timeseries",
    file_list_subdir: Path = Path("preprocessed/file_list.csv"),
    output_dir: Path = Path("../../project_datasets/tremor/preprocessed_datasets") 
    ):
    """
    Preprocesses the *Parkinson's Disease Smartwatch Dataset (PADS)* to create 
    22 separate datasets, one for each movement activity.

    Each dataset will contain all subjects' recordings for that specific movement,
    stored as compressed `.npz` files. Each file includes:
        - signal: np.ndarray of shape (N, 6), representing motion sensor values
        - label: integer (0 for Healthy, 1 for Parkinson)
        - subject_id: string (3-digit subject identifier)

    Parameters
    ----------
    root_dir : Path, optional
        Path to the root dataset directory containing `movement/timeseries/` and metadata.
    time_series_subdir : str, optional
        Subdirectory (relative to `root_dir`) where the raw `.txt` movement files are stored.
    file_list_path : Path, optional
        Path to the `file_list.csv` file containing subject IDs and health labels.
    output_dir : Path, optional
        Directory to save the preprocessed `.npz` datasets.

    Returns
    -------
    None
        Creates and saves 22 directories (one per movement), each containing subject-level `.npz` files.
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


    # 2. Final all movement names
    # ----------------------------
    movement_names = sorted(TIME_SERIES_DIR.glob("*.txt"))


    # 3. Group files by movement
    # ----------------------------

    # where files are named: SubjectID_Movement_Name.txt
    # Example: "001_CrossArms_LeftWrist.txt" -> subject_id="001", movement="CrossArms_LeftWrist"

    def parse_filename(fname:str):
        # 3.1. remove the extention from name
        stem = Path(fname).stem
        
        # 3.2. split id from movement name
        subject_id, movement_name = stem.split("_", 1) # 1: only the first instance of '_'
        
        # 3.3. return id and movement name
        return int(subject_id), movement_name


    # 4. Build dataset per movement
    # --------------------------------
    for movement_file in tqdm(movement_names, desc="Creating dataset..."):
        
        # 4.1. get id and movement name
        subject_id, movement_name = parse_filename(movement_file)
        
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
        label = id_to_label.get(subject_id)
        if label is None:
            continue # skip if not in file_list.csv
        
        # 4.5. save info in the output dir
        out_dir = OUTPUT_DIR / movement_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            out_dir / f"{subject_id}.npz",
            signal = data,
            label = label,
            subject_id = subject_id
        )
        
        
    # 5. To load any of them:
    # --------------------------

    # npz = np.load("processed_datasets/drink_glass/012.npz")
    # X = npz["signal"]
    # y = npz["label"]
