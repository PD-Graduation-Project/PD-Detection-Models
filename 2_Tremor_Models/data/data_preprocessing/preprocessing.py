import multiprocessing
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from l1_trend_filter import l1_trend_filter
from data_handling import load_all_files, get_data_from_observation, MOVEMENT_GROUPS

"""
Preprocessing pipeline for tremor dataset.

This file is responsible for:
    - Creating output directories for preprocessed data
    - Preprocessing raw movement (IMU) signals into ML-ready binary files
    - Generating subject-level metadata (labels, conditions)
    - Exporting questionnaire and movement data in a consistent binary format

All heavy signal processing happens here, while file I/O and grouping logic
are handled in data_handling.py.
"""

# ==========================================================
# 0. Dataset paths (raw)
# ==========================================================

# Directory containing raw movement (IMU) recordings
movement_dir = "../../../project_datasets/tremor/Tremor_dataset/movement/"

# Directory containing patient metadata JSON files
patient_dir = "../../../project_datasets/tremor/Tremor_dataset/patients/"

# ==========================================================
# 1. Output directories (preprocessed)
# ==========================================================

data_path = '../preprocessed_dataset/'
mov_path = data_path + 'movement/'

# Movement (time-series) data
Path(mov_path).mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. Movement preprocessing
# ==========================================================

def preprocess_movement(df: pd.DataFrame, overwrite: bool = False):
    """
    Preprocess raw IMU movement data for a single subject and store it as
    binary files (one per movement group) suitable for machine learning.

    Parameters
    ----------
    df : pandas.DataFrame
        Flattened observation metadata for a single subject.
        Expected to include information such as:
        - subject_id
        - record_name
        - channels
        - rows
        - file_name

    overwrite : bool, default=False
        If False, preprocessing is skipped when the output files already exist.

    Output
    ------
    Writes three binary files (one per movement group):
        <subject_id>_Postural tasks_ml.bin
        <subject_id>_Kinetic tasks_ml.bin
        <subject_id>_Resting tasks_ml.bin

    Each file contains:
    - Accelerometer signals only (gyro + time removed)
    - Gravity-compensated signals using L1 trend filtering
    - First 0.5 seconds removed (vibration / notification artifact)
    """

    # ------------------------------------------------------
    # 2.1 Load raw sensor data for one subject
    # ------------------------------------------------------

    subject_id = df['subject_id'].iloc[0]

    # Returns grouped movement data and channel names per group
    grouped_data, grouped_channels = get_data_from_observation(movement_dir, df)

    # ------------------------------------------------------
    # 2.2 Process each movement group separately
    # ------------------------------------------------------

    for group_name, data in grouped_data.items():
        
        # Skip if no data for this group
        if data is None:
            continue

        channels = np.array(grouped_channels[group_name])

        # ------------------------------------------------
        # 2.2.1 Enforce a fixed channel ordering
        # ------------------------------------------------
        # The model expects channels in a deterministic order:
        #   task -> wrist -> sensor -> axis

        channels_sorted = []

        tasks = [
            'Relaxed1', 'Relaxed2', 'RelaxedTask1', 'RelaxedTask2',
            'StretchHold', 'LiftHold', 'HoldWeight',
            'PointFinger', 'DrinkGlas', 'CrossArms',
            'TouchIndex', 'TouchNose',
            'Entrainment1', 'Entrainment2'
        ]

        for task in tasks:
            for wrist in ['LeftWrist', 'RightWrist']:
                for sensor in ['Time', 'Accelerometer', 'Gyroscope']:
                    if sensor == 'Time':
                        ch = '_'.join([task, wrist, sensor])
                        if ch in channels:
                            channels_sorted.append(ch)
                    else:
                        for axis in ['X', 'Y', 'Z']:
                            ch = '_'.join([task, wrist, sensor, axis])
                            if ch in channels:
                                channels_sorted.append(ch)

        # Reorder data and channel names
        sorting_indices = [list(channels).index(ch) for ch in channels_sorted]
        data = data[sorting_indices]
        channels = channels[sorting_indices]

        # ------------------------------------------------
        # 2.2.2 Remove unwanted channels
        # ------------------------------------------------
        # - Remove time channels (not used for ML)
        # - Remove tasks not included in final analysis

        to_remove = 'Time|LiftHold|PointFinger|TouchIndex'
        keep_mask = ~pd.Series(channels).str.contains(to_remove)

        data = data[keep_mask]
        channels = channels[keep_mask]

        # Accelerometer channels only (used for signal processing)
        process_mask = pd.Series(channels).str.contains('Accelerometer')

        # ------------------------------------------------
        # 2.2.3 Skip preprocessing if output already exists
        # ------------------------------------------------

        output_filename = f'{mov_path}{subject_id}_{group_name}_ml.bin'
        print(f'{mov_path}{subject_id}_{group_name}_ml.bin')
        if not overwrite and os.path.exists(output_filename):
            continue

        # ------------------------------------------------
        # 2.2.4 Signal preprocessing
        # ------------------------------------------------

        # Remove gravitational offset using L1 trend filtering
        acc_data = data[process_mask, :]

        for i in tqdm(
            range(acc_data.shape[0]),
            desc=f"L1 detrending | subject {subject_id}", leave=False ):
            acc_data[i] = acc_data[i] - l1_trend_filter(acc_data[i], vlambda=50, verbose=False)
            
        data[process_mask, :] = acc_data

        # Remove first 0.5 seconds (sensor vibration / notification artifact)
        data = data[:, 48:]

        # ------------------------------------------------
        # 2.2.5 Save preprocessed movement data
        # ------------------------------------------------

        data.tofile(output_filename)


# ==========================================================
# 3. Script entry point
# ==========================================================

if __name__ == '__main__':

    # ------------------------------------------------------
    # 3.1 Build subject-level metadata file
    # ------------------------------------------------------

    # Load patient metadata (id, condition, handedness)
    df = pd.concat(load_all_files(patient_dir))

    # Encode condition as numerical label
    df['label'] = df['condition'].map({
        'Healthy': 0,
        "Parkinson's": 1,
        'Other Movement Disorders': 2,
        'Essential Tremor': 2,
        'Multiple Sclerosis': 2,
        'Atypical Parkinsonism': 2,
    }).astype(int)

    # Save file list for ML experiments
    df.to_csv(f'{data_path}file_list.csv', index=False, sep=',')

    # ------------------------------------------------------
    # 3.2 Preprocess all movement recordings
    # ------------------------------------------------------

    df_list = load_all_files(movement_dir)

    # Run sequentially (safer)
    for df_element in tqdm(df_list, desc="Preprocessing subjects"):
        preprocess_movement(df_element)
