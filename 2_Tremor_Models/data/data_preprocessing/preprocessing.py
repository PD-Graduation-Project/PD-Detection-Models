import multiprocessing
import os
from pathlib import Path
import pandas as pd
import numpy as np

from l1_trend_filter import l1_trend_filter
from data_handling import load_all_files, get_data_from_observation

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

data_path = '../preprocessed/'

# Movement (time-series) data
Path(data_path).mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. Movement preprocessing
# ==========================================================

def preprocess_movement(df: pd.DataFrame, overwrite: bool = False):
    """
    Preprocess raw IMU movement data for a single subject and store it as a
    binary file suitable for machine learning.

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
        If False, preprocessing is skipped when the output file already exists.

    Output
    ------
    Writes a binary file:
        <subject_id>_ml.bin

    The file contains:
    - Accelerometer signals only (gyro + time removed)
    - Gravity-compensated signals using L1 trend filtering
    - First 0.5 seconds removed (vibration / notification artifact)
    """

    # ------------------------------------------------------
    # 2.1 Load raw sensor data for one subject
    # ------------------------------------------------------

    subject_id = df['subject_id'].iloc[0]

    # Returns grouped movement data and channel names
    # NOTE: data is already split and concatenated per movement group
    data, channels = get_data_from_observation(movement_dir, df)

    # ------------------------------------------------------
    # 2.2 Enforce a fixed channel ordering
    # ------------------------------------------------------
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
                    channels_sorted.append('_'.join([task, wrist, sensor]))
                else:
                    for axis in ['X', 'Y', 'Z']:
                        channels_sorted.append('_'.join([task, wrist, sensor, axis]))

    # Reorder data and channel names
    sorting_indices = [channels.index(ch) for ch in channels_sorted]
    data = data[sorting_indices]
    channels = np.array(channels)[sorting_indices]

    # ------------------------------------------------------
    # 2.3 Remove unwanted channels
    # ------------------------------------------------------
    # - Remove time channels (not used for ML)
    # - Remove tasks not included in final analysis

    to_remove = 'Time|LiftHold|PointFinger|TouchIndex'
    keep_mask = ~pd.Series(channels).str.contains(to_remove)

    data = data[keep_mask]
    channels = channels[keep_mask]

    # Accelerometer channels only (used for signal processing)
    process_mask = pd.Series(channels).str.contains('Accelerometer')

    # ------------------------------------------------------
    # 2.4 Skip preprocessing if output already exists
    # ------------------------------------------------------

    if not overwrite:
        existing_files = [f for f in os.listdir(mov_path) if f.endswith('.bin')]
        if f'{subject_id}_ml.bin' in existing_files:
            return

    # ------------------------------------------------------
    # 2.5 Signal preprocessing
    # ------------------------------------------------------

    # Remove gravitational offset using L1 trend filtering
    data[process_mask, :] = np.apply_along_axis(
        lambda x: x - l1_trend_filter(x, vlambda=50, verbose=False),
        axis=1,
        arr=data[process_mask, :]
    )

    # Remove first 0.5 seconds (sensor vibration / notification artifact)
    data = data[:, 48:]

    # ------------------------------------------------------
    # 2.6 Save preprocessed movement data
    # ------------------------------------------------------

    data.tofile(f'{mov_path}{subject_id}_ml.bin')


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
    df['label'] = df['condition']
    df.replace({
        'label': {
            'Healthy': 0,
            "Parkinson's": 1,
            'Other Movement Disorders': 2,
            'Essential Tremor': 2,
            'Multiple Sclerosis': 2,
            'Atypical Parkinsonism': 2,
        }
    }, inplace=True)

    # Save file list for ML experiments
    df.to_csv(f'{data_path}file_list.csv', index=False, sep=',')

    # ------------------------------------------------------
    # 3.3 Preprocess all movement recordings
    # ------------------------------------------------------

    df_list = load_all_files(movement_dir)

    # Sequential processing (can be parallelized if needed)
    for df_element in df_list:
        preprocess_movement(df_element)
