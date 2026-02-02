import multiprocessing
import os
from pathlib import Path
import pandas as pd
from l1_trend_filter import l1_trend_filter
from constants import movement_dir, questionnaire_dir, patient_dir
from data_handling import load_all_files, get_data, get_data_from_observation
import numpy as np

# 1. Create output folders for preprocessed data
data_path = '../../preprocessed/'
quest_path = data_path + '/questionnaire/'
Path(quest_path).mkdir(parents=True, exist_ok=True)
mov_path = data_path + '/movement/'
Path(mov_path).mkdir(parents=True, exist_ok=True)


def preprocess_movement(df, overwrite=False):
    # 2. Load raw sensor data for one subject
    id = df['subject_id'][0]
    data, channels = get_data_from_observation(movement_dir, df)

    channels_sorted = []
    
    # 3. Sort by the following pattern
    # 3.1. we ahve 14 tasks
    for task in ['Relaxed1', 'Relaxed2', 'RelaxedTask1', 'RelaxedTask2', 'StretchHold', 'LiftHold', 'HoldWeight',
                'PointFinger', 'DrinkGlas', 'CrossArms', 'TouchIndex', 'TouchNose', 'Entrainment1', 'Entrainment2']:
        # 3.2. for each there are 2 wrists
        for wrist in ['LeftWrist', 'RightWrist']:
            # 3.3. for each there are accel-data, gyro-data, and time stamps
            for sensor in ['Time', 'Accelerometer', 'Gyroscope']:
                # 3.4. handle time
                if sensor == 'Time':
                    channel_name = '_'.join([task, wrist, sensor])
                    channels_sorted.append(channel_name)
                # 3.5. handle accel, gyro
                else:
                    for axis in ['X', 'Y', 'Z']:
                        channel_name = '_'.join([task, wrist, sensor, axis])
                        channels_sorted.append(channel_name)

    sorting_indices = [channels.index(channel_name) for channel_name in channels_sorted]

    data = data[sorting_indices]
    channels = np.array(channels)[sorting_indices]

    # 4. Remove Unwanted Data -> Time columns (not needed for ML)
    to_remove = 'Time|LiftHold|PointFinger|TouchIndex'
    keep_mask = ~pd.Series(channels).str.contains(to_remove)
    channels = channels[keep_mask]

    to_process = 'Accelerometer'
    process_mask = pd.Series(channels).str.contains(to_process)

    # 4.1. Check if file already exists
    if not overwrite:
        all_files = os.listdir(mov_path)
        all_files = list(filter(lambda f: f.endswith('.bin'), all_files))
        if f'{id}_ml.bin' in all_files:
            return

    # 5. Apply L1 Trend Filter -> For each accelerometer channel
    # 5.1. Remove assessment steps
    data = data[keep_mask]
    # 5.2. Remove gravitational offset
    data[process_mask, :] = np.apply_along_axis(lambda x: x - l1_trend_filter(x, vlambda=50, verbose=False), 1,
                                                data[process_mask, :])
    # 5.3. Remove first half second of the signal (vibration notification)
    data = data[:, 48:]
    data.tofile(f'{mov_path}{id}_ml.bin')


if __name__ == '__main__':
    # Store file list for ml project
    df = pd.concat(load_all_files(patient_dir))
    df['label'] = df['condition']
    df.replace({'label': {'Healthy': 0,
                        "Parkinson's": 1,
                        'Other Movement Disorders': 2,
                        'Essential Tremor': 2,
                        'Multiple Sclerosis': 2,
                        'Atypical Parkinsonism': 2}},
            inplace=True)
    df.to_csv(f'{data_path}file_list.csv', index=False, sep=',')

    # Store all questionnaire data for ml project
    data, channels = get_data(questionnaire_dir)
    for idx, data_sample in enumerate(data):
        data_sample.tofile(f'{quest_path}{idx + 1:03d}_ml.bin')

    # Store file list for ml project
    df_list = load_all_files(movement_dir)
    # Run in parallel
    for df_element in df_list:
        preprocess_movement(df_element)
