import json
from glob import glob
import numpy as np
import pandas as pd

# ===================
# Meta-data loading
# ===================
def load_all_files(path, dataframe=True):
    """
    Load all .json files from the defined directory and return all the loaded meta data.
    
    What it does:
        - Finds all .json files in the given path
        - Loads each JSON file
        - Flattens nested JSON into a pandas DataFrame
        - Returns list of DataFrames

    Parameters
    ----------
    path : str
        Path to the directory holding the .json files.
    dataframe : bool, default = True
        Whether to flatten the meta data into dataframes.
    """
    data_list = []
    search_space = glob(path + '*json')
    search_space.sort()
    for f_name in search_space:
        with open(f_name, 'r') as f:
            data = json.load(f)
            if dataframe:
                data = flatten_dict(data)
                data = pd.DataFrame(data)
            data_list.append(data)
    return data_list

# ===================
# Raw signal loading
# ===================

def get_data_from_txt_file(path, n_channels):
    record = np.loadtxt(path, dtype=np.float32, delimiter=",")
    return record  # (Time, Acc_X, Acc_Y, Acc_Z, Gyro_X, Gyro_Y, Gyro_Z)


# ========================
# Movement grouping logic
# ========================

# Sorted according to discriminability of PD
MOVEMENT_GROUPS = {
    "Postural_tasks": ["StretchHold", "HoldWeight", "Entrainment1", "Entrainment2"],
    "Kinetic_tasks": ["DrinkGlas", "CrossArms", "TouchNose"],
    "Resting_tasks": ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2"],
}


def _movement_to_group(record_name: str):
    """
    Map a record_name to its movement group.
    Movements with suffix 1/2 are treated as the SAME movement.
    """
    base_name = record_name.rstrip("12")
    for group, movements in MOVEMENT_GROUPS.items():
        if base_name in [m.rstrip("12") for m in movements]:
            return group
    return None


# ===================
# Observation loading
# ===================

def get_data_from_observation(path, meta_file):
    """
    Returns data split into three numpy arrays:
        - Postural tasks
        - Kinetic tasks
        - Resting tasks
    
    All recordings within a group are padded to the same length.
    """

    grouped_records = {k: [] for k in MOVEMENT_GROUPS.keys()}
    grouped_channels = {k: [] for k in MOVEMENT_GROUPS.keys()}

    # 1. shortest recording as reference
    min_rows = meta_file['rows'].min()

    # 2. loop through each recording
    for _, meta_item in meta_file.iterrows():
        n_splits = meta_item['rows'] // min_rows

        # 3. load raw data
        file_path = meta_item['file_name']
        record = get_data_from_txt_file(path + file_path, len(meta_item['channels']))
        record = np.swapaxes(record, 0, 1) # Transposes so rows = channels, columns = time points

        channels = ['_'.join([meta_item['device_location'], ch]) for ch in meta_item['channels']]

        # 4. Re-organize the raw data so that each record has the same length and all records fit into one matrix
        step = record.shape[1] // n_splits # -> If a recording is 2× the minimum length, **split it in half**
        if n_splits > 1:
            split_records = [record[:, n:n + step] for n in range(0, record.shape[1], step)]
            split_channels_list = []
            for split_idx in range(len(split_records)):
                split_channels_list.append(['_'.join([f'{meta_item["record_name"]}{split_idx+1}', ch]) for ch in channels])
        else:
            split_records = [record]
            split_channels_list = [['_'.join([meta_item['record_name'], ch]) for ch in channels]]

        # 5. assign each split to its movement group
        group = _movement_to_group(meta_item['record_name'])
        if group is None:
            continue

        for split_rec, split_chans in zip(split_records, split_channels_list):
            grouped_records[group].append(split_rec)
            grouped_channels[group].extend(split_chans)

    # 6. concatenate per group (with padding to max length)
    for group in grouped_records:
        if len(grouped_records[group]) > 0:
            # Find max length in this group
            max_length = max(rec.shape[1] for rec in grouped_records[group])
            
            # Pad all records to max length
            padded_records = []
            for rec in grouped_records[group]:
                if rec.shape[1] < max_length:
                    padding = np.zeros((rec.shape[0], max_length - rec.shape[1]), dtype=rec.dtype)
                    rec = np.concatenate([rec, padding], axis=1)
                padded_records.append(rec)
            
            grouped_records[group] = np.concatenate(padded_records, axis=0)
        else:
            grouped_records[group] = None

    return grouped_records, grouped_channels


# ===================
# Public API
# ===================

def get_data(path):
    data_list = []
    channels_list = []

    meta_list = load_all_files(path, dataframe=True)

    for meta_file in meta_list:
        if meta_file is None:
            continue

        if 'resource_type' in meta_file.columns and meta_file['resource_type'].iloc[0] == 'observation':
            data, channels = get_data_from_observation(path, meta_file)
        else:
            raise Exception('Only observation resources are supported.')

        data_list.append(data)
        channels_list.append(channels)

    return data_list, channels_list

# ==================================== #
# ======== HELPER FUNCTIONS ========== #
# ==================================== #
def _flatten_dict(data_dict, tmp_fields, data_dict_flat):
    is_most_inner = True  # Assume we are at most inner element
    for key, item in data_dict.items():
        if isinstance(item, dict):
            is_most_inner = False
            data_dict, tmp_fields, data_dict_flat = _flatten_dict(item, tmp_fields.copy(), data_dict_flat.copy())
        elif isinstance(item, list):
            if isinstance(item[0], dict):
                is_most_inner = False
                for list_item in item:
                    data_dict, tmp_fields, data_dict_flat = _flatten_dict(list_item, tmp_fields.copy(),
                                                                        data_dict_flat.copy())
            else:
                tmp_fields[key] = item
        else:
            tmp_fields[key] = item

    if is_most_inner:
        data_dict_flat.append(tmp_fields)

    return data_dict, tmp_fields, data_dict_flat


def flatten_dict(data_dict):
    tmp_fields = {}
    data_dict_flat = []
    _, _, data_dict_flat = _flatten_dict(data_dict, tmp_fields, data_dict_flat)
    return data_dict_flat
