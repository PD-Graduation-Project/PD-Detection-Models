import numpy as np
import pandas as pd

def prepare_data_for_timegan(
        data_path: str,
        include_other: bool = True,
        print_details: bool = True
    ):
    """
    Prepares data for YData Synthetic TimeGAN model. 
    It combines temporal (12 channels) and static (2 features) data 
    into a single conditional 3D NumPy array (N, T, 14) per movement.
    
    Returns:
        movement_data (dict): 
            Keys: Movement names (str)
            Values: 3D NumPy array (N_samples, 1026, 14)
    """
    
    # 0. Import Dataset (Assuming it's in a path discoverable by Python)
    from dataset import TremorDataset # Using the existing TremorDataset
    
    if print_details:
        print("--- YData TimeGAN Data Prep: Using ALL Subjects for Training ---")

    # 1. Load the full dataset, including ALL subjects (subject_ids=None)
    full_training_dataset = TremorDataset(
        data_path=data_path,
        subject_ids=None, # Include all subjects
        include_other=include_other,
        print_details=False,
    )
    
    if print_details:
        print(f"Total subjects loaded: {len(full_training_dataset.get_unique_subjects())}")

    # 2. Dictionary to store the final NumPy arrays
    movement_data = {}
    
    # 3. Iterate over each movement type
    for movement_name in full_training_dataset.movement_names:
        movement_idx = full_training_dataset.movement_to_idx[movement_name]
        
        # Get all indices belonging to this movement
        indices = [i for i, m in enumerate(full_training_dataset.movements) if m == movement_idx]
        
        if len(indices) == 0:
            if print_details: print(f"[{movement_name}] Skipped (0 samples)")
            continue
            
        # --- PREPARE DATA CONTAINERS ---
        temporal_data_list = []
        static_data_list = [] # Used temporarily to track static features
        
        # 4. Extract and transform data for the current movement
        for idx in indices:
            # Load raw sample: signal, handedness, movement_idx, label
            signal, handedness, _, label = full_training_dataset[idx] 
            
            # --- A. TEMPORAL TRANSFORMATION (12 Channels) ---
            if hasattr(signal, 'numpy'):
                signal = signal.numpy()
                
            # Stack channels: shape becomes (T, 12) from (2, T, 6)
            combined_signal = np.concatenate([signal[0], signal[1]], axis=1) 
            temporal_data_list.append(combined_signal)
            
            # --- B. STATIC TRANSFORMATION (2 Features) ---
            # Stored as a simple list, will be broadcast later
            static_data_list.append([float(handedness), float(label)])
            
        # Convert to Numpy Arrays
        X_temporal = np.array(temporal_data_list) # Shape: (N_samples, 1026, 12)
        X_static = np.array(static_data_list)     # Shape: (N_samples, 2)
        
        # --- C. TIMEGAN CONDITIONAL INPUT CREATION ---
        N, T, C_temp = X_temporal.shape
        
        # 1. Expand static features (N, 2) to (N, 1026, 2)
        X_static_expanded = np.repeat(
            X_static[:, np.newaxis, :], 
            T, 
            axis=1
        ) 

        # 2. Concatenate Temporal and Static features: (N, 1026, 12 + 2 = 14)
        X_timegan = np.concatenate([X_temporal, X_static_expanded], axis=-1).astype(np.float32)
        
        movement_data[movement_name] = X_timegan
        
        if print_details:
            print(f"[{movement_name:20s}] Prepared Array | Samples: {N} | Final Shape: {X_timegan.shape}")

    return movement_data
