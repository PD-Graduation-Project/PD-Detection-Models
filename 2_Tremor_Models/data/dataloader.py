from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from sklearn.preprocessing import StandardScaler
import joblib

# dataloader creator function
# -----------------------------
def create_tremor_dataloaders(
        data_path: str = None,
        csv_path: str = None,
        batch_size: int = 32,
        train_val_split: float = 0.8,
        random_seed: int = 42,
        print_details: bool = False,
        per_movement: bool = False,
        split_by_subject: bool = True):
    """
    Creates PyTorch DataLoaders for tremor movement classification across all movements.
    
    IMPORTANT: Splits by SUBJECT, not by sample, to prevent data leakage.
    
    - Loads preprocessed .npz signals from all movement folders.
    - Each sample contains:
        (features, wrist_tensor, movement_tensor, label_tensor)
            - features   : IMU signal features
            - wrist_tensor    : scalar (0 = Left-handed, 1 = Right-handed)
            - movement_tensor : scalar (0-10), movement type
            - label_tensor    : scalar (0 = Healthy, 1 = Parkinson)
    - Splits dataset by subjects into train/validation (stratified by label).
    - Returns DataLoaders ready for model training.
    
    Supports two modes:
        A. Unified dataloaders across all movements.
        B. Per-movement dataloaders (train/val for each).

    Args:
        data_path : str
            Root directory containing all movement folders.
        batch_size : int, default=32
            Batch size for both train and validation dataloaders.
        train_val_split : float, default=0.8
            Fraction of subjects (not samples) used for training.
        random_seed : int, default=42
            Random seed for reproducibility.
        print_details : bool, default=False
            Whether to print dataset loading details.
        per_movement : bool, default=False
            If True, returns a dataloader dict for each movement.
        split_by_subject: bool, default=True
            Wether to split the data based on subject_id (real data) or not (generated data)

    Returns:
        - If per_movement=False:
            (train_dataloader, val_dataloader)
        - If per_movement=True:
            movement_dataloaders : dict of {movement_name: {"train": DataLoader, "val": DataLoader}}
    """
    
    # 0. Import here to avoid circular imports
    from .dataset import TremorDataset
    
    # 1. Load full dataset to get subject information
    # FIX: Pass both paths, let Dataset decide which to use
    temp_dataset = TremorDataset(
        data_path=data_path,
        csv_path=csv_path
    )
    
    # -------------------------------
    # SPLIT: by subject or by sample
    # -------------------------------
    if split_by_subject:
        # 2. Get unique subjects and labels
        all_subjects = temp_dataset.get_unique_subjects()
        subject_to_label = {}
        for i, subject_id in enumerate(temp_dataset.subject_ids_list):
            if subject_id not in subject_to_label:
                subject_to_label[subject_id] = temp_dataset.labels[i]

        subject_labels = [subject_to_label[s] for s in all_subjects]

        train_subjects, val_subjects = train_test_split(
            all_subjects,
            test_size=1 - train_val_split,
            stratify=subject_labels,
            random_state=random_seed
        )

        if print_details:
            print(f"\n{'='*60}")
            print(f"Subject-Level Split (prevents data leakage)")
            print(f"{'='*60}")
            print(f"Total subjects: {len(all_subjects)}")
            print(f"Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)}")
            
            # Count labels per split
            train_label_counts = {0: 0, 1: 0}
            val_label_counts = {0: 0, 1: 0}
            for s in train_subjects:
                train_label_counts[subject_to_label[s]] += 1
            for s in val_subjects:
                val_label_counts[subject_to_label[s]] += 1
            
            print(f"\nTrain subjects by label:")
            print(f"  Healthy: {train_label_counts[0]}, Parkinson: {train_label_counts[1]}")
            print(f"Val subjects by label:")
            print(f"  Healthy: {val_label_counts[0]}, Parkinson: {val_label_counts[1]}\n\n")

    else:
        # 2b. Random sample-based split
        all_indices = np.arange(len(temp_dataset))
        all_labels = np.array(temp_dataset.labels)
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=1 - train_val_split,
            stratify=all_labels,
            random_state=random_seed
        )
        
        if print_details:
            print(f"\n{'='*60}")
            print(f"Sample-Level Random Split")
            print(f"Train samples: {len(train_indices)} | Val samples: {len(val_indices)}")
            print(f"{'='*60}\n")
    
    
    # --------------------------------------------------------------
    # Option A: return all the movements in the same dataloader:
    # --------------------------------------------------------------
    if not per_movement:
        
        # 4. Create datasets with optional subject filtering
        if split_by_subject:
            # FIX: Pass both paths
            train_dataset = TremorDataset(data_path=data_path, csv_path=csv_path,
                                        subject_ids=train_subjects)
            val_dataset   = TremorDataset(data_path=data_path, csv_path=csv_path,
                                        subject_ids=val_subjects)
            
            train_labels = np.array(train_dataset.labels)
            
            # Fit scaler on train only, apply to both
            scaler = StandardScaler()
            train_dataset.features = scaler.fit_transform(train_dataset.features)
            val_dataset.features   = scaler.transform(val_dataset.features)
            
            # Save it
            joblib.dump(scaler, "tremor_scaler.pkl")

        else:
            # simple subset using indices
            train_dataset = Subset(temp_dataset, 
                                train_indices)
            val_dataset   = Subset(temp_dataset, 
                                val_indices)
            
            # Helper for subsets
            train_labels = np.array([temp_dataset.labels[i] for i in train_indices])
            
            scaler = StandardScaler()

            # fit on train indices only, transform both
            train_features = temp_dataset.features[train_indices]
            val_features   = temp_dataset.features[val_indices]

            temp_dataset.features[train_indices] = scaler.fit_transform(train_features)
            temp_dataset.features[val_indices]   = scaler.transform(val_features)
            
            # Save it
            joblib.dump(scaler, "tremor_scaler.pkl")

        

        # 5. WeightedRandomSampler for class imbalance in training set
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # 6. Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        # 7. Print dataset info
        if print_details:
            print(f"\nSample counts:")
            print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
            # Handle class distribution print safely for Subsets
            if hasattr(train_dataset, 'get_class_distribution'):
                dist = train_dataset.get_class_distribution()
            else:
                 dist = {'Healthy': np.sum(train_labels == 0), 'Parkinson': np.sum(train_labels == 1)}
            
            print(f"\nTrain Class Distribution:")
            for cls, count in dist.items():
                print(f"  {cls:12s}: {count:4d} ({count/len(train_dataset)*100:.1f}%)")
            print(f"{'='*60}\n")
            
        return train_dataloader, val_dataloader    
    
    # ----------------------------------------------------------
    # Option B: return each movement as a separate dataloader:
    # ----------------------------------------------------------
    else:
        # 4. Create train/val datasets with optional subject filtering
        if split_by_subject:
            train_dataset = TremorDataset(data_path=data_path, csv_path=csv_path,
                                        subject_ids=train_subjects)
            val_dataset   = TremorDataset(data_path=data_path, csv_path=csv_path,
                                        subject_ids=val_subjects)
            
            # Accessors for regular dataset
            train_all_movements = train_dataset.movements
            val_all_movements = val_dataset.movements
            train_all_labels = train_dataset.labels
            movement_names = train_dataset.movement_names
            movement_map = train_dataset.movement_to_idx
            
        else:
            # simple subset using indices
            train_dataset = Subset(temp_dataset, 
                                train_indices)
            val_dataset   = Subset(temp_dataset, 
                                val_indices)
            
            # Accessors for Subset (must access underlying dataset via indices)
            train_all_movements = [temp_dataset.movements[i] for i in train_indices]
            val_all_movements = [temp_dataset.movements[i] for i in val_indices]
            train_all_labels = [temp_dataset.labels[i] for i in train_indices]
            movement_names = temp_dataset.movement_names
            movement_map = temp_dataset.movement_to_idx
        
        # 5. Init movement dataloaders dict
        movement_dataloaders = {}

        # 6. Loop through every movement
        for movement_name in movement_names:
            movement_idx = movement_map[movement_name]
            
            # Get indices for this movement in train and val (relative to their specific datasets)
            # We filter the lists created above
            train_movement_indices = [i for i, m in enumerate(train_all_movements) if m == movement_idx]
            val_movement_indices = [i for i, m in enumerate(val_all_movements) if m == movement_idx]
            
            # Skip if no samples for this movement
            if len(train_movement_indices) == 0 or len(val_movement_indices) == 0:
                if print_details:
                    print(f"[{movement_name}] Skipped - insufficient samples")
                continue

            # Create subsets
            train_subset = Subset(train_dataset, train_movement_indices)
            val_subset = Subset(val_dataset, val_movement_indices)

            # Weighted sampler for this movement
            # Map subset indices back to our temporary label lists
            train_labels_subset = np.array([train_all_labels[i] for i in train_movement_indices])
            class_counts = np.bincount(train_labels_subset)
            class_weights = 1.0 / np.maximum(class_counts, 1)
            sample_weights = class_weights[train_labels_subset]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            # Dataloaders
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                pin_memory=True,
                drop_last=True,
                sampler=sampler
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
            )

            movement_dataloaders[movement_name] = {
                "train": train_loader,
                "val": val_loader,
            }

            if print_details:
                print(f"[{movement_name}]  Train: {len(train_subset)} | Val: {len(val_subset)}")

        if print_details:
            print(f"\nCreated dataloaders for {len(movement_dataloaders)} movements")

        return movement_dataloaders