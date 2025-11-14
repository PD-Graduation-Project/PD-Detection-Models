from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np

# dataloader creator function
# -----------------------------
def create_tremor_dataloaders(
        data_path: str,
        batch_size: int = 32,
        train_val_split: float = 0.8,
        random_seed: int = 42,
        include_other: bool = True,
        print_details: bool = False,
        per_movement: bool = False):
    """
    Creates PyTorch DataLoaders for tremor movement classification across all movements.
    
    IMPORTANT: Splits by SUBJECT, not by sample, to prevent data leakage.
    
    - Loads preprocessed .npz signals from all movement folders.
    - Each sample contains:
        (signal_tensor, wrist_tensor, movement_tensor, label_tensor, metadata_tensor)
            - signal_tensor   : shape (2, T, 6), IMU signal (left-signals, right-signals)
            - wrist_tensor    : scalar (0 = Left-handed, 1 = Right-handed)
            - movement_tensor : scalar (0-10), movement type
            - label_tensor    : scalar (0 = Healthy, 1 = Parkinson, 2 = Other)
            - metadata_tensor : shape (8,), [age_at_diagnosis, age, height, weight, gender,
                                          appearance_in_kinship, appearance_in_first_grade_kinship,
                                          effect_of_alcohol_on_tremor]
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
        include_other : bool, default=True
            Whether to include samples from the "Other" class (label=2).
        print_details : bool, default=False
            Whether to print dataset loading details.
        per_movement : bool, default=False
            If True, returns a dataloader dict for each movement.

    Returns:
        - If per_movement=False:
            (train_dataloader, val_dataloader)
        - If per_movement=True:
            movement_dataloaders : dict of {movement_name: {"train": DataLoader, "val": DataLoader}}
    """
    
    # 0. Import here to avoid circular imports
    from .dataset import TremorDataset
    
    # 1. Load full dataset to get subject information
    temp_dataset = TremorDataset(
        data_path=data_path,
        include_other=include_other,
        print_details=False,
    )
    
    # 2. Get all unique subjects and their labels
    all_subjects = temp_dataset.get_unique_subjects()
    
    # Create mapping: subject_id -> label (take first occurrence)
    subject_to_label = {}
    for i, subject_id in enumerate(temp_dataset.subject_ids_list):
        if subject_id not in subject_to_label:
            subject_to_label[subject_id] = temp_dataset.labels[i]
    
    # 3. Split subjects (not samples!) into train/val, stratified by label
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
        train_label_counts = {0: 0, 1: 0, 2: 0}
        val_label_counts = {0: 0, 1: 0, 2: 0}
        for s in train_subjects:
            train_label_counts[subject_to_label[s]] += 1
        for s in val_subjects:
            val_label_counts[subject_to_label[s]] += 1
        
        print(f"\nTrain subjects by label:")
        print(f"  Healthy: {train_label_counts[0]}, Parkinson: {train_label_counts[1]}, Other: {train_label_counts[2]}")
        print(f"Val subjects by label:")
        print(f"  Healthy: {val_label_counts[0]}, Parkinson: {val_label_counts[1]}, Other: {val_label_counts[2]}")
    
    # --------------------------------------------------------------
    # Option A: return all the movements in the same dataloader:
    # --------------------------------------------------------------
    if not per_movement:
        
        # 4. Create datasets with subject filtering
        train_dataset = TremorDataset(
            data_path=data_path,
            subject_ids=train_subjects,
            include_other=include_other,
            print_details=print_details,
        )
        
        val_dataset = TremorDataset(
            data_path=data_path,
            subject_ids=val_subjects,
            include_other=include_other,
            print_details=False,
        )
        
        # 5. WeightedRandomSampler for class imbalance in training set
        train_labels = np.array(train_dataset.labels)
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
            shuffle=False,
            pin_memory=True,
        )

        # 7. Print dataset info
        if print_details:
            print(f"\nSample counts:")
            print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
            print(f"\nTrain Class Distribution:")
            for cls, count in train_dataset.get_class_distribution().items():
                print(f"  {cls:12s}: {count:4d} ({count/len(train_dataset)*100:.1f}%)")
            print(f"{'='*60}\n")
            
        return train_dataloader, val_dataloader    
    
    # ----------------------------------------------------------
    # Option B: return each movement as a separate dataloader:
    # ----------------------------------------------------------
    else:
        # 4. Create train/val datasets with subject filtering
        train_dataset = TremorDataset(
            data_path=data_path,
            subject_ids=train_subjects,
            include_other=include_other,
            print_details=print_details,
        )
        
        val_dataset = TremorDataset(
            data_path=data_path,
            subject_ids=val_subjects,
            include_other=include_other,
            print_details=False,
        )
        
        # 5. Init movement dataloaders dict
        movement_dataloaders = {}

        # 6. Loop through every movement
        for movement_name in train_dataset.movement_names:
            movement_idx = train_dataset.movement_to_idx[movement_name]
            
            # Get indices for this movement in train and val
            train_movement_indices = [i for i, m in enumerate(train_dataset.movements) if m == movement_idx]
            val_movement_indices = [i for i, m in enumerate(val_dataset.movements) if m == movement_idx]
            
            # Skip if no samples for this movement
            if len(train_movement_indices) == 0 or len(val_movement_indices) == 0:
                if print_details:
                    print(f"[{movement_name}] Skipped - insufficient samples")
                continue

            # Create subsets
            from torch.utils.data import Subset
            train_subset = Subset(train_dataset, train_movement_indices)
            val_subset = Subset(val_dataset, val_movement_indices)

            # Weighted sampler for this movement
            train_labels = np.array([train_dataset.labels[i] for i in train_movement_indices])
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / np.maximum(class_counts, 1)
            sample_weights = class_weights[train_labels]

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