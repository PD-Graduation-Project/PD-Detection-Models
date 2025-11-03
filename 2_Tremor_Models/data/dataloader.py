from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
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
    
    - Loads preprocessed .npz signals from all movement folders.
    - Each sample contains:
        (signal_tensor, wrist_tensor, movement_tensor, label_tensor)
            - signal_tensor   : shape (2, T, 6), IMU signal (left-signals, right-signals)
            - wrist_tensor    : scalar (0 = Left-handed, 1 = Right-handed)
            - movement_tensor : scalar (0-10), movement type
            - label_tensor    : scalar (0 = Healthy, 1 = Parkinson, 2 = Other)
    - Splits dataset into train/validation subsets (stratified by label AND movement).
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
            Fraction of data used for training.
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
    
    # 1. Init unified dataset (loads all movements)
    full_dataset = TremorDataset(
        data_path=data_path,
        random_seed=random_seed,
        include_other= include_other,
        print_details = print_details,
    )
    
    # --------------------------------------------------------------
    # Option A: return all the movements in the same dataloader:
    # --------------------------------------------------------------
    if not per_movement:
        
        # 2.1. Create stratification key: combine label and movement
        # This ensures each split has good representation of all movements AND classes
        dataset_size = len(full_dataset)
        indices = list(range(dataset_size))
        
        # 2.2. Create combined stratification labels (label * num_movements + movement)
        # This ensures proportional split across both dimensions
        stratify_keys = [
            full_dataset.labels[i] * full_dataset.num_movements + full_dataset.movements[i]
            for i in indices
        ]

        # 3. Split dataset indices into train/val (stratified by label + movement)
        train_indices, val_indices = train_test_split(
            indices,
            test_size=1 - train_val_split,
            stratify=stratify_keys,
            random_state=random_seed
        )

        # 4. Create subsets
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        # 5. WeightedRandomSampler for class imbalance
        # ------------------------------------------------
        # 5.1.Extract labels for the training subset
        train_labels = np.array([full_dataset.labels[i] for i in train_indices])
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)  # avoid divide by zero
        sample_weights = class_weights[train_labels]

        # 5.2. create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # 6. Create DataLoaders
        train_dataloader = DataLoader(
            train_subset,
            batch_size=batch_size,
            # shuffle=True, # sampler option is mutually exclusive with shuffle
            pin_memory=True,
            drop_last=True,
            sampler= sampler # [NEW] using sampler for classes imbalance
        )

        val_dataloader = DataLoader( # no sampler always plain
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        # 6. Print dataset info
        if print_details:
            print(f"\n{'='*60}")
            print(f"Dataset Summary")
            print(f"{'='*60}")
            print(f"Total samples: {dataset_size}")
            print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")
            print(f"Movements: {full_dataset.num_movements}")
            print(f"\nClass Distribution:")
            for cls, count in full_dataset.get_class_distribution().items():
                print(f"  {cls:12s}: {count:4d} ({count/dataset_size*100:.1f}%)")
            print(f"{'='*60}\n")
            
        return train_dataloader, val_dataloader    
    
    # ----------------------------------------------------------
    # Option B: return each movement as a seperate dataloader:
    # ----------------------------------------------------------
    else:
        # 2. init movement dataloaders dict
        movement_dataloaders = {}

        # 3. loop through every movement
        for movement_name in full_dataset.movement_names:
            # 4. Get indices for this movement
            movement_idx = full_dataset.movement_to_idx[movement_name]
            movement_indices = [i for i, m in enumerate(full_dataset.movements) if m == movement_idx]

            # 5. Split train/val for this movement (stratify only by label)
            movement_labels = [full_dataset.labels[i] for i in movement_indices]

            train_idx, val_idx = train_test_split(
                movement_indices,
                test_size=1 - train_val_split,
                stratify=movement_labels if len(set(movement_labels)) > 1 else None,
                random_state=random_seed
            )

            # 6. Create subsets
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            # 7. Weighted sampler for movement
            train_labels = np.array([full_dataset.labels[i] for i in train_idx])
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / np.maximum(class_counts, 1)
            sample_weights = class_weights[train_labels]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            # 8. Dataloaders
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

        return movement_dataloaders
    
    
    