from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
import torch

# custom collate function
# -------------------------
def collate_variable_length(batch):
    """
    Custom collate function to handle variable-length sequences (1024 or 2048).
    
    Pads sequences to the maximum length in the batch.
    
    Args:
        batch: List of tuples (signal, wrist, movement, label)
    
    Returns:
        Tuple of (signals_padded, wrists, movements, labels)
    """
    
    # 1. unpack data in batch - each item is (signal, wrist, movement, label)
    signals = [item[0] for item in batch]
    wrists = torch.stack([item[1] for item in batch])
    movements = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])

    # 2. pad signals to max length in batch (if some samples are 1024 and there is a 2048 sample in the batch)
    # pad_sequence expects (batch, seq_len, features) format
    signals_padded = pad_sequence(
        signals,
        batch_first=True,
        padding_value=0.0
    )
    
    return signals_padded, wrists, movements, labels


# dataloader creator function
# -----------------------------
def create_tremor_dataloaders(
        data_path: str,
        batch_size: int = 32,
        train_val_split: float = 0.8,
        random_seed: int = 42):
    """
    Creates PyTorch DataLoaders for tremor movement classification across all movements.
    
    - Loads preprocessed .npz signals from all movement folders.
    - Each sample contains:
        (signal_tensor, wrist_tensor, movement_tensor, label_tensor)
            - signal_tensor   : shape (T, 6), IMU signal
            - wrist_tensor    : scalar (0 = Left, 1 = Right)
            - movement_tensor : scalar (0-10), movement type
            - label_tensor    : scalar (0 = Healthy, 1 = Parkinson, 2 = Other)
    - Splits dataset into train/validation subsets (stratified by label AND movement).
    - Returns DataLoaders ready for model training.

    Args:
        data_path (str): Path to root folder containing all movement subfolders.
        batch_size (int): Batch size for training (default: 32).
        train_val_split (float): Fraction of data to use for training (default: 0.8).
        random_seed (int): Random seed for reproducibility (default: 42).
        num_workers (int): Number of workers for DataLoader (default: 4).

    Returns:
        (DataLoader, DataLoader): train_loader, val_loader
    """
    
    # 0. Import here to avoid circular imports
    from dataset import TremorDataset
    
    # 1. Init unified dataset (loads all movements)
    full_dataset = TremorDataset(
        data_path=data_path,
        random_seed=random_seed
    )

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

    # 5. Create DataLoaders
    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_variable_length # NEW
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_variable_length # NEW
    )

    # 6. Print dataset info
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