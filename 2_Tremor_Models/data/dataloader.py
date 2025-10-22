from dataset import TremorDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def create_tremor_dataloaders(
        data_path: str,
        batch_size: int = 8,
        train_val_split: float = 0.8,
        random_seed: int = 42):
    """
    Creates PyTorch DataLoaders for tremor movement classification.

    - Loads preprocessed .npz signals from the given movement folder.
    - Each sample contains a motion signal (with wrist encoded as 7th feature) and a label (0=Healthy, 1=Parkinson, 2=Other).
    - Splits dataset into train/validation subsets (stratified by label).
    - Returns DataLoaders ready for model training.

    Args:
        data_path (str): Path to one movement folder (contains 'Healthy/', 'PD/', 'Other/').
        batch_size (int): Batch size for training (default: 8).
        train_val_split (float): Fraction of data to use for training (default: 0.8).
        random_seed (int): Random seed for reproducibility (default: 42).

    Returns:
        (DataLoader, DataLoader): train_dataloader, val_dataloader
    """

    # 1. init dataset (loads all .npz files under Healthy/Parkinson/Other)
    full_dataset = TremorDataset(data_path=data_path, random_seed=random_seed)

    # 2. Split dataset indices into train/val (stratified by labels)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    labels = full_dataset.labels

    train_indices, val_indices = train_test_split(
        indices,
        test_size=1 - train_val_split,
        stratify=labels,  # keeps label ratios same
        random_state=random_seed
    )

    # 3. Create subsets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    # 4. Create DataLoaders
    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,      # shuffle training data
        pin_memory=True,   # faster transfers to GPU
        drop_last=True     # drop incomplete batch
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,     # no need to shuffle validation data
        pin_memory=True
    )

    # 5. Print dataset info
    print(f"Train dataset size: {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)}")
    print('-' * 35)
    print(f"Train dataloader size: {len(train_dataloader)}")
    print(f"Validation dataloader size: {len(val_dataloader)}")

    return train_dataloader, val_dataloader
