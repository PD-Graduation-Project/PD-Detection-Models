from .dataset import SpectrogramDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def create_dataloaders(
        healthy_dir: str,
        pd_dir: str,
        batch_size: int = 8,
        train_val_split: float = 0.8,
        img_size: tuple = (512, 512),
        random_seed: int = 42,
    ):
    """
    Creates PyTorch DataLoaders for spectrogram images for Parkinson’s Disease (PD) classification.

    - Builds train and validation datasets without any augmentation.
    - Splits data randomly by given ratio while keeping class balance.
    - Returns DataLoaders ready for training.

    Args:
        healthy_dir (str): Path to Healthy spectrogram images.
        pd_dir (str): Path to PD spectrogram images.
        batch_size (int): Batch size (default: 8).
        train_val_split (float): Train/val split ratio (default: 0.8).
        img_size (tuple): Resize (H, W) (default: (512, 512)).
        random_seed (int): Random seed for reproducibility (default: 42).

    Returns:
        (DataLoader, DataLoader): train_dataloader, val_dataloader
    """

    # 1. Load full dataset (train and val will be subsets)
    full_dataset = SpectrogramDataset(
        healthy_dir=healthy_dir,
        pd_dir=pd_dir,
        img_size=img_size,
        random_seed=random_seed
    )

    # 2. Create stratified train/val split
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    labels = full_dataset.labels  # 0=Healthy, 1=PD

    train_indices, val_indices = train_test_split(
        indices,
        test_size=1-train_val_split,
        stratify=labels,
        random_state=random_seed
    )

    # 3. Create PyTorch subsets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    # 4. Create DataLoaders
    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    print(f"Train dataset size: {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)}")
    print('-'*35)
    print(f"Train dataloader size: {len(train_dataloader)}")
    print(f"Validation dataloader size: {len(val_dataloader)}")

    return train_dataloader, val_dataloader
