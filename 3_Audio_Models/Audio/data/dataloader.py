from .dataset import AudioDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def create_dataloaders(
    healthy_dir: str,
    pd_dir: str,
    batch_size: int = 8,
    train_val_split: float = 0.8,
    sample_rate: int = 16000,
    max_length: float = 5.0,   # seconds
    random_seed: int = 42,
):
    """
    Creates PyTorch DataLoaders for audio-based PD classification
    using raw waveform input (torchaudio pipelines).

    Args:
        healthy_dir (str): Path to Healthy wav files.
        pd_dir (str): Path to PD wav files.
        batch_size (int): Batch size.
        train_val_split (float): Train/val split ratio.
        sample_rate (int): Target sample rate.
        max_length (float): Max audio length in seconds.
        random_seed (int): Random seed.

    Returns:
        (DataLoader, DataLoader): train_dataloader, val_dataloader
    """

    # 1. create datasets (separate for transforms if needed later)
    train_dataset = AudioDataset(
        healthy_dir=healthy_dir,
        pd_dir=pd_dir,
        sample_rate=sample_rate,
        max_length=max_length,
        augment=True,     # optional (noise, gain, etc.)
    )

    val_dataset = AudioDataset(
        healthy_dir=healthy_dir,
        pd_dir=pd_dir,
        sample_rate=sample_rate,
        max_length=max_length,
        augment=False,
    )

    # 2. stratified split
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    labels = train_dataset.labels

    train_indices, val_indices = train_test_split(
        indices,
        test_size=1 - train_val_split,
        stratify=labels,
        random_state=random_seed,
    )

    # 3. subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # 4. dataloaders
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
    print("-" * 35)
    print(f"Train dataloader size: {len(train_dataloader)}")
    print(f"Validation dataloader size: {len(val_dataloader)}")

    return train_dataloader, val_dataloader
