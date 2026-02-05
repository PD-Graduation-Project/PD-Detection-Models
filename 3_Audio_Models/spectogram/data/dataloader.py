from .dataset import SpectrogramDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def create_dataloaders(
    healthy_dir: str,
    pd_dir: str,
    batch_size: int = 8,
    img_size:tuple = (512, 512),
    
    train_val_split: float = 0.8,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 64,
    
    spectrogram_type: str = 'mel',  # 'linear' or 'mel'
    random_seed: int = 42,
):
    """
    Creates DataLoaders for spectrogram-based PD classification from raw .wav files.

    Args:
        healthy_dir (str): Path to Healthy wav files.
        pd_dir (str): Path to PD wav files.
        batch_size (int): Batch size.
        train_val_split (float): Train/val split ratio.
        sample_rate (int): Target sample rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        n_mels (int): Number of mel bands.
        spectrogram_type (str): 'linear' or 'mel'.
        random_seed (int): Random seed.

    Returns:
        (DataLoader, DataLoader): train_dataloader, val_dataloader
    """

    # create datasets
    train_dataset = SpectrogramDataset(
        healthy_dir=healthy_dir,
        pd_dir=pd_dir,
        img_size=img_size,
        
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        spectrogram_type=spectrogram_type,
        augment=True,
    )

    val_dataset = SpectrogramDataset(
        healthy_dir=healthy_dir,
        pd_dir=pd_dir,
        img_size=img_size,
        
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        spectrogram_type=spectrogram_type,
        augment=False,
    )

    # stratified split
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    labels = train_dataset.labels

    train_indices, val_indices = train_test_split(
        indices,
        test_size=1 - train_val_split,
        stratify=labels,
        random_state=random_seed,
    )

    # subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # dataloaders
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