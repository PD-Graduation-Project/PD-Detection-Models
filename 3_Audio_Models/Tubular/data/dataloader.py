import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Parkinson Dataset
# -------------------
class ParkinsonDataset(Dataset):
    """
    A PyTorch Dataset for the Parkinson CSV dataset.

    Args:
        csv_path (str): Path to the CSV file.

    Behavior:
        - Loads the CSV using pandas.
        - Features = all columns except 'status'.
        - Target = 'status'.
        - Converts both to float32 PyTorch tensors.
    """
    def __init__(self, csv_path: str):
        super().__init__()

        # 1. Load CSV file
        df = pd.read_csv(csv_path)

        # 2. Separate features and labels
        self.y = df['status'].values.astype('float32')
        self.X = df.drop(columns=['status']).values.astype('float32')
        
        # 3. Normalizing data
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)

        # 4. Convert to torch tensors
        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)
        
        
    # Required Dataset method
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Function: Create Train/Test DataLoaders
# -----------------------------------------
def create_dataloaders(
        csv_path: str,
        train_val_split: float = 0.8,
        batch_size: int = 16,
        random_seed: int = 42
    ):
    """
    Creates training and validation DataLoaders from the CSV dataset.

    Args:
        csv_path (str): Path to CSV file.
        train_val_split (float): Fraction for training (e.g., 0.8 → 80% train).
        batch_size (int): Batch size for DataLoaders.
        random_seed (int): Seed for reproducible splitting.

    Returns:
        (train_loader, val_loader)
    """
    # 1. Load full datase
    dataset = ParkinsonDataset(csv_path)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # 2. labels needed for stratified split
    labels = dataset.y.numpy()
    
    # 3. Stratified train/validation spli
    train_indices, val_indices = train_test_split(
        indices,
        test_size=1 - train_val_split,
        stratify=labels,
        random_state=random_seed
    )
    
    # 4. Create Subset
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    # 5. Create DataLoader
    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,         # shuffle training batches
        pin_memory=True,      # performance optimization
        drop_last=True        # ensures fixed batch size
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,        # no shuffling for validation/testing
        pin_memory=True
    )
    
    # 6. Debug informatio
    print("=========================================")
    print(f"Train dataset size:      {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)}")
    print("-----------------------------------------")
    print(f"Train dataloader steps:  {len(train_dataloader)}")
    print(f"Val dataloader steps:    {len(val_dataloader)}")
    print("=========================================")

    return train_dataloader, val_dataloader
