from .dataset import TremorDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

def create_tremor_dataloaders(
        data_path: str,
        batch_size: int = 8,
        train_val_split: float = 0.8,
        random_seed: int = 42,
        load_all: bool = False):
    """
    Creates PyTorch DataLoaders for tremor movement classification.
    
    If `load_all=True`, it automatically loads dataloaders for **all movement folders**
    inside `data_path`, each containing its own Healthy/Parkinson/Other subfolders.

    - Loads preprocessed .npz signals from the given movement folder.
    - Each sample contains:
        (signal_tensor, wrist_tensor, label_tensor)
            - signal_tensor : shape (T, 6), IMU signal
            - wrist_tensor  : scalar (0 = Left, 1 = Right)
            - label_tensor  : scalar (0 = Healthy, 1 = Parkinson, 2 = Other)
    - Splits dataset into train/validation subsets (stratified by label).
    - Returns DataLoaders ready for model training.

    Args:
        data_path (str): 
            - If `load_all=False`: path to one movement folder.
            - If `load_all=True`: path to the root folder containing multiple movement folders.
        batch_size (int): Batch size for training (default: 8).
        train_val_split (float): Fraction of data to use for training (default: 0.8).
        random_seed (int): Random seed for reproducibility (default: 42).
        load_all (bool): If True, create dataloaders for all movement folders under `data_path`.

    Returns:
        If load_all=False:
            (DataLoader, DataLoader)
        If load_all=True:
            dict[str, tuple[DataLoader, DataLoader]]  # movement_name -> (train_loader, val_loader)
    """
    
    # Function to create dataloaders per folder
    # -------------------------------------------
    def make_loaders_for_folder(folder_path: str):
        # 1. init dataset (loads all .npz files under Healthy/Parkinson/Other)
        full_dataset = TremorDataset(data_path=folder_path, random_seed=random_seed)

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

        # 5. Print dataset info and return
        print(f"[{Path(folder_path).name}] -> Train: {len(train_subset)}, Val: {len(val_subset)}")
        
        return train_dataloader, val_dataloader

    # Single folder mode
    # -------------------
    if not load_all:
        return (make_loaders_for_folder(data_path))
    
    # Multi-folder mode
    # -------------------
    data_root = Path(data_path)
    loaders_dict = {}
    
    print(f"\nLoading dataloaders from {data_root}...\n")
    
    # iterate through the root directory
    movement_dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    for movement_dir in tqdm(movement_dirs, desc="Creating dataloaders", ncols=80):
        loaders_dict[movement_dir.name] = make_loaders_for_folder(movement_dir)

    print(f"\nLoaded dataloaders for {len(loaders_dict)} movements.")
    return loaders_dict