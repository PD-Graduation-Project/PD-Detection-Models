from .dataset import *
from numpy import random
from torch.utils.data import DataLoader, Subset

def create_dataloaders(
                healthy_dir:str,
                pd_dir:str,
                batch_size:int = 8,
                train_val_split:float = 0.8,
                img_size:tuple = (512, 512),
                random_seed :int = 42,):
    """
    Creates PyTorch DataLoaders for spiral & wave drawing PD classification.

    - Builds train and validation datasets with different transforms.
    - Splits data randomly by given ratio.
    - Returns DataLoaders ready for training.

    Args:
        healthy_dir (str): Path to Healthy images.
        pd_dir (str): Path to PD images.
        batch_size (int): Batch size (default: 8).
        train_val_split (float): Train/val split ratio (default: 0.8).
        img_size (tuple): Resize (H, W) (default: (512, 512)).
        random_seed (int): Random seed for reproducibility (default: 42).

    Returns:
        (DataLoader, DataLoader): train_dataloader, val_dataloader
    """
    
    # 1. get 2 datasets: one for training and the other for validating
    # This still scans the directory twice but is necessary to assign different transforms to training dataset.
    train_dataset = DrawingDataset(
        healthy_dir=healthy_dir,
        pd_dir=pd_dir,
        
        transforms=train_transforms(),
        img_size=img_size,
    )
    val_dataset = DrawingDataset( # using default transforms only
        healthy_dir=healthy_dir,
        pd_dir=pd_dir,
        
        img_size=img_size,
    )
    
    # 2. create random split of indices
    dataset_size = len(train_dataset) # total dataset size
    indices = list(range(dataset_size)) # list of all indices [0, 1, 2, ..., dataset_size - 1]
    
    split = int(train_val_split * dataset_size)
    random.seed(random_seed)
    random.shuffle(indices) # shuffle the dataset indices randomly
    train_indices, val_indices = indices[:split], indices[split:]
    
    # 3. create pytorch subsets of the original dataset
    train_subset = Subset(train_dataset, 
                            train_indices)
    val_subset = Subset(val_dataset, 
                            val_indices)
    
    # 4. create dataloaders
    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True, # shuffle training data
        pin_memory=True, # copy data to CUDA (faster training)
        drop_last=True, # drop the last incomplete batch
    )
    
    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False, # no need to shuffle testing data
        pin_memory=True,
    )
    
    print(f"Train dataset size: {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)}")
    print('-'*35)
    print(f"Train dataloader size: {len(train_dataloader)}")
    print(f"Validation dataloader size: {len(val_dataloader)}")
    
    return train_dataloader, val_dataloader