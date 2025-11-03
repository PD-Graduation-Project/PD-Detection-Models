import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random

class TremorDataset(Dataset):
    """
    PyTorch Dataset for loading tremor movement signals from ALL movements.

    Expected folder layout:
    
    data_path/
        ├── Movement1/  (e.g., "CrossArms")
        │   ├── Healthy/
        │   │     ├── 001.npz
        │   │     ├── 002.npz
        │   │     └── ...
        │   ├── Parkinson/
        │   │     ├── 003.npz
        │   │     └── ...
        │   └── Other/
        │         └── ...
        ├── Movement2/  (e.g., "FingerNose")
        │   ├── Healthy/
        │   ├── Parkinson/
        │   └── Other/
        └── ...

    Each .npz must contain:
        - signal : tuple of 2 np.ndarrays
                    ((1024, 6), (1024, 6)) -> (Left, Right)
        - label  : int (0 = Healthy, 1 = Parkinson, 2 = Other)
        - wrist  : int (0 = Left-handed, 1 = Right-handed) 
        - subject_id : int or str (optional)

    Parameters
    ----------
    data_path : str or Path
        Path to the root directory containing all movement folders.
        
    movement_names : list of str, optional
        List of movement folder names to include. If None, automatically detects
        all subdirectories in data_path.
        
    random_seed : int, optional (default=42)
        Seed used to shuffle the dataset for reproducibility.
        
    include_other : bool, default=True
        Whether to include the "Other" class (label=2)
        
    print_details : bool, default=False
        Print dataset info

    Returns
    -------
    tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        (signal_tensor, handedness_tensor, movement_tensor, label_tensor) per sample.
        - signal_tensor     : shape (2, T, 6), dtype=torch.float32  # (Left, Right)
        - handedness_tensor : scalar (0 = Left-handed, 1 = Right-handed), dtype=torch.long
        - movement_tensor   : scalar (movement index 0-10), dtype=torch.long
        - label_tensor      : scalar (0 = Healthy, 1 = Parkinson, 2 = Other), dtype=torch.long
    """
    def __init__(self,
                data_path: str,
                movement_names: list = None,
                random_seed: int = 42,
                include_other: bool = True,
                print_details: bool = False):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.include_other = include_other
        
        # 1. Movements inits
        # -------------------
        # 1.1. Automatically detect movement folders if not provided
        if movement_names is None:
            movement_names = sorted([
                d.name for d in self.data_path.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ])
        
        # 1.2. Create movement name to index mapping
        self.movement_names = movement_names
        self.num_movements = len(movement_names)
        self.movement_to_idx = {name: idx for idx, name in enumerate(movement_names)}
        
        if print_details:
            print(f"Found {self.num_movements} movements: {movement_names}")
        
        # 2. Init lists for all data
        all_samples = []
        
        # 3. Load data from each movement folder
        # ----------------------------------------
        for movement_idx, movement_name in enumerate(movement_names):
            movement_path = self.data_path / movement_name
            
            # 3.1. check for dir availability
            if not movement_path.exists():
                print(f"Warning: Movement folder '{movement_name}' not found, skipping...")
                continue
            
            # 3.2. init Healthy, Parkinson, Other subfolders dir
            dirs = {
                0: movement_path / "Healthy",
                1: movement_path / "Parkinson",
                2: movement_path / "Other",
            }
            
            # 4. Helper function to process .npz files
            # -------------------------------------------
            def process_npz(file, label):
                """Load (Left, Right) signals and handedness from .npz file"""
                npz = np.load(file, allow_pickle=True)
                
                # 4.1. extract tuple of both wrist signals
                left_signal, right_signal = npz["signal"]
                
                # 4.2. stack them into (2, T, 6)
                signal = np.stack([left_signal, right_signal], axis=0).astype(np.float32)
                
                # 4.3. extract handedness (0 = Left-handed, 1 = Right-handed)
                handedness = int(npz["wrist"])
                
                # 4.4. finally return everything separately
                return signal, handedness, movement_idx, label

            # 5. add all data
            # -----------------
            for label, dir_path in dirs.items():
                # 5.1. skip the 'other' label if 'include_other' is False
                if label == 2 and not include_other:
                    continue
                
                # 5.2. get all '.npz' data in each directory
                if dir_path.exists():
                    for file in dir_path.glob("*.npz"):
                        result = process_npz(file, label)
                        if result is not None:
                            all_samples.append(result)
            
            if print_details:
                print(f"  Loaded {movement_name}: {len([s for s in all_samples if s[2] == movement_idx])} samples")
        
        # 6. Shuffle all samples
        random.seed(random_seed)
        random.shuffle(all_samples)
        
        # 7. Split into separate lists
        self.signals = [s[0] for s in all_samples]
        self.handedness = [s[1] for s in all_samples]
        self.movements = [s[2] for s in all_samples]
        self.labels = [s[3] for s in all_samples]
        
        if print_details:
            print(f"\nTotal samples loaded: {len(self.signals)}")
            print(f"  Healthy: {self.labels.count(0)}")
            print(f"  Parkinson: {self.labels.count(1)}")
            if include_other:
                print(f"  Other: {self.labels.count(2)}")
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, index):
        """
        Returns
        -------
        tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            (signal_tensor, handedness_tensor, movement_tensor, label_tensor)
        """
        # 1. Signal 
        # ----------
        signal = torch.tensor(
            self.signals[index],
            dtype=torch.float32
        )  # shape: (2, T, 6)
        
        # 2. wrist (handedness)
        # ----------------------
        handedness = torch.tensor(
            self.handedness[index],
            dtype=torch.long
        )
        
        # 3. movement
        # -------------
        movement = torch.tensor(
            self.movements[index],
            dtype=torch.long
        )
        
        # 4. label
        # --------- 
        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )
        
        return signal, handedness, movement, label
    
    def get_movement_name(self, movement_idx):
        """Get movement name from index"""
        return self.movement_names[movement_idx]
    
    def get_class_distribution(self):
        """Get distribution of classes across dataset"""
        return {
            'Healthy': self.labels.count(0),
            'Parkinson': self.labels.count(1),
            'Other': self.labels.count(2)
        }
    
    def get_movement_distribution(self):
        """Get distribution of samples per movement"""
        movement_counts = {}
        for idx, name in enumerate(self.movement_names):
            movement_counts[name] = self.movements.count(idx)
        return movement_counts
