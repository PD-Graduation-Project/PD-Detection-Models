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
        │   │     ├── 001_L.npz
        │   │     ├── 002_R.npz
        │   │     └── ...
        │   ├── Parkinson/
        │   │     ├── 003_L.npz
        │   │     └── ...
        │   └── Other/
        │         └── ...
        ├── Movement2/  (e.g., "FingerNose")
        │   ├── Healthy/
        │   ├── Parkinson/
        │   └── Other/
        └── ...

    Each .npz must contain:
        - signal : np.ndarray, shape (T, 6)  # IMU channels (T can be 1024 or 2048)
        - label  : int (0 = Healthy, 1 = Parkinson, 2 = Other)
        - wrist  : int (0 = Left, 1 = Right) 
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

    Returns
    -------
    tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        (signal_tensor, wrist_tensor, movement_tensor, label_tensor) per sample.
        - signal_tensor   : shape (T, 6), dtype=torch.float32
        - wrist_tensor    : scalar (0 = Left, 1 = Right), dtype=torch.long
        - movement_tensor : scalar (movement index 0-10), dtype=torch.long
        - label_tensor    : scalar (0 = Healthy, 1 = Parkinson, 2 = Other), dtype=torch.long
    """
    def __init__(self,
                data_path: str,
                movement_names: list = None,
                random_seed: int = 42,
                print_details: bool = False):
        super().__init__()
        
        self.data_path = Path(data_path)
        
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
            healthy_dir = movement_path / "Healthy"
            parkinson_dir = movement_path / "Parkinson"
            other_dir = movement_path / "Other"
            
            # 4. Helper function to process .npz files
            # -------------------------------------------
            def process_npz(file, label):
                """Load signal and wrist from .npz file"""
                # 4.1. load the .npz file
                npz = np.load(file)
                
                # 4.2. extract the IMU signal
                signal = npz["signal"].astype(np.float32)
                # 4.3. extract wrist indicator (0 = Left, 1 = Right)
                wrist = int(npz["wrist"])
                
                # 4.4. # finally return everything separately
                return signal, wrist, movement_idx, label

            # 5. add all data
            # -----------------
            # 5.1. Load Healthy samples (label=0)
            if healthy_dir.exists():
                for file in healthy_dir.glob("*.npz"):
                    result = process_npz(file, label=0)
                    if result is not None:
                        all_samples.append(result)
            
            # 5.2. Load Parkinson samples (label=1)
            if parkinson_dir.exists():
                for file in parkinson_dir.glob("*.npz"):
                    result = process_npz(file, label=1)
                    if result is not None:
                        all_samples.append(result)
            
            # 5.3. Load Other samples (label=2)
            if other_dir.exists():
                for file in other_dir.glob("*.npz"):
                    result = process_npz(file, label=2)
                    if result is not None:
                        all_samples.append(result)
            
            if print_details:
                print(f"  Loaded {movement_name}: {len([s for s in all_samples if s[2] == movement_idx])} samples")
        
        # 6. Shuffle all samples
        random.seed(random_seed)
        random.shuffle(all_samples)
        
        # 7. Split into separate lists
        self.signals = [s[0] for s in all_samples]
        self.wrists = [s[1] for s in all_samples]
        self.movements = [s[2] for s in all_samples]
        self.labels = [s[3] for s in all_samples]
        
        if print_details:
            print(f"\nTotal samples loaded: {len(self.signals)}")
            print(f"  Healthy: {self.labels.count(0)}")
            print(f"  Parkinson: {self.labels.count(1)}")
            print(f"  Other: {self.labels.count(2)}")
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, index):
        """
        Returns
        -------
        tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            (signal_tensor, wrist_tensor, movement_tensor, label_tensor)
        """
        signal = torch.tensor(
            self.signals[index],
            dtype=torch.float32
        )
        wrist = torch.tensor(
            self.wrists[index],
            dtype=torch.long
        )
        movement = torch.tensor(
            self.movements[index],
            dtype=torch.long
        )
        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )
        
        return signal, wrist, movement, label
    
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
