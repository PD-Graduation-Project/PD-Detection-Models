import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random

class TremorDataset(Dataset):
    """
    PyTorch Dataset for loading tremor movement signals from preprocessed .npz files.

    Each dataset corresponds to a single movement (e.g., "CrossArms"), and the expected
    folder layout is:

    data_path/
        ├── Healthy/
        │     ├── 001_L.npz
        │     ├── 002_R.npz
        │     └── ...
        ├── Parkinson/
        │     ├── 003_L.npz
        │     └── ...
        └── Other/
              └── ...

    Each .npz must contain:
        - signal : np.ndarray, shape (T, 6)  # IMU channels
        - label  : int (0 = Healthy, 1 = Parkinson, 2 = Other)
        - wrist  : int (0 = Left, 1 = Right) 
        - subject_id : int or str

    Parameters
    ----------
    data_path : str or Path
        Path to the root directory for one movement (contains Healthy/ Parkinson/ Other).
        
    random_seed : int, optional (default=42)
        Seed used to shuffle the dataset for reproducibility.

    Returns
    -------
    tuple(torch.Tensor, torch.Tensor)
        (signal_tensor, label_tensor) per sample. signal_tensor dtype=torch.float32,
        label_tensor dtype=torch.long.
    """
    def __init__(self,
                data_path: str,
                random_seed:int = 42):
        super().__init__()
        
        # 0. init dirs with respect to root data dir
        healthy_dir = Path(data_path) / "Healthy"
        parkinson_dir = Path(data_path) / "Parkinson"
        other_dir = Path(data_path) / "Other"
        
        # 1. init empty lists of data
        healthy = []
        parkinson = []
        other = []
        
        # 2. create data tensors
        # ------------------------
        def process_npz(file):
            # load the whole .npz file
            npz = np.load(file)
            # load the signal
            signal = npz["signal"].astype(np.float32)
            # load the wrist
            wrist = int(npz["wrist"])
            # create a wrist col and add it to the signals (now the input is 7 instead of 6)
            wrist_col = np.full((signal.shape[0], 1), wrist, dtype=np.float32)
            signal = np.concatenate([signal, wrist_col], axis=1)
            
            # finally return the final combined signal
            return signal
        
        # 2.1 Healthy
        for h in healthy_dir.glob("*.npz"):
            signal = process_npz(h)
            healthy.append(signal)
        
        # 2.2. Parkinson
        for pd in parkinson_dir.glob("*.npz"):
            signal = process_npz(pd)
            parkinson.append(signal)
            
        # 2.3. Other
        for o in other_dir.glob("*.npz"):
            signal = process_npz(o)
            other.append(signal)
            
            
        # 3. combine them and create labels tensor
        signals = healthy + parkinson + other
        labels = [0]*len(healthy) + [1]*len(parkinson) + [2]*len(other)
        
        # 4. combine labels and signals
        combined = list( zip(signals, labels) )
        
        # 5, shuffle them
        random.seed(random_seed)
        random.shuffle(combined)
        
        # 6. split them again
        self.signals, self.labels = zip(*combined)
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, index):
        """
        returns (signal_tensor, label_tensor)
        """
        x = torch.tensor(
            self.signals[index],
            dtype=torch.float32
        )
        y = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )
        
        return x, y