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
        
    Notes
    -----
    The wrist information is **not concatenated** to the signal as an extra channel.
    Instead, it is stored and returned as a separate scalar tensor (0 or 1) to avoid
    redundant repetition across all timesteps.

    Parameters
    ----------
    data_path : str or Path
        Path to the root directory for one movement (contains Healthy/ Parkinson/ Other).
        
    random_seed : int, optional (default=42)
        Seed used to shuffle the dataset for reproducibility.

    Returns
    -------
    tuple(torch.Tensor, torch.Tensor, torch.Tensor)
        (signal_tensor, wrist_tensor, label_tensor) per sample.
        - signal_tensor : shape (T, 6), dtype=torch.float32
        - wrist_tensor  : scalar (0 = Left, 1 = Right), dtype=torch.long
        - label_tensor  : scalar (0 = Healthy, 1 = Parkinson, 2 = Other), dtype=torch.long
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
        
        # 2.0 function to load and process individual .npz files
        def process_npz(file):
            # load the .npz file
            npz = np.load(file)
            # extract the IMU signal
            signal = npz["signal"].astype(np.float32)
            # extract wrist indicator (0 = Left, 1 = Right)
            wrist = int(npz["wrist"])

            # finally return both separately
            return signal, wrist
        
        # 2.1 Healthy
        for h in healthy_dir.glob("*.npz"):
            signal, wrist = process_npz(h)
            healthy.append((signal, wrist))
        
        # 2.2. Parkinson
        for pd in parkinson_dir.glob("*.npz"):
            signal, wrist = process_npz(pd)
            parkinson.append((signal, wrist))
            
        # 2.3. Other
        for o in other_dir.glob("*.npz"):
            signal, wrist = process_npz(o)
            other.append((signal, wrist))
            
            
        # 3. combine all signals and wrists, and create label list
        signals, wrist = zip(*[x for x in healthy + parkinson + other])
        labels = [0]*len(healthy) + [1]*len(parkinson) + [2]*len(other)
        
        # 4. combine signals, wrists, and labels into a single list
        combined = list( zip(signals, wrist, labels) )
        
        # 5, shuffle them
        random.seed(random_seed)
        random.shuffle(combined)
        
        # 6. split them again
        self.signals, self.wrists, self.labels = zip(*combined)
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, index):
        """
        Returns
        -------
        tuple(torch.Tensor, torch.Tensor, torch.Tensor)
            (signal_tensor, wrist_tensor, label_tensor)
        """
        x = torch.tensor(
            self.signals[index],
            dtype=torch.float32
        )
        wrist = torch.tensor(
            self.wrists[index],
            dtype=torch.long
        )
        y = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )
        
        return x, wrist, y