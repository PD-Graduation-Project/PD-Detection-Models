import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm

class TremorDataset(Dataset):
    """
    MODIFIED: Now loads FEATURES instead of raw signals.
    
    Expected .npz format:
        - features   : 1D array of shape (num_features,)  # NEW
                        Size depends on preprocessing:
                        - Paper method (magnitude + 1 hand): 22 features
                        - Both hands magnitude: 44 features  
                        - Both hands all axes: 132 features
        - label      : int (0 = Healthy, 1 = Parkinson)
        - handedness : int (0 = Left, 1 = Right)
        - subject_id : int
        - movement_name, segment_idx : metadata

    Returns per sample:
        (features, handedness, movement, label)
        - features  : shape (num_features,) - variable based on preprocessing
        - handedness: scalar (0 or 1)
        - movement  : scalar (movement index)
        - label     : scalar (0 or 1)
    """
    def __init__(self,
                data_path: str,
                movement_names: list = None,
                subject_ids: list = None):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.subject_ids = set(subject_ids) if subject_ids is not None else None
        
        # 1. Auto-detect movements
        if movement_names is None:
            movement_names = sorted([
                d.name for d in self.data_path.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ])
        
        self.movement_names = movement_names
        all_samples = []
        
        # 2. Load data from each movement folder
        for movement_idx, movement_name in enumerate(
            tqdm(movement_names, desc="Loading features", total=len(movement_names))):
            
            movement_path = self.data_path / movement_name
            if not movement_path.exists():
                continue
            
            # 2.1. Load from Healthy and Parkinson folders
            dirs = {
                0: movement_path / "Healthy",
                1: movement_path / "Parkinson",
            }
            
            for label, dir_path in dirs.items():
                if dir_path.exists():
                    for file in dir_path.glob("*.npz"):
                        result = self._process_npz(file, label, movement_idx)
                        if result is not None:
                            all_samples.append(result)
        
        # 3. Store samples
        self.features = [s[0] for s in all_samples]  # CHANGED: features not signals
        self.handedness = [s[1] for s in all_samples]
        self.movements = [s[2] for s in all_samples]
        self.labels = [s[3] for s in all_samples]
        self.subject_ids_list = [s[4] for s in all_samples]
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        """
        Returns: (features, handedness, movement, label)
        """
        # NEW: Load feature vector instead of raw signal
        features = torch.tensor(
            self.features[index],
            dtype=torch.float32
        )  # shape: (num_features,) e.g., (264,)
        
        handedness = torch.tensor(self.handedness[index], dtype=torch.long)
        movement = torch.tensor(self.movements[index], dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        
        return features, handedness, movement, label
    
    def _process_npz(self, file, label, movement_idx):
        """
        MODIFIED: Load features instead of raw signals
        """
        npz = np.load(file, allow_pickle=True)
        
        # Filter by subject
        subject_id = int(npz["subject_id"])
        if self.subject_ids is not None and subject_id not in self.subject_ids:
            return None
        
        # NEW: Load feature vector instead of signal
        features = npz["features"].astype(np.float32)
        
        # Keep everything else the same
        handedness = int(npz["handedness"])
        
        return features, handedness, movement_idx, label, subject_id
    
    # Keep all helper methods unchanged
    def get_movement_name(self, movement_idx):
        return self.movement_names[movement_idx]
    
    def list_movements(self):
        return {name: idx for idx, name in enumerate(self.movement_names)}
    
    def get_class_distribution(self):
        return {
            'Healthy': self.labels.count(0),
            'Parkinson': self.labels.count(1),
        }
    
    def get_movement_distribution(self):
        movement_counts = {}
        for idx, name in enumerate(self.movement_names):
            movement_counts[name] = self.movements.count(idx)
        return movement_counts
    
    def get_unique_subjects(self):
        return sorted(set(self.subject_ids_list))