import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd 
from tqdm import tqdm

class TremorDataset(Dataset):
    """
    MODIFIED: Now loads FEATURES instead of raw signals.
    Supports loading from:
        - .npz feature files (original behavior)
        - CSV file (new, optional)
    
    Expected .npz format:
        - features   : 1D array of shape (num_features,)  # NEW
                        Size depends on preprocessing
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
                data_path: str = None,
                csv_path: str = None,
                movement_names: list = None,
                subject_ids: list = None):
        super().__init__()
        
        self.subject_ids = set(subject_ids) if subject_ids is not None else None

        # =========================
        # NEW: Load from CSV
        # =========================
        if csv_path is not None:
            df = pd.read_csv(csv_path)

            # 1. Ensure numeric
            df = df.apply(pd.to_numeric, errors="coerce")
            
            # 1.5 Filter by Subject ID if requested (CRITICAL FIX FOR LEAKAGE)
            if self.subject_ids is not None and "subject_id" in df.columns:
                df = df[df["subject_id"].isin(self.subject_ids)]

            # 2. Auto-detect feature columns
            feature_cols = [
                c for c in df.columns
                if c.startswith("lh_") or c.startswith("rh_")
            ]

            self.features = df[feature_cols].values.astype(np.float32)
            self.handedness = df["handedness"].astype(int).tolist()
            self.movements = df["movement"].astype(int).tolist()
            self.labels = df["label"].astype(int).tolist()

            # 3. Optional subject_id
            if "subject_id" in df.columns:
                self.subject_ids_list = df["subject_id"].astype(int).tolist()
            else:
                self.subject_ids_list = [-1] * len(df)

            # 4. Movement names (optional, for compatibility)
            if movement_names is None:
                if len(df) > 0:
                    max_movement = int(df["movement"].max())
                    self.movement_names = [f"movement_{i}" for i in range(max_movement + 1)]
                else:
                    self.movement_names = []
            else:
                self.movement_names = movement_names
            
            # Add mapping dict expected by dataloader
            self.movement_to_idx = {name: i for i, name in enumerate(self.movement_names)}

            return  # IMPORTANT: skip npz logic

        # =========================
        # ORIGINAL NPZ LOGIC
        # =========================
        self.data_path = Path(data_path)
        
        # 1. Auto-detect movements
        if movement_names is None:
            movement_names = sorted([
                d.name for d in self.data_path.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ])
        
        self.movement_names = movement_names
        self.movement_to_idx = {name: i for i, name in enumerate(self.movement_names)}

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
        if all_samples:
            self.features = [s[0] for s in all_samples]  # CHANGED: features not signals
            self.handedness = [s[1] for s in all_samples]
            self.movements = [s[2] for s in all_samples]
            self.labels = [s[3] for s in all_samples]
            self.subject_ids_list = [s[4] for s in all_samples]
        else:
            self.features = []
            self.handedness = []
            self.movements = []
            self.labels = []
            self.subject_ids_list = []
        
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
        return self.movement_to_idx
    
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