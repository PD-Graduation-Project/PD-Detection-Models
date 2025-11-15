import torch
from torch.utils.data import Dataset
    
# VAE
# ======
class SyntheticTremorDataset(Dataset):
    """
    Load synthetic tremor data saved in the new format:
    - samples: list of tuples (signal, handedness, movement, label, metadata)
    """
    def __init__(self, pt_path):
        data = torch.load(pt_path, weights_only=False)
        
        # Extract samples
        self.samples = data['samples']  # list of tuples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Each sample is a tuple: (signal, handedness, movement, label, metadata)
        signal, handedness, movement, label, metadata = self.samples[idx]

        return signal, handedness, movement, label, metadata

