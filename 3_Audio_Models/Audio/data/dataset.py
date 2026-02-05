import os
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self,
        healthy_dir: str,
        pd_dir: str,
        sample_rate: int = 16000,
        max_length: float = 5.0,
        augment: bool = False,
    ):
        self.sample_rate = sample_rate
        self.max_length = int(max_length * sample_rate)
        self.augment = augment
        
        # collect file paths
        self.file_paths = []
        self.labels = []
        
        # healthy (label=0)
        for f in os.listdir(healthy_dir):
            if f.endswith('.wav'):
                self.file_paths.append(os.path.join(healthy_dir, f))
                self.labels.append(0)
        
        # pd (label=1)
        for f in os.listdir(pd_dir):
            if f.endswith('.wav'):
                self.file_paths.append(os.path.join(pd_dir, f))
                self.labels.append(1)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. load audio
        waveform, sr = torchaudio.load(self.file_paths[idx])
        
        # 2. resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        # 3. mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 4. pad/trim to max_length
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_length - waveform.shape[1]))
        
        # 5. optional augmentation
        if self.augment:
            # 5.1. add gaussian noise
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
            
            # 5.2. random gain
            waveform = waveform * (0.8 + 0.4 * torch.rand(1))
        
        return waveform.squeeze(0), self.labels[idx]