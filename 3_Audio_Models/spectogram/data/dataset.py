import os
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """
    PyTorch dataset for converting raw .wav files to spectrograms for PD classification.
    
    Supports both linear and mel-scale spectrograms.
    
    Args:
        healthy_dir (str): Path to Healthy wav files.
        pd_dir (str): Path to PD wav files.
        sample_rate (int): Target sample rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        n_mels (int): Number of mel bands (only for mel spectrograms).
        spectrogram_type (str): 'linear' or 'mel'.
        augment (bool): Apply time/frequency masking augmentation.
    """
    def __init__(
        self,
        healthy_dir: str,
        pd_dir: str,
        
        sample_rate: int = 16000,
        max_length: float = 5.0,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 128,
        
        spectrogram_type: str = 'mel',  # 'linear' or 'mel'
        augment: bool = False,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.max_length = int(max_length * sample_rate)  # convert to samples
        self.spectrogram_type = spectrogram_type
        self.augment = augment
        
        # spectrogram transforms
        # --------------------------
        # 1. MEL_Scale 
        if spectrogram_type == 'mel':
            self.spec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
        # 2. linear
        else:  
            self.spec_transform = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
            )
        
        # convert to dB scale
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # augmentation (time/frequency masking)
        if augment:
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        
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
        # 1. load with soundfile (no FFmpeg needed)
        waveform, sr = sf.read(self.file_paths[idx])
        waveform = torch.from_numpy(waveform).float()
        
        # 2. Ensure shape is (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # mono: (1, samples)
        else:
            waveform = waveform.T  # stereo: (2, samples)
        
        # 3. resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        # 4. mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 5. PAD/TRIM to fixed length BEFORE spectrogram
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            waveform = torch.nn.functional.pad(
                waveform, 
                (0, self.max_length - waveform.shape[1])
            )
        
        # 6. convert to spectrogram
        spec = self.spec_transform(waveform)  # (1, freq, time)
        spec = self.amplitude_to_db(spec)
        
        # 7. augmentation
        if self.augment:
            spec = self.time_mask(spec)
            spec = self.freq_mask(spec)
        
        # 8. normalize to [0, 1]
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        
        # 9. convert to 3-channel (RGB-like) for pretrained models
        spec = spec.repeat(3, 1, 1)  # (3, freq, time)
        
        return {
            "image": spec.squeeze(0) if spec.shape[0] == 1 else spec,
            "label": self.labels[idx]
        }