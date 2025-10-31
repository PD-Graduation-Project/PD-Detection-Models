import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch

class FeatureExtractor:
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, signal):
        """Extract time and frequency domain features from 6-channel signal"""
        features = []
        
        for i in range(6):  # For each IMU channel
            channel_data = signal[:, i]
            
            # Time domain features
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                stats.skew(channel_data),
                stats.kurtosis(channel_data),
                np.median(channel_data),
                np.max(channel_data),
                np.min(channel_data),
                np.ptp(channel_data),  # peak-to-peak
            ])
            
            # Frequency domain features (simple FFT-based)
            fft_vals = np.abs(np.fft.rfft(channel_data))
            if len(fft_vals) > 0:
                features.extend([
                    np.mean(fft_vals),
                    np.std(fft_vals),
                    np.max(fft_vals),
                    np.argmax(fft_vals) / len(fft_vals),  # dominant frequency (normalized)
                ])
            else:
                features.extend([0, 0, 0, 0])
                
        return np.array(features)

class TremorMLP(nn.Module):
    def __init__(self, num_classes=3, num_movements=11, feature_dim=72):
        super().__init__()
        
        # Wrist and movement embeddings
        self.wrist_embed = nn.Embedding(2, 16)
        self.movement_embed = nn.Embedding(num_movements, 32)
        
        input_dim = feature_dim + 16 + 32
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, features, wrist, movement):
        wrist_emb = self.wrist_embed(wrist.long())
        movement_emb = self.movement_embed(movement.long())
        
        combined = torch.cat([features, wrist_emb, movement_emb], dim=1)
        return self.classifier(combined)