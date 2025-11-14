"""
Modular components for advanced tremor detection models.

Contains modular classes:
--------------------------
    - `FrequencyAnalyzer`: Enhanced FFT with 3 analysis methods:
            - Power spectrum (tremor magnitude)
            - Phase coherence (wrist synchronization)
            - Spectral envelope (frequency shape)
            
    - `StatisticalFeatureExtractor`: Moments for irregularity detection
"""

import torch
from torch import nn
import torch.nn.functional as F


class FrequencyAnalyzer(nn.Module):
    """
    Advanced FFT-based frequency domain analysis.
    Extracts tremor-specific frequency features with multiple analysis methods.
    """
    def __init__(self, num_freq_bins=30, output_dim=128, dropout=0.3):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        
        # Power spectrum projection
        self.power_proj = nn.Sequential(
            nn.Linear(num_freq_bins, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Phase coherence projection (measures wrist synchronization)
        self.phase_proj = nn.Sequential(
            nn.Linear(num_freq_bins, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Spectral envelope projection (overall frequency shape)
        self.envelope_proj = nn.Sequential(
            nn.Linear(num_freq_bins, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Final fusion
        self.fusion = nn.Linear(64 + 32 + 32, output_dim)
        
    def forward(self, left_signal, right_signal):
        """
        Args:
            left_signal: [B, 6, T]
            right_signal: [B, 6, T]
        Returns:
            freq_features: [B, output_dim]
        """
        B, C, T = left_signal.shape
        
        # 1. Compute FFT for both wrists
        left_fft = torch.fft.rfft(left_signal, dim=2)
        right_fft = torch.fft.rfft(right_signal, dim=2)
        
        # 2. Power spectrum (magnitude)
        left_power = torch.abs(left_fft)[:, :, :self.num_freq_bins]
        right_power = torch.abs(right_fft)[:, :, :self.num_freq_bins]
        combined_power = (left_power + right_power).mean(dim=1)  # [B, num_freq_bins]
        
        # 3. Phase coherence (how synchronized are the wrists?)
        phase_diff = torch.angle(left_fft) - torch.angle(right_fft)
        phase_coherence = torch.cos(phase_diff)[:, :, :self.num_freq_bins].mean(dim=1)  # [B, num_freq_bins]
        
        # 4. Spectral envelope (smoothed power spectrum)
        envelope = F.avg_pool1d(combined_power.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        
        # 5. Project all features
        power_feat = self.power_proj(combined_power)
        phase_feat = self.phase_proj(phase_coherence)
        envelope_feat = self.envelope_proj(envelope)
        
        # 6. Fuse
        freq_features = self.fusion(torch.cat([power_feat, phase_feat, envelope_feat], dim=-1))
        
        return freq_features
    
class StatisticalFeatureExtractor(nn.Module):
    """
    Extracts statistical moments that capture tremor irregularity.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T] - signal tensor
        Returns:
            stats: [B, 8] - statistical features
        """
        eps = 1e-5
        
        # Basic statistics
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        std_safe = torch.clamp(std, min=eps)
        
        # Higher-order moments
        x_centered = x - mean.unsqueeze(2)
        skew = (x_centered ** 3).mean(dim=2) / (std_safe ** 3)
        kurt = (x_centered ** 4).mean(dim=2) / (std_safe ** 4)
        
        # Aggregate across channels
        features = torch.stack([
            mean.mean(dim=1), std.mean(dim=1),
            skew.mean(dim=1), kurt.mean(dim=1),
            mean.std(dim=1), std.std(dim=1),
            skew.std(dim=1), kurt.std(dim=1)
        ], dim=1)
        
        return features