"""
Modular components for advanced tremor detection models.

Contains 9 modular classes:

    - `FrequencyAnalyzer`: Enhanced FFT with 3 analysis methods:
            - Power spectrum (tremor magnitude)
            - Phase coherence (wrist synchronization)
            - Spectral envelope (frequency shape)
            
    - `StatisticalFeatureExtractor`: Moments for irregularity detection
    - `EfficientCNNExtractor`: Depthwise separable convs + squeeze-excitation
    - `DominantHandCNNExtractor`: 3x capacity super-pathway
    - `TemporalGRUEncoder`: Bidirectional GRU with attention pooling
    - `BilateralCoordinationModule`: Cross-attention between wrists
    - `MovementContextEncoder`: NEW! Encodes movement type (0-10)
    - `HandednessContextEncoder`: Enhanced handedness embedding
    - `DepthwiseSeparableConv1d` + `SqueezeExcitation1d`: Building blocks
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


class DepthwiseSeparableConv1d(nn.Module):
    """Parameter-efficient convolution."""
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_c, in_c, kernel_size, stride, padding, 
                                    dilation=dilation, groups=in_c, bias=False)
        self.pointwise = nn.Conv1d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_c)
        
    def forward(self, x):
        return self.bn(self.pointwise(self.depthwise(x)))


class SqueezeExcitation1d(nn.Module):
    """Channel attention for 1D signals."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 4))
        self.fc2 = nn.Linear(max(channels // reduction, 4), channels)
        
    def forward(self, x):
        B, C, T = x.shape
        squeeze = x.mean(dim=2)
        excitation = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))
        return x * excitation.unsqueeze(2)


class EfficientCNNExtractor(nn.Module):
    """
    Efficient CNN with depthwise separable convolutions and squeeze-excitation.
    """
    def __init__(self, output_dim=128, dropout=0.3):
        super().__init__()
        
        # Multi-scale extraction
        self.conv1 = DepthwiseSeparableConv1d(6, 64, kernel_size=7, stride=2, padding=3)
        self.se1 = SqueezeExcitation1d(64)
        
        self.conv2 = DepthwiseSeparableConv1d(64, 96, kernel_size=5, stride=2, padding=2, dilation=2)
        self.se2 = SqueezeExcitation1d(96)
        
        self.conv3 = DepthwiseSeparableConv1d(96, output_dim, kernel_size=3, stride=1, padding=2, dilation=2)
        self.se3 = SqueezeExcitation1d(output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """x: [B, 6, T] -> [B, output_dim]"""
        x = F.gelu(self.se1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.gelu(self.se2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.gelu(self.se3(self.conv3(x)))
        
        # Adaptive pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        return x


class DominantHandCNNExtractor(nn.Module):
    """
    3x capacity CNN pathway for dominant hand processing.
    """
    def __init__(self, output_dim=256, dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(6, 96, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(96)
        
        self.conv2 = nn.Conv1d(96, 192, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(192)
        
        self.conv3 = nn.Conv1d(192, output_dim, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
        # Squeeze-excitation
        self.se = SqueezeExcitation1d(output_dim, reduction=8)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """x: [B, 6, T] -> [B, output_dim]"""
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.se(x)
        
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        return x


class TemporalGRUEncoder(nn.Module):
    """
    Bidirectional GRU for temporal sequence modeling.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.ln = nn.LayerNorm(hidden_dim * 2)
        
        # Attention over time
        self.temporal_attn = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, input_dim]
        Returns:
            features: [B, hidden_dim * 2]
        """
        gru_out, _ = self.gru(x)  # [B, T, hidden_dim*2]
        gru_out = self.ln(gru_out)
        
        # Attention pooling
        attn_weights = torch.softmax(self.temporal_attn(gru_out), dim=1)  # [B, T, 1]
        attended = (gru_out * attn_weights).sum(dim=1)  # [B, hidden_dim*2]
        
        # Also use mean and max pooling
        mean_pool = gru_out.mean(dim=1)
        max_pool = gru_out.max(dim=1)[0]
        
        # Combine
        features = attended + 0.3 * mean_pool + 0.2 * max_pool
        
        return features


class BilateralCoordinationModule(nn.Module):
    """
    Models coordination between left and right wrists using cross-attention.
    """
    def __init__(self, dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, left_seq, right_seq):
        """
        Args:
            left_seq: [B, T, dim]
            right_seq: [B, T, dim]
        Returns:
            bilateral_features: [B, dim]
        """
        # Stack both sequences
        combined = torch.cat([left_seq, right_seq], dim=1)  # [B, 2T, dim]
        
        # Self-attention over combined sequences
        attn_out, _ = self.cross_attn(combined, combined, combined)
        attn_out = self.ln(attn_out + combined)
        
        # Pool
        bilateral_features = attn_out.mean(dim=1)  # [B, dim]
        
        return bilateral_features


class MovementContextEncoder(nn.Module):
    """
    Encodes movement type information and projects it to feature space.
    Allows the model to learn movement-specific patterns.
    """
    def __init__(self, num_movements=11, embed_dim=64, dropout=0.2):
        super().__init__()
        
        self.movement_embed = nn.Embedding(num_movements, embed_dim)
        
        # Project to richer representation
        self.movement_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, movement_ids):
        """
        Args:
            movement_ids: [B] - movement type indices (0-10)
        Returns:
            movement_features: [B, embed_dim]
        """
        movement_emb = self.movement_embed(movement_ids.long())
        movement_features = self.movement_proj(movement_emb)
        return F.normalize(movement_features, dim=-1)


class HandednessContextEncoder(nn.Module):
    """
    Encodes handedness information with enhanced projection.
    """
    def __init__(self, embed_dim=48, dropout=0.2):
        super().__init__()
        
        self.hand_embed = nn.Embedding(2, embed_dim)
        
        self.hand_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, handedness):
        """
        Args:
            handedness: [B] - 0=left, 1=right
        Returns:
            hand_features: [B, embed_dim]
        """
        hand_emb = self.hand_embed(handedness.long())
        hand_features = self.hand_proj(hand_emb)
        return F.normalize(hand_features, dim=-1)
    
    def forward_mixed(self, handedness_onehot):
        """
        For mixup augmentation.
        Args:
            handedness_onehot: [B, 2] - soft one-hot encoding
        Returns:
            hand_features: [B, embed_dim]
        """
        hand_emb = handedness_onehot @ self.hand_embed.weight
        hand_features = self.hand_proj(hand_emb)
        return F.normalize(hand_features, dim=-1)