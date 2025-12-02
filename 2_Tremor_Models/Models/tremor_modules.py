# file: tremor_modules.py
import torch
from torch import nn
import torch.nn.functional as F

class ClinicalMetadataEncoder(nn.Module):
    """
    Encodes clinical metadata with learned embeddings and projections.
    Handles missing values (-1) gracefully with learned embeddings.
    
    Input metadata vector (8 dims):
        [age_at_diagnosis, age, height, weight, gender, 
         appearance_in_kinship, appearance_in_first_grade_kinship, 
         effect_of_alcohol_on_tremor]
    """
    def __init__(self, output_dim=64, dropout=0.3):
        super().__init__()
        
        # Continuous features projection (age, height, weight, BMI)
        self.continuous_proj = nn.Sequential(
            nn.Linear(5, 32),  # age_at_diagnosis, age, height, weight, BMI
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 48)
        )
        
        # Categorical embeddings
        self.gender_embed = nn.Embedding(3, 8)  # male=0, female=1, missing=-1->2
        self.kinship_embed = nn.Embedding(3, 8)  # yes=1, no=0, missing=-1->2
        self.first_kinship_embed = nn.Embedding(3, 8)  # yes=1, no=0, missing=-1->2
        self.alcohol_effect_embed = nn.Embedding(5, 12)  # 0-3 + missing=-1->4
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(48 + 8 + 8 + 8 + 12, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
        # Cross-attention for metadata-signal interaction
        self.metadata_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=dropout * 0.5,
            batch_first=True
        )
        
    def forward(self, metadata):
        """
        Args:
            metadata: [B, 8] tensor with values
        Returns:
            [B, output_dim] encoded metadata features
        """
        B = metadata.shape[0]
        
        # Extract individual features
        age_at_diagnosis = metadata[:, 0]
        age = metadata[:, 1]
        height = metadata[:, 2]
        weight = metadata[:, 3]
        gender = metadata[:, 4].long()
        kinship = metadata[:, 5].long()
        first_kinship = metadata[:, 6].long()
        alcohol = metadata[:, 7].long()
        
        # Compute BMI (handle missing values)
        height_m = height / 100.0  # cm to m
        bmi = torch.where(
            (height > 0) & (weight > 0),
            weight / (height_m ** 2 + 1e-6),
            torch.zeros_like(weight)
        )
        
        # Normalize continuous features (z-score with robust scaling)
        def robust_normalize(x):
            # Replace missing (-1) with 0 temporarily
            x_valid = torch.where(x > 0, x, torch.zeros_like(x))
            mean = x_valid[x_valid > 0].mean() if (x_valid > 0).any() else 0
            std = x_valid[x_valid > 0].std() if (x_valid > 0).any() else 1
            return torch.where(x > 0, (x - mean) / (std + 1e-6), torch.zeros_like(x))
        
        age_at_diagnosis_norm = robust_normalize(age_at_diagnosis)
        age_norm = robust_normalize(age)
        height_norm = robust_normalize(height)
        weight_norm = robust_normalize(weight)
        bmi_norm = robust_normalize(bmi)
        
        continuous = torch.stack([
            age_at_diagnosis_norm,
            age_norm,
            height_norm,
            weight_norm,
            bmi_norm
        ], dim=1)  # [B, 5]
        
        continuous_feat = self.continuous_proj(continuous)  # [B, 48]
        
        # Convert -1 to separate embedding index for missing values
        gender_idx = torch.where(gender == -1, torch.tensor(2, device=gender.device), gender)
        kinship_idx = torch.where(kinship == -1, torch.tensor(2, device=kinship.device), kinship)
        first_kinship_idx = torch.where(first_kinship == -1, torch.tensor(2, device=first_kinship.device), first_kinship)
        alcohol_idx = torch.where(alcohol == -1, torch.tensor(4, device=alcohol.device), alcohol)
        
        # Categorical embeddings
        gender_feat = self.gender_embed(gender_idx)  # [B, 8]
        kinship_feat = self.kinship_embed(kinship_idx)  # [B, 8]
        first_kinship_feat = self.first_kinship_embed(first_kinship_idx)  # [B, 8]
        alcohol_feat = self.alcohol_effect_embed(alcohol_idx)  # [B, 12]
        
        # Combine all features
        combined = torch.cat([
            continuous_feat,
            gender_feat,
            kinship_feat,
            first_kinship_feat,
            alcohol_feat
        ], dim=1)  # [B, 48+8+8+8+12=84]
        
        # Fusion
        metadata_feat = self.fusion(combined)  # [B, output_dim]
        
        return metadata_feat


class StatisticalFeatureExtractor(nn.Module):
    """
    Lightweight torch version of statistical moments and simple time-domain features.
    Input: x: [B, 6, T] where 6 = accel_x,y,z, gyro_x,y,z (we will typically pass per-hand [B, 6, T] or [B, 3, T])
    Output: [B, D] small vector (float)
    """
    def __init__(self, out_dim=32):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Sequential(
            nn.Linear(14, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        # x: [B, 6, T] or [B, 3, T] (we'll robustly handle both)
        B, C, T = x.shape
        if C == 6:
            accel = x[:, :3, :]
            gyro = x[:, 3:, :]
        else:
            accel = x
            gyro = torch.zeros_like(accel)

        def stats(sig):
            # sig: [B, 3, T] -> compute per-axis stats and then aggregate
            mean = sig.mean(dim=2)            # [B,3]
            std = sig.std(dim=2)
            mx = sig.max(dim=2)[0]
            mn = sig.min(dim=2)[0]
            rms = torch.sqrt(torch.mean(sig**2, dim=2))
            skew = torch.mean(((sig - mean.unsqueeze(2))**3), dim=2) / (std + 1e-6)**3
            kurt = torch.mean(((sig - mean.unsqueeze(2))**4), dim=2) / (std + 1e-6)**4
            # reduce to per-signal summary (mean across axes)
            return torch.cat([mean.mean(dim=1, keepdim=True),
                              std.mean(dim=1, keepdim=True),
                              mx.mean(dim=1, keepdim=True),
                              mn.mean(dim=1, keepdim=True),
                              rms.mean(dim=1, keepdim=True),
                              skew.mean(dim=1, keepdim=True),
                              kurt.mean(dim=1, keepdim=True)], dim=1)  # [B,7]

        a_stats = stats(accel)
        g_stats = stats(gyro)
        combined = torch.cat([a_stats, g_stats], dim=1)  # [B,14]
        # small projection
        out = self.proj(combined)
        return out


class FrequencyAnalyzer(nn.Module):
    """
    Frequency-based feature extractor.

    - Computes STFT/log-power spectrograms for: left hand, right hand, and their magnitudes.
    - Learns frequency-time patterns using a small CNN.
    - Computes coarse band energies + left/right coherence.
    - Outputs a fixed-size embedding of dimension output_dim.
    """

    def __init__(self, sample_rate=100, n_fft=256, hop_length=None,
                output_dim=128, dropout=0.2):

        super().__init__()
        self.fs = sample_rate
        self.n_fft = n_fft
        self.hop = hop_length or n_fft // 4
        self.output_dim = output_dim

        # ------------------------------------------------------------------
        # CNN encoder for spectrogram maps
        # Input channels: 8
        # (left x,y,z, left mag, right x,y,z, right mag)
        # ------------------------------------------------------------------
        self.spec_encoder = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, output_dim // 2),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # MLP for static pooled features (band energies, coherence, etc.)
        # Always projects to output_dim // 2
        # ------------------------------------------------------------------
        self.pool_proj = nn.Sequential(
            nn.Linear(12, output_dim // 2),
            nn.GELU()
        )

        # ------------------------------------------------------------------
        # Final fusion + projection
        # ------------------------------------------------------------------
        self.out_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    # ======================================================================
    # Compute STFT log-power spectrogram for each channel
    # Input:  [B, C, T]
    # Output: [B, C, Freq, Time]
    # ======================================================================
    def _compute_spectrogram(self, sig):
        stfts = []
        for ch in range(sig.shape[1]):
            x = sig[:, ch, :]  # [B, T]
            S = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                            return_complex=True, center=True, window=None)
            P = (S.abs() ** 2)
            logP = torch.log1p(P)
            stfts.append(logP)
        return torch.stack(stfts, dim=1)

    # ======================================================================
    # Forward Pass
    # ======================================================================
    def forward(self, left_raw, right_raw):
        B = left_raw.shape[0]

        # ------------------------------------------------------------------
        # Keep only accelerometer channels (first 3 if more exist)
        # ------------------------------------------------------------------
        def to_accel(x):
            return x[:, :3, :] if x.shape[1] >= 3 else x

        lx = to_accel(left_raw)
        rx = to_accel(right_raw)

        # ------------------------------------------------------------------
        # Compute spectrograms for x,y,z
        # ------------------------------------------------------------------
        spec_l = self._compute_spectrogram(lx)
        spec_r = self._compute_spectrogram(rx)

        # ------------------------------------------------------------------
        # Magnitude channel for each hand
        # ------------------------------------------------------------------
        mag_l = torch.sqrt((lx**2).sum(dim=1, keepdim=True))
        mag_r = torch.sqrt((rx**2).sum(dim=1, keepdim=True))

        mag_l_spec = self._compute_spectrogram(mag_l)
        mag_r_spec = self._compute_spectrogram(mag_r)

        # ------------------------------------------------------------------
        # Final stacked spectrogram channels
        # (left xyz + left mag + right xyz + right mag) = 8 channels
        # Shape = [B, 8, F, T]
        # ------------------------------------------------------------------
        spec_stack = torch.cat([spec_l, mag_l_spec, spec_r, mag_r_spec], dim=1)

        # ------------------------------------------------------------------
        # Normalize spectrogram per sample
        # ------------------------------------------------------------------
        mean = spec_stack.mean(dim=[2, 3], keepdim=True)
        std = spec_stack.std(dim=[2, 3], keepdim=True) + 1e-6
        spec_norm = (spec_stack - mean) / std

        # CNN features (output_dim // 2)
        encoded = self.spec_encoder(spec_norm)

        # ------------------------------------------------------------------
        # Compute band energies (coarse frequency bands)
        # ------------------------------------------------------------------
        Fdim = spec_stack.shape[2]
        freqs = torch.linspace(0, self.fs / 2, steps=Fdim, device=spec_stack.device)
        band_edges = [0.5, 3, 6, 12, 20, 40]

        band_energies = []
        for i in range(len(band_edges) - 1):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            if mask.sum() == 0:
                band_energies.append(torch.zeros(B, device=spec_stack.device))
            else:
                power = spec_stack[:, :, mask, :].sum(dim=(2, 3))
                band_energies.append(power.mean(dim=1))

        band_tensor = torch.stack(band_energies, dim=1)

        # ------------------------------------------------------------------
        # Compute crude left/right coherence per-band
        # ------------------------------------------------------------------
        left_bands = []
        right_bands = []

        for i in range(len(band_edges) - 1):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            if mask.sum() == 0:
                left_bands.append(torch.zeros(B, device=spec_stack.device))
                right_bands.append(torch.zeros(B, device=spec_stack.device))
            else:
                left_p = spec_l[:, :, mask, :].sum(dim=(2, 3)).mean(dim=1)
                right_p = spec_r[:, :, mask, :].sum(dim=(2, 3)).mean(dim=1)
                left_bands.append(left_p)
                right_bands.append(right_p)

        left_b = torch.stack(left_bands, dim=1)
        right_b = torch.stack(right_bands, dim=1)

        dot = (left_b * right_b).sum(dim=1)
        denom = (left_b.norm(dim=1) * right_b.norm(dim=1) + 1e-6)
        coherence = dot / denom

        # ------------------------------------------------------------------
        # Build pooled static feature vector
        # ------------------------------------------------------------------
        pooled = torch.cat([
            spec_stack.mean(dim=[2, 3]).view(B, -1).mean(dim=1, keepdim=True),
            band_tensor,
            coherence.unsqueeze(1),
        ], dim=1)

        pooled = torch.cat([pooled, torch.zeros(B, 12 - pooled.shape[1], device=pooled.device)], dim=1)

        pooled_proj = self.pool_proj(pooled)

        # ------------------------------------------------------------------
        # Combine CNN features + pooled features
        # ------------------------------------------------------------------
        feat = torch.cat([encoded, pooled_proj], dim=1)
        return self.out_proj(feat)
