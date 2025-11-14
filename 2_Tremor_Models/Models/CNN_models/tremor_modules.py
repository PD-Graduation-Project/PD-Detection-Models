# file: tremor_modules.py
import torch
from torch import nn
import torch.nn.functional as F

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
    Computes STFT/log-power spectrograms for each hand, pools band energies and computes
    cross-hand coherence. Returns a learned projection vector per-sample.
    Inputs expected: left: [B, C, T], right: [B, C, T] (C typically 3 or 6)
    Output: [B, output_dim]
    """
    def __init__(self, sample_rate=100, n_fft=256, hop_length=None, num_mels=40, output_dim=128, dropout=0.2):
        super().__init__()
        self.fs = sample_rate
        self.n_fft = n_fft
        self.hop = hop_length or n_fft // 4
        self.num_mels = num_mels
        self.output_dim = output_dim

        # small CNN to learn from spectrogram stacks
        # input channels will be 3 (x,y,z magnitude) * 2 hands = 6 (we'll stack)
        self.spec_encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, output_dim // 2),
            nn.GELU(),
        )

        # static pooled features -> small MLP
        self.pool_proj = nn.Sequential(
            nn.Linear(12, output_dim // 2),
            nn.GELU()
        )

        self.out_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _compute_spectrogram(self, sig):
        # sig: [B, 3, T]
        # compute magnitude per axis stft -> log power
        stfts = []
        for ch in range(sig.shape[1]):
            x = sig[:, ch, :]  # [B, T]
            # Torch STFT return complex if return_complex=True
            S = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, return_complex=True, center=True, window=None)
            # S: [B, F, time]
            P = (S.abs() ** 2)  # power
            # log-power (stabilize)
            logP = torch.log1p(P)
            stfts.append(logP)  # list of [B, F, time]
        # stack channels: [B, 3, F, time]
        spec = torch.stack(stfts, dim=1)
        return spec  # [B, C, F, Tfram]

    def forward(self, left_raw, right_raw):
        """
        left_raw/right_raw: [B, C, T] (C = 3 or 6 but expects accel channels first)
        Note: we'll compute magnitude and use x,y,z channels.
        """
        B = left_raw.shape[0]
        # if input is 6 channels, take first 3 as accel
        def to_accel(x):
            return x[:, :3, :] if x.shape[1] >= 3 else x

        lx = to_accel(left_raw)
        rx = to_accel(right_raw)

        # Spectrogram per-hand
        spec_l = self._compute_spectrogram(lx)  # [B,3,F,Frames]
        spec_r = self._compute_spectrogram(rx)

        # compute magnitude channel as well and stack: [B, 3, F, frames] -> compute mag plane
        mag_l = torch.sqrt((lx**2).sum(dim=1, keepdim=True))  # [B,1,T] -> stft
        mag_l_spec = self._compute_spectrogram(mag_l)
        mag_r = torch.sqrt((rx**2).sum(dim=1, keepdim=True))
        mag_r_spec = self._compute_spectrogram(mag_r)

        # final stack: left x,y,z + left mag + right x,y,z + right mag => channels = 8
        # but our spec_encoder expects 6 channels; we can pick left(x,y,z)+right(x,y,z) and mag channels averaged by axis
        # Stack: left x,y,z, right x,y,z  => 6 channels
        spec_stack = torch.cat([spec_l, spec_r], dim=1)  # [B,6,F,Frames]

        # encode spectrogram map
        # normalize per-sample for numerical stability
        spec_mean = spec_stack.mean(dim=[2,3], keepdim=True)
        spec_std = spec_stack.std(dim=[2,3], keepdim=True) + 1e-6
        spec_norm = (spec_stack - spec_mean) / spec_std
        encoded = self.spec_encoder(spec_norm)  # [B, output_dim//2]

        # pooled band energies & simple cross-hand coherence estimate
        # compute band energies from power (coarse bands)
        # reduce frequency axis
        Fdim = spec_stack.shape[2]
        # compute band energies for 6 bands (coarse)
        band_edges = [0.5, 3, 6, 12, 20, 40]  # Hz
        freqs = torch.linspace(0, self.fs/2, steps=Fdim, device=spec_stack.device)
        band_energies = []
        for i in range(len(band_edges)-1):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
            if mask.sum() == 0:
                band_energies.append(torch.zeros(B, device=spec_stack.device))
            else:
                power = spec_stack[:,:,mask,:].sum(dim=(2,3))  # sum across freq and frames -> [B, channels]
                # average across channels
                band_energies.append(power.mean(dim=1))
        # band_energies: list of [B], stack -> [B, num_bands]
        band_tensor = torch.stack(band_energies, dim=1)  # [B, num_bands]

        # simple inter-hand coherence: correlate left mag and right mag in frequency domain (coarse)
        # compute coherence as normalized cross-power in psd per-band using mag specs
        # For simplicity produce a few scalar coherence-like values by computing correlation of band energies left/right
        left_bands = []
        right_bands = []
        # compute band energies per-hand
        for i in range(len(band_edges)-1):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
            if mask.sum()==0:
                left_bands.append(torch.zeros(B, device=spec_stack.device))
                right_bands.append(torch.zeros(B, device=spec_stack.device))
            else:
                left_p = spec_l[:,:,mask,:].sum(dim=(2,3)).mean(dim=1)  # [B]
                right_p = spec_r[:,:,mask,:].sum(dim=(2,3)).mean(dim=1)
                left_bands.append(left_p)
                right_bands.append(right_p)
        left_b = torch.stack(left_bands, dim=1)
        right_b = torch.stack(right_bands, dim=1)
        # coherence-like: normalized dot product between band vectors
        dot = (left_b * right_b).sum(dim=1)
        denom = (left_b.norm(dim=1) * right_b.norm(dim=1) + 1e-6)
        coherence = dot / denom  # [B]

        # pooled static features vector
        # use simple summaries: mean/std of spectrogram per-hand + coherence and band ratios
        pooled = torch.cat([
            spec_stack.mean(dim=[2,3]).view(B, -1).mean(dim=1, keepdim=True),  # avg power scalar
            band_tensor,  # [B, num_bands]
            coherence.unsqueeze(1)
        ], dim=1)  # shape [B, 1+num_bands+1] ~ [B, 1+5+1=7] -> we'll expand to fixed size
        # to fixed size (12) pad or aggregate
        pooled = torch.cat([pooled, torch.zeros(B, 12 - pooled.shape[1], device=pooled.device)], dim=1)

        pooled_proj = self.pool_proj(pooled)  # [B, output_dim//2]

        # combine and project
        feat = torch.cat([encoded, pooled_proj], dim=1)  # [B, output_dim]
        out = self.out_proj(feat)
        return out
