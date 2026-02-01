import torch
from torch import nn
import torch.nn.functional as F

# ===================================================== #
# ========     Statistical Features       ============= #
# ===================================================== #
class StatisticalFeatureExtractor(nn.Module):
    """
    Extract statistical moments from time-domain IMU signals.
    
    For each signal it computes:
        1. mean: Center value of the signal over time
        2. std: Spread or variability around the mean
        3. max: Peak amplitude reached
        4. min: Lowest amplitude reached
        5. RMS: Average signal magnitude (energy)
        6. skewness: Asymmetry of the distribution
        7. kurtosis: Tail heaviness (presence of outliers/peaks) 
    
    For statistical features, per-second analysis doesn't add significant value
    since these metrics capture 'overall' signal properties.
    
    Input:  x: [B, C, T] where C=6 (accel_xyz + gyro_xyz)
    Output: [B, out_dim] feature vector
    """
    def __init__(self, out_dim=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(14, 64), # 14 statistical features (7 from accelerometer, 7 from gyroscope) -> embed_dim = 64
            nn.LayerNorm(64),
            nn.GELU(), # learn non-linear combinations 
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, T] sensor data
        
        Returns:
            [B, out_dim] statistical features
        """
        B, C, T = x.shape
        
        # Split accelerometer and gyroscope (accel_xyz + gyro_xyz)
        accel, gyro = x[:, :3, :], x[:, 3:6, :]
        
        # Compute 7 statistical moments for each signal type
        a_stats = self._compute_stats(accel)  # [B, 7]
        g_stats = self._compute_stats(gyro)   # [B, 7]
        
        combined = torch.cat([a_stats, g_stats], dim=1)  # [B, 14]
        return self.proj(combined) # Output: [B, 14] -> [B, out_dim]
    
    # ================================
    # ====   HELPER FUNCTIONS   ======
    # ================================
    def _compute_stats(self, sig):
        """
        Compute statistical moments across time dimension.
        
        Args:
            sig: [B, 3, T] signal (3 axes -> for accel or gyro)
        
        Returns:
            [B, 7] averaged stats across axes adn time
        """
        mean = sig.mean(dim=2) # -> First average across time (dim=2) -> [B, 3]
        std = sig.std(dim=2)
        mx = sig.max(dim=2)[0]
        mn = sig.min(dim=2)[0]
        rms = torch.sqrt((sig**2).mean(dim=2))
        
        centered = sig - mean.unsqueeze(2)
        skew = (centered**3).mean(dim=2) / (std + 1e-6)**3
        kurt = (centered**4).mean(dim=2) / (std + 1e-6)**4
        
        # Average across axes: [B, 3] -> [B, 7]
        return torch.stack([
            mean.mean(dim=1), # [B, 3] -> [B, 1]
            std.mean(dim=1), mx.mean(dim=1),
            mn.mean(dim=1), rms.mean(dim=1), skew.mean(dim=1), kurt.mean(dim=1)
        ], dim=1) # -> [B, 7] after stacking

# ===================================================== #
# ========      Frequency Features        ============= #
# ===================================================== #
class FrequencyAnalyzer(nn.Module):
    """
    Frequency-domain feature extractor with per-second temporal analysis.
    
    Extracts features for:
        - Each second of the 10-second signal (10 entries)
        - Full 10-second signal (1 entry)
        Total: 11 temporal segments
    
    For each segment, computes:
        - Spectrogram CNN features: 
            Identifies repeating shake patterns in the frequency data that are typical of different tremor types
        - Band energies (5 frequency bands): 
            Measures how much shaking happens at different speeds (slow vs fast tremors indicate different conditions)
        - Left-right coherence: 
            Checks if both hands shake together or differently (helps distinguish between tremor types)
    
    Input:  left_raw, right_raw: [B, C, T] where T=1024 samples (~10 sec at 100Hz)
    Output: [B, 11, output_dim] where 11 = 10 seconds + 1 full signal
    """

    def __init__(self, sample_rate=100, n_fft=128,
                hop_length=None, # The number of samples to skip between consecutive STFT windows.
                                #  Smaller hop = more time resolution but more computation.
                output_dim=128, dropout=0.2):

        super().__init__()
        self.fs = sample_rate
        self.n_fft = n_fft
        self.hop = hop_length or n_fft // 4 # (75% overlap between windows)
        self.output_dim = output_dim
        self.samples_per_second = sample_rate  # ~100 samples/sec

        # --------------------------------------------------------------------
        # Spectrogram CNN encoder -> Learns time–frequency patterns
        # Input channels: 8 -> (left x,y,z, left mag, right x,y,z, right mag)
        # --------------------------------------------------------------------
        self.spec_encoder = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, output_dim // 2), # -> Output size = [B, output_dim // 2]
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # MLP for static pooled features (band energies + coherence)
        # Input 11 features:
        #     global power -> Average energy across all frequencies and both hands
        #     5 band energies -> Shows how strong the tremor is in slow vs fast frequencies
        #     5 left-right coherences -> For the same 5 bands, measures how similar left and right hand shaking are
        # Output -> [B, output_dim // 2] 
        # ------------------------------------------------------------------
        self.pool_proj = nn.Sequential(
            nn.Linear(11, output_dim // 2),
            nn.GELU()
        )

        # --------------------------------------------------------------------------------
        # Final fusion + projection -> combines CNN and pooled features and refines them
        # --------------------------------------------------------------------------------
        self.out_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Frequency band edges (Hz)
        self.band_edges = [0.5, 3, 6, 12, 20, 40]

    # ======================================================================
    # Forward Pass
    # ======================================================================
    def forward(self, left_raw, right_raw):
        """
        Extract per-second frequency features.
        
        Args:
            left_raw: [B, C, T] left hand IMU data (T=1024)
            right_raw: [B, C, T] right hand IMU data
        
        Returns:
            [B, 11, output_dim] where:
                - [:, 0:10, :] = features for seconds 0-9
                - [:, 10, :] = features for full 10-second signal
        """
        B, C, T = left_raw.shape
        
        # Extract accelerometer channels only (Because accelerometer measures shaking directly.)
        lx = left_raw[:, :3, :] if C >= 3 else left_raw
        rx = right_raw[:, :3, :] if C >= 3 else right_raw
        
        all_features = []
        
        # Process each second independently
        for i in range(10):
            start_idx = i * self.samples_per_second
            end_idx = min(start_idx + self.samples_per_second, T)
            
            l_seg = lx[:, :, start_idx:end_idx]
            r_seg = rx[:, :, start_idx:end_idx]
            
            feat = self._extract_segment_features(l_seg, r_seg) # -> [B, 1, output_dim] * 10
            all_features.append(feat)
        
        # Process full 10-second signal
        full_feat = self._extract_segment_features(lx, rx) # using full lx and rx
        all_features.append(full_feat) # -> [B, 1, output_dim]
        
        # Stack: [B, 11, output_dim]
        return torch.stack(all_features, dim=1)
    
    # ================================
    # ====   HELPER FUNCTIONS   ======
    # ================================
    def _extract_segment_features(self, lx, rx):
        """
        Extract frequency features from a signal segment.
        
        Args:
            lx: [B, 3, T_seg] left accelerometer segment
            rx: [B, 3, T_seg] right accelerometer segment
        
        Returns:
            [B, output_dim] feature vector for this segment
        """
        
        # 1. Compute spectrograms for x,y,z channels
        spec_l = self._compute_spectrogram(lx)  # -> [B, 3, F, T]
        spec_r = self._compute_spectrogram(rx)
        
        # 2. Compute magnitude and its spectrogram -> Captures overall shaking intensity, ignoring direction.
        mag_l = torch.sqrt((lx**2).sum(dim=1, keepdim=True))
        mag_r = torch.sqrt((rx**2).sum(dim=1, keepdim=True))
        mag_l_spec = self._compute_spectrogram(mag_l)  # -> [B, 1, F, T]
        mag_r_spec = self._compute_spectrogram(mag_r)
        
        # 3. Stack all spectrograms: -> [B, 8, F, T]
        # (left x,y,z, left mag, right x,y,z, right mag)
        spec_stack = torch.cat([spec_l, mag_l_spec, spec_r, mag_r_spec], dim=1)
        
        # 4. Normalize -> [B, 8, F, T]
        spec_norm = self._normalize_spectrogram(spec_stack)
        
        # 5. CNN encoding
        encoded = self.spec_encoder(spec_norm)  # [B, output_dim//2]
        
        # 6. Compute pooled features (energy bands & coherence)
        pooled = self._compute_pooled_features(spec_stack, spec_l, spec_r)  # [B, 11]
        pooled_proj = self.pool_proj(pooled)  # [B, output_dim//2]
        
        # 7. Combine and project
        feat = torch.cat([encoded, pooled_proj], dim=1)  # [B, output_dim]
        return self.out_proj(feat)
    
    # --------------------------------------------------------
    # Compute STFT log-power spectrogram for each channel
    # --------------------------------------------------------
    def _compute_spectrogram(self, sig):
        """
        Args:
            sig: [B, C, T] multi-channel signal
        
        Returns:
            [B, C, F, T] spectrogram (F=freq bins, T=time frames)
        """
        stfts = []
        window = torch.hann_window(self.n_fft, device='cuda')  # smooth Hann window
        for ch in range(sig.shape[1]):
            x = sig[:, ch, :]  # [B, T]
            S = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                        return_complex=True, center=True, window=window)
            P = (S.abs() ** 2) # power
            logP = torch.log1p(P) # log (1+p) -> for stability
            stfts.append(logP)
        return torch.stack(stfts, dim=1)
    
    # --------------------------------------
    # Normalize spectrogram per sample
    # --------------------------------------
    def _normalize_spectrogram(self, spec):
        """
        Normalize spectrogram over freq & time per sample. 
        Stabilizes CNN training and makes features comparable across samples.
        
        Args:
            spec: [B, C, F, T]
        
        Returns:
            [B, C, F, T] normalized spectrogram
        """
        mean = spec.mean(dim=[2, 3], keepdim=True) # -> dim=[2, 3] -> over freq & time for each channel 
        std = spec.std(dim=[2, 3], keepdim=True) + 1e-6
        return (spec - mean) / std
    
    def _compute_pooled_features(self, spec_stack, spec_l, spec_r):
        """
        Compute band energies and left-right coherence.
        
        Args:
            spec_stack: [B, 8, F, T] all spectrograms
            spec_l: [B, 3, F, T] left hand spectrogram
            spec_r: [B, 3, F, T] right hand spectrogram
        
        Returns:
            [B, 11] feature vector:
                - [0]: global mean power -> from stack
                - [1:6]: energy in 5 frequency bands -> from stack
                - [6:11]: coherence in 5 frequency bands -> from left and right
        """
        B, _, Fdim, _ = spec_stack.shape
        
        # 0. Frequency axis (Frequencies for each spectrogram row) (0 -> Nyquist (fs/2))
        freqs = torch.linspace(0, self.fs / 2, Fdim, device=spec_stack.device)
        
        # 1. Global average power (how strong the motion is)
        global_mean = spec_stack.mean(dim=[1, 2, 3])
        
        # 2. Band energies (5 bands)
        band_energies = [] # energy per frequency band
        
        # 3. Left-right coherence per band (5 bands)
        coherences = [] # left–right similarity per band
        
        for i in range(len(self.band_edges) - 1):
            # mask -> only pick the frequencies in this band
            mask = (freqs >= self.band_edges[i]) & (freqs < self.band_edges[i + 1])
            
            if mask.sum() == 0: # no vlaues in this energy band -> all zeros
                band_energies.append(torch.zeros(B, device=spec_stack.device))
                coherences.append(torch.zeros(B, device=spec_stack.device))
            else:
                # total energy in this frequency  band
                power = spec_stack[:, :, mask, :].sum(dim=(2, 3)).mean(dim=1)
                band_energies.append(power) # -> store one number (energy) per band
                
                # left and right energy in this band (to get coherence)
                l_power = spec_l[:, :, mask, :].sum(dim=(2, 3)).mean(dim=1)
                r_power = spec_r[:, :, mask, :].sum(dim=(2, 3)).mean(dim=1)
                
                # dot porduct shows how similar/different they are from each other
                dot = l_power * r_power
                denom = torch.sqrt(l_power**2) * torch.sqrt(r_power**2) + 1e-6 # -> normalization
                coherences.append(dot / denom) # -> store one number (coherence) per band
        
        # Combine all features
        return torch.cat([
            global_mean.unsqueeze(1),
            torch.stack(band_energies, dim=1),
            torch.stack(coherences, dim=1)
        ], dim=1)
