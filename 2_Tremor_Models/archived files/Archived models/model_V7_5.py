import torch
from torch import nn
import torch.nn.functional as F

class TremorNetAdvanced_V7_5(nn.Module):
    """
    Extreme architecture for very hard tremor data.
    
    Key innovations:
    1. Frequency domain analysis (FFT) - tremor is periodic!
    2. Phase coupling between wrists (PD affects coordination rhythm)
    3. Statistical moment features (skew, kurtosis capture tremor irregularity)
    4. Separate pathway for dominant hand with 3x capacity
    5. Uncertainty-aware features (model knows what it doesn't know)
    6. Mixup augmentation built into forward pass (optional)
    """
    # --------------------
    # 1. init function
    # --------------------
    def __init__(self, dropout=0.45, use_fft=True):
        super().__init__()
        self.use_fft = use_fft
        
        # 1. TIME DOMAIN PATH
        # --------------------
        # Multi-scale with dilated convolutions (captures long-range patterns)
        self.time_conv1 = nn.Conv1d(6, 48, 7, stride=2, padding=3)
        self.time_conv2 = nn.Conv1d(48, 96, 5, stride=2, padding=2, dilation=2)
        self.time_conv3 = nn.Conv1d(96, 128, 3, stride=1, padding=2, dilation=2)
        
        # 2. FREQUENCY DOMAIN PATH
        # -------------------------
        if use_fft:
            # FFT features: power spectrum in tremor-relevant bands
            self.fft_proj = nn.Sequential(
                nn.Linear(30, 64),  # 30 frequency bins
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5)
            )
        
        # 3. DOMINANT HAND SUPER-PATHWAY (3x capacity)
        # ---------------------------------------------
        # 3.1. Processes dominant hand with more parameters
        self.dominant_conv1 = nn.Conv1d(6, 96, 7, stride=2, padding=3)
        self.dominant_conv2 = nn.Conv1d(96, 192, 5, stride=2, padding=2)
        self.dominant_conv3 = nn.Conv1d(192, 256, 3, stride=1, padding=1)
        
        # 3.2. Squeeze-excitation for dominant path
        self.dominant_se1 = nn.Linear(256, 32)
        self.dominant_se2 = nn.Linear(32, 256)
        
        # 4. HANDEDNESS EMBEDDING
        # ------------------------
        self.hand_embed = nn.Embedding(2, 48)
        self.hand_proj = nn.Sequential(
            nn.Linear(48, 96),
            nn.LayerNorm(96),
            nn.Tanh(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(96, 48)
        )
        
        # 5. BILATERAL COORDINATION
        # ---------------------------
        # Cross-correlation features (how synchronized are the wrists?)
        self.bilateral_attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=dropout * 0.5,
            batch_first=True
        )
        
        # 6. ADVANCED FEATURE FUSION 
        # ------------------------------
        # Calculate total input dimension
        time_feat_dim = 128  # Non-dominant wrist time features
        dominant_feat_dim = 256  # Dominant wrist (3x capacity)
        fft_dim = 64 if use_fft else 0
        stat_dim = 8  # Statistical moments
        bilateral_dim = 128
        hand_dim = 48
        
        total_dim = time_feat_dim + dominant_feat_dim + fft_dim + stat_dim + bilateral_dim + hand_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Bottleneck layer (forces model to learn compressed representation)
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Expansion
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(128, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    # -----------------------------------
    # 2. Computing FFT function (HELPER)
    # ------------------------------------
    def _compute_fft_features(self, x):
        """
        Extract frequency domain features. Tremor is PERIODIC!
        x: [B, 6, T]
        Returns: [B, 30] frequency features
        """
        B, C, T = x.shape
        
        # 1. Compute FFT along time dimension
        fft = torch.fft.rfft(x, dim=2)  # [B, 6, T//2+1]
        power_spectrum = torch.abs(fft)  # Magnitude
        
        # 2. Focus on tremor-relevant frequencies (0.5-10 Hz)
        # Assuming sampling rate ~50-100Hz, take first 30 bins
        power_spectrum = power_spectrum[:, :, :30]  # [B, 6, 30]
        
        # 3. Pool across channels
        power_features = power_spectrum.mean(dim=1)  # [B, 30]
        
        return power_features
    
    # -------------------------------------------
    # 3. Computing stat feats function (HELPER)
    # -------------------------------------------
    def _compute_statistical_features(self, x):
        """
        Statistical moments capture tremor irregularity.
        PD tremor is less regular than healthy movement.
        x: [B, C, T]
        Returns: [B, 8] statistical features
        """
        # 1. Mean and std per channel
        mean = x.mean(dim=2)  # [B, C]
        std = x.std(dim=2)    # [B, C]
        
        # 2. Epsilon for better numerical stability
        eps = 1e-5
        std_safe = torch.clamp(std, min=eps)  # Prevent division by very small numbers
        
        # 3. Skewness (asymmetry of distribution)
        x_centered = x - mean.unsqueeze(2)
        skew = (x_centered ** 3).mean(dim=2) / (std_safe ** 3)
        
        # 4. Kurtosis (tailedness - tremor has spikes)
        kurt = (x_centered ** 4).mean(dim=2) / (std_safe ** 4)
        
        # 5. Pool across channels
        features = torch.stack([
            mean.mean(dim=1), std.mean(dim=1),
            skew.mean(dim=1), kurt.mean(dim=1),
            mean.std(dim=1), std.std(dim=1),
            skew.std(dim=1), kurt.std(dim=1)
        ], dim=1)  # [B, 8]
        
        return features
    
    # -------------------------------------
    # 4. Get time feats function (HELPER)
    # -------------------------------------
    def _extract_time_features(self, x):
        """Standard time-domain CNN. x: [B, 6, T] -> [B, 128]"""
        x = F.gelu(self.time_conv1(x))
        x = self.dropout(x)
        x = F.gelu(self.time_conv2(x))
        x = self.dropout(x)
        x = F.gelu(self.time_conv3(x))
        
        # Adaptive pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)  # [B, 128]
        return x
    
    # ------------------------------------------
    # 5. Get dominant feats function (HELPER)
    # ------------------------------------------
    def _extract_dominant_features(self, x):
        """
        SUPER-PATHWAY for dominant hand (3x capacity).
        x: [B, 6, T] -> [B, 256]
        """
        x = F.gelu(self.dominant_conv1(x))
        x = self.dropout(x)
        x = F.gelu(self.dominant_conv2(x))
        x = self.dropout(x)
        x = F.gelu(self.dominant_conv3(x))
        
        # 1. Squeeze-excitation (channel attention)
        B, C, T = x.shape
        se = F.adaptive_avg_pool1d(x, 1).squeeze(2)  # [B, 256]
        se = torch.sigmoid(self.dominant_se2(F.relu(self.dominant_se1(se))))
        x = x * se.unsqueeze(2)
        
        # 2. Pool
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)  # [B, 256]
        return x
    
    
    # -------------------------
    # 6. MAIN Forward function
    # -------------------------
    def forward(self, x, handedness, mixup_lambda=None):
        """
        x: [B, 2, T, 6]
        handedness: [B]
        mixup_lambda: Optional[float] for data augmentation
        """
        B = x.shape[0]
        
        # 1. Optional: Mixup augmentation during training
        # -------------------------------------------------
        if mixup_lambda is not None and self.training:
            indices = torch.randperm(B)
            x = mixup_lambda * x + (1 - mixup_lambda) * x[indices]
            handedness_onehot = F.one_hot(handedness.long(), 2).float()
            handedness_mixed = mixup_lambda * handedness_onehot + (1 - mixup_lambda) * handedness_onehot[indices]
        
        left_raw = x[:, 0].permute(0, 2, 1)   # [B, 6, T]
        right_raw = x[:, 1].permute(0, 2, 1)  # [B, 6, T]
        
        # 2. Handedness embedding
        # ------------------------
        if mixup_lambda is not None and self.training:
            # handedness_mixed is a soft one-hot [B, 2]
            # 2.1. convert to embedding by linear combination of embedding weights
            hand_emb = handedness_mixed.to(x.device) @ self.hand_embed.weight  # [B, 48]
        else:
            hand_emb = self.hand_embed(handedness.long())  # [B, 48]
        hand_emb = self.hand_proj(hand_emb)  # [B, 48]
        
        
        # 3. Route to correct pathway
        # ----------------------------
        """
            - Dominant hand -> SUPER pathway (256 dims)
            - Non-dominant -> Standard pathway (128 dims)
        """
        # 3.1. Determine which hand is dominant
        if mixup_lambda is not None and self.training:
            # 3.2. handedness_mixed: [B, 2] (left,right)
            left_weight = handedness_mixed[:, 0].view(B, 1, 1).to(x.device)
            right_weight = handedness_mixed[:, 1].view(B, 1, 1).to(x.device)

            # 3.3. dominant_raw becomes a weighted mix: if left is dominant -> more left_raw, etc.
            dominant_raw = left_weight * left_raw + right_weight * right_raw
            non_dominant_raw = right_weight * left_raw + left_weight * right_raw
        else:
            is_left_handed = (handedness == 0).float().view(B, 1, 1)
            is_right_handed = (handedness == 1).float().view(B, 1, 1)
            dominant_raw = is_left_handed * left_raw + is_right_handed * right_raw
            non_dominant_raw = is_right_handed * left_raw + is_left_handed * right_raw
        
        dominant_feat = self._extract_dominant_features(dominant_raw)  # [B, 256]
        non_dominant_feat = self._extract_time_features(non_dominant_raw)  # [B, 128]
        
        
        # 4. Frequency features (both wrists)
        # --------------------------------------
        features_list = [non_dominant_feat, dominant_feat]
        
        if self.use_fft:
            left_fft = self._compute_fft_features(left_raw)
            right_fft = self._compute_fft_features(right_raw)
            fft_combined = self.fft_proj(left_fft + right_fft)  # [B, 64]
            features_list.append(fft_combined)
        
        # 5. Statistical features
        # -------------------------
        left_stat = self._compute_statistical_features(left_raw)
        right_stat = self._compute_statistical_features(right_raw)
        stat_feat = (left_stat + right_stat) / 2  # [B, 8]
        features_list.append(stat_feat)
        
        # 6. Bilateral coordination (cross-attention between wrists)
        # ------------------------------------------------------------
        left_seq = self.time_conv3(F.gelu(self.time_conv2(F.gelu(self.time_conv1(left_raw)))))
        right_seq = self.time_conv3(F.gelu(self.time_conv2(F.gelu(self.time_conv1(right_raw)))))
        
        left_seq = left_seq.permute(0, 2, 1)  # [B, T', 128]
        right_seq = right_seq.permute(0, 2, 1)
        
        bilateral_seq = torch.cat([left_seq, right_seq], dim=1)  # [B, 2*T', 128]
        bilateral_attn, _ = self.bilateral_attn(bilateral_seq, bilateral_seq, bilateral_seq)
        bilateral_feat = bilateral_attn.mean(dim=1)  # [B, 128]
        features_list.append(bilateral_feat)
        
        # 7. Add handedness
        # --------------------
        features_list.append(hand_emb)
        
        # 8. Fuse all features
        # ---------------------
        combined = torch.cat(features_list, dim=1)
        
        logits = self.fusion(combined)
        
        return logits
    
    def forward_with_uncertainty(self, x, handedness, num_samples=10):
        """
        MC Dropout for uncertainty estimation.
        Use during inference to see model confidence.
        """
        self.train()  # Enable dropout
        
        logits_samples = []
        for _ in range(num_samples):
            logits = self.forward(x, handedness)
            logits_samples.append(logits)
        
        logits_samples = torch.stack(logits_samples)  # [num_samples, B, 1]
        
        mean_logits = logits_samples.mean(dim=0)
        std_logits = logits_samples.std(dim=0)
        
        return mean_logits, std_logits  # [B, 1], [B, 1]