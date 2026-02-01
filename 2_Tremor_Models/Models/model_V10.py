import torch
import torch.nn.functional as F
from torch import nn
from .tremor_modules import FrequencyAnalyzer, StatisticalFeatureExtractor


class TremorNetV10(nn.Module):
    """
    Tremor detection network with single CNN path and dominant hand weighting.
    
    Architecture:
    - Single multi-scale CNN path (processes both hands with dominant hand emphasis)
    - Frequency analysis (spectrogram features)
    - Statistical features (time-domain moments)
    - Bilateral coordination analysis
    - Hand asymmetry features
    
    Input:  x: [B, 2, T, 6] where 2=hands, T=1024 samples, 6=accel_xyz+gyro_xyz
            handedness: [B] with 0=left-dominant, 1=right-dominant
            movements: [B] movement type indices (optional)
    Output: [B, 1] tremor severity prediction
    """
    def __init__(self, dropout=0.45, all_movements=False, num_movements=11,
                dom_hand_weight=3.0,
                non_dom_weight=1.0):
        super().__init__()
        self.all_movements = all_movements
        self.dropout = nn.Dropout(dropout)
        
        # 0. Dominant and non-dominant hand weights
        self.dom_hand_weight = dom_hand_weight
        self.non_dom_weight = non_dom_weight
        
        # 1. Multi-scale CNN (single path for both hands)
        self.cnn = MultiScaleCNN(dropout=dropout) # -> [B, 128]
        
        # 2. Frequency & statistical extractors
        self.frequency_analyzer = FrequencyAnalyzer(output_dim=128, dropout=dropout) # -> [B, 11, 128]
        self.stat_extractor = StatisticalFeatureExtractor(out_dim=32) # -> [B, 32]
        
        # 3. Bilateral coordination
        self.bilateral_attn = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, 
            dropout=dropout * 0.5, batch_first=True
        )
        
        # 4. Asymmetry projection
        self.contrast_proj = nn.Linear(128, 64)
        
        # 5. Embeddings (Handedness & movement type)
        self.hand_embed = nn.Embedding(2, 48)
        self.hand_proj = nn.Sequential(
            nn.Linear(48, 96), nn.LayerNorm(96), nn.Tanh(),
            nn.Dropout(dropout * 0.3), nn.Linear(96, 48)
        )
        
        if all_movements:
            self.movement_embed = nn.Embedding(num_movements, 32)
            self.movement_proj = nn.Sequential(
                nn.Linear(32, 64), nn.LayerNorm(64), nn.GELU(),
                nn.Dropout(dropout * 0.3), nn.Linear(64, 32)
            )
        
        # 6. Final fusion classifier
        # Features: weighted_cnn(128) + freq(128) + stat(64) + bilateral(64) + asymmetry(64) + hand(48) + movement(32 if enabled) = 528
        cnn_dim = 128
        freq_dim = 128
        stat_dim = 64
        bilateral_dim = 64
        asymmetry_dim = 64
        hand_dim = 48
        movement_dim = 32 if all_movements else 0
        total_dim = cnn_dim + freq_dim + stat_dim + bilateral_dim + asymmetry_dim + hand_dim + movement_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 640), nn.BatchNorm1d(640), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(640, 192), nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(192, 320), nn.BatchNorm1d(320), nn.GELU(), nn.Dropout(dropout * 0.7),
            nn.Linear(320, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, handedness, movements=None, mixup_lambda=None):
        """
        Forward pass with dominant hand weighting.
        
        Args:
            x: [B, 2, T, 6] IMU data (2 hands, T samples, 6 channels)
            handedness: [B] dominant hand (0=left, 1=right)
            movements: [B] movement type (optional, required if all_movements=True)
            mixup_lambda: float in [0,1] for mixup augmentation (optional)
        
        Returns:
            [B, 1] tremor severity scores
        """
        B = x.shape[0]
        
        # 0. Apply mixup if requested
        if mixup_lambda is not None and self.training:
            x, handedness_mix, movements_mix = self._apply_mixup(
                x, handedness, movements, mixup_lambda
            )
        else:
            handedness_mix, movements_mix = None, None
        
        # 1. Extract left/right signals: [B, 6, T]
        left_raw = x[:, 0].permute(0, 2, 1)   # [B, T, 6] -> [B, 6, T]
        right_raw = x[:, 1].permute(0, 2, 1)
        
        # 2. Extract CNN features from each hand "separately"
        left_cnn = self.cnn(left_raw)   # [B, 128]
        right_cnn = self.cnn(right_raw) # [B, 128]
        
        # 3. Compute dominant hand weights (3.0x for dominant, 1.0x for non-dominant)        
        if handedness_mix is not None: 
            # 3.1. Mixup case: soft weighting
            left_w = handedness_mix[:, 0].view(B, 1, 1)   # [B, 1, 1]
            right_w = handedness_mix[:, 1].view(B, 1, 1)
            
            left_scale = left_w * self.dom_hand_weight + right_w * self.non_dom_weight
            right_scale = right_w * self.dom_hand_weight + left_w * self.non_dom_weight
        else:
            # 3.2. Standard case: hard weighting
            is_left = (handedness == 0).float().view(B, 1, 1)
            is_right = (handedness == 1).float().view(B, 1, 1)
            
            left_scale = is_left * self.dom_hand_weight + is_right * self.non_dom_weight
            right_scale = is_right * self.dom_hand_weight + is_left * self.non_dom_weight
        
        # 3.3. Weight CNN "outputs" based on hand dominance
        cnn_feat = (left_scale * left_cnn + right_scale * right_cnn) / (dom_weight + non_dom_weight)  # [B, 128]
        
        # 4. Frequency analysis on unweighted left/right
        freq_feat_temporal = self.frequency_analyzer(left_raw, right_raw)  # [B, 11, 128]
        
        # 4.1. Attention pooling over 11 temporal segments
        attn_scores = torch.softmax(self.temporal_freq_attn(freq_feat_temporal), dim=1)  # [B, 11, 1]
        freq_feat = (freq_feat_temporal * attn_scores).sum(dim=1)  # [B, 128]
        
        # 5. Statistical features (for both hands separately)
        left_stat = self.stat_extractor(left_raw)   # [B, 32]
        right_stat = self.stat_extractor(right_raw) # [B, 32]
        stat_feat = torch.cat([left_stat, right_stat], dim=1)  # [B, 64]
        
        # 6. Hand asymmetry features
        asymmetry_feat = self.contrast_proj(torch.abs(left_cnn - right_cnn))  # [B, 64]
        
        # 7. Bilateral coordination via attention
        bilateral_feat = self._compute_bilateral_coordination(left_raw, right_raw)  # [B, 64]
        
        # 8. Hand embedding
        hand_emb = self._compute_hand_embedding(handedness, handedness_mix)  # [B, 48]
        
        # 8.1. Movement embedding (optional)
        if self.all_movements:
            if movements is None and movements_mix is None:
                raise ValueError("movements required when all_movements=True")
            move_emb = self._compute_movement_embedding(movements, movements_mix)  # [B, 32]
        
        # 9. Combine all features
        feat_list = [cnn_feat, freq_feat, stat_feat, bilateral_feat, asymmetry_feat, hand_emb]
        if self.all_movements:
            feat_list.append(move_emb)
        
        combined = torch.cat(feat_list, dim=-1)  # [B, total_dim]
        
        return self.fusion(combined)  # [B, 1]
    
    def _apply_mixup(self, x, handedness, movements, mixup_lambda):
        """
        Apply mixup data augmentation.
        
        Mixup is a data augmentation technique that creates virtual training examples by blending two samples together.
        
        How it works:
            1. Takes two random samples (x1, x2) and their labels (y1, y2)
            2. Blends them: x_new = λ*x1 + (1-λ)*x2 
                            y_new = λ*y1 + (1-λ)*y2
        Args:
            x: [B, 2, T, 6]
            handedness: [B]
            movements: [B] or None
            mixup_lambda: float
        
        Returns:
            x: [B, 2, T, 6] mixed data
            handedness_mix: [B, 2] one-hot mixed handedness
            movements_mix: [B, num_movements] one-hot mixed movements (or None)
        """
        B = x.shape[0]
        idx = torch.randperm(B, device=x.device)
        
        # Mix inputs
        x = mixup_lambda * x + (1 - mixup_lambda) * x[idx]
        
        # Mix handedness (one-hot)
        hand_oh = F.one_hot(handedness.long(), 2).float()
        handedness_mix = mixup_lambda * hand_oh + (1 - mixup_lambda) * hand_oh[idx]
        
        # Mix movements (one-hot)
        movements_mix = None
        if self.all_movements and movements is not None:
            move_oh = F.one_hot(movements.long(), self.movement_embed.num_embeddings).float()
            movements_mix = mixup_lambda * move_oh + (1 - mixup_lambda) * move_oh[idx]
        
        return x, handedness_mix, movements_mix
    
    def _compute_bilateral_coordination(self, left_raw, right_raw):
        """
        Compute bilateral coordination features via attention.
        Uses self-attention to see how left and right movements relate over time
        
        Args:
            left_raw: [B, 6, T]
            right_raw: [B, 6, T]
        
        Returns:
            [B, 64] bilateral coordination features
        """
        # 1. Extract sequences from CNN intermediate features
        left_seq = self.cnn.get_sequence(left_raw)    # [B, SeqLen, 128]
        right_seq = self.cnn.get_sequence(right_raw)  # [B, SeqLen, 128]
        
        # 2. Concatenate and apply self-attention
        bilateral_seq = torch.cat([left_seq, right_seq], dim=1)  # [B, 2*SeqLen, 128]
        bilateral_attn, _ = self.bilateral_attn(bilateral_seq, bilateral_seq, bilateral_seq)
        
        # 3. Pool and project
        bilateral_feat = self.contrast_proj(bilateral_attn.mean(dim=1))  # [B, 64]
        return bilateral_feat
    
    def _compute_hand_embedding(self, handedness, handedness_mix):
        """
        Compute hand dominance embedding.
        
        Args:
            handedness: [B] hand indices
            handedness_mix: [B, 2] one-hot weights (or None)
        
        Returns:
            [B, 48] hand embedding
        """
        if handedness_mix is not None:
            hand_emb = handedness_mix @ self.hand_embed.weight
        else:
            hand_emb = self.hand_embed(handedness.long())
        
        return self.hand_proj(hand_emb)
    
    def _compute_movement_embedding(self, movements, movements_mix):
        """
        Compute movement type embedding.
        
        Args:
            movements: [B] movement indices
            movements_mix: [B, num_movements] one-hot weights (or None)
        
        Returns:
            [B, 32] normalized movement embedding
        """
        if movements_mix is not None:
            move_emb = movements_mix @ self.movement_embed.weight
        else:
            move_emb = self.movement_embed(movements.long())
        
        move_emb = self.movement_proj(move_emb)
        return F.normalize(move_emb, dim=-1)


class MultiScaleCNN(nn.Module):
    """
    Multi-scale convolutional feature extractor with temporal attention.
    
    Uses three parallel convolutions with different kernel sizes to capture
    fast, medium, and slow tremor patterns, followed by squeeze-excitation
    and temporal attention pooling.
    
    Input:  [B, 6, T] IMU signal
    Output: [B, 128] feature vector
    """
    def __init__(self, dropout=0.45):
        super().__init__()
        
        # Multi-scale convolutions (fast/medium/slow)
        self.conv_fast = nn.Conv1d(6, 64, kernel_size=3, stride=2, padding=1) # -> small kernel
        self.conv_mid = nn.Conv1d(6, 64, kernel_size=7, stride=2, padding=3) # -> mid kernel
        self.conv_slow = nn.Conv1d(6, 64, kernel_size=15, stride=2, padding=7) # -> large kernel
        
        self.bn1 = nn.BatchNorm1d(192)
        self.conv2 = nn.Conv1d(192, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Squeeze-and-excitation (channel attention)
        self.se_fc1 = nn.Linear(128, 16)
        self.se_fc2 = nn.Linear(16, 128)
        
        # Temporal attention
        self.temporal_attn = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Extract multi-scale features.
        
        Args:
            x: [B, 6, T] IMU signal
        
        Returns:
            [B, 128] feature vector
        """
        # Multi-scale convolutions
        fast = self.conv_fast(x)   # [B, 64, T/2]
        mid = self.conv_mid(x)     # [B, 64, T/2]
        slow = self.conv_slow(x)   # [B, 64, T/2]
        
        # Concatenate and process
        x_cat = torch.cat([fast, mid, slow], dim=1)  # [B, 192, T/2]
        x_cat = F.gelu(self.bn1(x_cat))
        x_cat = self.dropout(x_cat)
        
        x_cat = F.gelu(self.bn2(self.conv2(x_cat)))  # [B, 128, T/4]
        
        # Squeeze-and-excitation (SE) -> channel attention.
        se_pool = x_cat.mean(dim=2)  # [B, 128]
        se_attn = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(se_pool))))  # [B, 128]
        x_se = x_cat * se_attn.unsqueeze(2)  # [B, 128, T/4]
        
        # Temporal attention pooling
        x_transpose = x_se.permute(0, 2, 1)  # [B, T/4, 128]
        attn_scores = torch.softmax(self.temporal_attn(x_transpose), dim=1)  # [B, T/4, 1]
        x_att = (x_transpose * attn_scores).sum(dim=1)  # [B, 128]
        
        # Add max pooling
        x_max = x_se.max(dim=2)[0]  # [B, 128]
        
        return x_att + 0.3 * x_max  # [B, 128]
    
    def get_sequence(self, x):
        """
        Get intermediate sequence for attention (used in bilateral coordination).
        Lets the model see how features change over time instead of just a single compressed vector.
        
        Args:
            x: [B, 6, T]
        
        Returns:
            [B, SeqLen, 128] sequence features
        """
        # Multi-scale convolutions
        fast = self.conv_fast(x)
        mid = self.conv_mid(x)
        slow = self.conv_slow(x)
        
        # Concatenate and process
        x_cat = torch.cat([fast, mid, slow], dim=1)
        x_cat = F.gelu(self.bn1(x_cat))
        x_seq = self.conv2(x_cat)  # [B, 128, SeqLen]
        
        return x_seq.permute(0, 2, 1)  # [B, SeqLen, 128]