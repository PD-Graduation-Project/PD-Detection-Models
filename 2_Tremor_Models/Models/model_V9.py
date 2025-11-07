"""
Combines:
1. Advanced FFT frequency analysis (mandatory)
2. CNN pathway (efficient depthwise separable)
3. GRU pathway (temporal sequences)
4. Dominant hand super-pathway (3x capacity)
5. Statistical features
6. Bilateral coordination
7. Optional movement-type conditioning

Usage:
    # Single movement training
    model = TremorNetV10(all_movements=False)
    output = model(signals, handedness)
    
    # Multi-movement training
    model = TremorNetV10(all_movements=True)
    output = model(signals, handedness, movements)
"""

import torch
from torch import nn
import torch.nn.functional as F
from tremor_modules import (
    FrequencyAnalyzer,
    StatisticalFeatureExtractor,
    EfficientCNNExtractor,
    DominantHandCNNExtractor,
    TemporalGRUEncoder,
    BilateralCoordinationModule,
    MovementContextEncoder,
    HandednessContextEncoder
)


class TremorNetV9(nn.Module):
    """
    Ultimate multi-modal tremor classification model.
    
    Architecture:
        1. Non-dominant wrist: Efficient CNN (128D) + GRU (256D)
        2. Dominant wrist: Super CNN (256D) + GRU (256D)
        3. Frequency analysis: FFT power + phase coherence (128D)
        4. Statistical features: Moments (8D)
        5. Bilateral coordination: Cross-attention (128D)
        6. Handedness context: Embedding (48D)
        7. [Optional] Movement context: Embedding (64D)
    """
    
    def __init__(self, 
                dropout=0.45,
                gru_hidden=128,
                all_movements=False,
                num_movements=11):
        super().__init__()
        
        self.all_movements = all_movements
        self.dropout_rate = dropout
        
        # ===== CONTEXT ENCODERS =====
        self.handedness_encoder = HandednessContextEncoder(embed_dim=48, dropout=dropout * 0.5)
        
        if all_movements:
            self.movement_encoder = MovementContextEncoder(
                num_movements=num_movements, 
                embed_dim=64, 
                dropout=dropout * 0.5
            )
        
        # ===== FREQUENCY ANALYSIS (MANDATORY) =====
        self.frequency_analyzer = FrequencyAnalyzer(
            num_freq_bins=40,  # Increased from 30
            output_dim=128,
            dropout=dropout
        )
        
        # ===== STATISTICAL FEATURES =====
        self.stat_extractor = StatisticalFeatureExtractor()
        
        # ===== NON-DOMINANT WRIST PATHWAY =====
        # CNN branch
        self.non_dominant_cnn = EfficientCNNExtractor(output_dim=128, dropout=dropout)
        
        # GRU branch (operates on raw CNN features before pooling)
        self.non_dominant_conv1 = nn.Conv1d(6, 48, 7, stride=2, padding=3)
        self.non_dominant_conv2 = nn.Conv1d(48, 96, 5, stride=2, padding=2, dilation=2)
        self.non_dominant_gru = TemporalGRUEncoder(
            input_dim=96,
            hidden_dim=gru_hidden,
            num_layers=2,
            dropout=dropout
        )
        
        # ===== DOMINANT WRIST SUPER-PATHWAY =====
        # CNN branch (3x capacity)
        self.dominant_cnn = DominantHandCNNExtractor(output_dim=256, dropout=dropout)
        
        # GRU branch
        self.dominant_conv1 = nn.Conv1d(6, 96, 7, stride=2, padding=3)
        self.dominant_conv2 = nn.Conv1d(96, 192, 5, stride=2, padding=2, dilation=2)
        self.dominant_gru = TemporalGRUEncoder(
            input_dim=192,
            hidden_dim=gru_hidden,
            num_layers=2,
            dropout=dropout
        )
        
        # ===== BILATERAL COORDINATION =====
        # Use GRU sequences for cross-attention
        self.bilateral_coord = BilateralCoordinationModule(
            dim=gru_hidden * 2,  # GRU is bidirectional
            num_heads=4,
            dropout=dropout
        )
        
        # ===== FEATURE FUSION =====
        # Calculate total feature dimension
        non_dom_dim = 128 + (gru_hidden * 2)  # CNN + GRU
        dom_dim = 256 + (gru_hidden * 2)       # CNN + GRU
        freq_dim = 128
        stat_dim = 8
        bilateral_dim = gru_hidden * 2
        hand_dim = 48
        movement_dim = 64 if all_movements else 0
        
        total_dim = non_dom_dim + dom_dim + freq_dim + stat_dim + bilateral_dim + hand_dim + movement_dim
        
        # Multi-stage classifier with bottleneck
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Bottleneck (compression)
            nn.Linear(768, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Expansion
            nn.Linear(192, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def _encode_wrist_cnn_gru(self, signal, is_dominant=False):
        """
        Dual-pathway encoding: CNN + GRU.
        
        Args:
            signal: [B, 6, T]
            is_dominant: bool - use dominant or non-dominant pathway
        Returns:
            cnn_features: [B, cnn_dim]
            gru_features: [B, gru_dim]
            gru_sequence: [B, T', gru_dim] - for bilateral coordination
        """
        if is_dominant:
            # CNN pathway
            cnn_features = self.dominant_cnn(signal)  # [B, 256]
            
            # GRU pathway
            x = F.gelu(self.dominant_conv1(signal))
            x = F.gelu(self.dominant_conv2(x))  # [B, 192, T']
            
            # Convert to sequence
            gru_input = x.permute(0, 2, 1)  # [B, T', 192]
            gru_features = self.dominant_gru(gru_input)  # [B, 256]
            
            # Return sequence for bilateral coordination
            gru_seq, _ = self.dominant_gru.gru(gru_input)  # [B, T', 256]
            
        else:
            # CNN pathway
            cnn_features = self.non_dominant_cnn(signal)  # [B, 128]
            
            # GRU pathway
            x = F.gelu(self.non_dominant_conv1(signal))
            x = F.gelu(self.non_dominant_conv2(x))  # [B, 96, T']
            
            gru_input = x.permute(0, 2, 1)  # [B, T', 96]
            gru_features = self.non_dominant_gru(gru_input)  # [B, 256]
            
            gru_seq, _ = self.non_dominant_gru.gru(gru_input)  # [B, T', 256]
        
        return cnn_features, gru_features, gru_seq
    
    def forward(self, x, handedness, movements=None, mixup_lambda=None):
        """
        Forward pass.
        
        Args:
            x: [B, 2, T, 6] - dual-wrist signals (0=left, 1=right)
            handedness: [B] - 0=left-handed, 1=right-handed
            movements: [B] - movement type indices (0-10), required if all_movements=True
            mixup_lambda: Optional[float] - mixup augmentation strength
        
        Returns:
            logits: [B, 1]
        """
        B = x.shape[0]
        
        # ===== OPTIONAL MIXUP AUGMENTATION =====
        if mixup_lambda is not None and self.training:
            indices = torch.randperm(B, device=x.device)
            x = mixup_lambda * x + (1 - mixup_lambda) * x[indices]
            
            handedness_onehot = F.one_hot(handedness.long(), 2).float()
            handedness_mixed = mixup_lambda * handedness_onehot + (1 - mixup_lambda) * handedness_onehot[indices]
            
            if self.all_movements:
                movements_onehot = F.one_hot(movements.long(), 11).float()
                movements_mixed = mixup_lambda * movements_onehot + (1 - mixup_lambda) * movements_onehot[indices]
        
        # ===== PREPARE SIGNALS =====
        left_raw = x[:, 0].permute(0, 2, 1)   # [B, 6, T]
        right_raw = x[:, 1].permute(0, 2, 1)  # [B, 6, T]
        
        # ===== CONTEXT ENCODING =====
        if mixup_lambda is not None and self.training:
            hand_features = self.handedness_encoder.forward_mixed(handedness_mixed.to(x.device))
            
            if self.all_movements:
                movement_features = movements_mixed.to(x.device) @ self.movement_encoder.movement_embed.weight
                movement_features = self.movement_encoder.movement_proj(movement_features)
                movement_features = F.normalize(movement_features, dim=-1)
        else:
            hand_features = self.handedness_encoder(handedness)
            
            if self.all_movements:
                if movements is None:
                    raise ValueError("movements must be provided when all_movements=True")
                movement_features = self.movement_encoder(movements)
        
        # ===== DETERMINE DOMINANT/NON-DOMINANT ROUTING =====
        if mixup_lambda is not None and self.training:
            left_weight = handedness_mixed[:, 0].view(B, 1, 1).to(x.device)
            right_weight = handedness_mixed[:, 1].view(B, 1, 1).to(x.device)
            
            dominant_raw = left_weight * left_raw + right_weight * right_raw
            non_dominant_raw = right_weight * left_raw + left_weight * right_raw
        else:
            is_left_handed = (handedness == 0).float().view(B, 1, 1)
            is_right_handed = (handedness == 1).float().view(B, 1, 1)
            
            dominant_raw = is_left_handed * left_raw + is_right_handed * right_raw
            non_dominant_raw = is_right_handed * left_raw + is_left_handed * right_raw
        
        # ===== EXTRACT FEATURES =====
        # Non-dominant wrist (CNN + GRU)
        non_dom_cnn, non_dom_gru, non_dom_seq = self._encode_wrist_cnn_gru(non_dominant_raw, is_dominant=False)
        
        # Dominant wrist (Super CNN + GRU)
        dom_cnn, dom_gru, dom_seq = self._encode_wrist_cnn_gru(dominant_raw, is_dominant=True)
        
        # Frequency analysis (both wrists)
        freq_features = self.frequency_analyzer(left_raw, right_raw)
        
        # Statistical features
        left_stat = self.stat_extractor(left_raw)
        right_stat = self.stat_extractor(right_raw)
        stat_features = (left_stat + right_stat) / 2
        
        # Bilateral coordination (cross-attention on GRU sequences)
        bilateral_features = self.bilateral_coord(non_dom_seq, dom_seq)
        
        # ===== COMBINE ALL FEATURES =====
        features_list = [
            non_dom_cnn,        # [B, 128]
            non_dom_gru,        # [B, 256]
            dom_cnn,            # [B, 256]
            dom_gru,            # [B, 256]
            freq_features,      # [B, 128]
            stat_features,      # [B, 8]
            bilateral_features, # [B, 256]
            hand_features       # [B, 48]
        ]
        
        if self.all_movements:
            features_list.append(movement_features)  # [B, 64]
        
        combined = torch.cat(features_list, dim=-1)
        
        # ===== CLASSIFICATION =====
        logits = self.classifier(combined)
        
        return logits
    
    def forward_with_uncertainty(self, x, handedness, movements=None, num_samples=10):
        """
        MC Dropout for uncertainty estimation.
        
        Args:
            x: [B, 2, T, 6]
            handedness: [B]
            movements: [B] - required if all_movements=True
            num_samples: int - number of forward passes
        
        Returns:
            mean_logits: [B, 1]
            std_logits: [B, 1]
        """
        self.train()  # Enable dropout
        
        logits_samples = []
        for _ in range(num_samples):
            logits = self.forward(x, handedness, movements)
            logits_samples.append(logits)
        
        logits_samples = torch.stack(logits_samples)
        
        mean_logits = logits_samples.mean(dim=0)
        std_logits = logits_samples.std(dim=0)
        
        return mean_logits, std_logits