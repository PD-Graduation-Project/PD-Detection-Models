import torch
from torch import nn
import torch.nn.functional as F

class TremorNetGRU_V6_5(nn.Module):
    """
    Enhanced CNN–GRU–Attention model for binary tremor classification (Healthy=0, PD=1),
    with improved architecture for dual-wrist signals and handedness context.
    
    Key improvements:
        - Residual connections in CNN blocks
        - Multi-head self-attention over temporal features
        - Cross-wrist attention to model bilateral coordination
        - Enhanced handedness embedding with projection
        - Multiple pooling strategies (attention, mean, max)
        - Streamlined classifier with proper residual connections
    """

    def __init__(self,
                hidden_size: int = 128,
                handedness_embed_dim: int = 32,
                dropout: float = 0.3,
                num_attention_heads: int = 4):
        super().__init__()

        self.hidden_size = hidden_size
        self.handedness_embed_dim = handedness_embed_dim

        # 1. CNN Feature Extractor with residual connections
        # ---------------------------------------------------
        # Extracts spatial features from 6-channel IMU signals
        self.conv1 = self._make_residual_block(6, 64, 7, 2, 3, dropout * 0.5)
        self.conv2 = self._make_residual_block(64, 128, 5, 2, 2, dropout * 0.5)
        self.conv3 = self._make_residual_block(128, 256, 3, 1, 1, dropout)

        # 2. Bidirectional GRU for temporal modeling
        # -------------------------------------------
        # Input: CNN features (256) + handedness embedding
        gru_input_size = 256 + handedness_embed_dim
        self.gru = nn.GRU(
            input_size=gru_input_size, # (cnn_channels + handedness)
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 2.1. GRU layer normalization
        self.gru_ln = nn.LayerNorm(hidden_size * 2) # GRU output dim

        # 3. Multi-Head Self-Attention over GRU output 
        # ----------------------------------------------
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, # (embed_dim == hidden_size*2)
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=dropout * 0.5
        )

        # 3.1. Enhanced attention projection with residual connection
        self.attention_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 4. Enhanced Handedness Embedding with projection
        # -------------------------------------------------
        # Encodes whether patient is left-handed (0) or right-handed (1)
        self.handedness_embed = nn.Embedding(
            num_embeddings=2, 
            embedding_dim=handedness_embed_dim)
        
        # 4.1. Project handedness to richer representation
        self.handedness_proj = nn.Sequential(
            nn.Linear(handedness_embed_dim, handedness_embed_dim * 2),
            nn.GELU(),
            nn.Linear(handedness_embed_dim * 2, handedness_embed_dim),
            nn.LayerNorm(handedness_embed_dim)
        )
        

        # 5. Cross-Wrist Attention
        # --------------------------
        # Models bilateral coordination between left and right wrist
        # Operates on pooled features from each wrist
        self.pooled_dim = (hidden_size * 2) * 3  # attention + mean + max pooling
        
        cross_heads = max(1, num_attention_heads // 2)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.pooled_dim,
            num_heads=cross_heads,
            batch_first=True,
            dropout=dropout * 0.5
        )

        # 6. Classifier Head with residual connections
        # ----------------------------------------------
        # Input: left_features + right_features + cross_features + handedness
        classifier_input_dim = self.pooled_dim * 3 + handedness_embed_dim
        
        # 6.1. First residual block
        self.fc1 = nn.Linear(classifier_input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.residual1 = nn.Linear(classifier_input_dim, 256)
        
        # 6.2. Second residual block
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.residual2 = nn.Linear(256, 128)
        
        # 6.3. Output layers
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
        self.dropout_classifier = nn.Dropout(dropout)

        # 7. Initialize weights for better convergence
        # ---------------------------------------------
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight) # Smart random starting values
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def _make_residual_block(self, in_c, out_c, kernel, stride, padding, dropout):
        """
        Create residual convolutional block with two conv layers.
        Helps with gradient flow and feature learning.
        """
        return nn.Sequential(
            # First convolution
            nn.Conv1d(in_c, out_c, kernel, stride, padding, bias=False),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Second convolution (residual path)
            nn.Conv1d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _encode_wrist(self, x, handed_embed):
        """
        Encode single wrist signal with CNN, GRU, and attention.
        
        Args:
            x: [B, T, 6] -> single wrist IMU signal
            handed_embed: [B, handedness_embed_dim] -> handedness context
            
        Returns:
            pooled_features: [B, pooled_dim] - aggregated temporal features
        """
        B, T, C = x.shape
        
        # 1. CNN feature extraction
        # --------------------------
        x = x.permute(0, 2, 1)  # [B, 6, T] - channel-first for Conv1d
        
        features1 = self.conv1(x)      # [B, 64, T/2]
        features2 = self.conv2(features1)  # [B, 128, T/4]
        features3 = self.conv3(features2)  # [B, 256, T/4]
        
        features = features3.permute(0, 2, 1)  # [B, T', 256] - time-first for GRU

        # 2. Fuse CNN features with handedness context
        # ----------------------------------------------
        # Expand handedness to match temporal dimension
        handed_expanded = handed_embed.unsqueeze(1).expand(-1, features.size(1), -1)
        fused_features = torch.cat([features, handed_expanded], dim=-1)  # [B, T', 256+handedness_dim]

        # 3. Bidirectional GRU for temporal modeling
        # -------------------------------------------
        gru_out, _ = self.gru(fused_features)  # [B, T', hidden_size*2]
        gru_out = self.gru_ln(gru_out)

        # 4. Multi-head self-attention with residual connection
        # ------------------------------------------------------
        attn_out, _ = self.multihead_attn(gru_out, gru_out, gru_out)
        gru_out = gru_out + (attn_out * 0.8)  # Residual connection
        gru_out = self.attention_proj(gru_out)  # [B, T', hidden_size*2]

        # 5. Multiple pooling strategies
        # --------------------------------
        # 5.1. Attention-weighted pooling (learned importance)
        attn_scores = torch.softmax(gru_out.mean(dim=-1, keepdim=True), dim=1)
        attended_pool = torch.sum(attn_scores * gru_out, dim=1)  # [B, hidden_size*2]
        
        # 5.2. Mean pooling (average over time)
        mean_pool = gru_out.mean(dim=1)  # [B, hidden_size*2]
        
        # 5.3. Max pooling (capture peak activations)
        max_pool = gru_out.max(dim=1)[0]  # [B, hidden_size*2]
        
        # 6. Concatenate all pooling strategies
        # ---------------------------------------
        pooled_features = torch.cat([attended_pool, mean_pool, max_pool], dim=1)
        # Shape: [B, (hidden_size*2) * 3] = [B, pooled_dim]
        
        return pooled_features

    def forward(self, x, handedness):
        """
        Forward pass through the network.
        
        Args:
            x: [B, 2, T, 6] - dual-wrist signals
                x[:, 0] -> left wrist
                x[:, 1] -> right wrist
            handedness: [B] - patient handedness (0=left-handed, 1=right-handed)
            
        Returns:
            logits: [B, 1] - classification logits
        """
        B = x.shape[0]

        # 1. Process handedness embedding
        # ---------------------------------
        handed_embed = self.handedness_embed(handedness.long())  # [B, handedness_dim]
        handed_embed = self.handedness_proj(handed_embed)        # [B, handedness_dim]
        handed_embed = F.normalize(handed_embed, dim=-1)         # Normalize for stability

        # 2. Encode both wrists independently
        # -------------------------------------
        left_feats = self._encode_wrist(x[:, 0], handed_embed)   # [B, pooled_dim]
        right_feats = self._encode_wrist(x[:, 1], handed_embed)  # [B, pooled_dim]

        # 3. Cross-wrist attention for bilateral coordination
        # -----------------------------------------------------
        # Model interactions between left and right wrist movements
        left_expanded = left_feats.unsqueeze(1)   # [B, 1, pooled_dim]
        right_expanded = right_feats.unsqueeze(1)  # [B, 1, pooled_dim]
        
        cross_features = torch.cat([left_expanded, right_expanded], dim=1)  # [B, 2, pooled_dim]
        
        attended_cross, _ = self.cross_attention(cross_features, cross_features, cross_features)
        cross_pool = attended_cross.mean(dim=1)  # [B, pooled_dim]

        # 4. Combine all features
        # ------------------------
        # Left + Right + Cross-wrist + Handedness context
        combined = torch.cat([left_feats, right_feats, cross_pool, handed_embed], dim=1)
        # Shape: [B, pooled_dim*3 + handedness_dim]

        # 5. Classification with residual connections
        # ---------------------------------------------
        # First residual block
        x1 = F.gelu(self.ln1(self.fc1(combined)))
        x1 = self.dropout_classifier(x1)
        residual1 = self.residual1(combined)
        x1 = x1 + residual1  # [B, 512]
        
        # Second residual block
        x2 = F.gelu(self.ln2(self.fc2(x1)))
        x2 = self.dropout_classifier(x2)
        residual2 = self.residual2(x1)
        x2 = x2 + residual2  # [B, 256]
        
        # Output layers
        x3 = F.gelu(self.fc3(x2))
        x3 = self.dropout_classifier(x3)  # [B, 128]
        
        out = self.fc4(x3)  # [B, 1]

        return out