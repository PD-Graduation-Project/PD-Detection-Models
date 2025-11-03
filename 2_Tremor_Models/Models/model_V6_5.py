import torch
from torch import nn
import torch.nn.functional as F

class TremorNetGRU_V6_5(nn.Module):
    """
    Enhanced CNN–GRU–Attention model for binary tremor classification (Healthy=0, PD=1),
    with improved architecture for dual-wrist signals and handedness context.
    
    Key improvements:
        - Residual connections in CNN
        - Multi-scale feature extraction
        - Channel attention mechanism
        - Learnable positional encoding
        - Improved attention mechanisms
        - Better feature fusion
    """

    def __init__(self,
                hidden_size: int = 128,
                handedness_embed_dim: int = 32,
                dropout: float = 0.3,
                num_attention_heads: int = 4):
        super().__init__()

        self.hidden_size = hidden_size
        self.handedness_embed_dim = handedness_embed_dim

        # 1. Enhanced CNN Feature Extractor with residual connections
        # ------------------------------------------------------------
        self.conv1 = self._make_residual_block(6, 64, 7, 2, 3, dropout * 0.5)
        self.conv2 = self._make_residual_block(64, 128, 5, 2, 2, dropout * 0.5)
        self.conv3 = self._make_residual_block(128, 256, 3, 1, 1, dropout)
        
        # 1.1. Multi-scale feature extraction
        # Looks at a wider time window without increasing computation.
        self.dilated_conv = nn.Conv1d(256, 256, 3, padding=2, dilation=2)
        self.conv_bn = nn.BatchNorm1d(256)
        
        # 1.2. Channel attention for feature recalibration
        # Learns which signal channels are most important and boosts them.
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Squeeze: global average pooling
            nn.Flatten(),
            nn.Linear(256, 256 // 16), # Compress channels
            nn.ReLU(),
            nn.Linear(256 // 16, 256), # Expand back
            nn.Sigmoid()               # Get 0-1 importance weights
        )

        # 2. Enhanced GRU with layer normalization
        # ------------------------------------------
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

        # 3. Enhanced Multi-Head Self-Attention over GRU output 
        # -------------------------------------------------------
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, # (embed_dim == hidden_size*2)
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=dropout * 0.5
        )

        # 3.1. Learnable positional encoding (must match fused_features dimension (cnn 256 + handedness))
        # Learns the best way to remember time order instead of using fixed math formulas.
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, gru_input_size))
        
        # 3.2. Enhanced attention projection with residual connection
        self.attention_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # 3.3. Cross-wrist attention
        # Lets left and right wrist features "talk" to each other 
        # so the model can learn relationships between both wrists.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256 + handedness_embed_dim,
            num_heads=num_attention_heads // 2,
            batch_first=True,
            dropout=dropout * 0.5
        )

        # 4. Enhanced Handedness Embedding with projection
        # -------------------------------------------------
        self.handedness_embed = nn.Embedding(num_embeddings=2, embedding_dim=handedness_embed_dim)
        self.handedness_proj = nn.Sequential(
            nn.Linear(handedness_embed_dim, handedness_embed_dim * 2),
            nn.GELU(),
            nn.Linear(handedness_embed_dim * 2, handedness_embed_dim),
            nn.LayerNorm(handedness_embed_dim)
        )
        
        # Pooled feature dimensionality:
        # we concatenate 5 pooled vectors each with dim (hidden_size*2)
        self.pooled_dim = (hidden_size * 2) * 5  # e.g., 5 * 256 = 1280

        # Cross-wrist attention must accept pooled vectors (embed_dim = pooled_dim)
        # Make sure num_heads divides pooled_dim; default uses num_attention_heads//2
        cross_heads = max(1, num_attention_heads // 2)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.pooled_dim,
            num_heads=cross_heads,
            batch_first=True,
            dropout=dropout * 0.5
        )

        # 5. Enhanced Classifier with residual connections
        # -------------------------------------------------
        classifier_input_dim = self.pooled_dim * 3 + handedness_embed_dim
        
        self.classifier = nn.Sequential(
            # First block with residual connection
            nn.Linear(classifier_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Second block with residual connection
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            
            # Third block
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            # Output
            nn.Linear(128, 1)
        )
        
        # 5.1. Residual connections for classifier
        self.classifier_residual1 = nn.Linear(classifier_input_dim, 512) if classifier_input_dim != 512 else nn.Identity()
        self.classifier_residual2 = nn.Linear(512, 256) if 512 != 256 else nn.Identity()

        # 6. Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence"""
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
        """Create residual block with batch norm and activation"""
        return nn.Sequential(
            # First conv
            nn.Conv1d(in_c, out_c, kernel, stride, padding, bias=False),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Second conv
            nn.Conv1d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _apply_channel_attention(self, x):
        """Apply channel attention to features"""
        attention_weights = self.channel_attention(x).unsqueeze(-1)
        return x * attention_weights

    def _encode_wrist(self, x, handed_embed):
        """
        Enhanced wrist encoding with multiple improvements
        """
        B, T, C = x.shape
        
        # 1. Enhanced CNN feature extraction
        x = x.permute(0, 2, 1)  # [B, 6, T]
        
        # 1.1. Multi-scale feature extraction with residual connections
        features1 = self.conv1(x)
        features2 = self.conv2(features1)
        features3 = self.conv3(features2)
        
        # 1.2. Dilated convolution for larger receptive field
        dilated_features = F.gelu(self.conv_bn(self.dilated_conv(features3)))
        features = features3 + dilated_features  # Residual connection [B, 256, T']
        
        # 1.3. Apply channel attention
        features = self._apply_channel_attention(features)
        features = features.permute(0, 2, 1)  # [B, T', 256]

        # 2. Enhanced handedness fusion
        # expand and concat -> [B, T', 256 + handedness]
        handed_expanded = handed_embed.unsqueeze(1).expand(-1, features.size(1), -1)
        fused_features = torch.cat([features, handed_expanded], dim=-1) # [B, T', 256 + handedness]

        # 3. Enhanced temporal modeling with positional encoding
        seq_len = fused_features.size(1)
        pos_enc = self.pos_encoding[:, :seq_len, :].expand(B, -1, -1)
        fused_features = fused_features + pos_enc # [B, T', 256+handedness]

        # 4. GRU processing
        gru_out, _ = self.gru(fused_features) # [B, T', 256]
        gru_out = self.gru_ln(gru_out) # [B, T', hidden_size*2]

        # 5. Enhanced self-attention with residual connection
        attn_out, attn_weights = self.multihead_attn(gru_out, gru_out, gru_out)
        gru_out = gru_out + attn_out  # Residual connection
        gru_out = self.attention_proj(gru_out) # [B, T', hidden_size*2]

        # 6. Multi-pooling strategy with learnable weights
        # 6.1. Attention pooling
        attn_scores = torch.softmax(gru_out.mean(dim=-1, keepdim=True), dim=1)
        attended_pool = torch.sum(attn_scores * gru_out, dim=1) # Weighted average
        
        # 6.2. Statistical pooling
        mean_pool = gru_out.mean(dim=1)  # Simple average  
        max_pool = gru_out.max(dim=1)[0] # Most important parts
        std_pool = gru_out.std(dim=1)    # How much variation
        
        # 6.3. Learnable pooling weights
        pool_weights = torch.softmax(torch.randn(3, device=gru_out.device), dim=0)
        weighted_pool = (attended_pool * pool_weights[0] + 
                        mean_pool * pool_weights[1] + 
                        max_pool * pool_weights[2])
        
        # 7. Concatenate all pooling strategies
        pooled_features = torch.cat([weighted_pool, attended_pool, mean_pool, max_pool, std_pool], dim=1)
        # pooled_features shape: [B, (hidden_size*2) * 5] == [B, self.pooled_dim]
        
        return pooled_features

    def forward(self, x, handedness):
        """
        Enhanced forward pass with cross-wrist attention
        """
        B = x.shape[0]

        # 1. Enhanced handedness embedding
        handed_embed = self.handedness_embed(handedness.long())  # [B, handed_dim]
        handed_embed = self.handedness_proj(handed_embed)        # [B, handed_dim]
        handed_embed = F.normalize(handed_embed, dim=-1)

        # 2. Encode both wrists
        left_feats = self._encode_wrist(x[:, 0], handed_embed)   # [B, hidden_size*6]
        right_feats = self._encode_wrist(x[:, 1], handed_embed)  # [B, hidden_size*6]

        # 3. Cross-wrist attention for interaction modeling
        left_expanded = left_feats.unsqueeze(1)  # [B, 1, hidden_size*6]
        right_expanded = right_feats.unsqueeze(1)  # [B, 1, hidden_size*6]
        
        cross_features = torch.cat([left_expanded, right_expanded], dim=1) # [B, 2, pooled_dim]
        
        # cross_attention embed_dim == pooled_dim
        attended_cross, _ = self.cross_attention(cross_features, cross_features, cross_features)
        cross_pool = attended_cross.mean(dim=1)  # [B, pooled_dim]

        # 4. Combined features: left + right + cross + handedness
        combined = torch.cat([left_feats, right_feats, cross_pool, handed_embed], dim=1) # [B, pooled_dim*3 + handed_dim]

        # 5. Enhanced classifier with residual connections
        # First block with residual
        x1 = self.classifier[0](combined)
        x1 = self.classifier[1](x1)
        x1 = self.classifier[2](x1)
        x1 = self.classifier[3](x1)
        residual1 = self.classifier_residual1(combined)
        x1 = x1 + residual1
        
        # Second block with residual
        x2 = self.classifier[4](x1)
        x2 = self.classifier[5](x2)
        x2 = self.classifier[6](x2)
        x2 = self.classifier[7](x2)
        residual2 = self.classifier_residual2(x1)
        x2 = x2 + residual2
        
        # Remaining layers
        out = self.classifier[8](x2)
        out = self.classifier[9](out)
        out = self.classifier[10](out)
        out = self.classifier[11](out)

        return out