import torch
from torch import nn
import torch.nn.functional as F

class TremorNetGRU_V4_5(nn.Module):
    """
    CNN–GRU–Attention model for binary tremor classification (Healthy=0, PD=1),
    without movement embedding.

    Differences from V4:
    - Movement embedding block removed.
    - Classifier input dimension adjusted.
    - Forward pass simplified accordingly.
    """

    def __init__(self,
                hidden_size: int = 128, 
                wrist_embed_dim: int = 32, 
                dropout: float = 0.3,
                num_attention_heads: int = 4):
        super().__init__()
        
        # 1. CNN Feature Extractor
        self.cnn = nn.Sequential(
            self._make_conv_block(6, 64, 7, 2, 3),   # [B, 64, T/2]
            nn.Dropout(dropout * 0.5),

            self._make_conv_block(64, 128, 5, 2, 2),  # [B, 128, T/4]
            nn.Dropout(dropout * 0.5),

            self._make_conv_block(128, 256, 3, 1, 1), # [B, 256, T/4]
            nn.Dropout(dropout)
        )
        
        # 2. GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 3. Multi-Head Attention mechanism
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=dropout * 0.5
        )

        self.attention_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.Tanh(),
            nn.Dropout(dropout * 0.5),
        )
        
        # 4. Wrist embedding only
        self.wrist_embed = nn.Embedding(
            num_embeddings=2,  # 0 = Left, 1 = Right
            embedding_dim=wrist_embed_dim
        )
        
        # 5. Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # 6. Classifier 
        # Input: 3*(hidden_size*2) + wrist_embed_dim
        classifier_input_dim = hidden_size * 6 + wrist_embed_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 1)  # binary classification
        )
        
    def _make_conv_block(self, in_c, out_c, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel, stride, padding),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, wrist, movement=None):
        """
        Args:
            x: [B, T, 6]
            wrist: [B] (0 or 1)
            movement: ignored (for compatibility)

        Returns:
            logits: [B, 1]
        """
        # 1. CNN
        x = x.permute(0, 2, 1)
        features = self.cnn(x).permute(0, 2, 1)
        
        # 2. GRU
        gru_out, _ = self.gru(features)
        gru_out = self.layer_norm(gru_out)
        
        # 3. Multi-Head Attention
        attended_out, _ = self.multihead_attn(gru_out, gru_out, gru_out)
        attended_out = self.attention_proj(attended_out)
        attn_weights = torch.softmax(attended_out.mean(dim=-1, keepdim=True), dim=1)
        attended = torch.sum(attn_weights * attended_out, dim=1)
        
        # 4. Global pooling
        pooled_mean = gru_out.mean(dim=1)
        pooled_max = gru_out.max(dim=1)[0]
        
        # 5. Temporal features
        time_features = torch.cat([attended, pooled_mean, pooled_max], dim=1)
        
        # 6. Wrist embedding
        wrist_embed = F.normalize(self.wrist_embed(wrist.long()), dim=-1)
        
        # 7. Combine features
        combined = torch.cat([time_features, wrist_embed], dim=1)
        
        # 8. Classifier
        out = self.classifier(combined)
        return out
