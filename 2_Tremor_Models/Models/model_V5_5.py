import torch
from torch import nn
import torch.nn.functional as F

class TremorNetGRU_V5_5(nn.Module):
    """
    CNN–GRU–Attention model for binary tremor classification (Healthy=0, PD=1),
    adapted to handle two-hand signals of shape [B, 2, T, 6].
    
    Differences from V4_5:
    - Input x: shape [B, 2, T, 6]
        - x[:, 0, :, :] -> always left wrist signal
        - x[:, 1, :, :] -> always right wrist signal

    - handedness: shape [B, 1]
        0 -> patient is left-handed
        1 -> patient is right-handed
    - Each wrist is processed by the same CNN–GRU–Attention stack.
    """

    def __init__(self,
                hidden_size: int = 128,
                handedness_embed_dim: int = 32,
                dropout: float = 0.3,
                num_attention_heads: int = 4):
        super().__init__()

        # 1. CNN Feature Extractor for per-wrist signals
        # ------------------------------------------------
        self.cnn = nn.Sequential(
            self._make_conv_block(6, 64, 7, 2, 3),   # [B, 64, T/2]
            nn.Dropout(dropout * 0.5),

            self._make_conv_block(64, 128, 5, 2, 2),  # [B, 128, T/4]
            nn.Dropout(dropout * 0.5),

            self._make_conv_block(128, 256, 3, 1, 1), # [B, 256, T/4]
            nn.Dropout(dropout)
        )

        # 2. GRU for temporal modeling (bidirectional, stacked)
        # -------------------------------------------------------
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 3. Multi-Head Attention over GRU outputs
        # -----------------------------------------
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

        # 4. handedness embedding
        # -------------------------
        self.handedness_embed = nn.Embedding(
            num_embeddings= 2, # 0=Left-handed, 1=Right-handed
            embedding_dim= handedness_embed_dim)

        # 5. Layer normalization for GRU outputs
        # -----------------------------------------
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # 6. Classifier head that combines both wrists + handedness
        # ------------------------------------------------------------
        # Each wrist -> 3 * hidden_size * 2 features (attended + mean + max)
        # Two wrists -> x2 + handedness embedding
        classifier_input_dim = (hidden_size * 6) * 2 + handedness_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1)
        )

    def _make_conv_block(self, in_c, out_c, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel, stride, padding),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def _encode_wrist(self, x):
        """Encode one wrist signal.

        Input: x with shape (B, T, 6).
        Returns: concatenated temporal features (attended, mean, max)
                 with shape (B, hidden_size * 6).
        """
        # 1. Convert to channel-first for CNN, then back to time-first
        x = x.permute(0, 2, 1)          # [B, 6, T]
        features = self.cnn(x).permute(0, 2, 1)  # [B, T', 256]

        # 2. Temporal modeling via GRU
        gru_out, _ = self.gru(features)
        gru_out = self.layer_norm(gru_out)

        # 3. Self-attention over GRU outputs
        attn_out, _ = self.multihead_attn(gru_out, gru_out, gru_out)
        attn_out = self.attention_proj(attn_out)

        # 3.1. Compute attention weights across time and apply
        attn_weights = torch.softmax(attn_out.mean(dim=-1, keepdim=True), dim=1)
        attended = torch.sum(attn_weights * attn_out, dim=1)

        # 4. Global pooling: mean and max over time
        pooled_mean = gru_out.mean(dim=1)
        pooled_max = gru_out.max(dim=1)[0]

        # 5. Concatenate attended + mean + max into final feature vector
        return torch.cat([attended, pooled_mean, pooled_max], dim=1)  # [B, hidden_size*6]

    def forward(self, x, handedness):
        """
        Args:
            x: [B, 2, T, 6]  -> left-hand (0), right-hand (1)
            handedness: [B, 1]
        Returns:
            logits: [B, 1]
        """

        # 1. Encode each wrist independently
        left_feats = self._encode_wrist(x[:, 0])
        right_feats = self._encode_wrist(x[:, 1])
        
        # 2. Embed and normalize handedness (learned influence)
        handed_embed = F.normalize(self.handedness_embed(handedness.squeeze(1).long()), dim=-1)

        # 3. Concatenate features and handedness, then classify
        combined = torch.cat([left_feats, right_feats, handed_embed], dim=1)

        # 4. Final classification head -> single logit
        out = self.classifier(combined)
        return out
