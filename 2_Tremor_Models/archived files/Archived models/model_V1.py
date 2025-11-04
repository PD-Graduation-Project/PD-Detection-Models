import torch
from torch import nn

class TremorNetGRU_V1(nn.Module):
    """
    Enhanced CNN–GRU–Attention model for tremor classification across multiple movements.

    Key improvements:
    - Larger CNN backbone (6→512 channels)
    - Deeper GRU layers with more hidden units
    - Movement type embedding (11 movements)
    - Enhanced wrist embedding (larger dimension)
    - Handles variable length inputs (1024 or 2048 timesteps)
    - More parameters for better capacity

    Args:
        num_classes (int): number of output classes (3: PD, Healthy, Other).
        num_movements (int): number of movement types (default 11).
        hidden_size (int): GRU hidden size per direction.
        wrist_embed_dim (int): wrist embedding dimension.
        movement_embed_dim (int): movement type embedding dimension.
        dropout (float): dropout probability.

    Input:
        x (Tensor): [B, T, 6] IMU signals (T can be 1024 or 2048).
        wrist (Tensor): [B] wrist side (0=Left, 1=Right).
        movement (Tensor): [B] movement type (0-10 for 11 movements).

    Output:
        Tensor: [B, num_classes] classification logits.
    """
    def __init__(self,
                num_classes: int = 3,
                num_movements: int = 11,
                hidden_size: int = 256,  # Doubled from 128
                wrist_embed_dim: int = 64,  # Increased from 16
                movement_embed_dim: int = 128,  # New: movement embedding
                dropout: float = 0.4,
                ):
        super().__init__()
        
        # 1. Enhanced CNN feature extractor (MUCH LARGER)
        # ------------------------------------------------
        # Input: [B, C=6, T] -> after 3 conv stride-2 layers -> approx T/8
        self.cnn = nn.Sequential(
            # 1.1. Conv layer 1: 6 -> 128 channels (doubled)
            nn.Conv1d(
                in_channels=6,
                out_channels=128,
                kernel_size=7, stride=2, padding=3),  # Larger kernel
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Light dropout early
            
            # 1.2. Conv layer 2: 128 -> 256 channels (doubled)
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            # 1.3. Conv layer 3: 256 -> 512 channels (doubled)
            nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            # 1.4. Additional conv layer for more depth (512 -> 512)
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=3, stride=1, padding=1),  # stride=1 maintains T/8
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Dropout(dropout)
        )
        
        # 2. Deeper GRU for temporal modeling
        # ------------------------------------
        self.gru = nn.GRU(
            input_size=512,  # Increased from 256
            hidden_size=hidden_size,  # 256 (doubled)
            num_layers=3,  # Increased from 2 for more depth
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 3. Enhanced Attention mechanism
        # --------------------------------
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # Larger intermediate layer
            nn.Tanh(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )
        
        # 4. Wrist embedding (MUCH LARGER)
        # ---------------------------------
        self.wrist_embed = nn.Embedding(
            num_embeddings=2,  # 0 = Left, 1 = Right
            embedding_dim=wrist_embed_dim  # Now 64 instead of 16
        )
        
        # 5. Movement type embedding (NEW)
        # --------------------------------
        self.movement_embed = nn.Embedding(
            num_embeddings=num_movements,  # 11 movements
            embedding_dim=movement_embed_dim  # 128-dim representation
        )
        
        # 6. Enhanced Classifier with more layers
        # ----------------------------------------
        # Input: (attended + mean + max) = 3 * (hidden_size*2) = hidden_size*6
        #        + wrist_embed_dim + movement_embed_dim
        classifier_input_dim = hidden_size * 6 + wrist_embed_dim + movement_embed_dim
        
        self.classifier = nn.Sequential(
            # First hidden layer
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second hidden layer
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Third hidden layer
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            # Output layer
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, wrist, movement, lengths=None):
        """
        Args:
            x: [B, T, C=6] (T can be 1024 or 2048)
            wrist: [B] or [B,1] integer indices (0 or 1)
            movement: [B] or [B,1] integer indices (0-10 for 11 movements)
            lengths: [B] original sequence lengths before padding (optional)
        
        Returns:
            logits: [B, num_classes]
        """
        # 1. Permute for CNN: [B, T, 6] -> [B, 6, T]
        x = x.permute(0, 2, 1)
        
        # 2. CNN: extract local motion features
        # ---------------------------------------
        features = self.cnn(x)  # [B, 512, T'], where T' ≈ T/8
        features = features.permute(0, 2, 1)  # [B, T', 512]
        
        # 3. GRU: capture temporal dependencies
        # --------------------------------------
        if lengths is not None:
            # 3.1. Convert lengths to match reduced temporal dimension (after CNN stride)
            downsample_factor = 8  # because of 3 stride-2 conv layers
            reduced_lengths = torch.clamp((lengths.float() / downsample_factor).long(), min=1)
            
            # 3.2. Pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                features, reduced_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            gru_out_packed, _ = self.gru(packed)
            
            # 3.3. Pad back to tensor form
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out_packed, batch_first=True
            )  # [B, T', hidden_size*2]
        else:
            gru_out, _ = self.gru(features)  # [B, T', hidden_size*2]
            reduced_lengths = torch.full(
                (gru_out.size(0),), gru_out.size(1),
                dtype=torch.long, device=gru_out.device
            )
        
        # 4. Attention-weighted pooling
        # -----------------------------------
        attn_logits = self.attention(gru_out).squeeze(-1)  # [B, T']
        
        # 4.1. Mask padded timesteps before attention
        mask = torch.arange(gru_out.size(1), device=gru_out.device)[None, :] < reduced_lengths[:, None]
        attn_logits[~mask] = float('-inf')  # ignore padding regions

        # 4.2. Apply attention weights to GRU output
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)  # [B, T', 1]
        attended = (gru_out * attn_weights).sum(dim=1)  # [B, hidden_size*2]
        
        # 5. Global pooling
        pooled_mean = gru_out.mean(dim=1)  # [B, hidden_size*2]
        pooled_max = gru_out.max(dim=1)[0]  # [B, hidden_size*2]
        
        # 6. Combine temporal features
        time_features = torch.cat([
            attended,
            pooled_mean,
            pooled_max
        ], dim=1)  # [B, hidden_size*6]
        
        # 7. Wrist embedding
        wrist_embed = self.wrist_embed(wrist.long())  # [B, wrist_embed_dim]
        
        # 8. Movement embedding (NEW)
        movement_embed = self.movement_embed(movement.long())  # [B, movement_embed_dim]
        
        # 9. Concatenate all features
        combined = torch.cat([
            time_features,
            wrist_embed,
            movement_embed
        ], dim=1)  # [B, hidden_size*6 + wrist_embed_dim + movement_embed_dim]
        
        # 10. Final classification
        out = self.classifier(combined)  # [B, num_classes]
        
        return out