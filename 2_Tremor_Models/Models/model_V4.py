import torch
from torch import nn

class TremorNetGRU_V4(nn.Module):
    """
    Balanced CNN–GRU–Attention model for binary tremor classification (Healthy=0, PD=1), across multiple movements.
    
    Key Improvements over V3:
    - Reduced over-compression in CNN (keeps more temporal resolution)
    - Fewer GRU layers for stability
    - Added LayerNorm for smoother training with small batches
    - Proper softmax attention pooling instead of mean pooling
    - Normalized wrist and movement embeddings
    - Increased dropout for regularization
    - Cleaner classifier with better balance between capacity and generalization

    Args:
        num_movements (int): number of movement types (default 11).
        hidden_size (int): GRU hidden size per direction.
        wrist_embed_dim (int): wrist embedding dimension.
        movement_embed_dim (int): movement type embedding dimension.
        dropout (float): dropout probability.
        num_attention_heads (int): number of attention heads for multi-head attention.

    Input:
        x (Tensor): [B, T, 6] IMU signals (T can be 1024 or 2048).
        wrist (Tensor): [B] wrist side (0=Left, 1=Right).
        movement (Tensor): [B] movement type (0-10 for 11 movements).

    Output:
        Tensor: [B, num_classes] classification logits.
    """
    def __init__(self,
                num_movements: int = 11,
                hidden_size: int = 128, 
                wrist_embed_dim: int = 32, 
                movement_embed_dim: int = 64, 
                dropout: float = 0.3,
                num_attention_heads: int = 4, 
                ):
        super().__init__()
        
        # 1. CNN Feature Extractor (lighter + less downsampling)
        # --------------------------------------------------------
        # Input: [B, 6, T]
        # Output: [B, 256, T/4]
        self.cnn = nn.Sequential(
            # 1.1. Initial conv layer
            self._make_conv_block(6, 64, 7, 2, 3),   # [B, 64, T/2]
            nn.Dropout(dropout * 0.5),
            
            # 1.2. Second conv layer
            self._make_conv_block(64, 128, 5, 2, 2),  # [B, 128, T/4]
            nn.Dropout(dropout * 0.5),
            
            # 1.3. Third conv layer
            self._make_conv_block(128, 256, 3, 1, 1), # No stride → preserves T/4
            
            nn.Dropout(dropout)
        )
        
        # 2. GRU for temporal modeling
        # ------------------------------
        # 2-layer bidirectional GRU (smaller and more stable)
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 3. Multi-Head Attention mechanism
        # --------------------------------------------
        # 3.1. Multi-head self-attention to capture temporal dependencies across the GRU outputs
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,   # GRU is bidirectional, so output dim = hidden_size*2
            num_heads=num_attention_heads,  # Number of attention heads
            batch_first=True,              # Input shape: [B, T, C]
            dropout=dropout * 0.5         
        )

        # 3.2. Attention projection
        # Helps model learn a weighted combination of time steps
        self.attention_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),  # project back to same dim
            nn.Tanh(),                                   # non-linearity
            nn.Dropout(dropout * 0.5),                  # regularization
        )
        
        # 4. Wrist embedding (MUCH LARGER)
        # ---------------------------------
        self.wrist_embed = nn.Embedding(
            num_embeddings=2,  # 0 = Left, 1 = Right
            embedding_dim=wrist_embed_dim 
        )
        
        # 5. Movement type embedding (NEW)
        # --------------------------------
        self.movement_embed = nn.Embedding(
            num_embeddings=num_movements,  # 11 movements
            embedding_dim=movement_embed_dim 
        )
        
        # 6. Classifier 
        # ---------------
        # Input: (attended + mean + max) = 3 * (hidden_size*2) = hidden_size*6
        #        + wrist_embed_dim + movement_embed_dim
        classifier_input_dim = hidden_size * 6 + wrist_embed_dim + movement_embed_dim
        
        self.classifier = nn.Sequential(
            # First hidden layer
            nn.Linear(classifier_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second hidden layer
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(128, 1) # single logit for binary classification
        )
        
    # Helper CNN funciton
    # ---------------------
    def _make_conv_block(self, in_c, out_c, kernel, stride, padding):
        """
        Modular conv block creation for consistent CNN architecture
        
        Creates a convolution block with:
            - Conv1d layer with specified parameters
            - Batch normalization for stable training
            - ReLU activation for non-linearity
            - Dropout for regularization
        
        Args:
            in_c (int): Input channels
            out_c (int): Output channels  
            kernel (int): Kernel size
            stride (int): Stride length
            padding (int): Padding size
            
        Returns:
            nn.Sequential: Complete conv block
        """
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel, stride, padding),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, wrist, movement):
        """
        Args:
            x: [B, T, C=6]
            wrist: [B] or [B,1] integer indices (0 or 1)
            movement: [B] or [B,1] integer indices (0-10 for 11 movements)
        
        Returns:
            logits: [B, num_classes]
        """
        # 1. Permute for CNN: [B, T, 6] -> [B, 6, T]
        x = x.permute(0, 2, 1)
        
        # 2. CNN: extract local motion features
        # --------------------------------------
        features = self.cnn(x)  # [B, 256, T'], where T' ≈ T/4
        features = features.permute(0, 2, 1)  # [B, T', 256]
        
        
        # 3. GRU: capture temporal dependencies
        # --------------------------------------
        gru_out, _ = self.gru(features)  # [B, T', hidden_size*2]
        gru_out = self.layer_norm(gru_out)   # stabilize activations
        
        # 4. Multi-Head Attention
        # -------------------------
        # 4.1. Apply multi-head self-attention on the GRU outputs
        # Query, Key, Value all come from GRU outputs
        attended_out, _ = self.multihead_attn(gru_out, gru_out, gru_out)  # [B, T', hidden_size*2]

        # 4.2. Project the attended outputs through a linear + non-linearity
        attended_out = self.attention_proj(attended_out)  # [B, T', hidden_size*2]

        # 4.3. Compute softmax attention weights over time
        attn_weights = torch.softmax(attended_out.mean(dim=-1, keepdim=True), dim=1)
        attended = torch.sum(attn_weights * attended_out, dim=1)  # [B, hidden_size*2]
        
    
        # 5. Global pooling
        pooled_mean = gru_out.mean(dim=1)  # [B, hidden_size*2]
        pooled_max = gru_out.max(dim=1)[0]  # [B, hidden_size*2]
        
        # 6. Combine temporal features
        time_features = torch.cat([
            attended,
            pooled_mean,
            pooled_max
        ], dim=1)  # [B, hidden_size*6]
        
        # 7. Wrist + movement embeddings (normalized)
        wrist_embed = F.normalize(self.wrist_embed(wrist.long()), dim=-1)           # [B, wrist_embed_dim]
        movement_embed = F.normalize(self.movement_embed(movement.long()), dim=-1)  # [B, wrist_embed_dim]
        
        # 8. Concatenate all features
        combined = torch.cat([
            time_features,
            wrist_embed,
            movement_embed
        ], dim=1)  # [B, hidden_size*6 + wrist_embed_dim + movement_embed_dim]
        
        # 9. Final classification
        out = self.classifier(combined)  # [B, num_classes]
        
        return out
